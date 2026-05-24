import os
import time
import json
import re
import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import sqlite3

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")  # api-football.com

if not all([ODDS_API_KEY, GROQ_API_KEY]):
    raise ValueError("❌ CRITICAL: API Keys missing!")

# ---------------------------------------------------------
# 2. DATABASE SETUP (Track predictions & results)
# ---------------------------------------------------------
def init_database():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  sport TEXT,
                  home_team TEXT,
                  away_team TEXT,
                  winner_pick TEXT,
                  winner_odds REAL,
                  goals_pick TEXT,
                  goals_odds REAL,
                  risk_level TEXT,
                  result TEXT DEFAULT 'Pending',
                  profit REAL DEFAULT 0)''')
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# 3. ENHANCED DATA COLLECTION
# ---------------------------------------------------------
def get_team_stats(team_name: str, sport: str) -> Dict:
    """جمع‌آوری آمار واقعی تیم (فرم، گل‌ها، نتایج)"""
    if sport != "Soccer" or not FOOTBALL_API_KEY:
        return {}
    
    # Example: API-Football integration
    url = "https://v3.football.api-sports.io/teams"
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    params = {"search": team_name}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # پردازش داده‌ها...
            return data.get('response', [{}])[0]
    except Exception as e:
        logger.warning(f"Could not fetch stats for {team_name}: {e}")
    
    return {}

def get_next_2_hours_events() -> List[Dict]:
    """دریافت رویدادها + غنی‌سازی با داده‌های آماری"""
    now_utc = datetime.now(timezone.utc)
    end_window_utc = now_utc + timedelta(hours=2)
    
    logger.info(f"🔍 Scanning matches: {now_utc.strftime('%H:%M')} - {end_window_utc.strftime('%H:%M')} UTC")
    
    url = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals,spreads",  # اضافه کردن spreads
        "oddsFormat": "decimal"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            all_events = response.json()
            break
        except Exception as e:
            logger.error(f"Odds API error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return []
            time.sleep(3)
    
    filtered_events = []
    
    for event in all_events:
        commence_time_str = event.get("commence_time")
        if not commence_time_str: 
            continue
            
        match_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        
        if now_utc <= match_time <= end_window_utc:
            # 🔥 Enrich with real stats
            sport = event.get("sport_title", "Unknown")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            
            event['home_stats'] = get_team_stats(home, sport)
            event['away_stats'] = get_team_stats(away, sport)
            
            filtered_events.append(event)
    
    logger.info(f"✅ Found {len(filtered_events)} matches starting soon")
    return filtered_events

# ---------------------------------------------------------
# 4. ADVANCED AI ANALYSIS (IMPROVED PROMPT)
# ---------------------------------------------------------
def analyze_with_groq(events: List[Dict]) -> List[Dict]:
    if not events: 
        return []

    prompt_data = "=== MATCH DATA & STATISTICS ===\n\n"
    
    for i, event in enumerate(events, 1):
        sport = event.get("sport_title", "Unknown")
        home = event.get("home_team", "Unknown")
        away = event.get("away_team", "Unknown")
        
        bookmakers = event.get("bookmakers", [])
        if not bookmakers: 
            continue
        
        # استخراج اددهای مختلف
        h2h_odds = {}
        totals_odds = {}
        
        for market in bookmakers[0].get("markets", []):
            if market["key"] == "h2h":
                for outcome in market.get("outcomes", []):
                    h2h_odds[outcome['name']] = outcome['price']
            elif market["key"] == "totals":
                for outcome in market.get("outcomes", []):
                    key = f"{outcome['name']} {outcome.get('point', '')}"
                    totals_odds[key] = outcome['price']
        
        # محاسبه Implied Probability
        total_prob = sum(1/odd for odd in h2h_odds.values()) if h2h_odds else 0
        margin = (total_prob - 1) * 100 if total_prob > 1 else 0
        
        # آمار تیم‌ها (اگر موجود باشد)
        home_stats = event.get('home_stats', {})
        away_stats = event.get('away_stats', {})
        
        prompt_data += f"""
📊 MATCH #{i}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 Sport: {sport}
⚔️  Teams: {home} (H) vs {away} (A)
🕒 Kickoff: {event.get('commence_time', 'N/A')}

💰 ODDS ANALYSIS:
   Match Winner (H2H): {json.dumps(h2h_odds, indent=6)}
   Total Goals: {json.dumps(totals_odds, indent=6)}
   Bookmaker Margin: {margin:.2f}%

📈 TEAM STATISTICS:
   {home}: {json.dumps(home_stats, indent=6) if home_stats else 'Limited data'}
   {away}: {json.dumps(away_stats, indent=6) if away_stats else 'Limited data'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    prompt_instructions = """
🎯 [SYSTEM ROLE]: Elite Quantitative Sports Betting Analyst with 15+ years experience

📋 [MISSION]: 
Analyze matches using:
1. **Odds Value Detection**: Find +EV opportunities where bookmaker odds > true probability
2. **Statistical Edge**: Use team stats, form, and historical patterns
3. **Risk Assessment**: Classify bets as Low/Medium/High risk
4. **Contrarian Thinking**: Identify public bias in odds

⚠️ [CRITICAL RULES]:
- Only pick bets with genuine statistical edge (minimum 5% +EV)
- Be SELECTIVE - if no value exists, skip the match
- Provide SPECIFIC reasoning with numbers
- Output MUST be valid JSON array (no markdown)

📊 [ANALYSIS FRAMEWORK]:
For each match evaluate:
✓ Implied probability vs true probability
✓ Team form & motivation
✓ H2H history & tactical matchup
✓ Bookmaker margin analysis
✓ Public betting bias

🎯 [JSON OUTPUT FORMAT]:
[
  {
    "sport": "Sport Name",
    "home_team": "Home Team",
    "away_team": "Away Team",
    "winner_pick": "Team Name or Draw",
    "winner_odds": 2.15,
    "winner_probability": "46.5% (True) vs 43.2% (Implied)",
    "winner_edge": "+7.6% EV",
    "goals_pick": "Over 2.5",
    "goals_odds": 1.95,
    "goals_probability": "51.3% (True) vs 47.6% (Implied)",
    "goals_edge": "+7.8% EV",
    "logic": "Tactical analysis: [specific reasoning]. Value explanation: [why odds are mispriced]. Statistical edge: [concrete numbers].",
    "risk_level": "Low/Medium/High",
    "confidence": "1-10 scale",
    "key_factors": ["Factor 1", "Factor 2", "Factor 3"]
  }
]

⚠️ If match has NO value, output empty array []
"""

    logger.info(f"🤖 Analyzing {len(events)} matches with Llama 3.3 70B...")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}", 
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": prompt_instructions},
            {"role": "user", "content": prompt_data}
        ],
        "temperature": 0.3,  # افزایش برای خلاقیت بیشتر
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        ai_raw_text = response.json()["choices"][0]["message"]["content"]
        
        logger.debug(f"Raw AI Response:\n{ai_raw_text}")
        
        # پاکسازی و استخراج JSON
        cleaned_text = ai_raw_text.replace("```json", "").replace("```", "").strip()
        
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', cleaned_text, re.DOTALL)
        if json_match:
            predictions = json.loads(json_match.group(0))
            logger.info(f"✅ AI generated {len(predictions)} high-value predictions")
            return predictions
        else:
            logger.warning("AI found no valuable bets (empty array)")
            return []
            
    except Exception as e:
        logger.error(f"❌ Groq API Error: {e}")
        return []

# ---------------------------------------------------------
# 5. ENHANCED TELEGRAM FORMATTER
# ---------------------------------------------------------
def format_and_send_to_telegram(predictions: List[Dict]):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials missing")
        return

    if not predictions:
        logger.info("No predictions to send")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # ارسال پیام هدر
    header_msg = f"""
🚨 <b>NEW BETTING ALERTS</b> 🚨
━━━━━━━━━━━━━━━━━━━━━━
📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
🎯 {len(predictions)} High-Value Picks Detected
━━━━━━━━━━━━━━━━━━━━━━
"""
    send_telegram_message(url, header_msg)
    time.sleep(2)
    
    for i, pick in enumerate(predictions, 1):
        # فرمت‌بندی پیشرفته
        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(pick.get('risk_level', 'Medium'), "⚪")
        
        html_message = f"""
{risk_emoji} <b>PICK #{i}</b> {risk_emoji}
━━━━━━━━━━━━━━━━━━━━━━
🏆 <b>{pick.get('sport')}</b>
⚔️  <b>{pick.get('home_team')}</b> vs <b>{pick.get('away_team')}</b>

🎯 <b>WINNER:</b> {pick.get('winner_pick')} @ <code>{pick.get('winner_odds')}</code>
   📊 {pick.get('winner_probability', 'N/A')}
   💰 Edge: <b>{pick.get('winner_edge', 'N/A')}</b>

⚽ <b>GOALS:</b> {pick.get('goals_pick')} @ <code>{pick.get('goals_odds')}</code>
   📊 {pick.get('goals_probability', 'N/A')}
   💰 Edge: <b>{pick.get('goals_edge', 'N/A')}</b>

💡 <b>ANALYSIS:</b>
{pick.get('logic', 'No analysis provided')}

🔑 <b>KEY FACTORS:</b>
{chr(10).join('   • ' + f for f in pick.get('key_factors', []))}

📈 <b>Confidence:</b> {pick.get('confidence', 'N/A')}/10
🎲 <b>Risk:</b> {pick.get('risk_level', 'N/A')}
━━━━━━━━━━━━━━━━━━━━━━
"""
        
        send_telegram_message(url, html_message)
        
        # ذخیره در دیتابیس
        save_prediction_to_db(pick)
        
        time.sleep(4)  # جلوگیری از Rate Limit

def send_telegram_message(url: str, message: str, max_retries: int = 3):
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            
            if resp.status_code == 200:
                return True
            elif resp.status_code == 429:
                retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                logger.warning(f"⏳ Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
            else:
                logger.error(f"❌ Telegram error [{resp.status_code}]: {resp.text}")
                return False
                
        except Exception as e:
            logger.error(f"Network error: {e}")
            time.sleep(2)
    
    return False

# ---------------------------------------------------------
# 6. DATABASE OPERATIONS
# ---------------------------------------------------------
def save_prediction_to_db(prediction: Dict):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    c.execute('''INSERT INTO predictions 
                 (timestamp, sport, home_team, away_team, winner_pick, winner_odds, 
                  goals_pick, goals_odds, risk_level)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().isoformat(),
               prediction.get('sport'),
               prediction.get('home_team'),
               prediction.get('away_team'),
               prediction.get('winner_pick'),
               prediction.get('winner_odds'),
               prediction.get('goals_pick'),
               prediction.get('goals_odds'),
               prediction.get('risk_level')))
    
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced Sports AI Predictor...")
    
    init_database()
    
    events = get_next_2_hours_events()
    
    if not events:
        logger.info("🛑 No upcoming matches. Exiting...")
    else:
        predictions = analyze_with_groq(events)
        
        if predictions:
            logger.info(f"✅ Generated {len(predictions)} predictions")
            format_and_send_to_telegram(predictions)
        else:
            logger.info("⚠️ No valuable bets found (AI was selective)")
    
    logger.info("🏁 Workflow completed")
