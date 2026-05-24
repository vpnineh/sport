import os
import time
import json
import re
import logging
import html as html_lib
import requests
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
Telegram_ID = "@zBET90"
if not ODDS_API_KEY or not GROQ_API_KEY:
    raise ValueError("CRITICAL ERROR: ODDS and GROQ API Keys are missing in GitHub Secrets!")

# ---------------------------------------------------------
# 2. DATA COLLECTION (The Odds API)
# ---------------------------------------------------------
def get_next_2_hours_events() -> list:
    now_utc = datetime.now(timezone.utc)
    end_window_utc = now_utc + timedelta(hours=2)
    
    logger.info(f"Scanning for matches between {now_utc.strftime('%H:%M')} and {end_window_utc.strftime('%H:%M')} UTC...")
    
    url = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        all_events = response.json()
    except Exception as e:
        logger.error(f"Failed to fetch data from Odds API: {e}")
        return []
        
    filtered_events = []
    
    for event in all_events:
        commence_time_str = event.get("commence_time")
        if not commence_time_str: continue
            
        match_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        
        if now_utc <= match_time <= end_window_utc:
            filtered_events.append(event)
            
    return filtered_events

# ---------------------------------------------------------
# 2.5. HELPER: EXTRACT BEST ODDS
# ---------------------------------------------------------
def extract_best_odds(bookmakers: list) -> tuple:
    best_h2h = {}     
    best_totals = {}  
    
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] == "h2h":
                for o in market.get("outcomes", []):
                    name, price = o["name"], o["price"]
                    if name not in best_h2h or price > best_h2h[name]:
                        best_h2h[name] = price
            elif market["key"] == "totals":
                for o in market.get("outcomes", []):
                    key = f"{o['name']} {o.get('point', '')}"
                    price = o["price"]
                    if key not in best_totals or price > best_totals[key]:
                        best_totals[key] = price
                        
    return best_h2h, best_totals

# ---------------------------------------------------------
# 3. AI ANALYSIS (Groq Llama 3.3)
# ---------------------------------------------------------
def analyze_with_groq(events: list) -> list:
    if not events: return []

    prompt_data = "Raw Data:\n"
    
    for i, event in enumerate(events):
        sport = event.get("sport_title", "Unknown")
        home = event.get("home_team", "Unknown")
        away = event.get("away_team", "Unknown")
        
        commence_time_str = event.get("commence_time")
        match_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        kickoff = match_time.strftime('%H:%M UTC')
        
        bookmakers = event.get("bookmakers", [])
        if not bookmakers: continue
        
        best_h2h, best_totals = extract_best_odds(bookmakers)
        
        h2h_str = " | ".join([f"{k}: {v}" for k, v in best_h2h.items()])
        totals_str = " | ".join([f"{k}: {v}" for k, v in best_totals.items()]) if best_totals else "N/A"
        market_depth = len(bookmakers)
        
        prompt_data += (
            f"Match {i+1}:\n"
            f"Sport: {sport}\n"
            f"Kickoff: {kickoff}\n"
            f"Commence Time: {commence_time_str}\n"
            f"Home: {home}\n"
            f"Away: {away}\n"
            f"Best H2H Odds: {h2h_str}\n"
            f"Best Totals: {totals_str}\n"
            f"Market Depth: {market_depth} bookmakers\n"
            f"---\n"
        )

    prompt_instructions = """
    [SYSTEM ROLE]: Elite Quantitative Sports Analyst & Value Bettor with deep knowledge of team tactical DNA, historical playstyles, and odds valuation.

    [OBJECTIVE]: Analyze the provided matches and their best available decimal odds. Identify matches where the odds offer genuine positive expected value (+EV). You are blind to today's live news — rely on historical team DNA, tactical matchups, and statistical patterns.

    [CRITICAL RULES]:
    1. Respond with ONLY a valid JSON array. No markdown, no ```json, no extra text before or after.
    2. If you cannot find genuine +EV in a match, EXCLUDE it from the response entirely.
    3. Do NOT force a pick on every match. Quality over quantity. Returning an empty array [] is perfectly acceptable if nothing has value.
    4. Only include matches where you have MODERATE to HIGH confidence.
    5. For sports with no draw possibility (tennis, basketball, baseball, etc.), NEVER pick "Draw".
    6. "Market Depth" indicates how many bookmakers offer this match — higher depth = more efficient odds = harder to find value.
    7. For football/soccer: use "goals_pick" as Over/Under X.X. For basketball: use Over/Under X.XX points. For others: use best alternative line or "N/A".
    8. "winner_odds" and "goals_odds" must be numbers (float), not strings. Use "N/A" only if the market doesn't exist.
    9. Include an appropriate "sport_emoji" (e.g., ⚽, 🏀, 🎾, 🏒).
    10. Include "home_flag" and "away_flag" with the exact country flag emoji for each team (e.g., 🇬🇧, 🇪🇸, 🇮🇹). If international, club, or unknown, use ⚽.
    11. "logic" must be clean plain text. Do NOT use HTML, markdown, or brackets inside the logic field.

    [JSON TEMPLATE]:
    [
      {
        "sport": "Sport Name",
        "sport_emoji": "⚽",
        "commence_time": "Exact string from Commence Time in prompt",
        "home_team": "Home Team Name",
        "home_flag": "🇬🇧",
        "away_team": "Away Team Name",
        "away_flag": "🇪🇸",
        "winner_pick": "Chosen Team Name (or Draw only for soccer/hockey)",
        "winner_odds": 1.85,
        "goals_pick": "Over 2.5",
        "goals_odds": 2.10,
        "logic": "1 sentence: tactical justification. 1 sentence: +EV value justification.",
        "risk_level": "Low" or "Medium" or "High"
      }
    ]
    """

    logger.info(f"Sending {len(events)} matches to Llama 3.3 70B...")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": prompt_instructions},
            {"role": "user", "content": prompt_data}
        ],
        "temperature": 0.15
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        ai_raw_text = response.json()["choices"][0]["message"]["content"]
        
        cleaned_text = ai_raw_text.replace("```json", "").replace("```", "").strip()
        
        json_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned_text, re.DOTALL)
        
        if json_match:
            predictions = json.loads(json_match.group(0))
            
            validated_predictions = []
            for p in predictions:
                if not p.get("winner_pick") or not p.get("home_team"):
                    continue
                
                try:
                    if p.get("winner_odds") and str(p["winner_odds"]).upper() != "N/A":
                        p["winner_odds"] = float(p["winner_odds"])
                    else:
                        p["winner_odds"] = "N/A"
                        
                    if p.get("goals_odds") and str(p["goals_odds"]).upper() != "N/A":
                        p["goals_odds"] = float(p["goals_odds"])
                    else:
                        p["goals_odds"] = "N/A"
                except (ValueError, TypeError):
                    p["winner_odds"] = "N/A"
                    p["goals_odds"] = "N/A"
                
                risk = str(p.get("risk_level", "Medium")).capitalize()
                if risk not in ["Low", "Medium", "High"]:
                    risk = "Medium"
                p["risk_level"] = risk
                
                # Ensure flags exist to prevent KeyError in Telegram formatter
                p.setdefault("home_flag", "⚽")
                p.setdefault("away_flag", "⚽")
                p.setdefault("sport_emoji", "🏆")
                p.setdefault("commence_time", "")
                
                validated_predictions.append(p)
                
            return validated_predictions
        else:
            logger.error("AI output did not contain a valid JSON array.")
            return []
            
    except Exception as e:
        logger.error(f"Groq API Error or Parsing Failed: {e}")
        return []

# ---------------------------------------------------------
# 4. PYTHON FORMATTER & SMART TELEGRAM BROADCASTER
# ---------------------------------------------------------
def format_and_send_to_telegram(predictions: list):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing. Skipping broadcast.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Safe time extraction function (Prevents crashes from malformed AI dates)
    def get_time(p):
        time_str = p.get('commence_time', '')
        if time_str:
            try:
                return datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                # If AI sends garbage date, push it to the end of the list
                return datetime.now(timezone.utc) + timedelta(hours=3)
        return datetime.now(timezone.utc) + timedelta(hours=3)
        
    predictions.sort(key=get_time)

    for pick in predictions:
        now_utc = datetime.now(timezone.utc) # Updated per loop for accurate countdown
        
        sport_emoji = pick.get('sport_emoji', '🏆')
        sport = html_lib.escape(str(pick.get('sport', '')))
        
        home = html_lib.escape(str(pick.get('home_team', '')))
        home_flag = pick.get('home_flag', '⚽')
        
        away = html_lib.escape(str(pick.get('away_team', '')))
        away_flag = pick.get('away_flag', '⚽')
        
        winner = html_lib.escape(str(pick.get('winner_pick', '')))
        goals_pick = html_lib.escape(str(pick.get('goals_pick', 'N/A')))
        
        winner_odds = pick.get('winner_odds', 'N/A')
        goals_odds = pick.get('goals_odds', 'N/A')
        
        # Risk Logic Mapping
        risk_raw = str(pick.get('risk_level', 'Medium')).capitalize()
        risk_icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(risk_raw, "🟠")
        risk = f"{risk_icon} {risk_raw}"
        
        # Clean and safely format the Logic text
        raw_logic = str(pick.get('logic', '')).replace('<', '').replace('>', '')
        logic_escaped = html_lib.escape(raw_logic)
        logic_formatted = re.sub(r'\b(\d+(?:\.\d+)?)\b', r'<code>\1</code>', logic_escaped)
        
        # Calculate time remaining
        match_time = get_time(pick)
        delta = match_time - now_utc
        minutes_left = int(delta.total_seconds() / 60)
        
        if minutes_left > 60:
            hours = minutes_left // 60
            mins = minutes_left % 60
            countdown_str = f"⏳ <b>Starts in:</b> {hours}h {mins}m"
        elif minutes_left > 0:
            countdown_str = f"⏳ <b>Starts in:</b> {minutes_left}m"
        else:
            countdown_str = f"⏳ <b>Status:</b> 🟢 LIVE / Started"
        
        html_message = (
            f"{sport_emoji} <b>Sport:</b> {sport}\n\n"
            f"⚔️ <b>Match:</b> <b>{home}</b> {home_flag} vs {away_flag} <b>{away}</b>\n\n"
            f"{countdown_str}\n\n"
            f"🎯 <b>Winner Pick:</b> <b>{winner}</b> <code>[{winner_odds}]</code>\n\n"
            f"🔥 <b>Risk Level:</b> {risk}\n\n"
            f"💡 <b>Logic:</b> {logic_formatted}\n\n"
            f"🆔 <b>Join:</b> {Telegram_ID}\n\n"
        )
        
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": html_message,
            "parse_mode": "HTML"
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                
                if resp.status_code == 200:
                    logger.info(f"✅ Posted: {home} vs {away}")
                    break
                    
                elif resp.status_code == 429: 
                    error_data = resp.json()
                    retry_after = error_data.get("parameters", {}).get("retry_after", 5)
                    logger.warning(f"⚠️ Telegram Flood Wait! Sleeping for {retry_after} seconds. (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_after)
                    
                else:
                    logger.error(f"❌ Telegram Error [{resp.status_code}]: {resp.text}")
                    break
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error connecting to Telegram: {e}")
                time.sleep(2)
                
        time.sleep(3.5)

# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Hourly Sports AI Predictor Workflow...")
    
    events = get_next_2_hours_events()
    
    if not events:
        logger.info("🛑 No matches starting soon. Workflow exiting cleanly.")
    else:
        logger.info(f"🔍 Found {len(events)} matches. Analyzing...")
        predictions = analyze_with_groq(events)
        
        if predictions:
            logger.info(f"Successfully generated {len(predictions)} VIP predictions. Broadcasting...")
            format_and_send_to_telegram(predictions)
        else:
            logger.warning("⚠️ No valid +EV predictions were generated by the AI for this batch.")
            
    logger.info("🏁 Workflow finished.")
