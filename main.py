import os
import time
import json
import re
import logging
import requests
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
# تنظیم سیستم لاگ‌گیری برای رصد دقیق اتفاقات در سرور گیت‌هاب
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# دریافت کلیدها از محیط امن گیت‌هاب
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
    
    # فیلتر کردن بازی‌هایی که دقیقاً در ۲ ساعت آینده شروع می‌شوند
    for event in all_events:
        commence_time_str = event.get("commence_time")
        if not commence_time_str: continue
            
        match_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        
        if now_utc <= match_time <= end_window_utc:
            filtered_events.append(event)
            
    return filtered_events

# ---------------------------------------------------------
# 3. AI ANALYSIS (Groq Llama 3.3 forcing JSON output)
# ---------------------------------------------------------
def analyze_with_groq(events: list) -> list:
    if not events: return []

    prompt_data = "Raw Data:\n"
    
    for i, event in enumerate(events):
        sport = event.get("sport_title", "Unknown")
        home = event.get("home_team", "Unknown")
        away = event.get("away_team", "Unknown")
        
        bookmakers = event.get("bookmakers", [])
        if not bookmakers: continue
        
        h2h_str, totals_str = "N/A", "N/A"
        
        for market in bookmakers[0].get("markets", []):
            if market["key"] == "h2h":
                h2h_str = " | ".join([f"{o['name']}: {o['price']}" for o in market.get("outcomes", [])])
            elif market["key"] == "totals":
                totals_str = " | ".join([f"{o['name']} {o.get('point', '')}: {o['price']}" for o in market.get("outcomes", [])])
        
        prompt_data += f"Match {i+1}:\nSport: {sport}\nTeams: {home} vs {away}\nH2H: {h2h_str}\nTotals: {totals_str}\n---\n"

    # دستورات فوق‌سخت‌گیرانه برای دریافت فقط و فقط JSON
    prompt_instructions = """
    You are an elite sports betting AI. Analyze the matches based ONLY on the provided decimal odds.
    
    CRITICAL RULE: You MUST respond with a valid JSON array of objects and NOTHING ELSE. No markdown formatting like ```json, no greetings. Just the raw JSON array.
    
    JSON Template:
    [
      {
        "sport": "Sport Name",
        "home_team": "Home Team",
        "away_team": "Away Team",
        "winner_pick": "Chosen Team Name or N/A",
        "goals_pick": "Over/Under X.X or N/A",
        "logic": "One short sentence explaining the logic based on the decimal odds.",
        "risk_level": "Low/Medium/High"
      }
    ]
    """

    logger.info(f"Sending {len(events)} matches to Llama 3.3 70B...")
    url = "[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": prompt_instructions},
            {"role": "user", "content": prompt_data}
        ],
        "temperature": 0.1 # دمای پایین برای جلوگیری از توهم و خروجی دقیق‌تر
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        ai_raw_text = response.json()["choices"][0]["message"]["content"]
        
        # استخراج هوشمندانه JSON حتی اگر هوش مصنوعی متن اضافه تولید کرده باشد
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', ai_raw_text, re.DOTALL)
        if json_match:
            predictions = json.loads(json_match.group(0))
            return predictions
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

    url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_BOT_TOKEN}/sendMessage"
    
    for pick in predictions:
        # کدگذاری خودکار اعداد (ضرایب) برای نمایش متفاوت در تلگرام
        logic_text = pick.get('logic', '')
        logic_formatted = re.sub(r'(\d+\.\d+|\d+)', r'<code>\1</code>', logic_text)
        
        # قالب‌بندی نهایی HTML بسیار شیک و فاصله‌دار
        html_message = (
            f"🏆 <b>Sport:</b> {pick.get('sport')}\n\n"
            f"⚔️ <b>Match:</b> <b>{pick.get('home_team')}</b> vs <b>{pick.get('away_team')}</b>\n\n"
            f"🎯 <b>Winner Pick:</b> <b>{pick.get('winner_pick')}</b>\n\n"
            f"⚽ <b>Goals/Points Pick:</b> {pick.get('goals_pick')}\n\n"
            f"💡 <b>Logic:</b> {logic_formatted}\n\n"
            f"🔥 <b>Risk Level:</b> {pick.get('risk_level')}"
        )
        
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": html_message,
            "parse_mode": "HTML"
        }
        
        # سیستم ضدمرگ برای تلگرام (Retry Mechanism)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                
                if resp.status_code == 200:
                    logger.info(f"✅ Posted: {pick.get('home_team')} vs {pick.get('away_team')}")
                    break
                    
                elif resp.status_code == 429: # برخورد به اسپم‌گارد تلگرام
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
                
        # استراحت استاندارد بین پیام‌ها
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
        predictions = analyze_with_groq(events)
        
        if predictions:
            logger.info(f"Successfully generated {len(predictions)} VIP predictions. Broadcasting...")
            format_and_send_to_telegram(predictions)
        else:
            logger.warning("No valid predictions were generated by the AI.")
            
    logger.info("🏁 Workflow finished.")
