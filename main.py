import os
import sys
import time
import json
import re
import logging
import html as html_lib
import requests
from functools import wraps
from datetime import datetime, timedelta, timezone

# =========================================================
# 1. ENTERPRISE SETUP & LOGGING
# =========================================================
CACHE_DIR = "api_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(CACHE_DIR, "execution_logs.log")

logger = logging.getLogger("ZBET90_ENGINE")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ID = "@zBET90"

if not all([ODDS_API_KEY, GROQ_API_KEY, RAPIDAPI_KEY]):
    logger.critical("❌ FATAL: Missing API Keys in GitHub Secrets.")
    sys.exit(1)

# =========================================================
# 2. BULLETPROOF UTILS & CLEANERS
# =========================================================
def retry_request(max_retries=3, delay=2, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        wait_time = int(e.response.headers.get("Retry-After", current_delay))
                        logger.warning(f"⚠️ Rate Limit (429). Sleeping {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"⚠️ HTTP Error in {func.__name__}: {e}")
                        if attempt == max_retries - 1: return None
                except requests.exceptions.RequestException as e:
                    logger.error(f"⚠️ Connection Error in {func.__name__}: {e}. Retrying {attempt+1}/{max_retries}")
                    if attempt == max_retries - 1: return None
                time.sleep(current_delay)
                current_delay *= backoff
            return None
        return wrapper
    return decorator

def robust_json_extractor(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.error(f"❌ JSON Extraction Failed. Error: {e}")
    return None

def clean_team_name(name):
    """پاک‌سازی اسامی تیم‌ها از پرانتزها برای سرچ دقیق در سوفاسکور"""
    return re.sub(r'\s*\([^)]*\)', '', str(name)).strip()

# =========================================================
# 3. EXTERNAL API ADAPTERS
# =========================================================
@retry_request(max_retries=3)
def fetch_odds_api():
    now_utc = datetime.now(timezone.utc)
    end_window = now_utc + timedelta(hours=2)
    url = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "eu,us,uk,au", "markets": "h2h", "oddsFormat": "decimal"}
    
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    events = res.json()
    return [e for e in events if e.get("commence_time") and now_utc <= datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= end_window]

@retry_request(max_retries=2)
def get_match_id_via_search(home, away):
    clean_home = clean_team_name(home)
    clean_away = clean_team_name(away)
    query = f"{clean_home} {clean_away}"
    
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "sofascore.p.rapidapi.com"}
    res = requests.get("https://sofascore.p.rapidapi.com/search", headers=headers, params={"q": query, "page": "0"}, timeout=10)
    res.raise_for_status()
    
    for result in res.json().get("results", []):
        if result.get("type") == "event":
            return result.get("entity", {}).get("id")
    return None

@retry_request(max_retries=3)
def fetch_deep_stats(match_id):
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "sofascore.p.rapidapi.com"}
    data = {}
    endpoints = {
        "h2h": "https://sofascore.p.rapidapi.com/matches/get-h2h-events",
        "lineups": "https://sofascore.p.rapidapi.com/matches/get-lineups",
        "streaks": "https://sofascore.p.rapidapi.com/matches/get-team-streaks"
    }
    for key, url in endpoints.items():
        try:
            res = requests.get(url, headers=headers, params={"matchId": match_id}, timeout=10)
            if res.status_code == 200: data[key] = res.json()
            time.sleep(0.5)
        except Exception: pass
        
    if data:
        try:
            debug_filepath = os.path.join(LOG_DIR, f"debug_deep_stats_{match_id}.json")
            with open(debug_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 دیتای خام بازی {match_id} جهت بررسی شما در پوشه 'log' ذخیره شد.")
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره فایل دیباگ: {e}")
            
    return data

# =========================================================
# 4. CORE MATH ENGINE (WITH ODDS FILTER)
# =========================================================
def calculate_sharp_ev(bookmakers: list):
    sharp_odds, best_market_odds = {}, {}
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] == "h2h":
                for o in market.get("outcomes", []):
                    name, price = o["name"], o["price"]
                    if name not in best_market_odds or price > best_market_odds[name]["price"]:
                        best_market_odds[name] = {"price": price, "bookmaker": bm["title"]}
        if bm["key"] == "pinnacle":
            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for o in market.get("outcomes", []):
                        sharp_odds[o["name"]] = o["price"]

    if not sharp_odds: return []
    
    implied_sum = sum(1 / p for p in sharp_odds.values())
    opportunities = []
    for name, sharp_p in sharp_odds.items():
        true_prob = (1 / sharp_p) / implied_sum
        best_p = best_market_odds.get(name, {}).get("price", 0)
        bookie = best_market_odds.get(name, {}).get("bookmaker", "Unknown")
        
        if best_p > 0:
            ev = (true_prob * best_p) - 1
            # شرط دوگانه: ضریب بزرگتر مساوی 1.50 و حداقل لبه سود 1.5%
            if best_p >= 1.50 and ev > 0.015: 
                opportunities.append({"pick": name, "prob": true_prob, "odds": best_p, "bookmaker": bookie, "ev": ev})
    return opportunities

# =========================================================
# 5. AI ANALYSIS (UI META-DATA + LOGIC)
# =========================================================
@retry_request(max_retries=3)
def generate_ai_analysis(home, away, sport, pick, ev_edge, deep_stats):
    if not deep_stats:
        compressed_stats = "NO EXTERNAL DATA AVAILABLE. Rely solely on the mathematical edge and general team knowledge."
    else:
        compressed_stats = json.dumps(deep_stats, separators=(',', ':'))[:3500]
    
    system_prompt = """You are an Elite Quantitative Sports Analyst.
A mathematical Sharp-Market Engine has identified a profitable +EV bet.

[YOUR OBJECTIVE]
1. Write exactly 2 sentences of tactical logic based strictly on the provided 'Deep Stats' justifying the Pick. No math or odds talk.
2. Determine an appropriate sport emoji.
3. Determine the correct country flag emojis for the home and away teams. Use ⚽ if unknown or international club.
4. Assign a Risk Level (Low, Medium, or High).

[FORMAT]
Output ONLY a JSON object matching this exact schema:
{
  "sport_emoji": "🏀",
  "home_flag": "🇺🇸",
  "away_flag": "🇺🇸",
  "risk_level": "Medium",
  "logic": "Your 2 sentence logic here."
}"""

    user_prompt = f"""[MATCH CONTEXT]
Sport: {sport}
Home Team: {home}
Away Team: {away}
The Pick: {pick}

[DEEP STATS]
{compressed_stats}"""
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.15,
        "response_format": {"type": "json_object"}
    }
    
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json=payload, timeout=20)
    res.raise_for_status()
    
    raw_content = res.json()["choices"][0]["message"]["content"]
    parsed_json = robust_json_extractor(raw_content)
    
    # مقادیر پیش‌فرض در صورت خرابی خروجی هوش مصنوعی
    default_resp = {
        "sport_emoji": "🏆",
        "home_flag": "🏳️",
        "away_flag": "🏳️",
        "risk_level": "Medium",
        "logic": "This selection has been verified by our proprietary mathematical model."
    }
    
    if parsed_json:
        return {**default_resp, **parsed_json}
    return default_resp

# =========================================================
# 6. TELEGRAM BROADCAST
# =========================================================
@retry_request(max_retries=3)
def send_telegram(message_html):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message_html, "parse_mode": "HTML"}
    res = requests.post(url, json=payload, timeout=10)
    res.raise_for_status()
    return True

# =========================================================
# 7. MASTER PIPELINE
# =========================================================
def main():
    logger.info("🚀 STARTING ZBET90 ENTERPRISE ENGINE")
    
    events = fetch_odds_api() or []
    
    if not events:
        logger.info("🛑 هیچ مسابقه‌ای در پنجره زمانی ۲ ساعت آینده پیدا نشد.")
        return

    now_utc = datetime.now(timezone.utc)
    total_ev_found = 0
    logger.info(f"🔍 تعداد {len(events)} مسابقه جهت فیلتر ریاضی بررسی می‌شود...")
    
    for event in events:
        home, away, sport = event.get("home_team"), event.get("away_team"), event.get("sport_title")
        bookmakers = event.get("bookmakers", [])
        
        # محاسبه زمان باقی مانده (Countdown)
        try:
            match_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
            delta = match_time - now_utc
            minutes_left = int(delta.total_seconds() / 60)
            if minutes_left > 60:
                countdown_str = f"{minutes_left // 60}h {minutes_left % 60}m"
            elif minutes_left > 0:
                countdown_str = f"{minutes_left}m"
            else:
                countdown_str = "🟢 LIVE"
        except Exception:
            countdown_str = "N/A"

        opportunities = calculate_sharp_ev(bookmakers)
        for opp in opportunities:
            total_ev_found += 1
            pick, ev_pct, odds = opp['pick'], opp['ev'] * 100, opp['odds']
            logger.info(f"💎 MATH VERIFIED -> {home} vs {away} | Pick: {pick} | Odds: {odds} | Edge: +{ev_pct:.1f}%")
            
            match_id = get_match_id_via_search(home, away)
            deep_stats = fetch_deep_stats(match_id) if match_id else {}
            
            if not match_id:
                logger.warning(f"⚠️ Match ID not found for {home} vs {away}. Proceeding with Math Edge only.")

            logger.info("🤖 Analyzing stats & generating UI elements via Groq...")
            ai_data = generate_ai_analysis(home, away, sport, pick, opp['ev'], deep_stats)
            
            # مپ کردن رنگ ریسک
            risk_raw = str(ai_data.get('risk_level', 'Medium')).capitalize()
            risk_icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(risk_raw, "🟠")
            
            # فرمت‌بندی ایمن متن لاجیک برای تلگرام (Highlight کردن اعداد)
            raw_logic = str(ai_data.get('logic', '')).replace('<', '').replace('>', '')
            logic_escaped = html_lib.escape(raw_logic)
            
            html_msg = (
                f"{ai_data.get('sport_emoji', '🏆')} <b>Sport:</b> {html_lib.escape(str(sport))}\n\n"
                f"⚔️ <b>Match:</b> <b>{html_lib.escape(home)}</b> {ai_data.get('home_flag', '🏳️')} vs {ai_data.get('away_flag', '🏳️')} <b>{html_lib.escape(away)}</b>\n\n"
                f"⏳ <b>Starts in:</b> {countdown_str}\n\n"
                f"🎯 <b>Winner Pick:</b> <b>{html_lib.escape(pick)}</b> <code>[{odds}]</code>\n\n"
                f"🔥 <b>Risk Level:</b> {risk_icon} {risk_raw}\n\n"
                f"💡 <b>Logic:</b> {logic_escaped}\n\n"
                f"🆔 <b>Join:</b> {TELEGRAM_ID}\n"
            )
            
            if send_telegram(html_msg):
                logger.info(f"✅ Telegram Broadcast Success: {home} vs {away}")
            time.sleep(3)

    if total_ev_found == 0:
        logger.info("⚖️ بررسی تمام شد. هیچ شرطی با ضریب بالای 1.50 و حاشیه سود (+EV > 1.5%) پیدا نشد.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"❌ SYSTEM FAILURE: {str(e)}", exc_info=True)
        sys.exit(1)
