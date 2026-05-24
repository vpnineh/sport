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
# 2. BULLETPROOF UTILS & DECORATORS
# =========================================================
def retry_request(max_retries=3, delay=2, backoff=2):
    """
    نسخه ضدتانک: هیچ اروری باعث توقف برنامه نمی‌شود.
    اگر API قطع باشد، مقدار None برمی‌گرداند تا سیستم مسیر جایگزین را طی کند.
    """
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
                        if attempt == max_retries - 1: return None # عدم توقف برنامه
                except requests.exceptions.RequestException as e:
                    logger.error(f"⚠️ Connection Error in {func.__name__}: {e}. Retrying {attempt+1}/{max_retries}")
                    if attempt == max_retries - 1: return None # عدم توقف برنامه
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

# =========================================================
# 3. DATA CACHING LAYER
# =========================================================
def get_cached_data(filename):
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_to_cache(filename, data):
    filepath = os.path.join(CACHE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================
# 4. EXTERNAL API ADAPTERS
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
def get_sofascore_team_id(team_name):
    cache = get_cached_data("team_mapping.json") or {}
    if team_name in cache: return cache[team_name]
        
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "sofascore.p.rapidapi.com"}
    res = requests.get("https://sofascore.p.rapidapi.com/search", headers=headers, params={"q": team_name, "page": "0"}, timeout=10)
    res.raise_for_status()
    
    for result in res.json().get("results", []):
        if result.get("type") == "team":
            tid = result.get("entity", {}).get("id")
            cache[team_name] = tid
            save_to_cache("team_mapping.json", cache)
            time.sleep(1)
            return tid
    return None

@retry_request(max_retries=2)
def fetch_sofascore_schedule(date_str):
    cache_name = f"schedule_{date_str}.json"
    if cached := get_cached_data(cache_name): return cached
    
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "sofascore.p.rapidapi.com"}
    res = requests.get("https://sofascore.p.rapidapi.com/matches/v2/list-by-date", headers=headers, params={"category": "all", "date": date_str}, timeout=15)
    res.raise_for_status()
    data = res.json().get("events", [])
    save_to_cache(cache_name, data)
    return data

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
    return data

# =========================================================
# 5. CORE MATH ENGINE
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
            if ev > 0.015: # لبه سود 1.5 درصد
                opportunities.append({"pick": name, "prob": true_prob, "odds": best_p, "bookmaker": bookie, "ev": ev})
    return opportunities

# =========================================================
# 6. AI ANALYSIS (ENTERPRISE PROMPT ENGINEERING)
# =========================================================
@retry_request(max_retries=3)
def generate_ai_logic(home, away, pick, ev_edge, deep_stats):
    # اگر آمار جانبی ارور داد، سیستم جایگزین متنی می‌گذارد
    if not deep_stats:
        compressed_stats = "NO EXTERNAL DATA AVAILABLE. Rely solely on the mathematical edge and general team knowledge."
    else:
        compressed_stats = json.dumps(deep_stats, separators=(',', ':'))[:3500]
    
    system_prompt = """You are an Elite Quantitative Sports Analyst and AI Betting Assistant.
Your core expertise lies in translating deep statistical data into crisp, authoritative tactical narratives.

[YOUR OBJECTIVE]
Our mathematical Sharp-Market Engine has already identified a highly profitable betting opportunity (+EV) for the provided Pick. 
Your ONLY task is to write a compelling, data-driven justification for WHY this team might win or perform well.

[CRITICAL RULES]
1. ZERO MATH: Do not mention Expected Value (EV), odds, or the betting market. Focus ONLY on the sport/tactics.
2. DATA-DRIVEN: Base your analysis purely on the provided JSON stats. If no stats are provided, write a generic tactical reason based on the teams.
3. LENGTH: Write exactly 2 to 3 concise sentences.
4. TONE: Professional, authoritative, and analytical. No fluff.
5. FORMAT: You MUST output ONLY a valid JSON object matching this exact schema: {"logic": "your plain text analysis here"}."""

    user_prompt = f"""[MATCH CONTEXT]
Home Team: {home}
Away Team: {away}
The Winning Pick: {pick}

[DEEP STATS (JSON)]
{compressed_stats}

Based on the [DEEP STATS] above, write the tactical logic for '{pick}'."""
    
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
    
    if parsed_json and "logic" in parsed_json:
        return str(parsed_json["logic"]).strip()
    return "This selection has been verified by our proprietary mathematical model, identifying significant market value."

# =========================================================
# 7. TELEGRAM BROADCAST
# =========================================================
@retry_request(max_retries=3)
def send_telegram(message_html):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message_html, "parse_mode": "HTML"}
    res = requests.post(url, json=payload, timeout=10)
    res.raise_for_status()
    return True

# =========================================================
# 8. MASTER PIPELINE
# =========================================================
def main():
    logger.info("🚀 STARTING ZBET90 ENTERPRISE ENGINE")
    
    # اگر API قطع باشد، یک لیست خالی برمی‌گرداند تا کِرَش نکند
    events = fetch_odds_api() or []
    
    if not events:
        logger.info("🛑 هیچ مسابقه‌ای در پنجره زمانی ۲ ساعت آینده پیدا نشد یا API پاسخ نداد.")
        return

    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    total_ev_found = 0
    logger.info(f"🔍 تعداد {len(events)} مسابقه جهت فیلتر ریاضی بررسی می‌شود...")
    
    for event in events:
        home, away, sport = event.get("home_team"), event.get("away_team"), event.get("sport_title")
        bookmakers = event.get("bookmakers", [])
        
        opportunities = calculate_sharp_ev(bookmakers)
        for opp in opportunities:
            total_ev_found += 1
            pick, ev_pct, odds, true_prob = opp['pick'], opp['ev'] * 100, opp['odds'], opp['prob'] * 100
            logger.info(f"💎 MATH VERIFIED -> {home} vs {away} | Pick: {pick} | Edge: +{ev_pct:.1f}%")
            
            # Contextual Data Fetching (با مدیریت قطعی)
            hid = get_sofascore_team_id(home)
            aid = get_sofascore_team_id(away)
            deep_stats = {}
            
            if hid and aid:
                # اگر schedule ارور داد، یک لیست خالی می‌گذارد به جای کِرَش
                schedule = fetch_sofascore_schedule(today_str) or []
                for match in schedule:
                    if match.get("homeTeam", {}).get("id") == hid and match.get("awayTeam", {}).get("id") == aid:
                        deep_stats = fetch_deep_stats(match.get("id")) or {}
                        break
            else:
                logger.warning(f"⚠️ Team IDs not found for {home} vs {away}. Proceeding with Math Edge only.")

            # AI Logic Generation
            logger.info("🤖 Analyzing stats via Groq...")
            logic = generate_ai_logic(home, away, pick, opp['ev'], deep_stats)
            
            # HTML Escaping & Broadcasting
            html_msg = (
                f"🏆 <b>Sport:</b> {html_lib.escape(str(sport))}\n\n"
                f"⚔️ <b>Match:</b> <b>{html_lib.escape(home)}</b> vs <b>{html_lib.escape(away)}</b>\n\n"
                f"🎯 <b>VIP Pick:</b> <b>{html_lib.escape(pick)}</b>\n"
                f"⚖️ <b>Fair Probability:</b> {true_prob:.1f}%\n"
                f"📈 <b>Best Market Odds:</b> <code>{odds}</code> <i>({html_lib.escape(opp['bookmaker'])})</i>\n"
                f"📊 <b>Calculated +EV:</b> <b>+{ev_pct:.1f}% Edge</b> 🟢\n\n"
                f"💡 <b>Analysis:</b>\n"
                f"<blockquote expandable>{html_lib.escape(logic)}</blockquote>\n\n"
                f"🆔 <b>Join:</b> {TELEGRAM_ID}\n"
            )
            
            if send_telegram(html_msg):
                logger.info(f"✅ Telegram Broadcast Success: {home} vs {away}")
            time.sleep(3)

    if total_ev_found == 0:
        logger.info("⚖️ بررسی تمام شد. مارکت کاملاً کارا بود و هیچ شرط ارزشمندی (+EV > 1.5%) پیدا نشد.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"❌ SYSTEM FAILURE: {str(e)}", exc_info=True)
        sys.exit(1)
