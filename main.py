import os
import sys
import time
import json
import re
import logging
import html as html_lib
import hashlib
import asyncio
import aiohttp
import requests
import pandas as pd
from io import StringIO
from groq import Groq
from functools import wraps
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# =========================================================
# 1. CONFIGURATION
# =========================================================
@dataclass
class Config:
    CACHE_DIR: Path = Path("api_cache")
    LOG_DIR: Path = Path("log")
    HISTORICAL_DIR: Path = Path("api_cache/historical")
    
    HISTORY_FILE: Path = Path("api_cache/sent_history.json")
    TEAM_ID_CACHE_FILE: Path = Path("api_cache/team_id_cache.json")
    MATCH_ID_CACHE_FILE: Path = Path("api_cache/match_id_cache.json")
    DAILY_STATS_CACHE_FILE: Path = Path("api_cache/daily_stats_cache.json")
    LOG_FILE: Path = Path("api_cache/execution_logs.log")

    MATCH_WINDOW_HOURS: float = 2.0
    TELEGRAM_SLEEP_BETWEEN: float = 3.0

    FOOTBALL_DATA_DAILY_LIMIT: int = 80
    ODDS_API_MARKETS: list = field(default_factory=lambda: ["h2h", "totals"])
    ODDS_API_REGIONS: str = "eu,us,uk,au"

    TTL_SENT_HISTORY: float = 48.0
    TTL_MATCH_ID: float = 24.0
    TTL_TEAM_FORM: float = 6.0
    TTL_H2H: float = 24.0

    H2H_MIN_ODDS: float = 1.50
    H2H_MIN_EV: float = 0.015
    TOTALS_MIN_ODDS: float = 1.60
    TOTALS_MIN_EV: float = 0.020
    MAX_REALISTIC_EV: float = 0.12

    MARKET_EXPECTED_OUTCOMES: dict = field(default_factory=lambda: {
        "h2h": {"min": 2, "max": 3},
        "totals": {"min": 2, "max": 2}
    })
    MAX_VALID_IMPLIED_SUM: float = 1.20
    MIN_VALID_IMPLIED_SUM: float = 0.80

    AI_MODEL_ANALYST: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    AI_MODEL_VALIDATOR: str = "openai/gpt-oss-20b"
    AI_MAX_TOKENS: int = 2048

    TELEGRAM_ID: str = "@zBET90"

    SHARP_BOOKMAKERS: list = field(default_factory=lambda: [
        "pinnacle", "betfair_ex_eu", "matchbook", "betfair_ex_uk"
    ])

CFG = Config()

# =========================================================
# 2. LOGGING SETUP
# =========================================================
CFG.CACHE_DIR.mkdir(exist_ok=True)
CFG.LOG_DIR.mkdir(exist_ok=True)
CFG.HISTORICAL_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("ZBET90_ENGINE")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(CFG.LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# در ابتدای اسکریپت
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    logger.info("--- DEBUG MODE ENABLED: Detailed logging active ---")
    
# =========================================================
# 3. API KEYS VALIDATION
# =========================================================
ODDS_API_KEY           = os.getenv("ODDS_API_KEY")
GROQ_API_KEY           = os.getenv("GROQ_API_KEY")
RAPIDAPI_KEY           = os.getenv("RAPIDAPI_KEY")
TELEGRAM_BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")
FOOTBALL_DATA_API_KEY  = os.getenv("FOOTBALL_DATA_API_KEY")

if not all([ODDS_API_KEY, GROQ_API_KEY, RAPIDAPI_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing critical API Keys. Ensure environment variables are set.")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_API_KEY, max_retries=3)

# =========================================================
# 4. NATIONALITY FLAGS & EMOJIS
# =========================================================
NATIONALITY_FLAGS: dict = {
    # Tennis ATP/WTA
    "bautista agut": "ES", "alcaraz": "ES", "nadal": "ES", "munar": "ES",
    "djokovic": "RS", "kecmanovic": "RS", "krajinovic": "RS",
    "sinner": "IT", "berrettini": "IT", "musetti": "IT", "sonego": "IT",
    "zverev": "DE", "struff": "DE", "koepfer": "DE", "altmaier": "DE",
    "tiafoe": "US", "fritz": "US", "paul": "US", "nakashima": "US",
    "sock": "US", "isner": "US", "spizzirri": "US", "korda": "US",
    "mmoh": "US", "eubanks": "US", "wolf": "US",
    "medvedev": "RU", "rublev": "RU", "khachanov": "RU", "karatsev": "RU",
    "tsitsipas": "GR", "ruud": "NO", "rune": "DK", "tauson": "DK",
    "hurkacz": "PL", "swiatek": "PL",
    "marcinko": "HR", "cilic": "HR", "gojo": "HR",
    "auger-aliassime": "CA", "shapovalov": "CA", "raonic": "CA", "pospisil": "CA",
    "kyrgios": "AU", "de minaur": "AU", "thompson": "AU", "kokkinakis": "AU",
    "lys": "DE", "sabalenka": "BY",
    "gauff": "US", "keys": "US", "pegula": "US", "collins": "US",
    "halep": "RO", "wozniacki": "DK", "kvitova": "CZ", "vondrousova": "CZ",
    "jabeur": "TN", "badosa": "ES", "muguruza": "ES",
    "dimitrov": "BG", "fucsovics": "HU",
    "norrie": "GB", "murray": "GB", "draper": "GB", "Edmund": "GB",
    "thiem": "AT", "ofner": "AT",
    "wawrinka": "CH", "federer": "CH",
    "monfils": "FR", "simon": "FR", "gasquet": "FR", "pouille": "FR",
    "davidovich fokina": "ES", "lopez": "ES", "carreno busta": "ES",
    "bublik": "KZ", "rybakina": "KZ",
    "basilashvili": "GE", "tabilo": "CL",
    "etcheverry": "AR", "cerundolo": "AR", "schwartzman": "AR",
    "monteiro": "BR", "seyboth wild": "BR", "auger": "CA",
    
    # Football Clubs
    "real madrid": "ES", "barcelona": "ES", "atletico": "ES", "sevilla": "ES",
    "bayern": "DE", "dortmund": "DE", "leipzig": "DE", "leverkusen": "DE",
    "manchester united": "GB", "manchester city": "GB", "liverpool": "GB",
    "arsenal": "GB", "chelsea": "GB", "tottenham": "GB", "newcastle": "GB",
    "juventus": "IT", "milan": "IT", "inter": "IT", "napoli": "IT", "roma": "IT",
    "psg": "FR", "marseille": "FR", "lyon": "FR", "monaco": "FR",
    "ajax": "NL", "psv": "NL", "feyenoord": "NL",
    "porto": "PT", "benfica": "PT", "sporting": "PT",
    
    # US Sports
    "lakers": "US", "celtics": "US", "warriors": "US", "bulls": "US",
    "yankees": "US", "dodgers": "US", "cubs": "US", "red sox": "US",
}

def _code_to_flag(code: str) -> str:
    """Converts a two-letter ISO country code to its corresponding Emoji flag."""
    code = code.upper().strip()
    if len(code) != 2:
        return "\U0001F3F3\uFE0F"
    offset = 0x1F1E6 - ord("A")
    return chr(ord(code[0]) + offset) + chr(ord(code[1]) + offset)

def get_flag_from_name(name: str) -> str:
    name_lower = name.lower()
    for keyword, code in NATIONALITY_FLAGS.items():
        if keyword in name_lower:
            return _code_to_flag(code)
    return "\U0001F3F3\uFE0F"

def validate_flag(flag: str, fallback_name: str) -> str:
    if not flag:
        return get_flag_from_name(fallback_name)
    
    stripped = flag.strip()
    placeholder_flags = ["\U0001F3F3\uFE0F", "\U0001F3C1", "\U0001F6A9", "", "🏁", "🏳️"]
    
    if stripped in placeholder_flags:
        return get_flag_from_name(fallback_name)
    return stripped

# =========================================================
# 5. CACHE MANAGEMENT
# =========================================================
class CacheManager:
    @staticmethod
    def load(filepath: Path) -> dict:
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Cache load error (%s): %s", filepath.name, e)
        return {}

    @staticmethod
    def save(filepath: Path, data: dict) -> None:
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Cache save error (%s): %s", filepath.name, e)

    @staticmethod
    def is_valid(cache: dict, key: str, ttl_hours: float) -> bool:
        if key not in cache:
            return False
            
        entry = cache[key]
        if not isinstance(entry, dict) or "timestamp" not in entry:
            return False
            
        try:
            cached_time = datetime.fromisoformat(entry["timestamp"])
            return datetime.now(timezone.utc) - cached_time < timedelta(hours=ttl_hours)
        except Exception:
            return False

    @staticmethod
    def set(cache: dict, key: str, value) -> dict:
        cache[key] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": value,
        }
        return cache

    @staticmethod
    def get(cache: dict, key: str):
        return cache.get(key, {}).get("data")

# =========================================================
# 6. SENT HISTORY TRACKER
# =========================================================
class SentHistory:
    def __init__(self):
        self.history = CacheManager.load(CFG.HISTORY_FILE)
        self._cleanup_old()

    def _cleanup_old(self):
        now = datetime.now(timezone.utc)
        to_delete = []
        for key, value in self.history.items():
            try:
                sent_at = value.get("sent_at", "2000-01-01T00:00:00+00:00")
                if now - datetime.fromisoformat(sent_at) > timedelta(hours=CFG.TTL_SENT_HISTORY):
                    to_delete.append(key)
            except Exception:
                to_delete.append(key)
                
        for key in to_delete:
            del self.history[key]

    @staticmethod
    def _make_key(home: str, away: str, market: str) -> str:
        raw_string = f"{home.lower()}|{away.lower()}|{market.lower()}"
        return hashlib.md5(raw_string.encode()).hexdigest()

    def was_sent(self, home: str, away: str, market: str) -> bool:
        return self._make_key(home, away, market) in self.history

    def mark_sent(self, home: str, away: str, pick: str, market: str) -> None:
        key = self._make_key(home, away, market)
        self.history[key] = {
            "match": f"{home} vs {away}",
            "pick": pick,
            "market": market,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        CacheManager.save(CFG.HISTORY_FILE, self.history)

# =========================================================
# 7. HISTORICAL DATA ENGINE (GITHUB INTEGRATION)
# =========================================================
class HistoricalDataEngine:
    """
    Downloads and parses historical CSVs from Github dynamically for GitHub Actions environments.
    Optimized memory footprint by dropping unused columns.
    """
    def __init__(self):
        self.atp_matches = None
        self.wta_matches = None
        # Recent years give the best indication of current form
        self.years_to_fetch = [2024, 2025, 2026] 

    def _download_github_csv(self, url: str, filepath: Path) -> bool:
        if filepath.exists():
            return True
            
        logger.info("[HISTORICAL] Fetching missing data from Github: %s", url.split('/')[-1])
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(res.text)
                return True
            else:
                logger.warning("[HISTORICAL] Received status %d for %s", res.status_code, url)
        except Exception as e:
            logger.warning("[HISTORICAL] Failed to download %s: %s", url, e)
        return False

    def sync_and_load_tennis(self):
        """Fetches and loads ATP and WTA match data locally."""
        atp_dfs = []
        wta_dfs = []
        
        # We only need specific columns to keep RAM usage low
        cols_to_use = ["surface", "winner_name", "loser_name"]
        
        for year in self.years_to_fetch:
            # ATP Sync
            atp_url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
            atp_path = CFG.HISTORICAL_DIR / f"atp_{year}.csv"
            
            if self._download_github_csv(atp_url, atp_path):
                try:
                    df = pd.read_csv(atp_path, usecols=cols_to_use)
                    atp_dfs.append(df)
                except Exception as e:
                    logger.error("Failed to parse ATP CSV %s: %s", atp_path, e)
            
            # WTA Sync
            wta_url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"
            wta_path = CFG.HISTORICAL_DIR / f"wta_{year}.csv"
            
            if self._download_github_csv(wta_url, wta_path):
                try:
                    df = pd.read_csv(wta_path, usecols=cols_to_use)
                    wta_dfs.append(df)
                except Exception as e:
                    logger.error("Failed to parse WTA CSV %s: %s", wta_path, e)

        # Concatenate dataframes
        if atp_dfs:
            self.atp_matches = pd.concat(atp_dfs, ignore_index=True)
            logger.info("[HISTORICAL] ATP Data loaded: %d matches", len(self.atp_matches))
            
        if wta_dfs:
            self.wta_matches = pd.concat(wta_dfs, ignore_index=True)
            logger.info("[HISTORICAL] WTA Data loaded: %d matches", len(self.wta_matches))

    def get_tennis_edge(self, player_a: str, player_b: str, is_wta: bool = False) -> dict:
        """Calculates win rates and H2H from historical data."""
        df = self.wta_matches if is_wta else self.atp_matches
        if df is None or df.empty:
            return {}

        def clean_name(n: str) -> str:
            # Match by last name primarily for robustness against API vs Github name mismatches
            return n.split()[-1].lower() 

        pa_clean = clean_name(player_a)
        pb_clean = clean_name(player_b)

        stats = {"player_a": {}, "player_b": {}, "h2h": {}}
        
        # Calculate individual win rates
        for p_clean, orig_name, key in [(pa_clean, player_a, "player_a"), (pb_clean, player_b, "player_b")]:
            wins = df[df['winner_name'].str.lower().str.contains(p_clean, na=False)]
            losses = df[df['loser_name'].str.lower().str.contains(p_clean, na=False)]
            
            total = len(wins) + len(losses)
            if total > 0:
                stats[key] = {
                    "name": orig_name,
                    "recent_win_rate": round(len(wins) / total, 2),
                    "matches_found": total
                }
        
        # Calculate Historical Head-to-Head
        h2h_a_wins = df[(df['winner_name'].str.lower().str.contains(pa_clean, na=False)) & 
                        (df['loser_name'].str.lower().str.contains(pb_clean, na=False))]
                        
        h2h_b_wins = df[(df['winner_name'].str.lower().str.contains(pb_clean, na=False)) & 
                        (df['loser_name'].str.lower().str.contains(pa_clean, na=False))]
        
        total_h2h = len(h2h_a_wins) + len(h2h_b_wins)
        if total_h2h > 0:
            stats["h2h"] = {
                "total": total_h2h,
                f"{player_a}_wins": len(h2h_a_wins),
                f"{player_b}_wins": len(h2h_b_wins)
            }
            logger.info("[HISTORICAL] Found H2H for %s vs %s: %d matches", player_a, player_b, total_h2h)

        return stats

# =========================================================
# 8. GENERAL UTILS & MATH ENGINE
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
                    status = e.response.status_code if e.response is not None else 0
                    if status == 429:
                        wait = int(e.response.headers.get("Retry-After", current_delay * 3))
                        logger.warning("Rate limit 429 in %s, sleeping %ds", func.__name__, wait)
                        time.sleep(wait)
                    elif status in [401, 403]:
                        logger.error("Auth error %d in %s", status, func.__name__)
                        return None
                    else:
                        logger.error("HTTP %d in %s: %s", status, func.__name__, e)
                        if attempt == max_retries - 1:
                            return None
                except requests.exceptions.Timeout:
                    logger.warning("Timeout in %s attempt %d/%d", func.__name__, attempt + 1, max_retries)
                    if attempt == max_retries - 1:
                        return None
                except requests.exceptions.RequestException as e:
                    logger.error("Request error in %s: %s", func.__name__, e)
                    if attempt == max_retries - 1:
                        return None
                
                time.sleep(current_delay)
                current_delay *= backoff
            return None
        return wrapper
    return decorator

def robust_json_extractor(raw_text: str) -> Optional[dict]:
    if not raw_text:
        return None
        
    clean = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE)
    clean = re.sub(r"<think>[\s\S]*", "", clean, flags=re.IGNORECASE).strip()
    
    try: 
        return json.loads(clean)
    except json.JSONDecodeError: 
        pass
        
    all_matches = list(re.finditer(r"\{[\s\S]*?\}", clean))
    for match in reversed(all_matches):
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict) and len(result) > 0:
                return result
        except json.JSONDecodeError:
            continue
            
    try:
        m = re.search(r"\{[\s\S]*\}", clean)
        if m: 
            return json.loads(m.group(0))
    except Exception: 
        pass
        
    return None

def clean_team_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(name)).strip()

def normalize_sport_key(sport_title: str) -> str:
    lower_title = sport_title.lower()
    if "tennis" in lower_title or "atp" in lower_title or "wta" in lower_title: 
        return "tennis"
        
    keywords = ["soccer", "football", "premier league", "la liga", "bundesliga", "serie a", "ligue 1", "champions league"]
    if any(kw in lower_title for kw in keywords):
        return "football"
        
    return "other"

def get_countdown_str(commence_time_str: str, now_utc: datetime) -> str:
    try:
        match_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        minutes_left = int((match_time - now_utc).total_seconds() / 60)
        
        if minutes_left > 60: 
            return f"{minutes_left // 60}h {minutes_left % 60}m"
        if minutes_left > 0: 
            return f"{minutes_left}m"
            
        return "LIVE"
    except Exception: 
        return "N/A"

def calculate_sharp_ev(markets_data: dict, bookmakers_raw: list) -> list:
    """Calculates the mathematical Edge against Sharp Bookmakers."""
    best_per_market: dict = {}
    
    for market_key, market_data_list in markets_data.items():
        sharp_odds = {}
        best_odds = {}
        has_real_sharp = False
        
        for entry in market_data_list:
            bk = entry.get("bookmaker_key", "")
            if bk in CFG.SHARP_BOOKMAKERS: 
                has_real_sharp = True
                
            for o in entry.get("outcomes", []):
                name = f"{o['name']} {o.get('point')}" if o.get('point') is not None else o['name']
                price = float(o["price"])
                
                if price <= 1.0:
                    continue
                    
                if bk in CFG.SHARP_BOOKMAKERS:
                    if name not in sharp_odds or price > sharp_odds.get(name, {}).get("price", 0):
                        sharp_odds[name] = {"price": price, "bookmaker": entry["bookmaker"]}
                        
                if name not in best_odds or price > best_odds.get(name, {}).get("price", 0):
                    best_odds[name] = {"price": price, "bookmaker": entry["bookmaker"]}

        if not sharp_odds and best_odds: 
            sharp_odds = dict(best_odds)
            
        if not sharp_odds: 
            continue
        
        try: 
            implied_sum = sum(1.0 / v["price"] for v in sharp_odds.values())
        except ZeroDivisionError: 
            continue
        
        if not (CFG.MIN_VALID_IMPLIED_SUM <= implied_sum <= CFG.MAX_VALID_IMPLIED_SUM): 
            continue

        min_odds = CFG.H2H_MIN_ODDS if market_key == "h2h" else CFG.TOTALS_MIN_ODDS
        min_ev = (CFG.H2H_MIN_EV if market_key == "h2h" else CFG.TOTALS_MIN_EV) * (1.0 if has_real_sharp else 2.0)

        best_opp = None
        for outcome_name, sharp_data in sharp_odds.items():
            true_prob = (1.0 / sharp_data["price"]) / implied_sum
            best_price = best_odds.get(outcome_name, {}).get("price", 0.0)
            
            if best_price <= 1.0: 
                continue
                
            ev = (true_prob * best_price) - 1.0

            if CFG.MAX_REALISTIC_EV >= ev > min_ev and best_price >= min_odds:
                opp = {
                    "pick": outcome_name, 
                    "market": market_key,
                    "market_label": "Winner" if market_key == "h2h" else "Over/Under",
                    "prob": round(true_prob, 4), 
                    "odds": round(best_price, 3),
                    "bookmaker": best_odds[outcome_name]["bookmaker"],
                    "ev": round(ev, 4), 
                    "edge_pct": round(ev * 100, 2),
                    "has_sharp_line": has_real_sharp,
                }
                
                if best_opp is None or opp["ev"] > best_opp["ev"]: 
                    best_opp = opp
                    
        if best_opp: 
            best_per_market[market_key] = best_opp

    all_opps = list(best_per_market.values())
    all_opps.sort(key=lambda x: x["ev"], reverse=True)
    return all_opps[:1]

# =========================================================
# 9. ASYNC ODDS API
# =========================================================
async def fetch_market_async(session: aiohttp.ClientSession, market: str, now_utc: datetime) -> list:
    end_window = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    params = {
        "apiKey": ODDS_API_KEY, 
        "regions": CFG.ODDS_API_REGIONS, 
        "markets": market, 
        "oddsFormat": "decimal", 
        "dateFormat": "iso"
    }
    
    try:
        async with session.get("https://api.the-odds-api.com/v4/sports/upcoming/odds", params=params, timeout=aiohttp.ClientTimeout(total=20)) as res:
            if res.status != 200: 
                return []
            events = await res.json()
            valid_events = []
            for e in events:
                try:
                    match_time = datetime.fromisoformat(e.get("commence_time", "").replace("Z", "+00:00"))
                    if now_utc <= match_time <= end_window:
                        valid_events.append(e)
                except Exception:
                    continue
            return valid_events
    except Exception as e:
        logger.error("Async fetch error market %s: %s", market, e)
        return []

async def fetch_all_odds_async() -> list:
    now_utc = datetime.now(timezone.utc)
    all_events: dict = {}
    
    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(
            *[fetch_market_async(session, m, now_utc) for m in CFG.ODDS_API_MARKETS], 
            return_exceptions=True
        )
    
    for market_events in results:
        if not isinstance(market_events, list): 
            continue
            
        for e in market_events:
            eid = e.get("id")
            if not eid: 
                continue
                
            if eid not in all_events: 
                all_events[eid] = {**e, "_markets_data": {}}
                
            for bm in e.get("bookmakers", []):
                for m in bm.get("markets", []):
                    mk = m["key"]
                    if mk not in all_events[eid]["_markets_data"]: 
                        all_events[eid]["_markets_data"][mk] = []
                        
                    all_events[eid]["_markets_data"][mk].append({
                        "bookmaker": bm["title"], 
                        "bookmaker_key": bm["key"], 
                        "outcomes": m.get("outcomes", []),
                    })
                    
    return list(all_events.values())

# =========================================================
# 10. STATS & FOOTBALL DATA ADAPTER
# =========================================================
class FootballDataAdapter:
    def __init__(self):
        self.headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY} if FOOTBALL_DATA_API_KEY else {}
        self.call_count = 0

    @retry_request(max_retries=2, delay=3)
    def get_team_id(self, team_name: str) -> Optional[int]:
        if not FOOTBALL_DATA_API_KEY or self.call_count > CFG.FOOTBALL_DATA_DAILY_LIMIT: 
            return None
            
        res = requests.get(
            f"https://api.football-data.org/v4/teams", 
            headers=self.headers, 
            params={"name": clean_team_name(team_name)}, 
            timeout=10
        )
        
        if res.ok and res.json().get("teams"):
            self.call_count += 1
            return res.json()["teams"][0]["id"]
        return None

async def search_sofascore_match_async(home: str, away: str) -> Optional[int]:
    query = f"{clean_team_name(home)} {clean_team_name(away)}"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY, 
        "x-rapidapi-host": "sofascore.p.rapidapi.com"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://sofascore.p.rapidapi.com/search", headers=headers, params={"q": query, "page": "0"}, timeout=10) as res:
                if res.status == 200:
                    data = await res.json()
                    for item in data.get("results", []):
                        if item.get("type") == "event" and item.get("entity", {}).get("id"): 
                            return item["entity"]["id"]
    except Exception: 
        pass
    return None

# =========================================================
# 11. ENTERPRISE AI ANALYSIS & SCORING
# =========================================================
def calculate_system_confidence(ev_edge: float, stats: dict, market: str) -> tuple[int, str]:
    """Pure Python calculation, zero LLM hallucination risk."""
    score = 50
    dq = stats.get("data_quality", "none")
    
    if dq == "high": 
        score += 15
    elif dq == "medium": 
        score += 8
    
    ev_pct = ev_edge * 100
    if ev_pct > 5.0: 
        score += 12
    elif ev_pct > 3.0: 
        score += 8
    elif ev_pct > 1.5: 
        score += 4
    
    # Evaluate Historical Edge from Github Data
    hist = stats.get("historical_data", {})
    if hist:
        pa_wr = hist.get("player_a", {}).get("recent_win_rate", 0)
        pb_wr = hist.get("player_b", {}).get("recent_win_rate", 0)
        
        if pa_wr >= 0.65 or pb_wr >= 0.65: 
            score += 5
            
        h2h_data = hist.get("h2h", {})
        if h2h_data and h2h_data.get("total", 0) > 0: 
            score += 4

    if market.lower() == "totals": 
        score += 3

    # Normalize Score
    score = max(50, min(93, int(score)))
    
    # Evaluate Risk Class
    if score >= 75: 
        risk = "Low"
    elif score >= 60: 
        risk = "Medium"
    else: 
        risk = "High"
        
    return score, risk

def call_groq_sdk(model: str, messages: list, temperature: float = 0.1) -> Optional[str]:
    kwargs = {
        "model": model, 
        "messages": messages, 
        "temperature": temperature, 
        "max_tokens": CFG.AI_MAX_TOKENS
    }
    
    if "qwen" not in model.lower() and "gpt-oss" not in model.lower(): 
        kwargs["response_format"] = {"type": "json_object"}
        
    try:
        res = groq_client.chat.completions.create(**kwargs)
        return res.choices[0].message.content
    except Exception as e:
        logger.error("Groq SDK error model=%s: %s", model, e)
        return None

def generate_dual_ai_analysis(home: str, away: str, sport: str, pick: str, market: str, ev_edge: float, stats: dict) -> dict:
    calc_conf, calc_risk = calculate_system_confidence(ev_edge, stats, market)
    
    default_response = {
        "sport_emoji": "\U0001F3C6", 
        "home_flag": get_flag_from_name(home), 
        "away_flag": get_flag_from_name(away),
        "risk_level": calc_risk, 
        "confidence": calc_conf,
        "logic": "The sharp market indicates significant value on this selection based on underlying metrics.",
    }

    # Format stats for the Prompt
    stats_str = json.dumps(stats, indent=2)[:1500]

    sys_analyst = (
        "You are an elite sports betting analyst for a premium syndicate.\n"
        "Write EXACTLY two punchy, insightful sentences justifying the given pick.\n"
        "RULES:\n"
        "- NEVER mention 'Expected Value', 'EV', 'data quality', 'points', or internal models.\n"
        "- Use the provided statistics (like historical win rates or H2H) to highlight dominance.\n"
        "- If stats are sparse, focus on the sharp market movement and value.\n"
        "- Determine the EXACT country flag emoji for home_flag and away_flag.\n"
        "OUTPUT JSON ONLY: {\"logic\": \"your 2 sentence analysis\", \"sport_emoji\": \"...\", \"home_flag\": \"...\", \"away_flag\": \"...\"}"
    )
    
    u1 = f"MATCH: {home} vs {away}\nSPORT: {sport}\nPICK: {pick} [{market}]\n\nSTATS:\n{stats_str}\n\nOUTPUT JSON ONLY:"

    analysis_1 = None
    try:
        raw1 = call_groq_sdk(CFG.AI_MODEL_ANALYST, [{"role": "system", "content": sys_analyst}, {"role": "user", "content": u1}], temperature=0.4)
        analysis_1 = robust_json_extractor(raw1)
    except Exception as e: 
        logger.warning("Model Analyst failed: %s", e)

    initial_logic = (analysis_1 or {}).get("logic", default_response["logic"])
    
    sys_editor = (
        "You are the Chief Editor for a high-end sports betting platform.\n"
        "Review the provided analysis. If it sounds robotic, mentions internal models, points, or 'EV', REWRITE it.\n"
        "It must sound like a professional sports pundit giving a confident tip. Max 3 sentences.\n"
        "OUTPUT JSON ONLY: {\"validated_logic\": \"...\"}"
    )
    
    try:
        raw2 = call_groq_sdk(CFG.AI_MODEL_VALIDATOR, [{"role": "system", "content": sys_editor}, {"role": "user", "content": f"DRAFT: {initial_logic}\nPICK: {pick}\nOUTPUT JSON ONLY:"}], temperature=0.2)
        analysis_2 = robust_json_extractor(raw2)
        if analysis_2 and analysis_2.get("validated_logic"): 
            initial_logic = analysis_2["validated_logic"]
    except Exception: 
        pass

    result = dict(default_response)
    if analysis_1:
        result["sport_emoji"] = analysis_1.get("sport_emoji", result["sport_emoji"])
        result["home_flag"] = validate_flag(analysis_1.get("home_flag", ""), home)
        result["away_flag"] = validate_flag(analysis_1.get("away_flag", ""), away)
    
    safe_logic = str(initial_logic).strip()
    result["logic"] = safe_logic[:597] + "..." if len(safe_logic) > 600 else safe_logic
    
    return result

# =========================================================
# 12. TELEGRAM INTEGRATION & CHUNKING
# =========================================================
def send_telegram(message_html: str) -> bool:
    """
    Sends a message to Telegram, implementing strict chunking 
    for texts longer than the 4096 character limit.
    """
    MAX_LEN = 4000  # Built-in buffer below 4096
    chunks = []
    
    if len(message_html) <= MAX_LEN: 
        chunks.append(message_html)
    else:
        lines = message_html.split('\n')
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) + 1 > MAX_LEN:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else: 
                current_chunk += line + "\n"
                
        if current_chunk: 
            chunks.append(current_chunk.strip())

    success = True
    for chunk in chunks:
        res = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
            json={
                "chat_id": TELEGRAM_CHAT_ID, 
                "text": chunk, 
                "parse_mode": "HTML", 
                "disable_web_page_preview": True
            }, 
            timeout=10
        )
        if not res.ok:
            logger.error("Telegram API Error: %s", res.text)
            success = False
            
    return success

# =========================================================
# 13. MAIN PIPELINE
# =========================================================
async def async_main():
    logger.info("=" * 60)
    logger.info("ZBET90 ENTERPRISE ENGINE v3.0 (w/ GitHub Historical) STARTING")
    logger.info("=" * 60)

    sent_history = SentHistory()
    now_utc = datetime.now(timezone.utc)
    
    # Init Historical Engine (GitHub Actions Friendly)
    historical_engine = HistoricalDataEngine()
    logger.info("[HISTORICAL] Syncing Github Data...")
    historical_engine.sync_and_load_tennis()

    logger.info("Fetching events (async)...")
    events = await fetch_all_odds_async()
    
    if not events:
        logger.info("No events found in the 2-hour window.")
        return

    logger.info("Analyzing %d events...", len(events))
    total_sent = 0

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        sport = event.get("sport_title", "Unknown")
        sport_key = normalize_sport_key(sport)
        
        if not home or not away: 
            continue

        opportunities = calculate_sharp_ev(event.get("_markets_data", {}), event.get("bookmakers", []))
        if not opportunities: 
            continue
            
        opp = opportunities[0]

        if sent_history.was_sent(home, away, opp["market"]): 
            continue

        stats = {"data_quality": "medium"} 
        
        # Inject Historical Data if Tennis
        if sport_key == "tennis":
            is_wta = "wta" in sport.lower()
            hist_stats = historical_engine.get_tennis_edge(home, away, is_wta)
            
            if hist_stats and (hist_stats.get("player_a") or hist_stats.get("player_b")):
                stats["historical_data"] = hist_stats
                stats["data_quality"] = "high"
                logger.info("[HISTORICAL] Injected Github Data for %s vs %s", home, away)

        ai_data = generate_dual_ai_analysis(home, away, sport, opp["pick"], opp["market"], opp["ev"], stats)
        
        # Format Msg
        conf_icon = "\U0001F525" if ai_data["confidence"] >= 75 else ("\U00002705" if ai_data["confidence"] >= 65 else "\U000026A1")
        risk_icon = {"Low": "\U0001F7E2", "Medium": "\U0001F7E0", "High": "\U0001F534"}.get(ai_data["risk_level"], "\U0001F7E0")
        
        msg = (
            f"{ai_data.get('sport_emoji', '🏆')} <b>{html_lib.escape(sport)}</b>\n\n"
            f"⚔️ <b>{html_lib.escape(home)}</b> {ai_data.get('home_flag', '🏳️')}  <b>vs</b>  {ai_data.get('away_flag', '🏳️')} <b>{html_lib.escape(away)}</b>\n\n"
            f"⏳ <b>Starts in:</b> {get_countdown_str(event.get('commence_time', ''), now_utc)}\n\n"
            f"🎯 <b>Pick [{opp['market_label']}]:</b> <b>{html_lib.escape(opp['pick'])}</b> @ <code>{opp['odds']}</code>\n\n"
            f"{risk_icon} <b>Risk:</b> {ai_data['risk_level']}  |  {conf_icon} <b>Confidence: {ai_data['confidence']}%</b>\n\n"
            f"💡 <b>Analysis:</b>\n<blockquote>{html_lib.escape(ai_data['logic'])}</blockquote>\n\n"
            f"🆔 <b>Channel:</b> {CFG.TELEGRAM_ID}"
        )

        if send_telegram(msg):
            sent_history.mark_sent(home, away, opp["pick"], opp["market"])
            total_sent += 1
            logger.info("Sent: %s vs %s | %s", home, away, opp["pick"])
        else: 
            logger.error("Telegram failed: %s vs %s", home, away)

        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    logger.info("=" * 60)
    if total_sent > 0:
        logger.info("Done! %d signal(s) sent.", total_sent)
    else:
        logger.info("No qualifying +EV opportunities found.")
    logger.info("=" * 60)

if __name__ == "__main__":
    try: 
        asyncio.run(async_main())
    except Exception as e:
        logger.critical("SYSTEM FAILURE: %s", str(e), exc_info=True)
        sys.exit(1)
