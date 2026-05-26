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
    MODELS_DIR: Path = Path("api_cache/models")
    HISTORICAL_DIR: Path = Path("api_cache/historical")
    
    HISTORY_FILE: Path = Path("api_cache/sent_history.json")
    TEAM_ID_CACHE_FILE: Path = Path("api_cache/team_id_cache.json")
    MATCH_ID_CACHE_FILE: Path = Path("api_cache/match_id_cache.json")
    DAILY_STATS_CACHE_FILE: Path = Path("api_cache/daily_stats_cache.json")
    LOG_FILE: Path = Path("api_cache/execution_logs.log")
    
    ELO_FOOTBALL_FILE: Path = Path("api_cache/models/elo_football.json")
    ELO_TENNIS_FILE: Path = Path("api_cache/models/elo_tennis.json")
    BOOTSTRAP_FLAG: Path = Path("api_cache/models/bootstrap_done.flag")

    MATCH_WINDOW_HOURS: float = 2.0
    RESULT_CHECK_HOURS: float = 3.0
    TELEGRAM_SLEEP_BETWEEN: float = 3.0

    FOOTBALL_DATA_DAILY_LIMIT: int = 80
    # OPTIMIZATION: Combine markets to save Odds API requests
    ODDS_API_MARKETS_STR: str = "h2h,totals"
    ODDS_API_REGIONS: str = "eu,us,uk,au"

    TTL_SENT_HISTORY: float = 72.0
    TTL_MATCH_ID: float = 24.0
    TTL_TEAM_FORM: float = 6.0
    TTL_H2H: float = 24.0

    H2H_MIN_ODDS: float = 1.50
    H2H_MIN_EV: float = 0.015
    TOTALS_MIN_ODDS: float = 1.60
    TOTALS_MIN_EV: float = 0.020
    MAX_REALISTIC_EV: float = 0.15

    MARKET_EXPECTED_OUTCOMES: dict = field(default_factory=lambda: {
        "h2h": {"min": 2, "max": 3},
        "totals": {"min": 2, "max": 2}
    })
    MAX_VALID_IMPLIED_SUM: float = 1.20
    MIN_VALID_IMPLIED_SUM: float = 0.80

    ELO_K_FACTOR_FOOTBALL: float = 32.0
    ELO_K_FACTOR_TENNIS: float = 40.0
    ELO_HOME_ADVANTAGE: float = 80.0
    ELO_DEFAULT: float = 1500.0

    AI_MODEL_ANALYST: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    AI_MODEL_VALIDATOR: str = "openai/gpt-oss-20b"
    AI_MAX_TOKENS: int = 1024

    TELEGRAM_ID: str = "@zBET90"

    SHARP_BOOKMAKERS: list = field(default_factory=lambda: [
        "pinnacle", "betfair_ex_eu", "matchbook", "betfair_ex_uk"
    ])

CFG = Config()

# =========================================================
# 2. LOGGING SETUP
# =========================================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

for d in [CFG.CACHE_DIR, CFG.LOG_DIR, CFG.MODELS_DIR, CFG.HISTORICAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ZBET90_ENGINE")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

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

def log_section(title: str):
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)

# =========================================================
# 3. API KEYS VALIDATION & ROBUST CLIENTS
# =========================================================
ODDS_API_KEY           = os.getenv("ODDS_API_KEY")
GROQ_API_KEY           = os.getenv("GROQ_API_KEY")
RAPIDAPI_KEY           = os.getenv("RAPIDAPI_KEY")
TELEGRAM_BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")
FOOTBALL_DATA_API_KEY  = os.getenv("FOOTBALL_DATA_API_KEY")
FORCE_BOOTSTRAP        = os.getenv("FORCE_BOOTSTRAP", "false").lower() == "true"

if not all([ODDS_API_KEY, GROQ_API_KEY, RAPIDAPI_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing critical API Keys. Ensure environment variables are set.")
    sys.exit(1)

timeout_settings = httpx.Timeout(25.0, connect=10.0)
groq_client = Groq(api_key=GROQ_API_KEY, max_retries=3, timeout=timeout_settings)

# =========================================================
# 4. NATIONALITY FLAGS
# =========================================================
NATIONALITY_FLAGS: dict = {
    "bautista agut": "ES", "alcaraz": "ES", "nadal": "ES",
    "djokovic": "RS", "sinner": "IT", "berrettini": "IT",
    "zverev": "DE", "tiafoe": "US", "fritz": "US", "gauff": "US",
    "medvedev": "RU", "rublev": "RU", "tsitsipas": "GR",
    "hurkacz": "PL", "swiatek": "PL", "kyrgios": "AU",
    "sabalenka": "BY", "kvitova": "CZ", "jabeur": "TN",
    "real madrid": "ES", "barcelona": "ES", "bayern": "DE",
    "manchester united": "GB", "manchester city": "GB", "liverpool": "GB",
    "arsenal": "GB", "chelsea": "GB", "juventus": "IT", "milan": "IT",
    "psg": "FR", "ajax": "NL", "porto": "PT",
    "lakers": "US", "celtics": "US", "warriors": "US",
}

def _code_to_flag(code: str) -> str:
    code = code.upper().strip()
    if len(code) != 2: return "\U0001F3F3\uFE0F"
    offset = 0x1F1E6 - ord("A")
    return chr(ord(code[0]) + offset) + chr(ord(code[1]) + offset)

def get_flag_from_name(name: str) -> str:
    name_lower = name.lower()
    for keyword, code in NATIONALITY_FLAGS.items():
        if keyword in name_lower:
            return _code_to_flag(code)
    return "\U0001F3F3\uFE0F"

def validate_flag(flag: str, fallback_name: str) -> str:
    if not flag: return get_flag_from_name(fallback_name)
    stripped = flag.strip()
    if stripped in ["\U0001F3F3\uFE0F", "\U0001F3C1", "\U0001F6A9", "", "🏁", "🏳️"]:
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
        except Exception:
            pass
        return {}

    @staticmethod
    def save(filepath: Path, data: dict) -> None:
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Cache save error: %s", e)

    @staticmethod
    def is_valid(cache: dict, key: str, ttl_hours: float) -> bool:
        if key not in cache: return False
        entry = cache[key]
        if not isinstance(entry, dict) or "timestamp" not in entry: return False
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
# 6. SENT HISTORY & PENDING RESULTS
# =========================================================
class SentHistory:
    def __init__(self):
        self.history = CacheManager.load(CFG.HISTORY_FILE)
        self._cleanup_old()

    def _cleanup_old(self):
        now = datetime.now(timezone.utc)
        to_delete = []
        for k, v in self.history.items():
            try:
                sent_at = v.get("sent_at", "2000-01-01T00:00:00+00:00")
                if now - datetime.fromisoformat(sent_at) > timedelta(hours=CFG.TTL_SENT_HISTORY):
                    to_delete.append(k)
            except Exception:
                to_delete.append(k)
        for k in to_delete:
            del self.history[k]

    @staticmethod
    def _make_key(home: str, away: str, market: str) -> str:
        raw = f"{home.lower()}|{away.lower()}|{market.lower()}"
        return hashlib.md5(raw.encode()).hexdigest()

    def was_sent(self, home: str, away: str, market: str) -> bool:
        return self._make_key(home, away, market) in self.history

    def mark_sent(self, home: str, away: str, pick: str, market: str, odds: float, commence_time: str) -> None:
        key = self._make_key(home, away, market)
        self.history[key] = {
            "match": f"{home} vs {away}",
            "home": home,
            "away": away,
            "pick": pick,
            "market": market,
            "odds": odds,
            "commence_time": commence_time,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "result_checked": False,
        }
        CacheManager.save(CFG.HISTORY_FILE, self.history)

    def get_pending_results(self) -> list:
        now = datetime.now(timezone.utc)
        pending = []
        for k, v in self.history.items():
            if v.get("result_checked"): continue
            try:
                ct = v.get("commence_time", "")
                match_time = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                elapsed = (now - match_time).total_seconds() / 3600
                if elapsed >= CFG.RESULT_CHECK_HOURS:
                    pending.append((k, v))
            except Exception:
                continue
        return pending

    def mark_result_checked(self, key: str, result: str, won: bool) -> None:
        if key in self.history:
            self.history[key]["result_checked"] = True
            self.history[key]["result"] = result
            self.history[key]["won"] = won
            CacheManager.save(CFG.HISTORY_FILE, self.history)

# =========================================================
# 7. ELO RATING SYSTEM & BOOTSTRAP
# =========================================================
class ELOSystem:
    def __init__(self, sport: str = "football"):
        self.sport = sport
        self.k = CFG.ELO_K_FACTOR_FOOTBALL if sport == "football" else CFG.ELO_K_FACTOR_TENNIS
        self.ratings: dict = {}
        self.match_count: dict = {}
        filepath = CFG.ELO_FOOTBALL_FILE if sport == "football" else CFG.ELO_TENNIS_FILE
        self._load(filepath)

    def _load(self, filepath: Path):
        data = CacheManager.load(filepath)
        if data:
            self.ratings = data.get("ratings", {})
            self.match_count = data.get("match_count", {})

    def save(self):
        filepath = CFG.ELO_FOOTBALL_FILE if self.sport == "football" else CFG.ELO_TENNIS_FILE
        CacheManager.save(filepath, {
            "ratings": self.ratings,
            "match_count": self.match_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def get_rating(self, name: str) -> float:
        return self.ratings.get(name.lower().strip(), CFG.ELO_DEFAULT)

    def expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def predict(self, home: str, away: str, apply_home_advantage: bool = True) -> dict:
        ra = self.get_rating(home)
        rb = self.get_rating(away)
        ra_adj = ra + (CFG.ELO_HOME_ADVANTAGE if apply_home_advantage else 0)
        home_prob = self.expected_score(ra_adj, rb)
        away_prob = 1.0 - home_prob
        draw_prob = 0.0
        
        if self.sport == "football":
            draw_factor = 0.22
            home_prob_adj = home_prob * (1 - draw_factor)
            away_prob_adj = away_prob * (1 - draw_factor)
            draw_prob = draw_factor
            total = home_prob_adj + away_prob_adj + draw_prob
            home_prob = home_prob_adj / total
            away_prob = away_prob_adj / total
            draw_prob = draw_prob / total
            
        return {
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
            "draw_prob": round(draw_prob, 4),
            "home_elo": round(ra, 1),
            "away_elo": round(rb, 1),
            "elo_diff": round(ra - rb, 1),
            "home_matches": self.match_count.get(home.lower().strip(), 0),
            "away_matches": self.match_count.get(away.lower().strip(), 0),
        }

# =========================================================
# 8. GENERAL UTILS & STRICT MATH ENGINE
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
                        time.sleep(wait)
                    elif status in [401, 403]:
                        return None
                    elif attempt == max_retries - 1:
                        return None
                except Exception:
                    if attempt == max_retries - 1: return None
                time.sleep(current_delay)
                current_delay *= backoff
            return None
        return wrapper
    return decorator

def robust_json_extractor(raw_text: str) -> Optional[dict]:
    if not raw_text: return None
    clean = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE)
    clean = re.sub(r"<think>[\s\S]*", "", clean, flags=re.IGNORECASE).strip()
    try: return json.loads(clean)
    except json.JSONDecodeError: pass
    
    all_matches = list(re.finditer(r"\{[\s\S]*?\}", clean))
    for match in reversed(all_matches):
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict) and len(result) > 0: return result
        except json.JSONDecodeError: continue
    return None

def clean_team_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(name)).strip()

def normalize_sport_key(sport_title: str) -> str:
    tl = sport_title.lower()
    if any(kw in tl for kw in ["tennis", "atp", "wta"]): return "tennis"
    if any(kw in tl for kw in ["soccer", "football", "premier league", "la liga", "bundesliga", "serie a"]): return "football"
    return "other"

def get_countdown_str(commence_time_str: str, now_utc: datetime) -> str:
    try:
        match_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        minutes_left = int((match_time - now_utc).total_seconds() / 60)
        if minutes_left > 60: return f"{minutes_left // 60}h {minutes_left % 60}m"
        if minutes_left > 0: return f"{minutes_left}m"
        return "LIVE"
    except Exception: return "N/A"

def get_market_label(market_key: str) -> str:
    mapping = {"h2h": "Match Winner", "totals": "Over/Under", "spreads": "Point Spread"}
    return mapping.get(market_key, market_key.replace("_", " ").title())

def calculate_combined_ev(markets_data: dict, elo_prediction: Optional[dict], sport_key: str) -> list:
    """Calculates strict EV combining Sharp Odds and ELO metrics natively in Python."""
    best_per_market: dict = {}
    
    for market_key, market_data_list in markets_data.items():
        sharp_odds = {}
        best_odds = {}
        has_real_sharp = False
        
        for entry in market_data_list:
            bk = entry.get("bookmaker_key", "")
            if bk in CFG.SHARP_BOOKMAKERS: has_real_sharp = True
                
            for o in entry.get("outcomes", []):
                name = f"{o['name']} {o.get('point')}" if o.get('point') is not None else o['name']
                price = float(o["price"])
                if price <= 1.0: continue
                    
                if bk in CFG.SHARP_BOOKMAKERS:
                    if name not in sharp_odds or price > sharp_odds.get(name, {}).get("price", 0):
                        sharp_odds[name] = {"price": price, "bookmaker": entry["bookmaker"]}
                        
                if name not in best_odds or price > best_odds.get(name, {}).get("price", 0):
                    best_odds[name] = {"price": price, "bookmaker": entry["bookmaker"]}

        if not sharp_odds and best_odds: sharp_odds = dict(best_odds)
        if not sharp_odds: continue
        
        try: implied_sum = sum(1.0 / v["price"] for v in sharp_odds.values())
        except ZeroDivisionError: continue
        
        if not (CFG.MIN_VALID_IMPLIED_SUM <= implied_sum <= CFG.MAX_VALID_IMPLIED_SUM): continue

        min_odds = CFG.H2H_MIN_ODDS if market_key == "h2h" else CFG.TOTALS_MIN_ODDS
        min_ev = (CFG.H2H_MIN_EV if market_key == "h2h" else CFG.TOTALS_MIN_EV) * (1.0 if has_real_sharp else 2.0)

        best_opp = None
        for outcome_name, sharp_data in sharp_odds.items():
            sharp_true_prob = (1.0 / sharp_data["price"]) / implied_sum
            
            elo_true_prob = None
            if elo_prediction and market_key == "h2h":
                name_lower = outcome_name.lower()
                if "draw" in name_lower or "tie" in name_lower: elo_true_prob = elo_prediction.get("draw_prob")
                elif elo_prediction.get("elo_diff", 0) > 0: elo_true_prob = elo_prediction.get("home_prob") if "home" not in name_lower else elo_prediction.get("home_prob")
                else: elo_true_prob = elo_prediction.get("away_prob")

            # Weighting System: 60% Sharp Market, 40% ELO
            if elo_true_prob is not None:
                true_prob = 0.6 * sharp_true_prob + 0.4 * elo_true_prob
            else:
                true_prob = sharp_true_prob

            best_price = best_odds.get(outcome_name, {}).get("price", 0.0)
            if best_price <= 1.0: continue
                
            ev = (true_prob * best_price) - 1.0

            if CFG.MAX_REALISTIC_EV >= ev > min_ev and best_price >= min_odds:
                opp = {
                    "pick": outcome_name, 
                    "market": market_key,
                    "market_label": get_market_label(market_key),
                    "prob": round(true_prob, 4), 
                    "odds": round(best_price, 3),
                    "bookmaker": best_odds[outcome_name]["bookmaker"],
                    "ev": round(ev, 4), 
                    "edge_pct": round(ev * 100, 2),
                    "has_sharp_line": has_real_sharp,
                }
                if best_opp is None or opp["ev"] > best_opp["ev"]: best_opp = opp
                    
        if best_opp: best_per_market[market_key] = best_opp

    all_opps = list(best_per_market.values())
    all_opps.sort(key=lambda x: x["ev"], reverse=True)
    return all_opps[:1]

# =========================================================
# 9. OPTIMIZED ODDS API (1 CALL)
# =========================================================
async def fetch_all_odds_async() -> list:
    now_utc = datetime.now(timezone.utc)
    end_window = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    
    params = {
        "apiKey": ODDS_API_KEY, 
        "regions": CFG.ODDS_API_REGIONS, 
        "markets": CFG.ODDS_API_MARKETS_STR, # Optimized: h2h and totals in one call
        "oddsFormat": "decimal", 
        "dateFormat": "iso"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.the-odds-api.com/v4/sports/upcoming/odds", params=params, timeout=25) as res:
                if res.status != 200: return []
                events = await res.json()
                
                all_events = {}
                for e in events:
                    try:
                        match_time = datetime.fromisoformat(e.get("commence_time", "").replace("Z", "+00:00"))
                        if now_utc <= match_time <= end_window:
                            eid = e.get("id")
                            if eid not in all_events: all_events[eid] = {**e, "_markets_data": {}}
                            
                            for bm in e.get("bookmakers", []):
                                for m in bm.get("markets", []):
                                    mk = m["key"]
                                    if mk not in all_events[eid]["_markets_data"]: all_events[eid]["_markets_data"][mk] = []
                                    all_events[eid]["_markets_data"][mk].append({
                                        "bookmaker": bm["title"], 
                                        "bookmaker_key": bm["key"], 
                                        "outcomes": m.get("outcomes", []),
                                    })
                    except Exception: continue
                return list(all_events.values())
    except Exception as e:
        if DEBUG_MODE: logger.debug("Async fetch odds error: %s", e)
        return []

# =========================================================
# 10. STATS & FOOTBALL DATA & SOFASCORE
# =========================================================
class FootballDataAdapter:
    def __init__(self):
        self.headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY} if FOOTBALL_DATA_API_KEY else {}
        self.call_count = 0
        self.daily_cache = CacheManager.load(CFG.DAILY_STATS_CACHE_FILE)

    @retry_request(max_retries=2, delay=3)
    def get_team_id(self, team_name: str) -> Optional[int]:
        if not FOOTBALL_DATA_API_KEY or self.call_count > CFG.FOOTBALL_DATA_DAILY_LIMIT: return None
        res = requests.get(
            f"https://api.football-data.org/v4/teams", 
            headers=self.headers, params={"name": clean_team_name(team_name)}, timeout=10
        )
        if res.ok and res.json().get("teams"):
            self.call_count += 1
            return res.json()["teams"][0]["id"]
        return None

    def get_team_recent_form(self, team_id: int, team_name: str) -> dict:
        cache_key = f"form_{team_id}"
        if CacheManager.is_valid(self.daily_cache, cache_key, CFG.TTL_TEAM_FORM):
            return CacheManager.get(self.daily_cache, cache_key) or {}
            
        res = requests.get(f"https://api.football-data.org/v4/teams/{team_id}/matches", headers=self.headers, params={"status": "FINISHED", "limit": 5}, timeout=10)
        if not res.ok or not res.json(): return {}
            
        form = self._parse_form(res.json(), team_id)
        self.daily_cache = CacheManager.set(self.daily_cache, cache_key, form)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)
        return form

    def _parse_form(self, data: dict, team_id: int) -> dict:
        results, goals_scored, goals_conceded = [], [], []
        for m in data.get("matches", [])[-5:]:
            home_id = m.get("homeTeam", {}).get("id")
            score = m.get("score", {}).get("fullTime", {})
            hg, ag = score.get("home", 0) or 0, score.get("away", 0) or 0
            
            if home_id == team_id:
                scored, conceded = hg, ag
                results.append("W" if hg > ag else ("D" if hg == ag else "L"))
            else:
                scored, conceded = ag, hg
                results.append("W" if ag > hg else ("D" if ag == hg else "L"))
            goals_scored.append(scored)
            goals_conceded.append(conceded)
            
        total = len(results)
        if total == 0: return {}
        return {
            "form_string": "".join(results),
            "win_rate": round(results.count("W") / total, 2),
            "avg_goals_scored": round(sum(goals_scored) / total, 2),
            "btts_rate": round(sum(1 for s, c in zip(goals_scored, goals_conceded) if s > 0 and c > 0) / total, 2),
            "over25_rate": round(sum(1 for s, c in zip(goals_scored, goals_conceded) if s + c > 2.5) / total, 2),
        }

    def get_h2h(self, team1_id: int, team2_id: int) -> dict:
        cache_key = f"h2h_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
        if CacheManager.is_valid(self.daily_cache, cache_key, CFG.TTL_H2H): return CacheManager.get(self.daily_cache, cache_key) or {}
            
        res = requests.get(f"https://api.football-data.org/v4/teams/{team1_id}/matches", headers=self.headers, params={"status": "FINISHED", "limit": 20}, timeout=10)
        if not res.ok or not res.json(): return {}
            
        h2h_matches = [m for m in res.json().get("matches", []) if {m.get("homeTeam", {}).get("id"), m.get("awayTeam", {}).get("id")} == {team1_id, team2_id}]
        result = self._parse_h2h(h2h_matches, team1_id)
        self.daily_cache = CacheManager.set(self.daily_cache, cache_key, result)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)
        return result

    def _parse_h2h(self, matches: list, team1_id: int) -> dict:
        t1_wins = t2_wins = draws = total_goals = btts = over25 = 0
        total = len(matches)
        for m in matches:
            score = m.get("score", {}).get("fullTime", {})
            hg, ag = score.get("home", 0) or 0, score.get("away", 0) or 0
            home_id = m.get("homeTeam", {}).get("id")
            if hg > ag:
                if home_id == team1_id: t1_wins += 1
                else: t2_wins += 1
            elif ag > hg:
                if home_id != team1_id: t1_wins += 1
                else: t2_wins += 1
            else: draws += 1
            
            total_goals += hg + ag
            if hg > 0 and ag > 0: btts += 1
            if hg + ag > 2.5: over25 += 1
            
        if total == 0: return {}
        return {
            "total_h2h": total, "team1_wins": t1_wins, "team2_wins": t2_wins,
            "avg_goals_per_game": round(total_goals / total, 2),
            "btts_rate": round(btts / total, 2), "over25_rate": round(over25 / total, 2),
        }

def _sofa_headers() -> dict:
    return {"x-rapidapi-key": RAPIDAPI_KEY or "", "x-rapidapi-host": "sofascore.p.rapidapi.com"}

async def fetch_sofascore_endpoint_async(session: aiohttp.ClientSession, url: str, params: dict) -> Optional[dict]:
    try:
        async with session.get(url, headers=_sofa_headers(), params=params, timeout=12) as res:
            if res.status == 200: return await res.json()
    except Exception: pass
    return None

async def fetch_sofascore_stats_async(match_id: int) -> dict:
    endpoints = {
        "h2h": "https://sofascore.p.rapidapi.com/matches/get-h2h-events",
        "lineups": "https://sofascore.p.rapidapi.com/matches/get-lineups",
        "streaks": "https://sofascore.p.rapidapi.com/matches/get-team-streaks",
    }
    params = {"matchId": str(match_id)}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=5, ssl=False)) as session:
        results = await asyncio.gather(*[fetch_sofascore_endpoint_async(session, url, params) for url in endpoints.values()], return_exceptions=True)
    data = {}
    for key, result in zip(endpoints.keys(), results):
        if not isinstance(result, Exception) and result is not None: data[key] = result
    return data

async def search_sofascore_match_async(home: str, away: str) -> Optional[int]:
    query = f"{clean_team_name(home)} {clean_team_name(away)}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://sofascore.p.rapidapi.com/search", headers=_sofa_headers(), params={"q": query, "page": "0"}, timeout=10) as res:
                if res.status == 200:
                    data = await res.json()
                    for item in data.get("results", []):
                        if item.get("type") == "event" and item.get("entity", {}).get("id"): return item["entity"]["id"]
    except Exception: pass
    return None

class MatchIDCache:
    def __init__(self):
        self.cache = CacheManager.load(CFG.MATCH_ID_CACHE_FILE)

    def get(self, home: str, away: str) -> Optional[int]:
        key = self._key(home, away)
        if CacheManager.is_valid(self.cache, key, CFG.TTL_MATCH_ID): return CacheManager.get(self.cache, key)
        return None

    def set(self, home: str, away: str, match_id: Optional[int]) -> None:
        key = self._key(home, away)
        self.cache = CacheManager.set(self.cache, key, match_id)
        CacheManager.save(CFG.MATCH_ID_CACHE_FILE, self.cache)

    @staticmethod
    def _key(home: str, away: str) -> str:
        return hashlib.md5(f"{home.lower()}|{away.lower()}".encode()).hexdigest()

async def get_stats_async(home: str, away: str, sport_key: str, football_adapter: FootballDataAdapter, match_id_cache: MatchIDCache, elo_system: ELOSystem) -> dict:
    stats = {"home_form": {}, "away_form": {}, "h2h": {}, "sofascore": {}, "elo": {}, "data_quality": "none"}

    elo_pred = elo_system.predict(home, away, apply_home_advantage=(sport_key == "football"))
    if elo_pred and elo_pred.get("home_matches", 0) > 0: stats["elo"] = elo_pred

    cached_mid = match_id_cache.get(home, away)
    match_id = cached_mid if cached_mid is != None else await search_sofascore_match_async(home, away)
    if cached_mid is None: match_id_cache.set(home, away, match_id if match_id else 0)

    task_names, coros = [], []
    if match_id:
        task_names.append("sofascore")
        coros.append(fetch_sofascore_stats_async(match_id))

    if sport_key == "football":
        loop = asyncio.get_running_loop()
        async def get_football_data():
            home_id = await loop.run_in_executor(None, football_adapter.get_team_id, home)
            away_id = await loop.run_in_executor(None, football_adapter.get_team_id, away)
            if not home_id or not away_id: return {}
            hf, af, h2h = await asyncio.gather(
                loop.run_in_executor(None, football_adapter.get_team_recent_form, home_id, home),
                loop.run_in_executor(None, football_adapter.get_team_recent_form, away_id, away),
                loop.run_in_executor(None, football_adapter.get_h2h, home_id, away_id),
                return_exceptions=True,
            )
            out = {}
            if not isinstance(hf, Exception) and hf: out["home_form"] = hf
            if not isinstance(af, Exception) and af: out["away_form"] = af
            if not isinstance(h2h, Exception) and h2h: out["h2h"] = h2h
            return out
        task_names.append("football")
        coros.append(get_football_data())

    if coros:
        gathered = await asyncio.gather(*coros, return_exceptions=True)
        for name, result in zip(task_names, gathered):
            if isinstance(result, Exception): continue
            if name == "sofascore" and result: stats["sofascore"] = result
            elif name == "football" and isinstance(result, dict): stats.update(result)

    has_football = bool(stats.get("home_form") or stats.get("h2h"))
    has_sofascore = bool(stats.get("sofascore"))
    
    if has_football and has_sofascore: stats["data_quality"] = "high"
    elif has_football or has_sofascore: stats["data_quality"] = "medium"

    return stats

# =========================================================
# 11. ENTERPRISE AI ANALYSIS & SCORING
# =========================================================
def calculate_system_confidence(ev_edge: float, stats: dict, market: str) -> tuple[int, str]:
    score = 50
    dq = stats.get("data_quality", "none")
    
    if dq == "high": score += 15
    elif dq == "medium": score += 8
    
    ev_pct = ev_edge * 100
    if ev_pct > 5.0: score += 12
    elif ev_pct > 3.0: score += 8
    elif ev_pct > 1.5: score += 4
    
    elo = stats.get("elo", {})
    if elo and elo.get("home_matches", 0) >= 5: score += 10

    if market.lower() == "totals": score += 3

    score = max(50, min(93, int(score)))
    
    if score >= 75: risk = "Low"
    elif score >= 60: risk = "Medium"
    else: risk = "High"
        
    return score, risk

def call_groq_sdk(model: str, messages: list, temperature: float = 0.1) -> Optional[str]:
    kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": CFG.AI_MAX_TOKENS}
    if "qwen" not in model.lower() and "gpt-oss" not in model.lower(): kwargs["response_format"] = {"type": "json_object"}
        
    try:
        res = groq_client.chat.completions.create(**kwargs)
        return res.choices[0].message.content
    except Exception as e:
        logger.error("Groq SDK error model=%s: %s", model, e)
        return None

def generate_dual_ai_analysis(home: str, away: str, sport: str, pick: str, market: str, ev_edge: float, stats: dict, has_real_stats: bool) -> dict:
    calc_conf, calc_risk = calculate_system_confidence(ev_edge, stats, market)
    
    default_response = {
        "sport_emoji": "\U0001F3C6", 
        "home_flag": get_flag_from_name(home), 
        "away_flag": get_flag_from_name(away),
        "risk_level": calc_risk, 
        "confidence": calc_conf,
        "logic": "The sharp market indicates significant value on this selection based on underlying metrics.",
    }

    stats_str = json.dumps(stats, indent=2)[:1500]

    sys_analyst = (
        "You are an elite, high-stakes sports betting analyst for a VIP syndicate.\n"
        "Write EXACTLY two punchy, professional sentences justifying the given pick.\n"
        "STRICT RULES:\n"
        "- NEVER mention 'Expected Value', 'EV', 'data quality', 'points', or models.\n"
        "- Determine the EXACT country flag emoji for home_flag and away_flag.\n"
    )
    
    if has_real_stats: sys_analyst += "- Use the provided historical/form statistics to highlight the team/player's dominance.\n"
    else: sys_analyst += "- IMPORTANT: DO NOT invent past matches. Explicitly state this pick is driven solely by odds discrepancies in the sharp betting market.\n"
        
    sys_analyst += "OUTPUT JSON ONLY: {\"logic\": \"your 2 sentence analysis\", \"sport_emoji\": \"...\", \"home_flag\": \"...\", \"away_flag\": \"...\"}"

    u1 = f"MATCH: {home} vs {away}\nSPORT: {sport}\nPICK: {pick}\nMARKET TYPE: {get_market_label(market)}\n\nSTATS:\n{stats_str}\n\nOUTPUT JSON ONLY:"

    analysis_1 = None
    try:
        raw1 = call_groq_sdk(CFG.AI_MODEL_ANALYST, [{"role": "system", "content": sys_analyst}, {"role": "user", "content": u1}], temperature=0.3)
        analysis_1 = robust_json_extractor(raw1)
    except Exception: pass

    initial_logic = (analysis_1 or {}).get("logic", default_response["logic"])
    
    sys_editor = (
        "You are the Chief Editor for a high-end sports betting platform.\n"
        "Review the drafted analysis. REWRITE it if it hallucinates form/momentum when none exists, or sounds robotic.\n"
        "Keep it strictly under 3 sentences. Professional Tipster Tone.\n"
        "OUTPUT JSON ONLY: {\"validated_logic\": \"...\"}"
    )
    
    try:
        raw2 = call_groq_sdk(CFG.AI_MODEL_VALIDATOR, [{"role": "system", "content": sys_editor}, {"role": "user", "content": f"DRAFT: {initial_logic}\nPICK: {pick}\nOUTPUT JSON ONLY:"}], temperature=0.2)
        analysis_2 = robust_json_extractor(raw2)
        if analysis_2 and analysis_2.get("validated_logic"): initial_logic = analysis_2["validated_logic"]
    except Exception: pass

    result = dict(default_response)
    if analysis_1:
        result["sport_emoji"] = analysis_1.get("sport_emoji", result["sport_emoji"])
        result["home_flag"] = validate_flag(analysis_1.get("home_flag", ""), home)
        result["away_flag"] = validate_flag(analysis_1.get("away_flag", ""), away)
    
    safe_logic = str(initial_logic).strip()
    result["logic"] = safe_logic[:597] + "..." if len(safe_logic) > 600 else safe_logic
    
    return result

# =========================================================
# 12. RESULT CHECKING SYSTEM
# =========================================================
@retry_request(max_retries=3)
def fetch_event_result(home: str, away: str) -> Optional[dict]:
    try:
        params = {"apiKey": ODDS_API_KEY, "daysFrom": 3, "dateFormat": "iso"}
        res = requests.get("https://api.the-odds-api.com/v4/sports/upcoming/scores", params=params, timeout=15)
        if res.status_code != 200: return None
        scores = res.json()
        for event in scores:
            if event.get("home_team", "").lower() == home.lower() and event.get("away_team", "").lower() == away.lower():
                if event.get("completed"): return event
        return None
    except Exception: return None

def _determine_win(pick: str, market: str, scores: dict, home: str, away: str) -> Optional[bool]:
    try:
        if market == "h2h":
            home_sc = int(scores.get(home, {}).get("score", -1))
            away_sc = int(scores.get(away, {}).get("score", -1))
            if home_sc < 0 or away_sc < 0: return None
            
            pick_lower = pick.lower()
            if home.lower() in pick_lower or "home" in pick_lower: return home_sc > away_sc
            if away.lower() in pick_lower or "away" in pick_lower: return away_sc > home_sc
            if "draw" in pick_lower or "tie" in pick_lower: return home_sc == away_sc

        elif market == "totals":
            home_sc = int(scores.get(home, {}).get("score", -1))
            away_sc = int(scores.get(away, {}).get("score", -1))
            if home_sc < 0 or away_sc < 0: return None
            total = home_sc + away_sc
            
            pick_lower = pick.lower()
            m = re.search(r"(over|under)\s*([\d.]+)", pick_lower)
            if m:
                direction, line = m.group(1), float(m.group(2))
                return total > line if direction == "over" else total < line
    except Exception: pass
    return None

def check_and_report_results(sent_history: SentHistory) -> Optional[str]:
    pending = sent_history.get_pending_results()
    if not pending: return None

    wins, losses = [], []

    for key, entry in pending:
        home, away = entry.get("home", ""), entry.get("away", "")
        pick, market = entry.get("pick", ""), entry.get("market", "")
        
        result_event = fetch_event_result(home, away)
        if not result_event: continue

        scores = result_event.get("scores", {})
        won = _determine_win(pick, market, scores, home, away)

        try:
            home_score = scores.get(home, {}).get("score", "?")
            away_score = scores.get(away, {}).get("score", "?")
            result_str = f"{home_score} - {away_score}"
        except Exception: result_str = "? - ?"

        sent_history.mark_result_checked(key, result_str, won)

        if won is True: wins.append({**entry, "result": result_str})
        elif won is False: losses.append({**entry, "result": result_str})

    if not wins and not losses: return None

    total = len(wins) + len(losses)
    win_rate = len(wins) / total if total > 0 else 0
    roi = sum([w.get("odds", 1.0) - 1.0 for w in wins] + [-1.0 for _ in losses]) / total if total > 0 else 0

    lines = ["\U0001F4CA <b>RESULTS REPORT</b>\n"]
    for w in wins:
        lines.append(f"\U00002705 <b>{html_lib.escape(w['home'])} vs {html_lib.escape(w['away'])}</b>\n   Pick: {html_lib.escape(w['pick'])} @ <code>{w['odds']}</code>\n   Result: {w.get('result', '?')} \u2014 <b>WIN</b>\n")
    for l in losses:
        lines.append(f"\u274C <b>{html_lib.escape(l['home'])} vs {html_lib.escape(l['away'])}</b>\n   Pick: {html_lib.escape(l['pick'])} @ <code>{l['odds']}</code>\n   Result: {l.get('result', '?')} \u2014 <b>LOSS</b>\n")

    lines.append(f"\n\U0001F3AF <b>Session:</b> {len(wins)}W / {len(losses)}L | Win Rate: {win_rate:.0%} | ROI: {roi:+.1%}\n\n\U0001F194 {CFG.TELEGRAM_ID}")
    return "\n".join(lines)

# =========================================================
# 13. TELEGRAM INTEGRATION & CHUNKING
# =========================================================
def send_telegram(message_html: str) -> bool:
    MAX_LEN = 4000
    chunks = []
    
    if len(message_html) <= MAX_LEN: chunks.append(message_html)
    else:
        current_chunk = ""
        for line in message_html.split('\n'):
            if len(current_chunk) + len(line) + 1 > MAX_LEN:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else: current_chunk += line + "\n"
        if current_chunk: chunks.append(current_chunk.strip())

    success = True
    for chunk in chunks:
        res = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
            json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True}, 
            timeout=10
        )
        if not res.ok: success = False
    return success

# =========================================================
# 14. MAIN PIPELINE
# =========================================================
async def async_main():
    log_section("ZBET90 ENTERPRISE ENGINE v4.0 (AI + ELO + REPORTS) STARTING")

    sent_history = SentHistory()
    football_adapter = FootballDataAdapter()
    match_id_cache = MatchIDCache()
    elo_system = ELOSystem("football")
    now_utc = datetime.now(timezone.utc)
    
    # --- PHASE 1: Results Check ---
    logger.info("Checking pending results...")
    results_msg = check_and_report_results(sent_history)
    if results_msg:
        send_telegram(results_msg)
        logger.info("Win/Loss report sent to Telegram.")
    
    # --- PHASE 2: Fetch Odds ---
    logger.info("Fetching combined events (h2h & totals)...")
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
        commence_time = event.get('commence_time', '')
        
        if not home or not away: continue

        # Predict using strict Python ELO
        elo_pred = elo_system.predict(home, away) if sport_key == "football" else None
        
        opportunities = calculate_combined_ev(event.get("_markets_data", {}), elo_pred, sport_key)
        if not opportunities: continue
            
        opp = opportunities[0]

        if sent_history.was_sent(home, away, opp["market"]): continue

        # Fetch Live APIs
        stats = await get_stats_async(home, away, sport_key, football_adapter, match_id_cache, elo_system)
        
        has_real_stats = bool(stats.get("elo")) or bool(stats.get("home_form")) or bool(stats.get("sofascore"))
        
        if has_real_stats: logger.info("✅ [STATS VERIFIED] Found rich data for %s vs %s", home, away)
        else: logger.info("⚠️ [STATS WARNING] Relying strictly on Sharp EV math for %s vs %s", home, away)

        ai_data = generate_dual_ai_analysis(home, away, sport, opp["pick"], opp["market"], opp["ev"], stats, has_real_stats)
        
        # VIP Telegram Format
        conf_icon = "\U0001F525" if ai_data["confidence"] >= 75 else ("\U00002705" if ai_data["confidence"] >= 65 else "\U000026A1")
        risk_icon = {"Low": "\U0001F7E2", "Medium": "\U0001F7E0", "High": "\U0001F534"}.get(ai_data["risk_level"], "\U0001F7E0")
        
        msg = (
            f"{ai_data.get('sport_emoji', '🏆')} <b>{html_lib.escape(sport)}</b>\n\n"
            f"⚔️ <b>{html_lib.escape(home)}</b> {ai_data.get('home_flag', '🏳️')}  vs  {ai_data.get('away_flag', '🏳️')} <b>{html_lib.escape(away)}</b>\n"
            f"⏳ <b>Starts in:</b> {get_countdown_str(commence_time, now_utc)}\n\n"
            f"🎯 <b>PICK: {html_lib.escape(opp['pick'])}</b> @ <code>{opp['odds']}</code>\n\n"
            f"📊 <b>MARKET:</b> {html_lib.escape(opp['market_label'])}\n"
            f"{risk_icon} <b>Risk:</b> {ai_data['risk_level']}  |  {conf_icon} <b>Confidence: {ai_data['confidence']}%</b>\n\n"
            f"💡 <b>EXPERT ANALYSIS:</b>\n<blockquote>{html_lib.escape(ai_data['logic'])}</blockquote>\n\n"
            f"🔍 <i>Curated by {CFG.TELEGRAM_ID}</i>"
        )

        if send_telegram(msg):
            sent_history.mark_sent(home, away, opp["pick"], opp["market"], opp["odds"], commence_time)
            total_sent += 1
            logger.info("Sent: %s vs %s | %s", home, away, opp["pick"])
        else: logger.error("Telegram failed: %s vs %s", home, away)

        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    log_section("RUN COMPLETE")
    if total_sent > 0: logger.info("Done! %d signal(s) sent.", total_sent)
    else: logger.info("No qualifying +EV opportunities found.")
    logger.info("=" * 60)

if __name__ == "__main__":
    try: asyncio.run(async_main())
    except Exception as e:
        logger.critical("SYSTEM FAILURE: %s", str(e), exc_info=True)
        sys.exit(1)
