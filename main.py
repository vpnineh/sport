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
from groq import Groq
from functools import wraps
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# =========================================================
# 1. CONFIG
# =========================================================
@dataclass
class Config:
    CACHE_DIR: Path = Path("api_cache")
    LOG_DIR: Path = Path("log")
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
# 2. LOGGING
# =========================================================
CFG.CACHE_DIR.mkdir(exist_ok=True)
CFG.LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("ZBET90_ENGINE")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler(CFG.LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# =========================================================
# 3. API KEYS
# =========================================================
ODDS_API_KEY           = os.getenv("ODDS_API_KEY")
GROQ_API_KEY           = os.getenv("GROQ_API_KEY")
RAPIDAPI_KEY           = os.getenv("RAPIDAPI_KEY")
TELEGRAM_BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")
FOOTBALL_DATA_API_KEY  = os.getenv("FOOTBALL_DATA_API_KEY")

if not all([ODDS_API_KEY, GROQ_API_KEY, RAPIDAPI_KEY,
            TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing critical API Keys.")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_API_KEY, max_retries=3)

# =========================================================
# 4. NATIONALITY FLAGS
# =========================================================
NATIONALITY_FLAGS: dict = {
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
    "monteiro": "BR", "seyboth wild": "BR",
    "auger": "CA",
    "manchester united": "GB", "manchester city": "GB",
    "liverpool": "GB", "chelsea": "GB", "arsenal": "GB",
    "tottenham": "GB", "newcastle": "GB", "west ham": "GB",
    "aston villa": "GB", "everton": "GB", "leicester": "GB",
    "wolves": "GB", "brighton": "GB", "brentford": "GB",
    "crystal palace": "GB", "fulham": "GB", "bournemouth": "GB",
    "nottingham": "GB", "burnley": "GB", "luton": "GB",
    "celtic": "GB", "rangers": "GB", "hearts": "GB",
    "real madrid": "ES", "barcelona": "ES", "atletico": "ES",
    "sevilla": "ES", "valencia": "ES", "villarreal": "ES",
    "real sociedad": "ES", "athletic bilbao": "ES", "betis": "ES",
    "osasuna": "ES", "celta": "ES", "getafe": "ES",
    "bayern": "DE", "dortmund": "DE", "leipzig": "DE",
    "leverkusen": "DE", "frankfurt": "DE", "wolfsburg": "DE",
    "freiburg": "DE", "gladbach": "DE", "hoffenheim": "DE",
    "juventus": "IT", "milan": "IT", "inter": "IT",
    "napoli": "IT", "roma": "IT", "lazio": "IT",
    "atalanta": "IT", "fiorentina": "IT", "torino": "IT",
    "psg": "FR", "marseille": "FR", "lyon": "FR", "monaco": "FR",
    "lille": "FR", "lens": "FR", "nice": "FR", "rennes": "FR",
    "ajax": "NL", "psv": "NL", "feyenoord": "NL",
    "az alkmaar": "NL", "utrecht": "NL",
    "porto": "PT", "benfica": "PT", "sporting": "PT", "braga": "PT",
    "galatasaray": "TR", "fenerbahce": "TR", "besiktas": "TR",
    "trabzonspor": "TR",
    "shakhtar": "UA", "dynamo kyiv": "UA",
    "salzburg": "AT", "rapid wien": "AT", "lask": "AT",
    "anderlecht": "BE", "club brugge": "BE", "gent": "BE",
    "zenit": "RU", "spartak": "RU", "cska moscow": "RU",
    "vaalerenga": "NO", "valerenga": "NO", "brann": "NO",
    "rosenborg": "NO", "molde": "NO", "bodo": "NO",
    "malmo": "SE", "ifk gothenburg": "SE", "djurgarden": "SE",
    "copenhagen": "DK", "midtjylland": "DK", "brondby": "DK",
    "lakers": "US", "celtics": "US", "warriors": "US",
    "bulls": "US", "heat": "US", "nets": "US", "knicks": "US",
    "bucks": "US", "suns": "US", "nuggets": "US",
    "yankees": "US", "dodgers": "US", "cubs": "US",
    "red sox": "US", "giants": "US", "mets": "US",
}


def _code_to_flag(code: str) -> str:
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
    if stripped in ["\U0001F3F3\uFE0F", "\U0001F3C1", "\U0001F6A9", "", "🏁", "🏳️"]:
        return get_flag_from_name(fallback_name)
    return stripped

# =========================================================
# 5. CACHE MANAGER
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
# 6. SENT HISTORY
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
# 7. UTILS
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

    try:
        m = re.search(r"\{[\s\S]*\}", raw_text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass

    logger.error("JSON parse failed: %s", raw_text[:300])
    return None


def clean_team_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(name)).strip()


def normalize_sport_key(sport_title: str) -> str:
    keywords = [
        "soccer", "football", "premier league", "la liga", "bundesliga",
        "serie a", "ligue 1", "champions league", "europa league",
        "mls", "eredivisie", "primeira liga", "championship", "league cup",
        "fa cup", "copa del rey", "dfb pokal",
    ]
    return "football" if any(kw in sport_title.lower() for kw in keywords) else "other"


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

# =========================================================
# 8. MATH ENGINE
# =========================================================
def validate_sharp_odds(sharp_odds: dict, market_key: str) -> tuple:
    if not sharp_odds:
        return False, "empty sharp_odds"
    n = len(sharp_odds)
    expected = CFG.MARKET_EXPECTED_OUTCOMES.get(market_key, {"min": 2, "max": 3})
    if n < expected["min"]:
        return False, f"only {n} outcomes, need {expected['min']}"
    try:
        implied_sum = sum(1.0 / v["price"] for v in sharp_odds.values())
    except (KeyError, ZeroDivisionError) as e:
        return False, f"price error: {e}"
    if implied_sum < CFG.MIN_VALID_IMPLIED_SUM:
        return False, f"implied_sum too low ({implied_sum:.3f})"
    if implied_sum > CFG.MAX_VALID_IMPLIED_SUM:
        return False, f"implied_sum too high ({implied_sum:.3f})"
    return True, "valid"


def calculate_sharp_ev(markets_data: dict, bookmakers_raw: list) -> list:
    best_per_market: dict = {}

    for market_key, market_data_list in markets_data.items():
        sharp_odds: dict = {}
        best_odds: dict = {}
        has_real_sharp = False

        for entry in market_data_list:
            bk = entry.get("bookmaker_key", "")
            if bk in CFG.SHARP_BOOKMAKERS:
                has_real_sharp = True
                for o in entry.get("outcomes", []):
                    base_name = o["name"]
                    point = o.get("point")
                    name = f"{base_name} {point}" if point is not None else base_name
                    price = float(o["price"])
                    if price <= 1.0:
                        continue
                    if name not in sharp_odds or price > sharp_odds[name]["price"]:
                        sharp_odds[name] = {"price": price, "bookmaker": entry["bookmaker"]}

            for o in entry.get("outcomes", []):
                base_name = o["name"]
                point = o.get("point")
                name = f"{base_name} {point}" if point is not None else base_name
                price = float(o["price"])
                if price <= 1.0:
                    continue
                if name not in best_odds or price > best_odds[name]["price"]:
                    best_odds[name] = {"price": price, "bookmaker": entry["bookmaker"]}

        if not sharp_odds and best_odds:
            sharp_odds = dict(best_odds)

        is_valid, reason = validate_sharp_odds(sharp_odds, market_key)
        if not is_valid:
            continue

        implied_sum = sum(1.0 / v["price"] for v in sharp_odds.values())

        if market_key == "h2h":
            min_odds, min_ev = CFG.H2H_MIN_ODDS, CFG.H2H_MIN_EV
        else:
            min_odds, min_ev = CFG.TOTALS_MIN_ODDS, CFG.TOTALS_MIN_EV

        if not has_real_sharp:
            min_ev = min_ev * 2.0

        best_opp = None
        for outcome_name, sharp_data in sharp_odds.items():
            true_prob = (1.0 / sharp_data["price"]) / implied_sum
            best = best_odds.get(outcome_name, {})
            best_price = best.get("price", 0.0)
            best_bookie = best.get("bookmaker", "Unknown")

            if best_price <= 1.0:
                continue

            ev = (true_prob * best_price) - 1.0

            if ev > CFG.MAX_REALISTIC_EV:
                continue

            if best_price >= min_odds and ev > min_ev:
                market_label = {
                    "h2h": "Winner",
                    "totals": "Over/Under",
                }.get(market_key, market_key.upper())

                opp = {
                    "pick": outcome_name,
                    "market": market_key,
                    "market_label": market_label,
                    "prob": round(true_prob, 4),
                    "odds": round(best_price, 3),
                    "bookmaker": best_bookie,
                    "ev": round(ev, 4),
                    "edge_pct": round(ev * 100, 2),
                    "implied_sum": round(implied_sum, 4),
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
async def fetch_market_async(
    session: aiohttp.ClientSession,
    market: str,
    now_utc: datetime,
) -> list:
    end_window = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": CFG.ODDS_API_REGIONS,
        "markets": market,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        async with session.get(
            "https://api.the-odds-api.com/v4/sports/upcoming/odds",
            params=params,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as res:
            if res.status == 422:
                return []
            if res.status == 429:
                return []
            if res.status != 200:
                return []
            events = await res.json()
            filtered = []
            for e in events:
                try:
                    ct = e.get("commence_time", "")
                    match_time = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    if now_utc <= match_time <= end_window:
                        filtered.append(e)
                except Exception:
                    continue
            logger.info("Market '%s': %d events in window", market, len(filtered))
            return filtered
    except asyncio.TimeoutError:
        return []
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
            return_exceptions=True,
        )

    for i, market_events in enumerate(results):
        if isinstance(market_events, Exception):
            continue
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

    result = list(all_events.values())
    logger.info("Total unique events in window: %d", len(result))
    return result

# =========================================================
# 10. SOFASCORE ASYNC
# =========================================================
def _sofa_headers() -> dict:
    return {
        "x-rapidapi-key": RAPIDAPI_KEY or "",
        "x-rapidapi-host": "sofascore.p.rapidapi.com",
    }


async def fetch_sofascore_endpoint_async(
    session: aiohttp.ClientSession,
    url: str,
    params: dict,
) -> Optional[dict]:
    try:
        async with session.get(
            url,
            headers=_sofa_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=12),
        ) as res:
            if res.status == 200:
                return await res.json()
    except Exception as e:
        pass
    return None


async def fetch_sofascore_stats_async(match_id: int) -> dict:
    endpoints = {
        "h2h": "https://sofascore.p.rapidapi.com/matches/get-h2h-events",
        "lineups": "https://sofascore.p.rapidapi.com/matches/get-lineups",
        "streaks": "https://sofascore.p.rapidapi.com/matches/get-team-streaks",
    }
    params = {"matchId": str(match_id)}
    connector = aiohttp.TCPConnector(limit=5, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(
            *[fetch_sofascore_endpoint_async(session, url, params)
              for url in endpoints.values()],
            return_exceptions=True,
        )
    data = {}
    for key, result in zip(endpoints.keys(), results):
        if not isinstance(result, Exception) and result is not None:
            data[key] = result
    return data


async def search_sofascore_match_async(home: str, away: str) -> Optional[int]:
    query = f"{clean_team_name(home)} {clean_team_name(away)}"
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            async with session.get(
                "https://sofascore.p.rapidapi.com/search",
                headers=_sofa_headers(),
                params={"q": query, "page": "0"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as res:
                if res.status != 200:
                    return None
                data = await res.json()
                for item in data.get("results", []):
                    if item.get("type") == "event":
                        mid = item.get("entity", {}).get("id")
                        if mid:
                            return mid
        except Exception as e:
            pass
    return None

# =========================================================
# 11. FOOTBALL-DATA ADAPTER
# =========================================================
class FootballDataAdapter:
    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self):
        self.headers = (
            {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
            if FOOTBALL_DATA_API_KEY else {}
        )
        self.daily_cache = CacheManager.load(CFG.DAILY_STATS_CACHE_FILE)
        self._init_call_counter()

    def _init_call_counter(self):
        entry = self.daily_cache.get("_call_count_today", {})
        self.call_count = entry.get("data", 0)
        last_ts = entry.get("timestamp", "2000-01-01T00:00:00+00:00")
        try:
            if datetime.now(timezone.utc).date() > datetime.fromisoformat(last_ts).date():
                self.call_count = 0
                logger.info("Football-Data call counter reset for new day")
        except Exception:
            self.call_count = 0

    def _can_call(self) -> bool:
        return (
            self.call_count < CFG.FOOTBALL_DATA_DAILY_LIMIT
            and bool(FOOTBALL_DATA_API_KEY)
        )

    def _increment_call(self):
        self.call_count += 1
        self.daily_cache = CacheManager.set(
            self.daily_cache, "_call_count_today", self.call_count
        )
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)

    @retry_request(max_retries=2, delay=3)
    def _raw_get(self, endpoint: str, params: dict = None) -> dict:
        if not self._can_call():
            return {}
        res = requests.get(
            f"{self.BASE_URL}{endpoint}",
            headers=self.headers,
            params=params,
            timeout=12,
        )
        res.raise_for_status()
        self._increment_call()
        return res.json()

    def find_team_id(self, team_name: str) -> Optional[int]:
        cache = CacheManager.load(CFG.TEAM_ID_CACHE_FILE)
        key = team_name.lower().strip()
        if key in cache:
            return cache[key]
        if not self._can_call():
            return None
        data = self._raw_get("/teams", {"name": clean_team_name(team_name)})
        team_id = None
        if data and data.get("teams"):
            team_id = data["teams"][0]["id"]
        cache[key] = team_id
        CacheManager.save(CFG.TEAM_ID_CACHE_FILE, cache)
        return team_id

    def get_team_recent_form(self, team_id: int, team_name: str) -> dict:
        cache_key = f"form_{team_id}"
        if CacheManager.is_valid(self.daily_cache, cache_key, CFG.TTL_TEAM_FORM):
            return CacheManager.get(self.daily_cache, cache_key) or {}
        data = self._raw_get(
            f"/teams/{team_id}/matches",
            {"status": "FINISHED", "limit": 5},
        )
        if not data:
            return {}
        form = self._parse_form(data, team_id)
        self.daily_cache = CacheManager.set(self.daily_cache, cache_key, form)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)
        return form

    def _parse_form(self, data: dict, team_id: int) -> dict:
        results, goals_scored, goals_conceded = [], [], []
        for m in data.get("matches", [])[-5:]:
            home_id = m.get("homeTeam", {}).get("id")
            score = m.get("score", {}).get("fullTime", {})
            hg = score.get("home", 0) or 0
            ag = score.get("away", 0) or 0
            if home_id == team_id:
                scored, conceded = hg, ag
                results.append("W" if hg > ag else ("D" if hg == ag else "L"))
            else:
                scored, conceded = ag, hg
                results.append("W" if ag > hg else ("D" if ag == hg else "L"))
            goals_scored.append(scored)
            goals_conceded.append(conceded)
        total = len(results)
        if total == 0:
            return {}
        return {
            "form_string": "".join(results),
            "win_rate": round(results.count("W") / total, 2),
            "draw_rate": round(results.count("D") / total, 2),
            "avg_goals_scored": round(sum(goals_scored) / total, 2),
            "avg_goals_conceded": round(sum(goals_conceded) / total, 2),
            "btts_rate": round(
                sum(1 for s, c in zip(goals_scored, goals_conceded)
                    if s > 0 and c > 0) / total, 2
            ),
            "over25_rate": round(
                sum(1 for s, c in zip(goals_scored, goals_conceded)
                    if s + c > 2.5) / total, 2
            ),
            "matches_analyzed": total,
        }

    def get_h2h(self, team1_id: int, team2_id: int) -> dict:
        cache_key = f"h2h_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
        if CacheManager.is_valid(self.daily_cache, cache_key, CFG.TTL_H2H):
            return CacheManager.get(self.daily_cache, cache_key) or {}
        data = self._raw_get(
            f"/teams/{team1_id}/matches",
            {"status": "FINISHED", "limit": 20},
        )
        if not data:
            return {}
        h2h_matches = [
            m for m in data.get("matches", [])
            if {m.get("homeTeam", {}).get("id"), m.get("awayTeam", {}).get("id")}
            == {team1_id, team2_id}
        ]
        result = self._parse_h2h(h2h_matches, team1_id)
        self.daily_cache = CacheManager.set(self.daily_cache, cache_key, result)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)
        return result

    def _parse_h2h(self, matches: list, team1_id: int) -> dict:
        t1_wins = t2_wins = draws = total_goals = btts = over25 = 0
        total = len(matches)
        for m in matches:
            score = m.get("score", {}).get("fullTime", {})
            hg = score.get("home", 0) or 0
            ag = score.get("away", 0) or 0
            home_id = m.get("homeTeam", {}).get("id")
            if hg > ag:
                if home_id == team1_id:
                    t1_wins += 1
                else:
                    t2_wins += 1
            elif ag > hg:
                if home_id != team1_id:
                    t1_wins += 1
                else:
                    t2_wins += 1
            else:
                draws += 1
            total_goals += hg + ag
            if hg > 0 and ag > 0:
                btts += 1
            if hg + ag > 2.5:
                over25 += 1
        if total == 0:
            return {}
        return {
            "total_h2h": total,
            "team1_wins": t1_wins,
            "team2_wins": t2_wins,
            "draws": draws,
            "avg_goals_per_game": round(total_goals / total, 2),
            "btts_rate": round(btts / total, 2),
            "over25_rate": round(over25 / total, 2),
        }

# =========================================================
# 12. MATCH ID CACHE
# =========================================================
class MatchIDCache:
    def __init__(self):
        self.cache = CacheManager.load(CFG.MATCH_ID_CACHE_FILE)

    def get(self, home: str, away: str) -> Optional[int]:
        key = self._key(home, away)
        if CacheManager.is_valid(self.cache, key, CFG.TTL_MATCH_ID):
            return CacheManager.get(self.cache, key)
        return None

    def set(self, home: str, away: str, match_id: Optional[int]) -> None:
        key = self._key(home, away)
        self.cache = CacheManager.set(self.cache, key, match_id)
        CacheManager.save(CFG.MATCH_ID_CACHE_FILE, self.cache)

    @staticmethod
    def _key(home: str, away: str) -> str:
        return hashlib.md5(f"{home.lower()}|{away.lower()}".encode()).hexdigest()

# =========================================================
# 13. STATS AGGREGATOR
# =========================================================
async def get_stats_async(
    home: str,
    away: str,
    sport_key: str,
    football_adapter: FootballDataAdapter,
    match_id_cache: MatchIDCache,
) -> dict:
    stats = {
        "home_form": {},
        "away_form": {},
        "h2h": {},
        "sofascore": {},
        "data_quality": "none",
    }

    cached_mid = match_id_cache.get(home, away)
    if cached_mid is not None:
        match_id = cached_mid if cached_mid != 0 else None
    else:
        match_id = await search_sofascore_match_async(home, away)
        match_id_cache.set(home, away, match_id if match_id else 0)

    task_names = []
    coros = []

    if match_id:
        task_names.append("sofascore")
        coros.append(fetch_sofascore_stats_async(match_id))

    if sport_key == "football":
        loop = asyncio.get_running_loop()

        async def get_football_data():
            home_id = await loop.run_in_executor(None, football_adapter.find_team_id, home)
            away_id = await loop.run_in_executor(None, football_adapter.find_team_id, away)
            if not home_id or not away_id:
                return {}
            hf, af, h2h = await asyncio.gather(
                loop.run_in_executor(
                    None, football_adapter.get_team_recent_form, home_id, home
                ),
                loop.run_in_executor(
                    None, football_adapter.get_team_recent_form, away_id, away
                ),
                loop.run_in_executor(
                    None, football_adapter.get_h2h, home_id, away_id
                ),
                return_exceptions=True,
            )
            out = {}
            if not isinstance(hf, Exception) and hf:
                out["home_form"] = hf
            if not isinstance(af, Exception) and af:
                out["away_form"] = af
            if not isinstance(h2h, Exception) and h2h:
                out["h2h"] = h2h
            return out

        task_names.append("football")
        coros.append(get_football_data())

    if coros:
        gathered = await asyncio.gather(*coros, return_exceptions=True)
        for name, result in zip(task_names, gathered):
            if isinstance(result, Exception):
                continue
            if name == "sofascore" and result:
                stats["sofascore"] = result
            elif name == "football" and isinstance(result, dict):
                stats.update(result)

    has_football = bool(stats.get("home_form") or stats.get("h2h"))
    has_sofascore = bool(stats.get("sofascore"))
    if has_football and has_sofascore:
        stats["data_quality"] = "high"
    elif has_football or has_sofascore:
        stats["data_quality"] = "medium"

    return stats

# =========================================================
# 14. ENTERPRISE AI ANALYSIS & SCORING
# =========================================================
def build_stats_summary(stats: dict, home: str, away: str) -> str:
    parts = []
    hf = stats.get("home_form", {})
    af = stats.get("away_form", {})
    h2h = stats.get("h2h", {})

    if hf:
        parts.append(
            f"HOME ({home}): Form={hf.get('form_string', 'N/A')} | "
            f"WR={hf.get('win_rate', 0):.0%} | "
            f"AvgGF={hf.get('avg_goals_scored', 0)} | "
            f"AvgGA={hf.get('avg_goals_conceded', 0)} | "
            f"BTTS={hf.get('btts_rate', 0):.0%} | "
            f"O2.5={hf.get('over25_rate', 0):.0%}"
        )
    if af:
        parts.append(
            f"AWAY ({away}): Form={af.get('form_string', 'N/A')} | "
            f"WR={af.get('win_rate', 0):.0%} | "
            f"AvgGF={af.get('avg_goals_scored', 0)} | "
            f"AvgGA={af.get('avg_goals_conceded', 0)} | "
            f"BTTS={af.get('btts_rate', 0):.0%} | "
            f"O2.5={af.get('over25_rate', 0):.0%}"
        )
    if h2h and h2h.get("total_h2h", 0) > 0:
        parts.append(
            f"H2H (n={h2h['total_h2h']}): "
            f"HomeW={h2h.get('team1_wins', 0)} | "
            f"AwayW={h2h.get('team2_wins', 0)} | "
            f"D={h2h.get('draws', 0)} | "
            f"AvgGoals={h2h.get('avg_goals_per_game', 0)} | "
            f"BTTS={h2h.get('btts_rate', 0):.0%} | "
            f"O2.5={h2h.get('over25_rate', 0):.0%}"
        )
    ss = stats.get("sofascore", {})
    if ss:
        parts.append(f"SOFASCORE: {json.dumps(ss, separators=(',', ':'))[:1500]}")

    return "\n".join(parts) if parts else "NO STATISTICAL DATA AVAILABLE"


def call_groq_sdk(model: str, messages: list, temperature: float = 0.1) -> Optional[str]:
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": CFG.AI_MAX_TOKENS,
    }
    if "qwen" not in model.lower() and "gpt-oss" not in model.lower():
        kwargs["response_format"] = {"type": "json_object"}
    try:
        res = groq_client.chat.completions.create(**kwargs)
        return res.choices[0].message.content
    except Exception as e:
        logger.error("Groq SDK error model=%s: %s", model, e)
        return None


def calculate_system_confidence(ev_edge: float, stats: dict, market: str) -> tuple[int, str]:
    score = 50
    dq = stats.get("data_quality", "none")
    
    if dq == "high": score += 15
    elif dq == "medium": score += 8
    
    ev_pct = ev_edge * 100
    if ev_pct > 5.0: score += 12
    elif ev_pct > 3.0: score += 8
    elif ev_pct > 1.5: score += 4
    
    hf = stats.get("home_form", {})
    af = stats.get("away_form", {})
    if hf and hf.get("win_rate", 0) >= 0.6: score += 4
    if af and af.get("win_rate", 0) >= 0.6: score += 4

    if market.lower() == "totals": score += 3

    score = max(50, min(93, int(score)))

    if score >= 75: risk = "Low"
    elif score >= 60: risk = "Medium"
    else: risk = "High"

    return score, risk


def generate_dual_ai_analysis(
    home: str,
    away: str,
    sport: str,
    pick: str,
    market: str,
    ev_edge: float,
    stats: dict,
) -> dict:
    stats_summary = build_stats_summary(stats, home, away)
    calc_conf, calc_risk = calculate_system_confidence(ev_edge, stats, market)

    default_response = {
        "sport_emoji": "\U0001F3C6",
        "home_flag": get_flag_from_name(home),
        "away_flag": get_flag_from_name(away),
        "risk_level": calc_risk,
        "confidence": calc_conf,
        "logic": "The sharp market indicates significant value on this selection based on underlying metrics.",
    }

    sys_analyst = (
        "You are an elite, professional sports betting analyst for a premium syndicate.\n"
        "Your task is to write EXACTLY two punchy, insightful sentences justifying the given pick.\n"
        "CRITICAL RULES:\n"
        "- NEVER mention 'Expected Value', 'EV', 'data quality', 'points', or internal model metrics.\n"
        "- DO NOT say 'The model indicates'. Sound like a human expert.\n"
        "- Use the provided statistics to highlight form, scoring trends, or historical dominance.\n"
        "- If stats are sparse, focus purely on the mismatch in the odds and sharp market movement.\n"
        "- Determine the EXACT country flag emoji for home_flag and away_flag.\n\n"
        "OUTPUT JSON ONLY: {\"logic\": \"your 2 sentence analysis\", \"sport_emoji\": \"...\", \"home_flag\": \"...\", \"away_flag\": \"...\"}"
    )
    u1 = (
        f"MATCH: {home} vs {away}\n"
        f"SPORT: {sport}\n"
        f"PICK: {pick} [{market}]\n\n"
        f"STATISTICS:\n{stats_summary[:1500]}\n\n"
        "OUTPUT JSON ONLY:"
    )

    analysis_1 = None
    try:
        raw1 = call_groq_sdk(
            CFG.AI_MODEL_ANALYST,
            [{"role": "system", "content": sys_analyst}, {"role": "user", "content": u1}],
            temperature=0.4,
        )
        analysis_1 = robust_json_extractor(raw1)
    except Exception as e:
        logger.warning("Model Analyst failed: %s", e)

    initial_logic = (analysis_1 or {}).get("logic", default_response["logic"])
    
    sys_editor = (
        "You are the Chief Editor for a high-end sports betting platform.\n"
        "Review the provided analysis. If it sounds robotic, mentions internal models, points, or 'EV', REWRITE it.\n"
        "It must sound like a professional sports pundit giving a confident tip.\n"
        "Keep it strictly under 3 sentences.\n"
        "OUTPUT JSON ONLY: {\"validated_logic\": \"...\"}"
    )
    u2 = f"DRAFT ANALYSIS: {initial_logic}\nPICK: {pick}\nOUTPUT JSON ONLY:"

    try:
        raw2 = call_groq_sdk(
            CFG.AI_MODEL_VALIDATOR,
            [{"role": "system", "content": sys_editor}, {"role": "user", "content": u2}],
            temperature=0.2,
        )
        analysis_2 = robust_json_extractor(raw2)
        if analysis_2 and analysis_2.get("validated_logic"):
            initial_logic = analysis_2["validated_logic"]
    except Exception as e:
        logger.warning("Model Editor failed: %s", e)

    result = dict(default_response)
    if analysis_1:
        result["sport_emoji"] = analysis_1.get("sport_emoji", result["sport_emoji"])
        result["home_flag"] = validate_flag(analysis_1.get("home_flag", ""), home)
        result["away_flag"] = validate_flag(analysis_1.get("away_flag", ""), away)
    
    safe_logic = str(initial_logic).strip()
    if len(safe_logic) > 600:
        safe_logic = safe_logic[:597] + "..."
    result["logic"] = safe_logic

    return result

# =========================================================
# 15. TELEGRAM & MAIN PIPELINE
# =========================================================
def send_telegram(message_html: str) -> bool:
    """Sends a message to Telegram, implementing chunking for long texts."""
    MAX_LEN = 4000
    chunks = []
    
    if len(message_html) <= MAX_LEN:
        chunks.append(message_html)
    else:
        # Safe chunking algorithm to prevent HTML tag breaking and 4096 limit
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
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if not res.ok:
            logger.error("Telegram API Error: %s", res.text)
            success = False
            
    return success


def format_message(
    home: str,
    away: str,
    sport: str,
    opp: dict,
    ai_data: dict,
    countdown_str: str,
) -> str:
    risk_raw = str(ai_data.get("risk_level", "Medium")).capitalize()
    risk_icon = {
        "Low": "\U0001F7E2",
        "Medium": "\U0001F7E0",
        "High": "\U0001F534",
    }.get(risk_raw, "\U0001F7E0")

    confidence = ai_data.get("confidence", 60)
    conf_icon = (
        "\U0001F525" if confidence >= 75
        else ("\U00002705" if confidence >= 65 else "\U000026A1")
    )

    raw_logic = str(ai_data.get("logic", "")).replace("<", "").replace(">", "")
    logic_escaped = html_lib.escape(raw_logic)

    sport_emoji = ai_data.get("sport_emoji", "\U0001F3C6")
    home_flag = ai_data.get("home_flag", "\U0001F3F3\uFE0F")
    away_flag = ai_data.get("away_flag", "\U0001F3F3\uFE0F")

    pick_line = (
        f"\U0001F3AF <b>Pick [{opp['market_label']}]:</b> "
        f"<b>{html_lib.escape(opp['pick'])}</b> "
        f"@ <code>{opp['odds']}</code>"
    )

    risk_conf_line = (
        f"{risk_icon} <b>Risk:</b> {risk_raw}"
        f"  |  {conf_icon} <b>Confidence: {confidence}%</b>"
    )

    msg = (
        f"{sport_emoji} <b>{html_lib.escape(sport)}</b>\n\n"
        f"\u2694\uFE0F <b>{html_lib.escape(home)}</b> {home_flag}"
        f"  <b>vs</b>  "
        f"{away_flag} <b>{html_lib.escape(away)}</b>\n\n"
        f"\u23F3 <b>Starts in:</b> {countdown_str}\n\n"
        f"{pick_line}\n\n"
        f"{risk_conf_line}\n\n"
        f"\U0001F4A1 <b>Analysis:</b>\n"
        f"<blockquote>{logic_escaped}</blockquote>\n\n"
        f"\U0001F194 <b>Channel:</b> {CFG.TELEGRAM_ID}"
    )
    return msg


async def async_main():
    logger.info("=" * 60)
    logger.info("ZBET90 ENTERPRISE ENGINE v2.5 STARTING")
    logger.info("=" * 60)

    sent_history = SentHistory()
    football_adapter = FootballDataAdapter()
    match_id_cache = MatchIDCache()
    now_utc = datetime.now(timezone.utc)

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
        markets_data = event.get("_markets_data", {})
        bookmakers_raw = event.get("bookmakers", [])
        countdown_str = get_countdown_str(event.get("commence_time", ""), now_utc)

        if not home or not away:
            continue

        opportunities = calculate_sharp_ev(markets_data, bookmakers_raw)
        if not opportunities:
            continue

        opp = opportunities[0]

        if sent_history.was_sent(home, away, opp["market"]):
            continue

        stats = await get_stats_async(
            home, away, sport_key, football_adapter, match_id_cache
        )

        ai_data = generate_dual_ai_analysis(
            home, away, sport,
            opp["pick"], opp["market"], opp["ev"], stats,
        )
        msg = format_message(home, away, sport, opp, ai_data, countdown_str)

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


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("SYSTEM FAILURE: %s", str(e), exc_info=True)
        sys.exit(1)
