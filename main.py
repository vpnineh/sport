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
from functools import wraps
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# =========================================================
# 1. CENTRALIZED CONFIG CLASS
# =========================================================
@dataclass
class Config:
    # -- Paths --
    CACHE_DIR: Path = Path("api_cache")
    LOG_DIR: Path = Path("log")
    HISTORY_FILE: Path = Path("api_cache/sent_history.json")
    TEAM_ID_CACHE_FILE: Path = Path("api_cache/team_id_cache.json")
    MATCH_ID_CACHE_FILE: Path = Path("api_cache/match_id_cache.json")
    DAILY_STATS_CACHE_FILE: Path = Path("api_cache/daily_stats_cache.json")
    LOG_FILE: Path = Path("api_cache/execution_logs.log")

    # -- Timing --
    MATCH_WINDOW_HOURS: float = 2.0
    TELEGRAM_SLEEP_BETWEEN: float = 3.0

    # -- API Limits --
    FOOTBALL_DATA_DAILY_LIMIT: int = 80  # Free plan 100/day with margin
    # Removed "btts" to prevent 422 errors on cross-sport upcoming endpoints
    ODDS_API_MARKETS: list = field(default_factory=lambda: ["h2h", "totals"])
    ODDS_API_REGIONS: str = "eu,us,uk,au"

    # -- Cache TTLs (hours) --
    TTL_SENT_HISTORY: float = 48.0
    TTL_MATCH_ID: float = 24.0
    TTL_TEAM_FORM: float = 6.0
    TTL_H2H: float = 24.0

    # -- EV Thresholds --
    H2H_MIN_ODDS: float = 1.50
    H2H_MIN_EV: float = 0.015
    TOTALS_MIN_ODDS: float = 1.60
    TOTALS_MIN_EV: float = 0.020

    # -- Sharp Market Validation --
    MARKET_EXPECTED_OUTCOMES: dict = field(default_factory=lambda: {
        "h2h": {"min": 2, "max": 3},
        "totals": {"min": 2, "max": 2}
    })
    MAX_VALID_IMPLIED_SUM: float = 1.20
    MIN_VALID_IMPLIED_SUM: float = 0.80

    # -- AI Models --
    AI_MODEL_ANALYST: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    AI_MODEL_VALIDATOR: str = "qwen-qwq-32b"
    # Increased tokens to prevent QwQ reasoning truncation
    AI_MAX_TOKENS: int = 4096

    # -- Telegram --
    TELEGRAM_ID: str = "@zBET90"

    # -- Sharp Bookmakers --
    SHARP_BOOKMAKERS: list = field(default_factory=lambda: [
        "pinnacle", "betfair_ex_eu", "matchbook", "betfair_ex_uk"
    ])

CFG = Config()

# =========================================================
# 2. ENTERPRISE LOGGING
# =========================================================
CFG.CACHE_DIR.mkdir(exist_ok=True)
CFG.LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("ZBET90_ENGINE")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler(CFG.LOG_FILE, mode='a', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# =========================================================
# 3. API KEYS
# =========================================================
ODDS_API_KEY          = os.getenv("ODDS_API_KEY")
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")
RAPIDAPI_KEY          = os.getenv("RAPIDAPI_KEY")
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID")
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

if not all([ODDS_API_KEY, GROQ_API_KEY, RAPIDAPI_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing critical API Keys in GitHub Secrets.")
    sys.exit(1)

# =========================================================
# 4. CACHE MANAGER
# =========================================================
class CacheManager:
    @staticmethod
    def load(filepath: Path) -> dict:
        try:
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Cache load error ({filepath.name}): {e}")
        return {}

    @staticmethod
    def save(filepath: Path, data: dict):
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Cache save error ({filepath.name}): {e}")

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
            "data": value
        }
        return cache

    @staticmethod
    def get(cache: dict, key: str):
        return cache.get(key, {}).get("data")

# =========================================================
# 5. SENT HISTORY (ANTI-DUPLICATE)
# =========================================================
class SentHistory:
    def __init__(self):
        self.history = CacheManager.load(CFG.HISTORY_FILE)
        self._cleanup_old()

    def _cleanup_old(self):
        now = datetime.now(timezone.utc)
        to_delete = [
            k for k, v in self.history.items()
            if now - datetime.fromisoformat(
                v.get("sent_at", "2000-01-01T00:00:00+00:00")
            ) > timedelta(hours=CFG.TTL_SENT_HISTORY)
        ]
        for k in to_delete:
            del self.history[k]

    def _make_key(self, home: str, away: str, pick: str, market: str) -> str:
        raw = f"{home.lower()}|{away.lower()}|{pick.lower()}|{market.lower()}"
        return hashlib.md5(raw.encode()).hexdigest()

    def was_sent(self, home: str, away: str, pick: str, market: str) -> bool:
        return self._make_key(home, away, pick, market) in self.history

    def mark_sent(self, home: str, away: str, pick: str, market: str):
        key = self._make_key(home, away, pick, market)
        self.history[key] = {
            "match": f"{home} vs {away}",
            "pick": pick,
            "market": market,
            "sent_at": datetime.now(timezone.utc).isoformat()
        }
        CacheManager.save(CFG.HISTORY_FILE, self.history)

# =========================================================
# 6. UTILS
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
                        wait_time = int(e.response.headers.get("Retry-After", current_delay * 3))
                        logger.warning(f"Rate Limit (429) in {func.__name__}. Sleeping {wait_time}s...")
                        time.sleep(wait_time)
                    elif status in [401, 403]:
                        logger.error(f"Auth Error in {func.__name__}: HTTP {status}")
                        return None
                    else:
                        logger.error(f"HTTP {status} in {func.__name__}: {e}")
                        if attempt == max_retries - 1:
                            return None
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout in {func.__name__}. Attempt {attempt+1}/{max_retries}")
                    if attempt == max_retries - 1:
                        return None
                except requests.exceptions.RequestException as e:
                    logger.error(f"Connection Error in {func.__name__}: {e}")
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
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r'\{[\s\S]*\}', raw_text)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    logger.error(f"JSON parse failed: {raw_text[:300]}")
    return None

def clean_team_name(name: str) -> str:
    cleaned = re.sub(r'\s*\([^)]*\)', '', str(name)).strip()
    return cleaned

def normalize_sport_key(sport_title: str) -> str:
    football_kws = [
        "soccer", "football", "premier league", "la liga", "bundesliga",
        "serie a", "ligue 1", "champions league", "europa league",
        "mls", "eredivisie", "primeira liga", "championship", "league cup"
    ]
    tl = sport_title.lower()
    return "football" if any(kw in tl for kw in football_kws) else "other"

def get_countdown_str(commence_time_str: str, now_utc: datetime) -> str:
    try:
        match_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        delta = match_time - now_utc
        minutes_left = int(delta.total_seconds() / 60)
        if minutes_left > 60:
            return f"{minutes_left // 60}h {minutes_left % 60}m"
        elif minutes_left > 0:
            return f"{minutes_left}m"
        else:
            return "LIVE"
    except Exception:
        return "N/A"

# =========================================================
# 7. CORE MATH ENGINE
# =========================================================
def validate_sharp_odds(sharp_odds: dict, market_key: str) -> tuple[bool, str]:
    if not sharp_odds:
        return False, "empty sharp_odds"

    n_outcomes = len(sharp_odds)
    expected = CFG.MARKET_EXPECTED_OUTCOMES.get(market_key, {"min": 2, "max": 3})

    if n_outcomes < expected["min"]:
        return False, (
            f"Incomplete market: only {n_outcomes} outcomes "
            f"(expected min {expected['min']}) for {market_key}"
        )

    try:
        implied_sum = sum(1.0 / v["price"] for v in sharp_odds.values())
    except (KeyError, ZeroDivisionError) as e:
        return False, f"Price calculation error: {e}"

    if implied_sum < CFG.MIN_VALID_IMPLIED_SUM:
        return False, f"implied_sum too low ({implied_sum:.3f})"

    if implied_sum > CFG.MAX_VALID_IMPLIED_SUM:
        return False, f"implied_sum too high ({implied_sum:.3f})"

    return True, "valid"

def calculate_sharp_ev(markets_data: dict, bookmakers_raw: list) -> list:
    opportunities = []

    for market_key, market_data_list in markets_data.items():
        sharp_odds = {}
        best_odds = {}

        for entry in market_data_list:
            bk = entry.get("bookmaker_key", "")
            if bk in CFG.SHARP_BOOKMAKERS:
                for o in entry.get("outcomes", []):
                    name, price = o["name"], float(o["price"])
                    if price <= 1.0:
                        continue
                    if name not in sharp_odds or price > sharp_odds[name]["price"]:
                        sharp_odds[name] = {
                            "price": price,
                            "bookmaker": entry["bookmaker"]
                        }

        for entry in market_data_list:
            for o in entry.get("outcomes", []):
                name, price = o["name"], float(o["price"])
                if price <= 1.0:
                    continue
                if name not in best_odds or price > best_odds[name]["price"]:
                    best_odds[name] = {
                        "price": price,
                        "bookmaker": entry["bookmaker"]
                    }

        if not sharp_odds and best_odds:
            sharp_odds = {k: v for k, v in best_odds.items()}

        is_valid, reason = validate_sharp_odds(sharp_odds, market_key)
        if not is_valid:
            logger.debug(f"Skipping {market_key}: {reason}")
            continue

        implied_sum = sum(1.0 / v["price"] for v in sharp_odds.values())

        market_opportunities = []
        for outcome_name, sharp_data in sharp_odds.items():
            true_prob = (1.0 / sharp_data["price"]) / implied_sum
            best = best_odds.get(outcome_name, {})
            best_price = best.get("price", 0.0)
            best_bookie = best.get("bookmaker", "Unknown")

            if best_price <= 1.0:
                continue

            ev = (true_prob * best_price) - 1.0

            if market_key == "h2h":
                min_odds, min_ev = CFG.H2H_MIN_ODDS, CFG.H2H_MIN_EV
            else:
                min_odds, min_ev = CFG.TOTALS_MIN_ODDS, CFG.TOTALS_MIN_EV

            if best_price >= min_odds and ev > min_ev:
                market_label = {
                    "h2h": "Winner",
                    "totals": "Over/Under"
                }.get(market_key, market_key.upper())

                market_opportunities.append({
                    "pick": outcome_name,
                    "market": market_key,
                    "market_label": market_label,
                    "prob": round(true_prob, 4),
                    "odds": round(best_price, 3),
                    "bookmaker": best_bookie,
                    "ev": round(ev, 4),
                    "edge_pct": round(ev * 100, 2),
                    "implied_sum": round(implied_sum, 4)
                })

        opportunities.extend(market_opportunities)

    opportunities.sort(key=lambda x: x["ev"], reverse=True)
    return opportunities[:2]

# =========================================================
# 8. ASYNC ODDS API FETCHER
# =========================================================
async def fetch_market_async(session: aiohttp.ClientSession, market: str, now_utc: datetime) -> list:
    end_window = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    url = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": CFG.ODDS_API_REGIONS,
        "markets": market,
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as res:
            if res.status == 429:
                logger.warning(f"Odds API rate limit for market {market}")
                return []
            if res.status != 200:
                logger.error(f"Odds API HTTP {res.status} for market {market}")
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
            logger.info(f"Market '{market}': {len(filtered)} events in window")
            return filtered
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching market {market}")
        return []
    except Exception as e:
        logger.error(f"Async fetch error for market {market}: {e}")
        return []

async def fetch_all_odds_async() -> list:
    now_utc = datetime.now(timezone.utc)
    all_events: dict[str, dict] = {}

    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_market_async(session, market, now_utc)
            for market in CFG.ODDS_API_MARKETS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, market_events in enumerate(results):
        if isinstance(market_events, Exception):
            logger.error(f"Market fetch exception: {market_events}")
            continue
        for e in market_events:
            eid = e["id"]
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
                        "outcomes": m.get("outcomes", [])
                    })

    result = list(all_events.values())
    logger.info(f"Total unique events in window: {len(result)}")
    return result

# =========================================================
# 9. ASYNC STATS FETCHER
# =========================================================
async def fetch_sofascore_endpoint_async(
    session: aiohttp.ClientSession,
    url: str, params: dict
) -> Optional[dict]:
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY or "",
        "x-rapidapi-host": "sofascore.p.rapidapi.com"
    }
    try:
        async with session.get(
            url, headers=headers, params=params,
            timeout=aiohttp.ClientTimeout(total=12)
        ) as res:
            if res.status == 200:
                return await res.json()
    except Exception as e:
        logger.debug(f"SofaScore async error {url}: {e}")
    return None

async def fetch_sofascore_stats_async(match_id: int) -> dict:
    endpoints = {
        "h2h":     "https://sofascore.p.rapidapi.com/matches/get-h2h-events",
        "lineups": "https://sofascore.p.rapidapi.com/matches/get-lineups",
        "streaks": "https://sofascore.p.rapidapi.com/matches/get-team-streaks",
    }
    params = {"matchId": str(match_id)}
    connector = aiohttp.TCPConnector(limit=5, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = {
            key: fetch_sofascore_endpoint_async(session, url, params)
            for key, url in endpoints.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    data = {}
    for key, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.debug(f"SofaScore {key} exception: {result}")
        elif result is not None:
            data[key] = result
    return data

async def search_sofascore_match_async(home: str, away: str) -> Optional[int]:
    clean_home = clean_team_name(home)
    clean_away = clean_team_name(away)
    query = f"{clean_home} {clean_away}"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY or "",
        "x-rapidapi-host": "sofascore.p.rapidapi.com"
    }
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            async with session.get(
                "https://sofascore.p.rapidapi.com/search",
                headers=headers,
                params={"q": query, "page": "0"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as res:
                if res.status != 200:
                    return None
                data = await res.json()
                for result in data.get("results", []):
                    if result.get("type") == "event":
                        return result.get("entity", {}).get("id")
        except Exception as e:
            logger.debug(f"SofaScore search error: {e}")
    return None

# =========================================================
# 10. FOOTBALL-DATA.ORG ADAPTER
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
        last_reset = entry.get("timestamp", "2000-01-01T00:00:00+00:00")
        try:
            last_reset_dt = datetime.fromisoformat(last_reset)
            if datetime.now(timezone.utc).date() > last_reset_dt.date():
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
            logger.warning(
                f"Football-Data daily limit ({self.call_count}/{CFG.FOOTBALL_DATA_DAILY_LIMIT})"
            )
            return {}
        res = requests.get(
            f"{self.BASE_URL}{endpoint}",
            headers=self.headers,
            params=params,
            timeout=12
        )
        res.raise_for_status()
        self._increment_call()
        logger.debug(f"Football-Data call #{self.call_count}: {endpoint}")
        return res.json()

    def find_team_id(self, team_name: str) -> Optional[int]:
        team_id_cache = CacheManager.load(CFG.TEAM_ID_CACHE_FILE)
        key = team_name.lower().strip()
        if key in team_id_cache:
            return team_id_cache[key]
        if not self._can_call():
            return None
        logger.info(f"Searching team ID: '{team_name}'")
        data = self._raw_get("/teams", {"name": clean_team_name(team_name)})
        team_id = None
        if data and data.get("teams"):
            team_id = data["teams"][0]["id"]
            logger.info(f"Team found: {team_name} -> ID:{team_id}")
        team_id_cache[key] = team_id
        CacheManager.save(CFG.TEAM_ID_CACHE_FILE, team_id_cache)
        return team_id

    def get_team_recent_form(self, team_id: int, team_name: str) -> dict:
        cache_key = f"form_{team_id}"
        if CacheManager.is_valid(self.daily_cache, cache_key, CFG.TTL_TEAM_FORM):
            return CacheManager.get(self.daily_cache, cache_key) or {}
        logger.info(f"Fetching form: {team_name} (id:{team_id})")
        data = self._raw_get(f"/teams/{team_id}/matches", {"status": "FINISHED", "limit": 5})
        if not data:
            return {}
        form = self._parse_form(data, team_id)
        self.daily_cache = CacheManager.set(self.daily_cache, cache_key, form)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)
        return form

    def _parse_form(self, data: dict, team_id: int) -> dict:
        matches = data.get("matches", [])
        results, goals_scored, goals_conceded = [], [], []
        for m in matches[-5:]:
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
                sum(1 for s, c in zip(goals_scored, goals_conceded) if s > 0 and c > 0) / total, 2
            ),
            "over25_rate": round(
                sum(1 for s, c in zip(goals_scored, goals_conceded) if s + c > 2.5) / total, 2
            ),
            "matches_analyzed": total
        }

    def get_h2h(self, team1_id: int, team2_id: int) -> dict:
        cache_key = f"h2h_{min(team1_id,team2_id)}_{max(team1_id,team2_id)}"
        if CacheManager.is_valid(self.daily_cache, cache_key, CFG.TTL_H2H):
            return CacheManager.get(self.daily_cache, cache_key) or {}
        logger.info(f"Fetching H2H: {team1_id} vs {team2_id}")
        data = self._raw_get(f"/teams/{team1_id}/matches", {"status": "FINISHED", "limit": 20})
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
                (t1_wins if home_id == team1_id else t2_wins).__add__(1)
                if home_id == team1_id: t1_wins += 1
                else: t2_wins += 1
            elif ag > hg:
                if home_id != team1_id: t1_wins += 1
                else: t2_wins += 1
            else:
                draws += 1
            total_goals += hg + ag
            if hg > 0 and ag > 0: btts += 1
            if hg + ag > 2.5: over25 += 1
        if total == 0:
            return {}
        return {
            "total_h2h": total,
            "team1_wins": t1_wins,
            "team2_wins": t2_wins,
            "draws": draws,
            "avg_goals_per_game": round(total_goals / total, 2),
            "btts_rate": round(btts / total, 2),
            "over25_rate": round(over25 / total, 2)
        }

# =========================================================
# 11. MATCH ID CACHE
# =========================================================
class MatchIDCache:
    def __init__(self):
        self.cache = CacheManager.load(CFG.MATCH_ID_CACHE_FILE)

    def get(self, home: str, away: str) -> Optional[int]:
        key = self._key(home, away)
        if CacheManager.is_valid(self.cache, key, CFG.TTL_MATCH_ID):
            return CacheManager.get(self.cache, key)
        return None

    def set(self, home: str, away: str, match_id: Optional[int]):
        key = self._key(home, away)
        self.cache = CacheManager.set(self.cache, key, match_id)
        CacheManager.save(CFG.MATCH_ID_CACHE_FILE, self.cache)

    @staticmethod
    def _key(home: str, away: str) -> str:
        raw = f"{home.lower()}|{away.lower()}"
        return hashlib.md5(raw.encode()).hexdigest()

# =========================================================
# 12. ASYNC STATS AGGREGATOR
# =========================================================
async def get_stats_async(
    home: str, away: str,
    sport_key: str,
    football_adapter: FootballDataAdapter,
    match_id_cache: MatchIDCache
) -> dict:
    stats = {
        "home_form": {},
        "away_form": {},
        "h2h": {},
        "sofascore": {},
        "data_quality": "none"
    }

    cached_mid = match_id_cache.get(home, away)
    if cached_mid is not None:
        match_id = cached_mid if cached_mid != 0 else None
    else:
        logger.info(f"Searching SofaScore match ID: {home} vs {away}")
        match_id = await search_sofascore_match_async(home, away)
        match_id_cache.set(home, away, match_id if match_id else 0)

    tasks = []

    if match_id:
        tasks.append(("sofascore", fetch_sofascore_stats_async(match_id)))

    if sport_key == "football":
        loop = asyncio.get_event_loop()

        async def get_football_data():
            home_id = await loop.run_in_executor(None, football_adapter.find_team_id, home)
            away_id = await loop.run_in_executor(None, football_adapter.find_team_id, away)
            if home_id and away_id:
                home_form = await loop.run_in_executor(
                    None, football_adapter.get_team_recent_form, home_id, home
                )
                await asyncio.sleep(0.3)
                away_form = await loop.run_in_executor(
                    None, football_adapter.get_team_recent_form, away_id, away
                )
                await asyncio.sleep(0.3)
                h2h = await loop.run_in_executor(
                    None, football_adapter.get_h2h, home_id, away_id
                )
                return {"home_form": home_form, "away_form": away_form, "h2h": h2h}
            return {}

        tasks.append(("football", get_football_data()))

    if tasks:
        names = [t[0] for t in tasks]
        coros = [t[1] for t in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.warning(f"Stats fetch error ({name}): {result}")
                continue
            if name == "sofascore" and result:
                stats["sofascore"] = result
            elif name == "football" and result:
                stats.update(result)

    has_football = bool(stats.get("home_form") or stats.get("h2h"))
    has_sofascore = bool(stats.get("sofascore"))
    if has_football and has_sofascore:
        stats["data_quality"] = "high"
    elif has_football or has_sofascore:
        stats["data_quality"] = "medium"

    logger.info(f"Stats quality for {home} vs {away}: {stats['data_quality']}")
    return stats

# =========================================================
# 13. DUAL-AI ANALYSIS
# =========================================================
def build_stats_summary(stats: dict, home: str, away: str) -> str:
    parts = []
    hf = stats.get("home_form", {})
    af = stats.get("away_form", {})
    h2h = stats.get("h2h", {})

    if hf:
        parts.append(
            f"HOME ({home}): Form={hf.get('form_string','N/A')} | "
            f"WR={hf.get('win_rate',0):.0%} | "
            f"AvgGF={hf.get('avg_goals_scored',0)} | "
            f"AvgGA={hf.get('avg_goals_conceded',0)} | "
            f"BTTS={hf.get('btts_rate',0):.0%} | "
            f"O2.5={hf.get('over25_rate',0):.0%}"
        )
    if af:
        parts.append(
            f"AWAY ({away}): Form={af.get('form_string','N/A')} | "
            f"WR={af.get('win_rate',0):.0%} | "
            f"AvgGF={af.get('avg_goals_scored',0)} | "
            f"AvgGA={af.get('avg_goals_conceded',0)} | "
            f"BTTS={af.get('btts_rate',0):.0%} | "
            f"O2.5={af.get('over25_rate',0):.0%}"
        )
    if h2h and h2h.get("total_h2h", 0) > 0:
        parts.append(
            f"H2H (n={h2h['total_h2h']}): "
            f"HomeW={h2h.get('team1_wins',0)} | "
            f"AwayW={h2h.get('team2_wins',0)} | "
            f"D={h2h.get('draws',0)} | "
            f"AvgGoals={h2h.get('avg_goals_per_game',0)} | "
            f"BTTS={h2h.get('btts_rate',0):.0%} | "
            f"O2.5={h2h.get('over25_rate',0):.0%}"
        )
    ss = stats.get("sofascore", {})
    if ss:
        ss_str = json.dumps(ss, separators=(',', ':'))[:1500]
        parts.append(f"SOFASCORE: {ss_str}")
    return "\n".join(parts) if parts else "NO STATISTICAL DATA AVAILABLE"

@retry_request(max_retries=3, delay=2)
def call_groq(model: str, messages: list, temperature: float = 0.1) -> Optional[str]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": CFG.AI_MAX_TOKENS
    }
    
    # Conditional JSON mode to prevent 400 Bad Request from reasoning models
    if "qwen" not in model.lower():
        payload["response_format"] = {"type": "json_object"}
        
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def generate_dual_ai_analysis(
    home: str, away: str, sport: str,
    pick: str, market: str, ev_edge: float, stats: dict
) -> dict:
    stats_summary = build_stats_summary(stats, home, away)
    data_quality = stats.get("data_quality", "none")

    default_response = {
        "sport_emoji": "🏆",
        "home_flag": "🏳️",
        "away_flag": "🏳️",
        "risk_level": "Medium",
        "confidence": 55,
        "logic": "Mathematical edge confirmed by Sharp Market Model.",
        "key_stat": "Positive EV detected vs sharp lines"
    }

    sys1 = (
        "You are an Elite Quantitative Sports Analyst.\n"
        "A mathematical model found a +EV betting opportunity.\n\n"
        "YOUR TASKS:\n"
        "1. Write EXACTLY 2 sentences justifying the Pick using specific numbers from the stats. No generic statements.\n"
        "2. Extract the single most impactful statistic supporting this pick (key_stat).\n"
        "3. Choose the correct sport_emoji.\n"
        "4. Determine the nationality of the teams/players and output the correct country flag emojis for home_flag and away_flag (e.g. 🇪🇸, 🇺🇸). Do not use generic flags.\n"
        "5. Assign risk_level: Low, Medium, High.\n\n"
        "OUTPUT: valid JSON object only. No markdown.\n"
        '{"sport_emoji":"⚽","home_flag":"🇪🇸","away_flag":"🇺🇸",'
        '"risk_level":"Medium","logic":"sentence1. sentence2.",'
        '"key_stat":"Specific stat here."}'
    )
    u1 = (
        f"MATCH: {home} vs {away}\n"
        f"SPORT: {sport}\n"
        f"PICK: {pick} [{market}]\n"
        f"EV EDGE: +{ev_edge:.1%}\n"
        f"DATA QUALITY: {data_quality}\n\n"
        f"STATISTICS:\n{stats_summary}\n\n"
        "Respond with JSON only."
    )

    analysis_1 = None
    try:
        raw1 = call_groq(
            CFG.AI_MODEL_ANALYST,
            [{"role": "system", "content": sys1}, {"role": "user", "content": u1}],
            temperature=0.1
        )
        analysis_1 = robust_json_extractor(raw1)
        logger.info("Model 1 (llama-4-scout) complete")
    except Exception as e:
        logger.warning(f"Model 1 failed: {e}")

    time.sleep(1.5)

    initial_logic = (analysis_1 or {}).get("logic", "No initial analysis")
    initial_risk = (analysis_1 or {}).get("risk_level", "Medium")

    sys2 = (
        "You are a Senior Betting Risk Analyst validating a prediction.\n\n"
        "SCORING RUBRIC for confidence (integer 50-95):\n"
        "+ Data quality: high=+15, medium=+8, none=0\n"
        "+ EV edge: >5%=+10, 3-5%=+7, 1.5-3%=+4\n"
        "+ Form consistency (3+ same results): +8\n"
        "+ H2H history supports pick: +7, neutral: +2, against: -5\n"
        "+ Market type bonus: totals=+3, btts=+2, h2h=0\n"
        "Cap final score at 93 (never claim 100% certainty).\n\n"
        "If initial logic is vague or generic, rewrite it with specific stats.\n\n"
        "OUTPUT: valid JSON object only. No markdown.\n"
        '{"confidence":72,"validated_logic":"Rewritten or approved logic here.",'
        '"risk_adjustment":"Medium"}\n\n'
        "Respond with JSON only."
    )
    u2 = (
        f"MATCH: {home} vs {away} | PICK: {pick} [{market}] | EV: +{ev_edge:.1%}\n"
        f"DATA QUALITY: {data_quality}\n"
        f"STATS: {stats_summary[:2000]}\n"
        f"INITIAL LOGIC: {initial_logic}\n"
        f"INITIAL RISK: {initial_risk}\n\n"
        "Respond with JSON only."
    )

    analysis_2 = None
    try:
        raw2 = call_groq(
            CFG.AI_MODEL_VALIDATOR,
            [{"role": "system", "content": sys2}, {"role": "user", "content": u2}],
            temperature=0.05
        )
        analysis_2 = robust_json_extractor(raw2)
        logger.info("Model 2 (qwen-qwq-32b) complete")
    except Exception as e:
        logger.warning(f"Model 2 failed: {e}")

    result = {**default_response}
    if analysis_1:
        result.update({k: v for k, v in analysis_1.items() if v})
    if analysis_2:
        if analysis_2.get("validated_logic"):
            result["logic"] = analysis_2["validated_logic"]
        if analysis_2.get("confidence"):
            result["confidence"] = max(50, min(93, int(analysis_2["confidence"])))
        if analysis_2.get("risk_adjustment"):
            result["risk_level"] = analysis_2["risk_adjustment"]

    if result["confidence"] == 55:
        result["confidence"] = min(70, 55 + int(ev_edge * 300))

    return result

# =========================================================
# 14. TELEGRAM
# =========================================================
@retry_request(max_retries=3)
def send_telegram(message_html: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    res = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }, timeout=10)
    res.raise_for_status()
    return True

def format_message(
    home: str, away: str, sport: str,
    opp: dict, ai_data: dict, countdown_str: str
) -> str:
    risk_raw = str(ai_data.get("risk_level", "Medium")).capitalize()
    risk_icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(risk_raw, "🟠")
    confidence = ai_data.get("confidence", 60)
    conf_icon = "🔥" if confidence >= 75 else ("✅" if confidence >= 65 else "⚡")

    logic = html_lib.escape(
        str(ai_data.get("logic", "")).replace('<', '').replace('>', '')
    )
    key_stat = html_lib.escape(
        str(ai_data.get("key_stat", "")).replace('<', '').replace('>', '')
    )

    msg = (
        f"{ai_data.get('sport_emoji','🏆')} <b>{html_lib.escape(sport)}</b>\n\n"
        f"⚔️ <b>{html_lib.escape(home)}</b> {ai_data.get('home_flag','🏳️')}"
        f"  <b>vs</b>  "
        f"{ai_data.get('away_flag','🏳️')} <b>{html_lib.escape(away)}</b>\n\n"
        f"⏳ <b>Starts in:</b> {countdown_str}\n\n"
        f"🎯 <b>Pick [{opp['market_label']}]:</b> <b>{html_lib.escape(opp['pick'])}</b>\n"
        f"💰 <b>Best Odds:</b> <code>{opp['odds']}</code>"
        f" <i>@ {html_lib.escape(opp['bookmaker'])}</i>\n\n"
        f"📊 <b>EV Edge:</b> +{opp['edge_pct']:.1f}%\n"
        f"{risk_icon} <b>Risk:</b> {risk_raw}"
        f"  |  {conf_icon} <b>Confidence: {confidence}%</b>\n\n"
        f"💡 <b>Analysis:</b>\n{logic}\n\n"
    )
    if key_stat:
        msg += f"📌 <b>Key Stat:</b> <i>{key_stat}</i>\n\n"
    msg += f"🆔 <b>Channel:</b> {CFG.TELEGRAM_ID}"
    return msg

# =========================================================
# 15. ASYNC MAIN PIPELINE
# =========================================================
async def async_main():
    logger.info("=" * 60)
    logger.info("ZBET90 ENTERPRISE ENGINE v2.1 STARTING")
    logger.info("=" * 60)

    sent_history = SentHistory()
    football_adapter = FootballDataAdapter()
    match_id_cache = MatchIDCache()
    now_utc = datetime.now(timezone.utc)

    logger.info("Fetching events (async, all markets simultaneously)...")
    events = await fetch_all_odds_async()

    if not events:
        logger.info("No events found in the 2-hour window.")
        return

    logger.info(f"Analyzing {len(events)} events for +EV opportunities...")

    total_sent = 0

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        sport = event.get("sport_title", "Unknown")
        sport_key = normalize_sport_key(sport)
        markets_data = event.get("_markets_data", {})
        bookmakers_raw = event.get("bookmakers", [])
        countdown_str = get_countdown_str(
            event.get("commence_time", ""), now_utc
        )

        if not home or not away:
            continue

        opportunities = calculate_sharp_ev(markets_data, bookmakers_raw)
        if not opportunities:
            continue

        logger.info(f"{len(opportunities)} EV opportunity(ies): {home} vs {away}")

        stats = await get_stats_async(
            home, away, sport_key, football_adapter, match_id_cache
        )

        for opp in opportunities:
            if sent_history.was_sent(home, away, opp["pick"], opp["market"]):
                logger.info(f"SKIP duplicate: {home} vs {away} | {opp['pick']}")
                continue

            logger.info(
                f"Dual-AI analysis: {home} vs {away} | "
                f"{opp['pick']} | EV:+{opp['edge_pct']:.1f}%"
            )
            ai_data = generate_dual_ai_analysis(
                home, away, sport,
                opp["pick"], opp["market"], opp["ev"], stats
            )
            msg = format_message(home, away, sport, opp, ai_data, countdown_str)

            if send_telegram(msg):
                sent_history.mark_sent(home, away, opp["pick"], opp["market"])
                total_sent += 1
                logger.info(f"Sent: {home} vs {away} | {opp['pick']}")
            else:
                logger.error(f"Telegram failed: {home} vs {away}")

            await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    logger.info("=" * 60)
    logger.info(
        f"Done! {total_sent} signal(s) sent."
        if total_sent > 0
        else "No qualifying +EV opportunities found."
    )
    logger.info("=" * 60)

# =========================================================
# 16. ENTRY POINT
# =========================================================
def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"SYSTEM FAILURE: {str(e)}", exc_info=True)
        sys.exit(1)
