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
# 1. CONFIG
# =========================================================
@dataclass
class Config:
    CACHE_DIR: Path = Path("api_cache")
    LOG_DIR: Path = Path("log")
    MODELS_DIR: Path = Path("api_cache/models")
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
    MAX_REALISTIC_EV: float = 0.12

    MARKET_EXPECTED_OUTCOMES: dict = field(default_factory=lambda: {
        "h2h": {"min": 2, "max": 3},
        "totals": {"min": 2, "max": 2},
    })
    MAX_VALID_IMPLIED_SUM: float = 1.20
    MIN_VALID_IMPLIED_SUM: float = 0.80

    ELO_K_FACTOR_FOOTBALL: float = 32.0
    ELO_K_FACTOR_TENNIS: float = 40.0
    ELO_HOME_ADVANTAGE: float = 80.0
    ELO_DEFAULT: float = 1500.0

    AI_MODEL_ANALYST: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    AI_MODEL_VALIDATOR: str = "llama-3.1-8b-instant"
    AI_MAX_TOKENS: int = 1024

    TELEGRAM_ID: str = "@zBET90"

    SHARP_BOOKMAKERS: list = field(default_factory=lambda: [
        "pinnacle", "betfair_ex_eu", "matchbook", "betfair_ex_uk"
    ])

    MARKET_DISPLAY: dict = field(default_factory=lambda: {
        "h2h":     "1X2 Full Time",
        "totals":  "Over / Under Goals",
        "spreads": "Asian Handicap",
    })

    # football-data.org competition IDs (tier-based)
    FD_COMPETITION_IDS: list = field(default_factory=lambda: [
        2021,  # Premier League
        2014,  # La Liga
        2002,  # Bundesliga
        2019,  # Serie A
        2015,  # Ligue 1
        2003,  # Eredivisie
        2017,  # Primeira Liga
        2016,  # Championship
        2018,  # European Championship (fallback)
        2001,  # Champions League
    ])


CFG = Config()

# =========================================================
# 2. LOGGING  — خیلی مهم: همه API call ها لاگ می‌شن
# =========================================================
for d in [CFG.CACHE_DIR, CFG.LOG_DIR, CFG.MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ZBET90")
logger.setLevel(logging.DEBUG)   # DEBUG تا همه چیز دیده شود

_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

_fh = logging.FileHandler(CFG.LOG_FILE, mode="a", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


def log_section(title: str):
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def log_api_call(api_name: str, url: str, params: dict,
                 status: int, records: int, sample=None):
    """
    هر API call را با جزئیات کامل لاگ می‌کند.
    status=-1 یعنی خطا قبل از دریافت response.
    """
    logger.info(
        "API▶ %-18s | %s | status=%s | records=%d | params=%s",
        api_name,
        url[:80],
        status if status != -1 else "ERR",
        records,
        str(params)[:120],
    )
    if sample is not None:
        logger.debug("API▶ %-18s | sample=%s", api_name,
                     str(sample)[:200])


def log_check(label: str, value, warn_if_none: bool = True):
    if value is None or value == {} or value == [] or value == "":
        if warn_if_none:
            logger.warning("CHECK | %-42s | EMPTY/NONE", label)
        else:
            logger.info("CHECK | %-42s | EMPTY (ok)", label)
    else:
        display = str(value)[:100]
        logger.info("CHECK | %-42s | OK | %s", label, display)

# =========================================================
# 3. API KEYS
# =========================================================
ODDS_API_KEY           = os.getenv("ODDS_API_KEY")
GROQ_API_KEY           = os.getenv("GROQ_API_KEY")
RAPIDAPI_KEY           = os.getenv("RAPIDAPI_KEY")
TELEGRAM_BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")
FOOTBALL_DATA_API_KEY  = os.getenv("FOOTBALL_DATA_API_KEY")
FORCE_BOOTSTRAP        = os.getenv("FORCE_BOOTSTRAP", "false").lower() == "true"

log_check("ODDS_API_KEY",          bool(ODDS_API_KEY))
log_check("GROQ_API_KEY",          bool(GROQ_API_KEY))
log_check("RAPIDAPI_KEY",          bool(RAPIDAPI_KEY))
log_check("TELEGRAM_BOT_TOKEN",    bool(TELEGRAM_BOT_TOKEN))
log_check("TELEGRAM_CHAT_ID",      bool(TELEGRAM_CHAT_ID))
log_check("FOOTBALL_DATA_API_KEY", bool(FOOTBALL_DATA_API_KEY),
          warn_if_none=False)

if not all([ODDS_API_KEY, GROQ_API_KEY, RAPIDAPI_KEY,
            TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing critical API Keys.")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_API_KEY, max_retries=3)

# =========================================================
# 4. NATIONALITY FLAGS
# =========================================================
NATIONALITY_FLAGS: dict = {
    # Tennis
    "bautista agut": "ES", "alcaraz": "ES", "nadal": "ES",
    "munar": "ES", "davidovich": "ES", "carreno": "ES", "badosa": "ES",
    "djokovic": "RS", "kecmanovic": "RS",
    "sinner": "IT", "berrettini": "IT", "musetti": "IT",
    "zverev": "DE", "struff": "DE", "koepfer": "DE",
    "tiafoe": "US", "fritz": "US", "paul": "US", "nakashima": "US",
    "sock": "US", "isner": "US", "korda": "US",
    "gauff": "US", "keys": "US", "pegula": "US", "collins": "US",
    "medvedev": "RU", "rublev": "RU", "khachanov": "RU",
    "tsitsipas": "GR", "ruud": "NO", "rune": "DK",
    "hurkacz": "PL", "swiatek": "PL",
    "cilic": "HR",
    "auger-aliassime": "CA", "shapovalov": "CA", "raonic": "CA",
    "kyrgios": "AU", "de minaur": "AU", "thompson": "AU",
    "sabalenka": "BY",
    "kvitova": "CZ", "vondrousova": "CZ",
    "jabeur": "TN", "rybakina": "KZ", "bublik": "KZ",
    "norrie": "GB", "murray": "GB", "draper": "GB",
    "thiem": "AT", "wawrinka": "CH",
    "monfils": "FR", "gasquet": "FR",
    "dimitrov": "BG",
    "etcheverry": "AR", "cerundolo": "AR", "schwartzman": "AR",
    # Football clubs
    "manchester united": "GB", "manchester city": "GB",
    "liverpool": "GB", "chelsea": "GB", "arsenal": "GB",
    "tottenham": "GB", "newcastle": "GB", "west ham": "GB",
    "aston villa": "GB", "everton": "GB", "brighton": "GB",
    "celtic": "GB", "rangers": "GB",
    "real madrid": "ES", "barcelona": "ES", "atletico": "ES",
    "sevilla": "ES", "valencia": "ES", "villarreal": "ES",
    "real sociedad": "ES", "athletic bilbao": "ES",
    "bayern": "DE", "dortmund": "DE", "leipzig": "DE",
    "leverkusen": "DE", "frankfurt": "DE",
    "juventus": "IT", "milan": "IT", "inter": "IT",
    "napoli": "IT", "roma": "IT", "lazio": "IT", "atalanta": "IT",
    "psg": "FR", "marseille": "FR", "lyon": "FR", "monaco": "FR",
    "ajax": "NL", "psv": "NL", "feyenoord": "NL",
    "porto": "PT", "benfica": "PT", "sporting": "PT",
    "galatasaray": "TR", "fenerbahce": "TR", "besiktas": "TR",
    "shakhtar": "UA", "dynamo kyiv": "UA",
    "salzburg": "AT", "rapid wien": "AT",
    "anderlecht": "BE", "club brugge": "BE",
    "copenhagen": "DK", "midtjylland": "DK", "brondby": "DK",
    "malmo": "SE", "djurgarden": "SE",
    "rosenborg": "NO", "brann": "NO",
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
    BAD = {"\U0001F3F3\uFE0F", "\U0001F3C1", "\U0001F6A9",
           "", "🏁", "🏳️", "🏳"}
    if flag.strip() in BAD:
        return get_flag_from_name(fallback_name)
    return flag.strip()

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
            tmp = filepath.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(filepath)
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
            return (datetime.now(timezone.utc) - cached_time
                    < timedelta(hours=ttl_hours))
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
                if (now - datetime.fromisoformat(sent_at)
                        > timedelta(hours=CFG.TTL_SENT_HISTORY)):
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

    def mark_sent(self, home: str, away: str, pick: str,
                  market: str, odds: float, commence_time: str) -> None:
        key = self._make_key(home, away, market)
        self.history[key] = {
            "match":          f"{home} vs {away}",
            "home":           home,
            "away":           away,
            "pick":           pick,
            "market":         market,
            "odds":           odds,
            "commence_time":  commence_time,
            "sent_at":        datetime.now(timezone.utc).isoformat(),
            "result_checked": False,
        }
        CacheManager.save(CFG.HISTORY_FILE, self.history)

    def get_pending_results(self) -> list:
        now = datetime.now(timezone.utc)
        pending = []
        for k, v in self.history.items():
            if v.get("result_checked"):
                continue
            try:
                ct = v.get("commence_time", "")
                match_time = datetime.fromisoformat(
                    ct.replace("Z", "+00:00"))
                elapsed = (now - match_time).total_seconds() / 3600
                if elapsed >= CFG.RESULT_CHECK_HOURS:
                    pending.append((k, v))
            except Exception:
                continue
        return pending

    def mark_result_checked(self, key: str, result: str,
                             won: Optional[bool]) -> None:
        if key in self.history:
            self.history[key]["result_checked"] = True
            self.history[key]["result"] = result
            self.history[key]["won"]    = won
            CacheManager.save(CFG.HISTORY_FILE, self.history)

# =========================================================
# 7. ELO SYSTEM
# =========================================================
class ELOSystem:
    def __init__(self, sport: str = "football"):
        self.sport = sport
        self.k = (CFG.ELO_K_FACTOR_FOOTBALL if sport == "football"
                  else CFG.ELO_K_FACTOR_TENNIS)
        self.ratings: dict     = {}
        self.match_count: dict = {}
        filepath = (CFG.ELO_FOOTBALL_FILE if sport == "football"
                    else CFG.ELO_TENNIS_FILE)
        self._load(filepath)

    def _load(self, filepath: Path):
        data = CacheManager.load(filepath)
        if data:
            self.ratings     = data.get("ratings", {})
            self.match_count = data.get("match_count", {})
            log_check(f"ELO {self.sport} loaded",
                      f"{len(self.ratings)} entities",
                      warn_if_none=False)
        else:
            logger.info("ELO %s: no data yet (bootstrap needed)",
                        self.sport)

    def save(self):
        filepath = (CFG.ELO_FOOTBALL_FILE if self.sport == "football"
                    else CFG.ELO_TENNIS_FILE)
        CacheManager.save(filepath, {
            "ratings":     self.ratings,
            "match_count": self.match_count,
            "updated_at":  datetime.now(timezone.utc).isoformat(),
        })

    def get_rating(self, name: str) -> float:
        return self.ratings.get(name.lower().strip(), CFG.ELO_DEFAULT)

    def expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def update(self, name_a: str, name_b: str,
               score_a: float, is_home_a: bool = False):
        key_a  = name_a.lower().strip()
        key_b  = name_b.lower().strip()
        ra     = self.get_rating(name_a)
        rb     = self.get_rating(name_b)
        ra_adj = ra + (CFG.ELO_HOME_ADVANTAGE if is_home_a else 0)
        ea     = self.expected_score(ra_adj, rb)
        score_b = 1.0 - score_a
        n_a    = self.match_count.get(key_a, 0)
        n_b    = self.match_count.get(key_b, 0)
        k_a    = self.k * (1.5 if n_a < 20 else 1.0)
        k_b    = self.k * (1.5 if n_b < 20 else 1.0)
        self.ratings[key_a]     = ra + k_a * (score_a - ea)
        self.ratings[key_b]     = rb + k_b * (score_b - (1.0 - ea))
        self.match_count[key_a] = n_a + 1
        self.match_count[key_b] = n_b + 1

    def predict(self, home: str, away: str,
                apply_home_advantage: bool = True) -> dict:
        ra     = self.get_rating(home)
        rb     = self.get_rating(away)
        ra_adj = ra + (CFG.ELO_HOME_ADVANTAGE
                       if apply_home_advantage else 0)
        home_prob = self.expected_score(ra_adj, rb)
        away_prob = 1.0 - home_prob
        draw_prob = 0.0
        if self.sport == "football":
            draw_factor = 0.22
            hp = home_prob * (1 - draw_factor)
            ap = away_prob * (1 - draw_factor)
            dp = draw_factor
            total = hp + ap + dp
            home_prob = hp / total
            away_prob = ap / total
            draw_prob = dp / total
        return {
            "home_prob":    round(home_prob, 4),
            "away_prob":    round(away_prob, 4),
            "draw_prob":    round(draw_prob, 4),
            "home_elo":     round(ra, 1),
            "away_elo":     round(rb, 1),
            "elo_diff":     round(ra - rb, 1),
            "home_matches": self.match_count.get(home.lower().strip(), 0),
            "away_matches": self.match_count.get(away.lower().strip(), 0),
        }

# =========================================================
# 8. BOOTSTRAP
# =========================================================
class DataBootstrap:
    FOOTBALL_LEAGUES = [
        ("E0",  "England Premier League"),
        ("E1",  "England Championship"),
        ("SP1", "Spain La Liga"),
        ("D1",  "Germany Bundesliga"),
        ("I1",  "Italy Serie A"),
        ("F1",  "France Ligue 1"),
        ("N1",  "Netherlands Eredivisie"),
        ("P1",  "Portugal Primeira Liga"),
        ("B1",  "Belgium Pro League"),
        ("T1",  "Turkey Super Lig"),
        ("G1",  "Greece Super League"),
        ("SP2", "Spain La Liga 2"),
        ("D2",  "Germany 2. Bundesliga"),
        ("I2",  "Italy Serie B"),
        ("F2",  "France Ligue 2"),
    ]
    TENNIS_FILES = [
        "atp_matches_2021.csv", "atp_matches_2022.csv",
        "atp_matches_2023.csv", "atp_matches_2024.csv",
        "wta_matches_2021.csv", "wta_matches_2022.csv",
        "wta_matches_2023.csv", "wta_matches_2024.csv",
    ]

    def __init__(self):
        self.elo_football = ELOSystem("football")
        self.elo_tennis   = ELOSystem("tennis")

    def should_run(self) -> bool:
        if FORCE_BOOTSTRAP:
            logger.info("Bootstrap: forced via env")
            return True
        if not CFG.BOOTSTRAP_FLAG.exists():
            logger.info("Bootstrap: first run")
            return True
        try:
            flag_time = datetime.fromisoformat(
                CFG.BOOTSTRAP_FLAG.read_text().strip())
            age_days = (datetime.now(timezone.utc) - flag_time).days
            if age_days >= 7:
                logger.info("Bootstrap: %d days old — refreshing",
                            age_days)
                return True
        except Exception:
            return True
        logger.info("Bootstrap: data is fresh")
        return False

    def run(self):
        log_section("BOOTSTRAP — BUILDING ELO MODELS")
        self._build_football_elo()
        self._build_tennis_elo()
        self.elo_football.save()
        self.elo_tennis.save()
        CFG.BOOTSTRAP_FLAG.write_text(
            datetime.now(timezone.utc).isoformat())
        log_check("Bootstrap football teams",
                  len(self.elo_football.ratings))
        log_check("Bootstrap tennis players",
                  len(self.elo_tennis.ratings))

    def _download_csv(self, url: str) -> Optional[pd.DataFrame]:
        try:
            res = requests.get(url, timeout=30)
            log_api_call("Bootstrap-CSV", url, {}, res.status_code,
                         0 if res.status_code != 200 else 1)
            if res.status_code == 200:
                try:
                    return pd.read_csv(StringIO(res.text))
                except Exception:
                    return pd.read_csv(StringIO(res.text),
                                       encoding="latin-1")
        except Exception as e:
            logger.debug("Download error %s: %s", url, e)
        return None

    def _build_football_elo(self):
        log_section("Building Football ELO")
        total   = 0
        seasons = ["2122", "2223", "2324", "2425"]
        for code, name in self.FOOTBALL_LEAGUES:
            count = 0
            for season in seasons:
                url = (f"https://www.football-data.co.uk/mmz4281/"
                       f"{season}/{code}.csv")
                df = self._download_csv(url)
                if df is None or df.empty:
                    continue
                if not {"HomeTeam", "AwayTeam", "FTR"}.issubset(
                        df.columns):
                    continue
                df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
                for _, row in df.iterrows():
                    try:
                        ftr   = str(row["FTR"]).strip().upper()
                        score = (1.0 if ftr == "H"
                                 else (0.0 if ftr == "A" else 0.5))
                        self.elo_football.update(
                            str(row["HomeTeam"]).strip(),
                            str(row["AwayTeam"]).strip(),
                            score, is_home_a=True)
                        count += 1
                    except Exception:
                        continue
                time.sleep(0.15)
            total += count
            if count:
                logger.info("Football ELO: %-30s -> %d matches",
                            name, count)
        log_check("Football ELO total matches", total)

    def _build_tennis_elo(self):
        log_section("Building Tennis ELO")
        total = 0
        for filename in self.TENNIS_FILES:
            tour = "atp" if filename.startswith("atp") else "wta"
            url  = (f"https://raw.githubusercontent.com/JeffSackmann/"
                    f"tennis_{tour}/master/{filename}")
            df   = self._download_csv(url)
            if df is None or df.empty:
                continue
            if not {"winner_name", "loser_name"}.issubset(df.columns):
                continue
            df    = df.dropna(subset=["winner_name", "loser_name"])
            count = 0
            for _, row in df.iterrows():
                try:
                    self.elo_tennis.update(
                        str(row["winner_name"]).strip(),
                        str(row["loser_name"]).strip(),
                        score_a=1.0)
                    count += 1
                except Exception:
                    continue
            total += count
            if count:
                logger.info("Tennis ELO: %-30s -> %d matches",
                            filename, count)
            time.sleep(0.2)
        log_check("Tennis ELO total matches", total)

# =========================================================
# 9. UTILS
# =========================================================
def retry_request(max_retries: int = 3, delay: float = 2,
                  backoff: float = 2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    status = (e.response.status_code
                              if e.response is not None else 0)
                    if status == 429:
                        wait = int(e.response.headers.get(
                            "Retry-After", current_delay * 3))
                        logger.warning(
                            "Rate limit 429 in %s, sleeping %ds",
                            func.__name__, wait)
                        time.sleep(wait)
                    elif status in [401, 403]:
                        logger.error("Auth error %d in %s",
                                     status, func.__name__)
                        return None
                    else:
                        logger.error("HTTP %d in %s (attempt %d/%d)",
                                     status, func.__name__,
                                     attempt + 1, max_retries)
                        if attempt == max_retries - 1:
                            return None
                except requests.exceptions.Timeout:
                    logger.warning("Timeout in %s (attempt %d/%d)",
                                   func.__name__,
                                   attempt + 1, max_retries)
                    if attempt == max_retries - 1:
                        return None
                except requests.exceptions.RequestException as e:
                    logger.warning("RequestException in %s: %s",
                                   func.__name__, e)
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
    clean = re.sub(r"<think>[\s\S]*?</think>", "",
                   raw_text, flags=re.IGNORECASE)
    clean = re.sub(r"<think>[\s\S]*", "",
                   clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"```(?:json)?", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    for m in reversed(list(re.finditer(r"\{[\s\S]*?\}", clean))):
        try:
            r = json.loads(m.group(0))
            if isinstance(r, dict) and r:
                return r
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
    tl = sport_title.lower()
    if any(kw in tl for kw in ["tennis", "atp", "wta",
                                "wimbledon", "roland garros"]):
        return "tennis"
    if any(kw in tl for kw in ["soccer", "football", "premier league",
                                "la liga", "bundesliga", "serie a",
                                "ligue 1", "champions league",
                                "europa league", "mls", "eredivisie"]):
        return "football"
    return "other"


def get_countdown_str(commence_time_str: str,
                      now_utc: datetime) -> str:
    try:
        match_time = datetime.fromisoformat(
            commence_time_str.replace("Z", "+00:00"))
        diff_sec = (match_time - now_utc).total_seconds()
        if diff_sec <= 0:
            return "⚡ Starting now"
        minutes_left = int(diff_sec / 60)
        if minutes_left > 60:
            h = minutes_left // 60
            m = minutes_left % 60
            return f"{h}h {m:02d}m"
        return f"{minutes_left}m"
    except Exception:
        return "N/A"


def get_display_pick(raw_pick: str, market: str,
                     home_team: str, away_team: str) -> str:
    """
    تبدیل pick خام به متن حرفه‌ای.
    از نام واقعی تیم استفاده می‌شود نه home/away.
    """
    pick_lower = raw_pick.lower().strip()

    if market == "h2h":
        if "draw" in pick_lower or "tie" in pick_lower:
            return "Draw (X)"
        # اگر نام تیم home در pick هست
        if home_team.lower() in pick_lower:
            return f"{home_team} to Win"
        # اگر نام تیم away در pick هست
        if away_team.lower() in pick_lower:
            return f"{away_team} to Win"
        # fallback: pick خودش اسم تیم است
        return f"{raw_pick} to Win"

    if market == "totals":
        m = re.match(r"(over|under)\s*([\d.]+)", pick_lower)
        if m:
            direction = m.group(1).capitalize()
            line      = m.group(2)
            return f"{direction} {line} Goals"
        return raw_pick.title()

    return raw_pick


def get_market_label(market_key: str) -> str:
    return CFG.MARKET_DISPLAY.get(
        market_key,
        market_key.replace("_", " ").title()
    )

# =========================================================
# 10. MATH ENGINE
# =========================================================
def calculate_combined_ev(
    markets_data: dict,
    elo_prediction: Optional[dict],
    sport_key: str,
    home_team: str,
    away_team: str,
) -> list:
    best_per_market: dict = {}

    for market_key, market_data_list in markets_data.items():
        sharp_odds: dict = {}
        best_odds:  dict = {}
        has_real_sharp   = False

        for entry in market_data_list:
            bk = entry.get("bookmaker_key", "")
            if bk in CFG.SHARP_BOOKMAKERS:
                has_real_sharp = True
            for o in entry.get("outcomes", []):
                base  = o["name"]
                point = o.get("point")
                name  = f"{base} {point}" if point is not None else base
                price = float(o["price"])
                if price <= 1.0:
                    continue
                if bk in CFG.SHARP_BOOKMAKERS:
                    if (name not in sharp_odds
                            or price > sharp_odds[name]["price"]):
                        sharp_odds[name] = {
                            "price":     price,
                            "bookmaker": entry["bookmaker"],
                        }
                if (name not in best_odds
                        or price > best_odds[name]["price"]):
                    best_odds[name] = {
                        "price":     price,
                        "bookmaker": entry["bookmaker"],
                    }

        if not has_real_sharp:
            logger.debug("No sharp line for %s — skipping", market_key)
            continue
        if not sharp_odds:
            continue

        try:
            implied_sum = sum(1.0 / v["price"]
                              for v in sharp_odds.values())
        except ZeroDivisionError:
            continue

        if not (CFG.MIN_VALID_IMPLIED_SUM <= implied_sum
                <= CFG.MAX_VALID_IMPLIED_SUM):
            logger.debug("Invalid implied_sum %.3f for %s",
                         implied_sum, market_key)
            continue

        expected = CFG.MARKET_EXPECTED_OUTCOMES.get(
            market_key, {"min": 2})
        if len(sharp_odds) < expected["min"]:
            logger.debug("Not enough outcomes for %s", market_key)
            continue

        min_odds = (CFG.H2H_MIN_ODDS if market_key == "h2h"
                    else CFG.TOTALS_MIN_ODDS)
        min_ev   = (CFG.H2H_MIN_EV if market_key == "h2h"
                    else CFG.TOTALS_MIN_EV)

        best_opp = None
        for outcome_name, sharp_data in sharp_odds.items():
            sharp_true_prob = (1.0 / sharp_data["price"]) / implied_sum

            elo_true_prob = None
            if elo_prediction and market_key == "h2h":
                name_l   = outcome_name.lower()
                hm       = elo_prediction.get("home_matches", 0)
                am       = elo_prediction.get("away_matches", 0)
                elo_diff = elo_prediction.get("elo_diff", 0)
                if "draw" in name_l or "tie" in name_l:
                    elo_true_prob = elo_prediction.get("draw_prob")
                elif hm >= 5 and am >= 5:
                    # تشخیص home/away از نام تیم
                    if home_team.lower() in name_l:
                        elo_true_prob = elo_prediction.get("home_prob")
                    elif away_team.lower() in name_l:
                        elo_true_prob = elo_prediction.get("away_prob")
                    elif elo_diff > 0:
                        elo_true_prob = elo_prediction.get("home_prob")
                    else:
                        elo_true_prob = elo_prediction.get("away_prob")

            true_prob = (0.6 * sharp_true_prob + 0.4 * elo_true_prob
                         if elo_true_prob is not None
                         else sharp_true_prob)

            best       = best_odds.get(outcome_name, {})
            best_price = best.get("price", 0.0)
            best_bookie = best.get("bookmaker", "Unknown")

            if best_price <= 1.0:
                continue

            ev = (true_prob * best_price) - 1.0

            if ev > CFG.MAX_REALISTIC_EV:
                logger.warning(
                    "Rejected unrealistic EV=%.1f%% for %s",
                    ev * 100, outcome_name)
                continue

            if best_price >= min_odds and ev > min_ev:
                opp = {
                    "pick":           outcome_name,
                    "market":         market_key,
                    "market_label":   get_market_label(market_key),
                    "prob":           round(true_prob, 4),
                    "odds":           round(best_price, 3),
                    "bookmaker":      best_bookie,
                    "ev":             round(ev, 4),
                    "edge_pct":       round(ev * 100, 2),
                    "has_sharp_line": has_real_sharp,
                    "elo_used":       elo_true_prob is not None,
                }
                if best_opp is None or opp["ev"] > best_opp["ev"]:
                    best_opp = opp

        if best_opp:
            best_per_market[market_key] = best_opp
            log_check(
                f"EV opp [{market_key}]",
                f"pick='{best_opp['pick']}' "
                f"EV={best_opp['edge_pct']:.1f}% "
                f"odds={best_opp['odds']} "
                f"bookie={best_opp['bookmaker']} "
                f"elo={best_opp['elo_used']}",
            )

    all_opps = list(best_per_market.values())
    all_opps.sort(key=lambda x: x["ev"], reverse=True)
    return all_opps[:1]

# =========================================================
# 11. ODDS API
# =========================================================
async def fetch_all_odds_async(now_utc: datetime) -> list:
    end_window = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    url    = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    CFG.ODDS_API_REGIONS,
        "markets":    CFG.ODDS_API_MARKETS_STR,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(
                url, params=params,
                timeout=aiohttp.ClientTimeout(total=25),
            ) as res:
                remaining = res.headers.get(
                    "x-requests-remaining", "?")
                used = res.headers.get("x-requests-used", "?")

                if res.status != 200:
                    body = await res.text()
                    log_api_call("OddsAPI", url, params,
                                 res.status, 0, body[:200])
                    logger.error("Odds API HTTP %d", res.status)
                    return []

                events      = await res.json()
                all_events: dict = {}

                for e in events:
                    try:
                        ct = e.get("commence_time", "")
                        match_time = datetime.fromisoformat(
                            ct.replace("Z", "+00:00"))
                        if not (now_utc <= match_time <= end_window):
                            continue
                        eid = e.get("id")
                        if not eid:
                            continue
                        if eid not in all_events:
                            all_events[eid] = {
                                **e, "_markets_data": {}}
                        for bm in e.get("bookmakers", []):
                            for m in bm.get("markets", []):
                                mk = m["key"]
                                if mk not in all_events[eid][
                                        "_markets_data"]:
                                    all_events[eid][
                                        "_markets_data"][mk] = []
                                all_events[eid][
                                    "_markets_data"][mk].append({
                                        "bookmaker":     bm["title"],
                                        "bookmaker_key": bm["key"],
                                        "outcomes":      m.get(
                                            "outcomes", []),
                                    })
                    except Exception:
                        continue

                result = list(all_events.values())
                log_api_call(
                    "OddsAPI", url, params, res.status,
                    len(result),
                    f"remaining={remaining} used={used} "
                    f"window_events={len(result)}",
                )
                log_check("Odds API requests remaining", remaining)
                return result
    except Exception as e:
        logger.error("Odds API fetch error: %s", e)
        return []

# =========================================================
# 12. SOFASCORE  — endpoint صحیح: /search?q=...&type=all
# =========================================================
def _sofa_headers() -> dict:
    return {
        "x-rapidapi-key":  RAPIDAPI_KEY or "",
        "x-rapidapi-host": "sofascore.p.rapidapi.com",
    }


async def _sofa_get_async(
    session: aiohttp.ClientSession,
    url: str,
    params: dict,
    label: str,
) -> Optional[dict]:
    """
    یک درخواست async به SofaScore با لاگ کامل.
    """
    try:
        async with session.get(
            url,
            headers=_sofa_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=12),
        ) as res:
            if res.status == 200:
                data = await res.json()
                # لاگ تعداد رکوردهای دریافتی
                record_count = (
                    len(data) if isinstance(data, list)
                    else sum(
                        len(v) for v in data.values()
                        if isinstance(v, list)
                    ) if isinstance(data, dict) else 1
                )
                log_api_call("SofaScore", url, params,
                             res.status, record_count,
                             str(data)[:150])
                return data
            else:
                body = await res.text()
                log_api_call("SofaScore", url, params,
                             res.status, 0, body[:150])
                logger.warning("SofaScore %s HTTP %d — %s",
                               label, res.status, body[:100])
    except Exception as e:
        log_api_call("SofaScore", url, params, -1, 0, str(e))
        logger.debug("SofaScore error [%s]: %s", label, e)
    return None


async def search_sofascore_match_async(
    home_team: str, away_team: str
) -> Optional[int]:
    """
    جستجوی match_id با endpoint رسمی:
    GET /search?q={query}&type=all&page=0
    """
    query = f"{clean_team_name(home_team)} {clean_team_name(away_team)}"
    url   = "https://sofascore.p.rapidapi.com/search"
    params = {"q": query, "type": "all", "page": "0"}

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        data = await _sofa_get_async(
            session, url, params,
            f"search:{home_team} vs {away_team}")

    if not data:
        return None

    # SofaScore search response: {"results": [{"type":"event","entity":{...}},..]}
    results = data.get("results", [])
    logger.debug("SofaScore search returned %d results for '%s'",
                 len(results), query)

    for item in results:
        if item.get("type") != "event":
            continue
        entity = item.get("entity", {})
        mid    = entity.get("id")
        if not mid:
            continue
        # تأیید که اسم تیم‌ها در event هست
        h_name = (entity.get("homeTeam", {}).get("name", "")
                  or entity.get("home_team", {}).get("name", "")).lower()
        a_name = (entity.get("awayTeam", {}).get("name", "")
                  or entity.get("away_team", {}).get("name", "")).lower()

        home_lower = clean_team_name(home_team).lower()
        away_lower = clean_team_name(away_team).lower()

        name_match = (
            (home_lower in h_name or h_name in home_lower) and
            (away_lower in a_name or a_name in away_lower)
        )
        if name_match or not (h_name or a_name):
            logger.info(
                "SofaScore: '%s' vs '%s' → match_id=%s "
                "(home_found='%s' away_found='%s')",
                home_team, away_team, mid, h_name, a_name)
            return int(mid)

    logger.info("SofaScore: no event match found for '%s' vs '%s'",
                home_team, away_team)
    return None


async def fetch_sofascore_stats_async(match_id: int,
                                      home_team: str,
                                      away_team: str) -> dict:
    """
    دریافت آمار مسابقه از SofaScore.
    endpoint های معتبر:
      /matches/get-pregame-form?matchId=X
      /matches/get-h2h?matchId=X
      /matches/get-lineups?matchId=X
    """
    mid_str = str(match_id)
    BASE    = "https://sofascore.p.rapidapi.com"
    endpoints = {
        "pregame_form": (f"{BASE}/matches/get-pregame-form",
                         {"matchId": mid_str}),
        "h2h":          (f"{BASE}/matches/get-h2h",
                         {"matchId": mid_str}),
        "lineups":      (f"{BASE}/matches/get-lineups",
                         {"matchId": mid_str}),
    }

    connector = aiohttp.TCPConnector(limit=3, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _sofa_get_async(session, url, params, label)
            for label, (url, params) in endpoints.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    data = {}
    for label, result in zip(endpoints.keys(), results):
        if isinstance(result, Exception):
            logger.warning("SofaScore [%s] exception: %s",
                           label, result)
            continue
        if result is not None:
            data[label] = result

    # خلاصه لاگ
    log_check(
        f"SofaScore match_id={match_id} "
        f"({home_team[:10]} vs {away_team[:10]})",
        f"sections_received={list(data.keys())}",
        warn_if_none=False,
    )

    # استخراج فرم اخیر برای AI
    summary = _parse_sofascore_summary(data, home_team, away_team)
    if summary:
        logger.info(
            "SofaScore parsed | %s vs %s | %s",
            home_team, away_team, str(summary)[:200])
    return summary


def _parse_sofascore_summary(data: dict,
                              home_team: str,
                              away_team: str) -> dict:
    """
    استخراج اطلاعات مفید از response SofaScore برای AI.
    """
    summary = {}

    # ── pregame_form ──────────────────────────────────────
    pgf = data.get("pregame_form", {})
    if pgf:
        for side, team_name in [("homeTeam", home_team),
                                 ("awayTeam", away_team)]:
            form_data = pgf.get(side, {})
            if not form_data:
                continue
            value = form_data.get("value", "")   # e.g. "WWDLW"
            avg_rating = form_data.get("avgRating", None)
            position   = form_data.get("position", None)
            key = "home_form" if side == "homeTeam" else "away_form"
            summary[key] = {
                "team":       team_name,
                "form":       value,
                "avg_rating": avg_rating,
                "position":   position,
            }
            logger.debug("SofaScore pregame_form [%s]: %s",
                         team_name, summary[key])

    # ── h2h ──────────────────────────────────────────────
    h2h_data = data.get("h2h", {})
    if h2h_data:
        home_wins  = h2h_data.get("homeTeamWins", 0)
        away_wins  = h2h_data.get("awayTeamWins", 0)
        draws      = h2h_data.get("draws", 0)
        total      = home_wins + away_wins + draws
        summary["h2h"] = {
            f"{home_team}_wins": home_wins,
            f"{away_team}_wins": away_wins,
            "draws":             draws,
            "total":             total,
        }
        logger.debug("SofaScore h2h: %s", summary["h2h"])

    # ── lineups ──────────────────────────────────────────
    lineups = data.get("lineups", {})
    if lineups:
        home_lu = lineups.get("home", {})
        away_lu = lineups.get("away", {})
        summary["lineups"] = {
            "home_formation": home_lu.get("formation", "N/A"),
            "away_formation": away_lu.get("formation", "N/A"),
            "home_avg_age":   home_lu.get("avgAge", None),
            "away_avg_age":   away_lu.get("avgAge", None),
        }
        logger.debug("SofaScore lineups: %s", summary["lineups"])

    return summary

# =========================================================
# 13. FOOTBALL-DATA ADAPTER  — endpoints صحیح
# =========================================================
class FootballDataAdapter:
    BASE_URL = "https://api.football-data.org/v4"

    # نگاشت competition_id به کد رشته‌ای
    COMP_MAP = {
        2021: "PL",   # Premier League
        2014: "PD",   # La Liga
        2002: "BL1",  # Bundesliga
        2019: "SA",   # Serie A
        2015: "FL1",  # Ligue 1
        2003: "DED",  # Eredivisie
        2017: "PPL",  # Primeira Liga
        2016: "ELC",  # Championship
        2001: "CL",   # Champions League
        2018: "EC",   # Euros
    }

    def __init__(self):
        self.headers = (
            {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
            if FOOTBALL_DATA_API_KEY else {}
        )
        self.daily_cache = CacheManager.load(CFG.DAILY_STATS_CACHE_FILE)
        self.call_count  = 0
        entry = self.daily_cache.get("_call_count_today", {})
        self.call_count = (entry.get("data", 0)
                           if isinstance(entry.get("data"), int) else 0)
        last_ts = entry.get("timestamp", "2000-01-01T00:00:00+00:00")
        try:
            if (datetime.now(timezone.utc).date()
                    > datetime.fromisoformat(last_ts).date()):
                self.call_count = 0
                logger.info("Football-Data call counter reset (new day)")
        except Exception:
            self.call_count = 0
        log_check("Football-Data calls today",
                  self.call_count, warn_if_none=False)

    def _can_call(self) -> bool:
        ok = (self.call_count < CFG.FOOTBALL_DATA_DAILY_LIMIT
              and bool(FOOTBALL_DATA_API_KEY))
        if not ok:
            logger.debug(
                "Football-Data: cannot call "
                "(count=%d limit=%d key=%s)",
                self.call_count, CFG.FOOTBALL_DATA_DAILY_LIMIT,
                bool(FOOTBALL_DATA_API_KEY))
        return ok

    def _increment(self):
        self.call_count += 1
        self.daily_cache = CacheManager.set(
            self.daily_cache, "_call_count_today", self.call_count)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)

    @retry_request(max_retries=2, delay=3)
    def _raw_get(self, endpoint: str,
                 params: dict = None) -> Optional[dict]:
        if not self._can_call():
            return None
        url = f"{self.BASE_URL}{endpoint}"
        res = requests.get(
            url, headers=self.headers,
            params=params, timeout=12)
        log_api_call(
            "FootballData",
            url, params or {},
            res.status_code,
            0,   # records اینجا نامشخص — بعد از parse مشخص می‌شود
        )
        res.raise_for_status()
        self._increment()
        data = res.json()
        logger.debug("FootballData response keys: %s",
                     list(data.keys()) if isinstance(data, dict)
                     else type(data))
        return data

    # ── Team lookup ──────────────────────────────────────
    def find_team_id(self, team_name: str) -> Optional[int]:
        """
        پیدا کردن team_id از طریق:
          GET /v4/competitions/{id}/teams
        برای هر competition تا team پیدا شود.
        """
        cache = CacheManager.load(CFG.TEAM_ID_CACHE_FILE)
        key   = team_name.lower().strip()

        if key in cache:
            cached_id = cache[key]
            logger.debug("FootballData team_id cache hit: "
                         "'%s' -> %s", team_name, cached_id)
            return cached_id

        if not self._can_call():
            return None

        clean_name = clean_team_name(team_name).lower()
        team_id    = None

        for comp_id, comp_code in self.COMP_MAP.items():
            # GET /v4/competitions/{id}/teams
            data = self._raw_get(
                f"/competitions/{comp_id}/teams",
                {"season": "2024"},
            )
            if not data or not data.get("teams"):
                logger.debug(
                    "FootballData /competitions/%d/teams: "
                    "no teams returned", comp_id)
                continue

            teams = data["teams"]
            log_api_call(
                "FootballData",
                f"/competitions/{comp_id}/teams",
                {"season": "2024"},
                200,
                len(teams),
                f"comp={comp_code} first={teams[0].get('name','?') if teams else 'N/A'}",
            )

            for t in teams:
                t_name  = t.get("name", "").lower()
                t_short = t.get("shortName", "").lower()
                t_tla   = t.get("tla", "").lower()

                if (clean_name == t_name
                        or clean_name == t_short
                        or clean_name == t_tla
                        or clean_name in t_name
                        or t_name in clean_name
                        or clean_name in t_short):
                    team_id = t["id"]
                    logger.info(
                        "FootballData: '%s' found in comp=%s "
                        "→ id=%d (matched='%s')",
                        team_name, comp_code, team_id, t_name)
                    break
            if team_id:
                break

        if team_id is None:
            logger.warning(
                "FootballData: team '%s' NOT found in any competition",
                team_name)

        cache[key] = team_id
        CacheManager.save(CFG.TEAM_ID_CACHE_FILE, cache)
        return team_id

    # ── Recent form ──────────────────────────────────────
    def get_team_recent_form(self, team_id: int,
                              team_name: str) -> dict:
        cache_key = f"form_{team_id}"
        if CacheManager.is_valid(self.daily_cache, cache_key,
                                 CFG.TTL_TEAM_FORM):
            cached = CacheManager.get(self.daily_cache, cache_key) or {}
            logger.debug(
                "FootballData form cache hit: '%s' (id=%d)",
                team_name, team_id)
            return cached

        # GET /v4/teams/{id}/matches/?status=FINISHED&limit=5
        data = self._raw_get(
            f"/teams/{team_id}/matches/",
            {"status": "FINISHED", "limit": "5"},
        )
        if not data:
            logger.warning(
                "FootballData: no match data for team '%s' (id=%d)",
                team_name, team_id)
            return {}

        matches = data.get("matches", [])
        log_api_call(
            "FootballData",
            f"/teams/{team_id}/matches/",
            {"status": "FINISHED", "limit": "5"},
            200,
            len(matches),
            f"team='{team_name}'",
        )

        form = self._parse_form(matches, team_id, team_name)
        log_check(
            f"FootballData form '{team_name[:20]}'",
            form.get("form_string"),
            warn_if_none=False,
        )
        self.daily_cache = CacheManager.set(
            self.daily_cache, cache_key, form)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)
        return form

    def _parse_form(self, matches: list, team_id: int,
                    team_name: str) -> dict:
        results, gs, gc = [], [], []
        for m in matches[-5:]:
            home_id = m.get("homeTeam", {}).get("id")
            away_id = m.get("awayTeam", {}).get("id")
            score   = m.get("score", {}).get("fullTime", {})
            hg      = int(score.get("home") or 0)
            ag      = int(score.get("away") or 0)

            if home_id == team_id:
                s, c = hg, ag
                r = "W" if hg > ag else ("D" if hg == ag else "L")
            elif away_id == team_id:
                s, c = ag, hg
                r = "W" if ag > hg else ("D" if ag == hg else "L")
            else:
                continue

            results.append(r)
            gs.append(s)
            gc.append(c)
            logger.debug(
                "FootballData form [%s]: %s vs %s → %d-%d (%s)",
                team_name,
                m.get("homeTeam", {}).get("name", "?"),
                m.get("awayTeam", {}).get("name", "?"),
                hg, ag, r)

        total = len(results)
        if total == 0:
            return {}

        form_dict = {
            "form_string":        "".join(results),
            "win_rate":           round(results.count("W") / total, 2),
            "draw_rate":          round(results.count("D") / total, 2),
            "avg_goals_scored":   round(sum(gs) / total, 2),
            "avg_goals_conceded": round(sum(gc) / total, 2),
            "btts_rate":          round(
                sum(1 for a, b in zip(gs, gc)
                    if a > 0 and b > 0) / total, 2),
            "over25_rate":        round(
                sum(1 for a, b in zip(gs, gc)
                    if a + b > 2.5) / total, 2),
            "matches_analyzed":   total,
        }
        logger.info(
            "FootballData form parsed [%s]: %s | "
            "WR=%.0f%% AvgGF=%.1f AvgGA=%.1f",
            team_name,
            form_dict["form_string"],
            form_dict["win_rate"] * 100,
            form_dict["avg_goals_scored"],
            form_dict["avg_goals_conceded"],
        )
        return form_dict

    # ── Head-to-Head ─────────────────────────────────────
    def get_h2h(self, team1_id: int, team2_id: int,
                team1_name: str, team2_name: str) -> dict:
        cache_key = (f"h2h_{min(team1_id, team2_id)}_"
                     f"{max(team1_id, team2_id)}")
        if CacheManager.is_valid(self.daily_cache, cache_key,
                                 CFG.TTL_H2H):
            cached = CacheManager.get(self.daily_cache, cache_key) or {}
            logger.debug("FootballData H2H cache hit: "
                         "%s vs %s", team1_name, team2_name)
            return cached

        # GET /v4/teams/{id}/matches/?status=FINISHED&limit=20
        data = self._raw_get(
            f"/teams/{team1_id}/matches/",
            {"status": "FINISHED", "limit": "20"},
        )
        if not data:
            return {}

        all_matches = data.get("matches", [])
        h2h_matches = [
            m for m in all_matches
            if {m.get("homeTeam", {}).get("id"),
                m.get("awayTeam", {}).get("id")}
            == {team1_id, team2_id}
        ]
        log_api_call(
            "FootballData",
            f"/teams/{team1_id}/matches/ [H2H filter]",
            {"status": "FINISHED", "limit": "20"},
            200,
            len(h2h_matches),
            f"{team1_name} vs {team2_name}",
        )
        logger.info(
            "FootballData H2H: %s vs %s → %d encounters found",
            team1_name, team2_name, len(h2h_matches))

        result = self._parse_h2h(h2h_matches, team1_id,
                                 team1_name, team2_name)
        self.daily_cache = CacheManager.set(
            self.daily_cache, cache_key, result)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)
        return result

    def _parse_h2h(self, matches: list, team1_id: int,
                   team1_name: str, team2_name: str) -> dict:
        t1 = t2 = draws = tg = btts = over25 = 0
        total = len(matches)
        for m in matches:
            score = m.get("score", {}).get("fullTime", {})
            hg    = int(score.get("home") or 0)
            ag    = int(score.get("away") or 0)
            hid   = m.get("homeTeam", {}).get("id")
            if hg > ag:
                if hid == team1_id:
                    t1 += 1
                else:
                    t2 += 1
            elif ag > hg:
                if hid != team1_id:
                    t1 += 1
                else:
                    t2 += 1
            else:
                draws += 1
            tg += hg + ag
            if hg > 0 and ag > 0:
                btts   += 1
            if hg + ag > 2.5:
                over25 += 1

        if total == 0:
            return {}

        h2h_dict = {
            "total_h2h":          total,
            f"{team1_name}_wins": t1,
            f"{team2_name}_wins": t2,
            "draws":              draws,
            "avg_goals_per_game": round(tg / total, 2),
            "btts_rate":          round(btts / total, 2),
            "over25_rate":        round(over25 / total, 2),
        }
        logger.info(
            "FootballData H2H parsed: %s %dW / %s %dW / D=%d "
            "| AvgGoals=%.1f BTTS=%.0f%% O2.5=%.0f%%",
            team1_name, t1, team2_name, t2, draws,
            h2h_dict["avg_goals_per_game"],
            h2h_dict["btts_rate"] * 100,
            h2h_dict["over25_rate"] * 100,
        )
        return h2h_dict

# =========================================================
# 14. MATCH ID CACHE
# =========================================================
class MatchIDCache:
    def __init__(self):
        self.cache = CacheManager.load(CFG.MATCH_ID_CACHE_FILE)

    def get(self, home_team: str, away_team: str) -> Optional[int]:
        key = self._key(home_team, away_team)
        if CacheManager.is_valid(self.cache, key, CFG.TTL_MATCH_ID):
            return CacheManager.get(self.cache, key)
        return None

    def set(self, home_team: str, away_team: str,
            match_id: Optional[int]) -> None:
        key        = self._key(home_team, away_team)
        self.cache = CacheManager.set(self.cache, key, match_id)
        CacheManager.save(CFG.MATCH_ID_CACHE_FILE, self.cache)

    @staticmethod
    def _key(home_team: str, away_team: str) -> str:
        return hashlib.md5(
            f"{home_team.lower()}|{away_team.lower()}".encode()
        ).hexdigest()

# =========================================================
# 15. STATS AGGREGATOR
# =========================================================
async def get_stats_async(
    home_team: str,
    away_team: str,
    sport_key: str,
    football_adapter: FootballDataAdapter,
    match_id_cache: MatchIDCache,
    elo_football: ELOSystem,
    elo_tennis: ELOSystem,
) -> tuple:
    log_section(f"STATS: {home_team} vs {away_team}")

    stats = {
        "home_form":    {},
        "away_form":    {},
        "h2h":          {},
        "sofascore":    {},
        "elo":          {},
        "data_quality": "none",
    }

    # ── ELO ──────────────────────────────────────────────
    if sport_key == "football":
        elo_pred = elo_football.predict(
            home_team, away_team, apply_home_advantage=True)
    elif sport_key == "tennis":
        elo_pred = elo_tennis.predict(
            home_team, away_team, apply_home_advantage=False)
    else:
        elo_pred = None

    if elo_pred and (elo_pred.get("home_matches", 0) >= 3
                     or elo_pred.get("away_matches", 0) >= 3):
        stats["elo"] = elo_pred
        logger.info(
            "ELO | %s vs %s | H=%.1f%% D=%.1f%% A=%.1f%% "
            "| hm=%d am=%d diff=%.0f",
            home_team, away_team,
            elo_pred["home_prob"] * 100,
            elo_pred["draw_prob"] * 100,
            elo_pred["away_prob"] * 100,
            elo_pred["home_matches"],
            elo_pred["away_matches"],
            elo_pred["elo_diff"],
        )
    else:
        logger.warning(
            "ELO insufficient data for %s vs %s "
            "(hm=%d am=%d)",
            home_team, away_team,
            (elo_pred or {}).get("home_matches", 0),
            (elo_pred or {}).get("away_matches", 0),
        )

    # ── SofaScore match ID ───────────────────────────────
    cached_mid = match_id_cache.get(home_team, away_team)
    if cached_mid is not None:
        match_id = cached_mid if cached_mid != 0 else None
        logger.debug("SofaScore match_id cache hit: %s", match_id)
    else:
        match_id = await search_sofascore_match_async(
            home_team, away_team)
        match_id_cache.set(
            home_team, away_team,
            match_id if match_id else 0)

    task_names: list = []
    coros:      list = []

    # ── SofaScore stats ──────────────────────────────────
    if match_id:
        task_names.append("sofascore")
        coros.append(fetch_sofascore_stats_async(
            match_id, home_team, away_team))
    else:
        logger.info("SofaScore: no match_id for %s vs %s — skipping",
                    home_team, away_team)

    # ── Football-Data ────────────────────────────────────
    if sport_key == "football":
        loop = asyncio.get_running_loop()

        async def get_football_data():
            log_section(
                f"Football-Data: {home_team} vs {away_team}")
            home_id = await loop.run_in_executor(
                None, football_adapter.find_team_id, home_team)
            away_id = await loop.run_in_executor(
                None, football_adapter.find_team_id, away_team)

            log_check(f"FD team_id '{home_team}'", home_id)
            log_check(f"FD team_id '{away_team}'", away_id)

            if not home_id or not away_id:
                logger.warning(
                    "Football-Data: could not resolve IDs "
                    "for %s (id=%s) or %s (id=%s)",
                    home_team, home_id, away_team, away_id)
                return {}

            hf, af, h2h = await asyncio.gather(
                loop.run_in_executor(
                    None,
                    football_adapter.get_team_recent_form,
                    home_id, home_team),
                loop.run_in_executor(
                    None,
                    football_adapter.get_team_recent_form,
                    away_id, away_team),
                loop.run_in_executor(
                    None,
                    football_adapter.get_h2h,
                    home_id, away_id, home_team, away_team),
                return_exceptions=True,
            )
            out = {}
            if not isinstance(hf, Exception) and hf:
                out["home_form"] = hf
                log_check(f"FD home form '{home_team}'",
                          hf.get("form_string"))
            else:
                logger.warning("FD home form MISSING for '%s'",
                               home_team)

            if not isinstance(af, Exception) and af:
                out["away_form"] = af
                log_check(f"FD away form '{away_team}'",
                          af.get("form_string"))
            else:
                logger.warning("FD away form MISSING for '%s'",
                               away_team)

            if not isinstance(h2h, Exception) and h2h:
                out["h2h"] = h2h
                log_check(f"FD H2H '{home_team}' vs '{away_team}'",
                          h2h.get("total_h2h", 0))
            else:
                logger.warning("FD H2H MISSING for '%s' vs '%s'",
                               home_team, away_team)
            return out

        task_names.append("football")
        coros.append(get_football_data())

    # ── Gather all ───────────────────────────────────────
    if coros:
        gathered = await asyncio.gather(*coros,
                                        return_exceptions=True)
        for name, result in zip(task_names, gathered):
            if isinstance(result, Exception):
                logger.warning("Stats gather error [%s]: %s",
                               name, result)
                continue
            if name == "sofascore" and result:
                stats["sofascore"] = result
            elif name == "football" and isinstance(result, dict):
                stats.update(result)

    # ── Data quality ─────────────────────────────────────
    has_fb  = bool(stats.get("home_form") or stats.get("h2h"))
    has_ss  = bool(stats.get("sofascore"))
    has_elo = bool(stats.get("elo"))

    if (has_fb or has_elo) and has_ss:
        stats["data_quality"] = "high"
    elif has_fb or has_ss or has_elo:
        stats["data_quality"] = "medium"

    logger.info(
        "DATA QUALITY | %s vs %s | quality=%s "
        "(fb=%s ss=%s elo=%s)",
        home_team, away_team,
        stats["data_quality"],
        has_fb, has_ss, has_elo,
    )
    return stats, elo_pred

# =========================================================
# 16. CONFIDENCE ENGINE
# =========================================================
def calculate_confidence(
    ev_edge: float, stats: dict,
    market: str, has_sharp: bool
) -> tuple:
    score = 50
    dq    = stats.get("data_quality", "none")

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

    elo = stats.get("elo", {})
    hm  = elo.get("home_matches", 0)
    am  = elo.get("away_matches", 0)
    if hm >= 10 and am >= 10:
        score += 10
    elif hm >= 5 and am >= 5:
        score += 6

    if has_sharp:
        score += 5
    if market == "totals":
        score += 3

    hf = stats.get("home_form", {})
    af = stats.get("away_form", {})
    if hf.get("form_string") and af.get("form_string"):
        if hf["form_string"].count("W") >= 3:
            score += 5
        if af["form_string"].count("L") >= 3:
            score += 3

    # SofaScore bonus
    ss = stats.get("sofascore", {})
    if ss.get("home_form") and ss.get("away_form"):
        score += 4

    score = max(50, min(93, score))
    risk  = ("Low" if score >= 75
             else ("Medium" if score >= 60 else "High"))

    logger.info(
        "Confidence score=%d risk=%s "
        "(dq=%s ev=%.1f%% elo_hm=%d elo_am=%d sharp=%s)",
        score, risk, dq, ev_pct, hm, am, has_sharp)
    return score, risk

# =========================================================
# 17. DUAL-AI ANALYSIS
# =========================================================
def build_stats_summary(stats: dict,
                         home_team: str,
                         away_team: str) -> str:
    parts = []
    elo   = stats.get("elo", {})
    hf    = stats.get("home_form", {})
    af    = stats.get("away_form", {})
    h2h   = stats.get("h2h", {})
    ss    = stats.get("sofascore", {})

    if elo and elo.get("home_matches", 0) >= 3:
        parts.append(
            f"[ELO MODEL]\n"
            f"  {home_team} ELO={elo['home_elo']:.0f} "
            f"({elo['home_matches']} matches)\n"
            f"  {away_team} ELO={elo['away_elo']:.0f} "
            f"({elo['away_matches']} matches)\n"
            f"  ELO Diff={elo['elo_diff']:.0f}\n"
            f"  Win probabilities: "
            f"{home_team}={elo['home_prob']:.1%} | "
            f"Draw={elo['draw_prob']:.1%} | "
            f"{away_team}={elo['away_prob']:.1%}"
        )

    if hf:
        parts.append(
            f"[RECENT FORM — {home_team}]\n"
            f"  Last 5: {hf.get('form_string', 'N/A')}\n"
            f"  Win rate: {hf.get('win_rate', 0):.0%} | "
            f"Avg scored: {hf.get('avg_goals_scored', 0)} | "
            f"Avg conceded: {hf.get('avg_goals_conceded', 0)}\n"
            f"  BTTS: {hf.get('btts_rate', 0):.0%} | "
            f"Over 2.5: {hf.get('over25_rate', 0):.0%}"
        )

    if af:
        parts.append(
            f"[RECENT FORM — {away_team}]\n"
            f"  Last 5: {af.get('form_string', 'N/A')}\n"
            f"  Win rate: {af.get('win_rate', 0):.0%} | "
            f"Avg scored: {af.get('avg_goals_scored', 0)} | "
            f"Avg conceded: {af.get('avg_goals_conceded', 0)}\n"
            f"  BTTS: {af.get('btts_rate', 0):.0%} | "
            f"Over 2.5: {af.get('over25_rate', 0):.0%}"
        )

    if h2h and h2h.get("total_h2h", 0) > 0:
        # کلیدهای داینامیک (نام تیم)
        t1_w = h2h.get(f"{home_team}_wins",
                        h2h.get("team1_wins", 0))
        t2_w = h2h.get(f"{away_team}_wins",
                        h2h.get("team2_wins", 0))
        parts.append(
            f"[HEAD TO HEAD — last {h2h['total_h2h']} games]\n"
            f"  {home_team} wins: {t1_w} | "
            f"{away_team} wins: {t2_w} | "
            f"Draws: {h2h.get('draws', 0)}\n"
            f"  Avg goals/game: {h2h.get('avg_goals_per_game', 0)} | "
            f"BTTS: {h2h.get('btts_rate', 0):.0%} | "
            f"Over 2.5: {h2h.get('over25_rate', 0):.0%}"
        )

    if ss:
        # SofaScore pregame form
        ss_hf = ss.get("home_form", {})
        ss_af = ss.get("away_form", {})
        if ss_hf or ss_af:
            parts.append("[SOFASCORE PREGAME FORM]")
        if ss_hf:
            parts.append(
                f"  {home_team}: form={ss_hf.get('form', 'N/A')} | "
                f"avg_rating={ss_hf.get('avg_rating', 'N/A')} | "
                f"league_pos={ss_hf.get('position', 'N/A')}"
            )
        if ss_af:
            parts.append(
                f"  {away_team}: form={ss_af.get('form', 'N/A')} | "
                f"avg_rating={ss_af.get('avg_rating', 'N/A')} | "
                f"league_pos={ss_af.get('position', 'N/A')}"
            )
        ss_h2h = ss.get("h2h", {})
        if ss_h2h and ss_h2h.get("total", 0) > 0:
            parts.append(
                f"[SOFASCORE H2H — last {ss_h2h['total']}]\n"
                f"  {home_team} wins: "
                f"{ss_h2h.get(f'{home_team}_wins', 'N/A')} | "
                f"{away_team} wins: "
                f"{ss_h2h.get(f'{away_team}_wins', 'N/A')} | "
                f"Draws: {ss_h2h.get('draws', 'N/A')}"
            )
        lu = ss.get("lineups", {})
        if lu:
            parts.append(
                f"[LINEUPS]\n"
                f"  {home_team} formation: "
                f"{lu.get('home_formation', 'N/A')}\n"
                f"  {away_team} formation: "
                f"{lu.get('away_formation', 'N/A')}"
            )

    if not parts:
        return "NO STATISTICAL DATA AVAILABLE"

    summary = "\n\n".join(parts)
    logger.debug("Stats summary for AI (%d chars):\n%s",
                 len(summary), summary[:500])
    return summary


def call_groq_sdk(
    model: str, messages: list, temperature: float = 0.1
) -> Optional[str]:
    SUPPORTS_JSON = ["llama-3", "llama3", "mixtral",
                     "gemma", "llama-4", "scout"]
    use_json = any(kw in model.lower() for kw in SUPPORTS_JSON)

    kwargs: dict = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  CFG.AI_MAX_TOKENS,
    }
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        res     = groq_client.chat.completions.create(**kwargs)
        content = res.choices[0].message.content
        logger.info("Groq %-35s | tokens_used=%s | preview=%s",
                    model,
                    getattr(res.usage, "total_tokens", "?"),
                    (content or "")[:80])
        return content
    except Exception as e:
        logger.error("Groq error model=%s: %s", model, e)
        return None


def generate_dual_ai_analysis(
    home_team: str, away_team: str, sport: str,
    display_pick: str, market: str,
    ev_edge: float, stats: dict,
    confidence: int, risk: str,
) -> dict:
    stats_summary  = build_stats_summary(stats, home_team, away_team)
    data_quality   = stats.get("data_quality", "none")
    has_real_stats = data_quality in ["medium", "high"]

    default_response = {
        "sport_emoji": "\U0001F3C6",
        "home_flag":   get_flag_from_name(home_team),
        "away_flag":   get_flag_from_name(away_team),
        "risk_level":  risk,
        "confidence":  confidence,
        "logic":       ("Sharp bookmaker lines show clear value "
                        "on this selection."),
    }

    sys1 = (
        "You are an elite sports betting analyst.\n"
        "Write EXACTLY 2 punchy professional sentences "
        "justifying the pick.\n"
        "RULES:\n"
        "- Use ONLY statistics provided. Never invent numbers.\n"
        "- Never mention EV, models, algorithms, data quality.\n"
        "- If no stats: state pick is driven by sharp market line.\n"
        "- Provide EXACT country flag emoji for home_flag, away_flag.\n"
        "- Choose correct sport_emoji.\n"
        "OUTPUT: valid JSON only. No markdown.\n"
        '{"sport_emoji":"...","home_flag":"...","away_flag":"...",'
        '"logic":"sentence1. sentence2."}'
    )
    u1 = (
        f"MATCH: {home_team} vs {away_team}\n"
        f"SPORT: {sport}\n"
        f"PICK: {display_pick}\n"
        f"MARKET: {get_market_label(market)}\n"
        f"DATA QUALITY: {data_quality}\n\n"
        f"STATISTICS:\n{stats_summary}\n\n"
        "OUTPUT JSON ONLY:"
    )

    analysis_1 = None
    try:
        raw1 = call_groq_sdk(
            CFG.AI_MODEL_ANALYST,
            [{"role": "system", "content": sys1},
             {"role": "user",   "content": u1}],
            temperature=0.2,
        )
        analysis_1 = robust_json_extractor(raw1)
        log_check("AI Model 1 (analyst)",
                  "OK" if analysis_1 else "FAILED")
        if analysis_1:
            logger.debug("Model 1 output: %s", analysis_1)
    except Exception as e:
        logger.warning("Model 1 error: %s", e)

    initial_logic = (
        (analysis_1 or {}).get("logic") or default_response["logic"]
    )

    sys2 = (
        "You are a professional sports content editor.\n"
        "Review and improve the analysis if needed.\n"
        "Max 2 sentences. Professional tipster tone.\n"
        "Do NOT fabricate stats if none provided.\n"
        "OUTPUT: valid JSON only.\n"
        '{"validated_logic":"..."}'
    )
    try:
        raw2 = call_groq_sdk(
            CFG.AI_MODEL_VALIDATOR,
            [{"role": "system", "content": sys2},
             {"role": "user", "content": (
                 f"DRAFT: {initial_logic}\n"
                 f"PICK: {display_pick}\n"
                 f"HAS REAL STATS: {has_real_stats}\n"
                 "OUTPUT JSON ONLY:"
             )}],
            temperature=0.15,
        )
        analysis_2 = robust_json_extractor(raw2)
        if analysis_2 and analysis_2.get("validated_logic"):
            initial_logic = analysis_2["validated_logic"]
        log_check("AI Model 2 (validator)",
                  "OK" if analysis_2 else "FAILED")
    except Exception as e:
        logger.warning("Model 2 error: %s", e)

    result = dict(default_response)
    if analysis_1:
        if analysis_1.get("sport_emoji"):
            result["sport_emoji"] = analysis_1["sport_emoji"]
        if analysis_1.get("home_flag"):
            result["home_flag"] = validate_flag(
                analysis_1["home_flag"], home_team)
        if analysis_1.get("away_flag"):
            result["away_flag"] = validate_flag(
                analysis_1["away_flag"], away_team)

    safe_logic      = str(initial_logic).strip()
    result["logic"] = (safe_logic[:600] + "..."
                       if len(safe_logic) > 600 else safe_logic)

    logger.info(
        "AI final | confidence=%d risk=%s | logic='%s'",
        result["confidence"], result["risk_level"],
        result["logic"][:80])
    return result

# =========================================================
# 18. RESULTS CHECKER
# =========================================================
@retry_request(max_retries=2)
def fetch_event_result(home_team: str,
                       away_team: str) -> Optional[dict]:
    url    = "https://api.the-odds-api.com/v4/sports/upcoming/scores"
    params = {
        "apiKey":    ODDS_API_KEY,
        "daysFrom":  3,
        "dateFormat": "iso",
    }
    res = requests.get(url, params=params, timeout=15)
    log_api_call("OddsAPI-Scores", url, params,
                 res.status_code, 0)
    if res.status_code != 200:
        return None

    events = res.json()
    log_api_call("OddsAPI-Scores", url, params,
                 res.status_code, len(events))

    for event in events:
        ht = event.get("home_team", "")
        at = event.get("away_team", "")
        if (ht.lower() == home_team.lower()
                and at.lower() == away_team.lower()
                and event.get("completed")):
            logger.info(
                "Result found: %s vs %s | scores=%s",
                home_team, away_team,
                event.get("scores"))
            return event
    return None


def _determine_win(
    pick: str, market: str, scores,
    home_team: str, away_team: str,
) -> Optional[bool]:
    try:
        # The-Odds-API v4: scores = [{"name":"Team","score":"2"},...]
        if isinstance(scores, list):
            score_map = {s["name"]: s.get("score")
                         for s in scores}
        else:
            score_map = scores

        home_sc = int(score_map.get(home_team, -1) or -1)
        away_sc = int(score_map.get(away_team, -1) or -1)

        logger.debug(
            "Win check: pick='%s' market=%s "
            "%s=%d %s=%d",
            pick, market,
            home_team, home_sc,
            away_team, away_sc)

        if home_sc < 0 or away_sc < 0:
            logger.warning(
                "Win check: scores not found in map=%s",
                score_map)
            return None

        pick_lower = pick.lower()
        if market == "h2h":
            if home_team.lower() in pick_lower:
                return home_sc > away_sc
            if away_team.lower() in pick_lower:
                return away_sc > home_sc
            if "draw" in pick_lower or "tie" in pick_lower:
                return home_sc == away_sc

        elif market == "totals":
            total = home_sc + away_sc
            m = re.search(r"(over|under)\s*([\d.]+)", pick_lower)
            if m:
                direction = m.group(1)
                line      = float(m.group(2))
                won = (total > line if direction == "over"
                       else total < line)
                logger.debug(
                    "Totals: total=%d line=%.1f dir=%s → won=%s",
                    total, line, direction, won)
                return won
    except Exception as e:
        logger.debug("Win check error: %s", e)
    return None


def check_and_report_results(
    sent_history: SentHistory
) -> Optional[str]:
    log_section("PHASE 1 — RESULTS CHECK")
    pending = sent_history.get_pending_results()
    log_check("Pending results to check",
              len(pending), warn_if_none=False)

    if not pending:
        return None

    wins:   list = []
    losses: list = []

    for key, entry in pending:
        home_team = entry.get("home", "")
        away_team = entry.get("away", "")
        pick      = entry.get("pick", "")
        market    = entry.get("market", "")
        odds      = entry.get("odds", 1.0)

        logger.info("Checking result: %s vs %s | pick=%s",
                    home_team, away_team, pick)

        result_event = fetch_event_result(home_team, away_team)
        if not result_event:
            logger.info("Result not found yet: %s vs %s",
                        home_team, away_team)
            continue

        scores = result_event.get("scores", [])
        won    = _determine_win(pick, market, scores,
                                home_team, away_team)

        try:
            if isinstance(scores, list):
                sm  = {s["name"]: s.get("score", "?") for s in scores}
                hs  = sm.get(home_team, "?")
                aws = sm.get(away_team, "?")
            else:
                hs  = scores.get(home_team, {}).get("score", "?")
                aws = scores.get(away_team, {}).get("score", "?")
            result_str = f"{hs} - {aws}"
        except Exception:
            result_str = "? - ?"

        sent_history.mark_result_checked(key, result_str, won)
        logger.info(
            "Result: %s vs %s | score=%s | won=%s",
            home_team, away_team, result_str, won)

        if won is True:
            wins.append({**entry, "result": result_str})
        elif won is False:
            losses.append({**entry, "result": result_str})

    if not wins and not losses:
        return None

    total    = len(wins) + len(losses)
    win_rate = len(wins) / total if total > 0 else 0
    roi_vals = ([w.get("odds", 1.0) - 1.0 for w in wins]
                + [-1.0] * len(losses))
    roi      = sum(roi_vals) / len(roi_vals) if roi_vals else 0

    lines = ["\U0001F4CA <b>RESULTS REPORT</b>\n"]
    for w in wins:
        lines.append(
            f"\u2705 <b>{html_lib.escape(w['home'])} vs "
            f"{html_lib.escape(w['away'])}</b>\n"
            f"   Pick: {html_lib.escape(w['pick'])} "
            f"@ <code>{w['odds']:.2f}</code>\n"
            f"   Score: {w.get('result', '?')} — <b>WIN ✅</b>\n"
        )
    for lo in losses:
        lines.append(
            f"\u274C <b>{html_lib.escape(lo['home'])} vs "
            f"{html_lib.escape(lo['away'])}</b>\n"
            f"   Pick: {html_lib.escape(lo['pick'])} "
            f"@ <code>{lo['odds']:.2f}</code>\n"
            f"   Score: {lo.get('result', '?')} — <b>LOSS ❌</b>\n"
        )
    lines.append(
        f"\n\U0001F3AF <b>Session:</b> "
        f"{len(wins)}W / {len(losses)}L | "
        f"Win Rate: {win_rate:.0%} | ROI: {roi:+.1%}\n\n"
        f"\U0001F194 {CFG.TELEGRAM_ID}"
    )
    return "\n".join(lines)

# =========================================================
# 19. TELEGRAM
# =========================================================
def send_telegram(message_html: str) -> bool:
    MAX_LEN = 4000
    if len(message_html) <= MAX_LEN:
        chunks = [message_html]
    else:
        chunks, current = [], ""
        for line in message_html.split("\n"):
            if len(current) + len(line) + 1 > MAX_LEN:
                chunks.append(current.strip())
                current = line + "\n"
            else:
                current += line + "\n"
        if current:
            chunks.append(current.strip())

    url     = (f"https://api.telegram.org/bot"
               f"{TELEGRAM_BOT_TOKEN}/sendMessage")
    success = True
    for i, chunk in enumerate(chunks):
        try:
            res = requests.post(
                url,
                json={
                    "chat_id":                  TELEGRAM_CHAT_ID,
                    "text":                     chunk,
                    "parse_mode":               "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            log_api_call("Telegram", url,
                         {"chunk": i + 1, "len": len(chunk)},
                         res.status_code, 1 if res.ok else 0)
            if not res.ok:
                logger.error("Telegram error: %d — %s",
                             res.status_code, res.text[:150])
                success = False
        except Exception as e:
            logger.error("Telegram send error: %s", e)
            success = False
    return success

# =========================================================
# 20. MESSAGE BUILDER
# =========================================================
SEP = "━" * 28

def build_telegram_message(
    sport: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    now_utc: datetime,
    opp: dict,
    display_pick: str,
    confidence: int,
    risk: str,
    ai_data: dict,
) -> str:
    conf_icon = (
        "\U0001F525" if confidence >= 75
        else ("\u2705" if confidence >= 60 else "\u26A1")
    )
    risk_icon = {
        "Low":    "\U0001F7E2",
        "Medium": "\U0001F7E0",
        "High":   "\U0001F534",
    }.get(risk, "\U0001F7E0")

    sport_emoji = ai_data.get("sport_emoji", "\U0001F3C6")
    home_flag   = ai_data.get("home_flag", "\U0001F3F3\uFE0F")
    away_flag   = ai_data.get("away_flag", "\U0001F3F3\uFE0F")
    logic       = str(ai_data.get("logic", "")).strip()
    logic_esc   = html_lib.escape(
        logic.replace("<", "").replace(">", ""))

    market_label = get_market_label(opp["market"])
    odds_display = f"{opp['odds']:.2f}"
    bookie       = opp.get("bookmaker", "Best Available")
    countdown    = get_countdown_str(commence_time, now_utc)

    return (
        f"{sport_emoji} <b>{html_lib.escape(sport)}</b>\n"
        f"{SEP}\n"
        f"{home_flag} <b>{html_lib.escape(home_team)}</b>"
        f"  vs  "
        f"<b>{html_lib.escape(away_team)}</b> {away_flag}\n"
        f"⏱ <b>Kick-off in:</b> {countdown}\n"
        f"{SEP}\n"
        f"📌 <b>Market:</b> {html_lib.escape(market_label)}\n"
        f"🎯 <b>Pick:</b> "
        f"<code>{html_lib.escape(display_pick)}</code>\n"
        f"💰 <b>Best Odds:</b> "
        f"<code>{odds_display}</code> "
        f"<i>({html_lib.escape(bookie)})</i>\n"
        f"{SEP}\n"
        f"{risk_icon} <b>Risk:</b> {risk}  "
        f"{conf_icon} <b>Confidence:</b> {confidence}%\n"
        f"{SEP}\n"
        f"💡 <b>Analysis:</b>\n"
        f"<blockquote>{logic_esc}</blockquote>\n"
        f"{SEP}\n"
        f"🆔 {CFG.TELEGRAM_ID}"
    )

# =========================================================
# 21. MAIN PIPELINE
# =========================================================
async def async_main():
    log_section("ZBET90 ENTERPRISE ENGINE v3.3 STARTING")

    bootstrap = DataBootstrap()
    if bootstrap.should_run():
        bootstrap.run()

    elo_football = ELOSystem("football")
    elo_tennis   = ELOSystem("tennis")

    log_check("ELO football teams", len(elo_football.ratings))
    log_check("ELO tennis players", len(elo_tennis.ratings))

    sent_history     = SentHistory()
    football_adapter = FootballDataAdapter()
    match_id_cache   = MatchIDCache()
    now_utc          = datetime.now(timezone.utc)

    # ── Phase 1: Results ─────────────────────────────────
    results_msg = check_and_report_results(sent_history)
    if results_msg:
        if send_telegram(results_msg):
            logger.info("Results report sent to Telegram")
        await asyncio.sleep(2)

    # ── Phase 2: Odds ────────────────────────────────────
    log_section("PHASE 2 — FETCHING ODDS")
    events = await fetch_all_odds_async(now_utc)

    if not events:
        logger.info("No events in %.0fh window — exiting",
                    CFG.MATCH_WINDOW_HOURS)
        return

    log_check("Total events to analyse", len(events))

    # ── Phase 3: Signals ─────────────────────────────────
    log_section("PHASE 3 — ANALYSIS & SIGNALS")
    total_sent = 0

    for event in events:
        home_team     = event.get("home_team", "")
        away_team     = event.get("away_team", "")
        sport         = event.get("sport_title", "Unknown")
        sport_key     = normalize_sport_key(sport)
        commence_time = event.get("commence_time", "")
        markets_data  = event.get("_markets_data", {})

        if not home_team or not away_team:
            continue

        logger.info(
            "Processing: %s vs %s [%s]",
            home_team, away_team, sport)

        elo_pred = None
        if sport_key == "football":
            elo_pred = elo_football.predict(home_team, away_team)
        elif sport_key == "tennis":
            elo_pred = elo_tennis.predict(
                home_team, away_team,
                apply_home_advantage=False)

        opportunities = calculate_combined_ev(
            markets_data, elo_pred, sport_key,
            home_team, away_team)

        if not opportunities:
            logger.debug("No EV opportunity: %s vs %s",
                         home_team, away_team)
            continue

        opp = opportunities[0]

        if sent_history.was_sent(home_team, away_team, opp["market"]):
            logger.info("SKIP duplicate: %s vs %s [%s]",
                        home_team, away_team, opp["market"])
            continue

        stats, _ = await get_stats_async(
            home_team, away_team, sport_key,
            football_adapter, match_id_cache,
            elo_football, elo_tennis,
        )

        confidence, risk = calculate_confidence(
            opp["ev"], stats,
            opp["market"], opp["has_sharp_line"])

        display_pick = get_display_pick(
            opp["pick"], opp["market"], home_team, away_team)

        ai_data = generate_dual_ai_analysis(
            home_team, away_team, sport,
            display_pick, opp["market"],
            opp["ev"], stats, confidence, risk,
        )

        msg = build_telegram_message(
            sport, home_team, away_team,
            commence_time, now_utc,
            opp, display_pick,
            confidence, risk, ai_data,
        )

        logger.info(
            "SIGNAL | %s vs %s | pick=%s | "
            "odds=%.2f | ev=%.1f%% | conf=%d%%",
            home_team, away_team,
            display_pick, opp["odds"],
            opp["edge_pct"], confidence)

        if send_telegram(msg):
            sent_history.mark_sent(
                home_team, away_team,
                opp["pick"], opp["market"],
                opp["odds"], commence_time,
            )
            total_sent += 1
            logger.info("✅ Sent: %s vs %s", home_team, away_team)
        else:
            logger.error("❌ Telegram failed: %s vs %s",
                         home_team, away_team)

        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    log_section("RUN COMPLETE")
    log_check("Signals sent this run", total_sent,
              warn_if_none=False)


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except Exception as e:
        logger.critical("SYSTEM FAILURE: %s", str(e), exc_info=True)
        sys.exit(1)
