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
import numpy as np
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
    RESULTS_FILE: Path = Path("api_cache/pending_results.json")
    TEAM_ID_CACHE_FILE: Path = Path("api_cache/team_id_cache.json")
    MATCH_ID_CACHE_FILE: Path = Path("api_cache/match_id_cache.json")
    DAILY_STATS_CACHE_FILE: Path = Path("api_cache/daily_stats_cache.json")
    ELO_FOOTBALL_FILE: Path = Path("api_cache/models/elo_football.json")
    ELO_TENNIS_FILE: Path = Path("api_cache/models/elo_tennis.json")
    BOOTSTRAP_FLAG: Path = Path("api_cache/models/bootstrap_done.flag")
    LOG_FILE: Path = Path("api_cache/execution_logs.log")

    MATCH_WINDOW_HOURS: float = 2.0
    RESULT_CHECK_HOURS: float = 3.0
    TELEGRAM_SLEEP_BETWEEN: float = 3.0

    FOOTBALL_DATA_DAILY_LIMIT: int = 80
    ODDS_API_REGIONS: str = "eu,us,uk,au"

    TTL_SENT_HISTORY: float = 72.0
    TTL_MATCH_ID: float = 24.0
    TTL_TEAM_FORM: float = 6.0
    TTL_H2H: float = 24.0
    TTL_DAILY_STATS: float = 6.0

    H2H_MIN_ODDS: float = 1.50
    H2H_MIN_EV: float = 0.015
    TOTALS_MIN_ODDS: float = 1.60
    TOTALS_MIN_EV: float = 0.020
    MAX_REALISTIC_EV: float = 0.15
    REQUIRE_SHARP_LINE: bool = True

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
    AI_MODEL_VALIDATOR: str = "openai/gpt-oss-20b"
    AI_MAX_TOKENS: int = 1024

    TELEGRAM_ID: str = "@zBET90"

    SHARP_BOOKMAKERS: list = field(default_factory=lambda: [
        "pinnacle", "betfair_ex_eu", "matchbook", "betfair_ex_uk"
    ])


CFG = Config()

# =========================================================
# 2. LOGGING
# =========================================================
for d in [CFG.CACHE_DIR, CFG.LOG_DIR, CFG.MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ZBET90")
logger.setLevel(logging.DEBUG)
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
    logger.info("=" * 55)
    logger.info("  %s", title)
    logger.info("=" * 55)


def log_check(label: str, value, expected=None, warn_if_none=True):
    if value is None or value == {} or value == []:
        if warn_if_none:
            logger.warning("CHECK | %-35s | EMPTY/NONE", label)
        else:
            logger.info("CHECK | %-35s | EMPTY (ok)", label)
    else:
        display = str(value)[:80] if not isinstance(value, (int, float, bool)) else value
        status = "OK"
        if expected is not None and value != expected:
            status = "MISMATCH"
        logger.info("CHECK | %-35s | %s | %s", label, status, display)

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

log_check("ODDS_API_KEY set", bool(ODDS_API_KEY))
log_check("GROQ_API_KEY set", bool(GROQ_API_KEY))
log_check("RAPIDAPI_KEY set", bool(RAPIDAPI_KEY))
log_check("TELEGRAM_BOT_TOKEN set", bool(TELEGRAM_BOT_TOKEN))
log_check("TELEGRAM_CHAT_ID set", bool(TELEGRAM_CHAT_ID))
log_check("FOOTBALL_DATA_API_KEY set", bool(FOOTBALL_DATA_API_KEY), warn_if_none=False)

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
    "davidovich": "ES", "carreno": "ES", "lopez": "ES", "badosa": "ES",
    "djokovic": "RS", "kecmanovic": "RS", "krajinovic": "RS",
    "sinner": "IT", "berrettini": "IT", "musetti": "IT", "sonego": "IT",
    "zverev": "DE", "struff": "DE", "koepfer": "DE", "altmaier": "DE",
    "tiafoe": "US", "fritz": "US", "paul": "US", "nakashima": "US",
    "sock": "US", "isner": "US", "spizzirri": "US", "korda": "US",
    "mmoh": "US", "eubanks": "US", "wolf": "US", "gauff": "US",
    "keys": "US", "pegula": "US", "collins": "US",
    "medvedev": "RU", "rublev": "RU", "khachanov": "RU",
    "tsitsipas": "GR", "ruud": "NO", "rune": "DK", "tauson": "DK",
    "wozniacki": "DK", "halep": "RO",
    "hurkacz": "PL", "swiatek": "PL",
    "marcinko": "HR", "cilic": "HR",
    "auger-aliassime": "CA", "shapovalov": "CA", "raonic": "CA",
    "kyrgios": "AU", "de minaur": "AU", "thompson": "AU",
    "lys": "DE", "sabalenka": "BY",
    "kvitova": "CZ", "vondrousova": "CZ",
    "jabeur": "TN", "rybakina": "KZ", "bublik": "KZ",
    "norrie": "GB", "murray": "GB", "draper": "GB",
    "thiem": "AT", "ofner": "AT",
    "wawrinka": "CH", "federer": "CH",
    "monfils": "FR", "gasquet": "FR",
    "dimitrov": "BG",
    "etcheverry": "AR", "cerundolo": "AR", "schwartzman": "AR",
    "manchester united": "GB", "manchester city": "GB",
    "liverpool": "GB", "chelsea": "GB", "arsenal": "GB",
    "tottenham": "GB", "newcastle": "GB", "west ham": "GB",
    "aston villa": "GB", "everton": "GB", "brighton": "GB",
    "celtic": "GB", "rangers": "GB",
    "real madrid": "ES", "barcelona": "ES", "atletico": "ES",
    "sevilla": "ES", "valencia": "ES", "villarreal": "ES",
    "real sociedad": "ES", "athletic bilbao": "ES", "betis": "ES",
    "bayern": "DE", "dortmund": "DE", "leipzig": "DE",
    "leverkusen": "DE", "frankfurt": "DE", "wolfsburg": "DE",
    "juventus": "IT", "milan": "IT", "inter": "IT",
    "napoli": "IT", "roma": "IT", "lazio": "IT", "atalanta": "IT",
    "psg": "FR", "marseille": "FR", "lyon": "FR", "monaco": "FR",
    "lille": "FR", "lens": "FR", "nice": "FR",
    "ajax": "NL", "psv": "NL", "feyenoord": "NL",
    "porto": "PT", "benfica": "PT", "sporting": "PT",
    "galatasaray": "TR", "fenerbahce": "TR", "besiktas": "TR",
    "shakhtar": "UA", "dynamo kyiv": "UA",
    "salzburg": "AT", "rapid wien": "AT",
    "anderlecht": "BE", "club brugge": "BE",
    "zenit": "RU", "spartak": "RU",
    "valerenga": "NO", "brann": "NO", "rosenborg": "NO",
    "molde": "NO", "bodo": "NO",
    "malmo": "SE", "djurgarden": "SE",
    "copenhagen": "DK", "midtjylland": "DK", "brondby": "DK",
    "lakers": "US", "celtics": "US", "warriors": "US",
    "bulls": "US", "heat": "US", "nets": "US",
    "yankees": "US", "dodgers": "US", "cubs": "US",
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

    def mark_sent(self, home: str, away: str, pick: str,
                  market: str, odds: float, commence_time: str) -> None:
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
            if v.get("result_checked"):
                continue
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
# 7. ELO RATING SYSTEM
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
            logger.info("ELO loaded (%s): %d teams/players", self.sport, len(self.ratings))
        else:
            logger.info("ELO: no existing data for %s", self.sport)

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

    def update(self, name_a: str, name_b: str,
               score_a: float, is_home_a: bool = False):
        key_a = name_a.lower().strip()
        key_b = name_b.lower().strip()
        ra = self.get_rating(name_a)
        rb = self.get_rating(name_b)
        if is_home_a:
            ra_adj = ra + CFG.ELO_HOME_ADVANTAGE
        else:
            ra_adj = ra
        ea = self.expected_score(ra_adj, rb)
        eb = 1.0 - ea
        score_b = 1.0 - score_a
        n_a = self.match_count.get(key_a, 0)
        n_b = self.match_count.get(key_b, 0)
        k_a = self.k * (1.5 if n_a < 20 else 1.0)
        k_b = self.k * (1.5 if n_b < 20 else 1.0)
        self.ratings[key_a] = ra + k_a * (score_a - ea)
        self.ratings[key_b] = rb + k_b * (score_b - eb)
        self.match_count[key_a] = n_a + 1
        self.match_count[key_b] = n_b + 1

    def predict(self, home: str, away: str,
                apply_home_advantage: bool = True) -> dict:
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
        result = {
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
            "draw_prob": round(draw_prob, 4),
            "home_elo": round(ra, 1),
            "away_elo": round(rb, 1),
            "elo_diff": round(ra - rb, 1),
            "home_matches": self.match_count.get(home.lower().strip(), 0),
            "away_matches": self.match_count.get(away.lower().strip(), 0),
        }
        log_check(
            f"ELO predict {home[:15]} vs {away[:15]}",
            f"H:{home_prob:.2%} D:{draw_prob:.2%} A:{away_prob:.2%}"
        )
        return result

# =========================================================
# 8. BOOTSTRAP - DATASET DOWNLOAD & ELO TRAINING
# =========================================================
class DataBootstrap:
    FOOTBALL_LEAGUES = [
        ("E0", "England Premier League"),
        ("E1", "England Championship"),
        ("SP1", "Spain La Liga"),
        ("SP2", "Spain La Liga 2"),
        ("D1", "Germany Bundesliga"),
        ("D2", "Germany 2. Bundesliga"),
        ("I1", "Italy Serie A"),
        ("I2", "Italy Serie B"),
        ("F1", "France Ligue 1"),
        ("F2", "France Ligue 2"),
        ("N1", "Netherlands Eredivisie"),
        ("P1", "Portugal Primeira Liga"),
        ("B1", "Belgium Pro League"),
        ("T1", "Turkey Super Lig"),
        ("G1", "Greece Super League"),
    ]

    TENNIS_FILES = [
        "atp_matches_2020.csv", "atp_matches_2021.csv",
        "atp_matches_2022.csv", "atp_matches_2023.csv",
        "atp_matches_2024.csv",
        "wta_matches_2020.csv", "wta_matches_2021.csv",
        "wta_matches_2022.csv", "wta_matches_2023.csv",
        "wta_matches_2024.csv",
    ]

    def __init__(self):
        self.elo_football = ELOSystem("football")
        self.elo_tennis = ELOSystem("tennis")

    def should_run(self) -> bool:
        if FORCE_BOOTSTRAP:
            logger.info("Bootstrap: forced via env var")
            return True
        if not CFG.BOOTSTRAP_FLAG.exists():
            logger.info("Bootstrap: flag not found - first run")
            return True
        try:
            flag_time = datetime.fromisoformat(
                CFG.BOOTSTRAP_FLAG.read_text().strip()
            )
            age_days = (datetime.now(timezone.utc) - flag_time).days
            if age_days >= 7:
                logger.info("Bootstrap: data is %d days old - refreshing", age_days)
                return True
        except Exception:
            return True
        logger.info("Bootstrap: data is fresh - skipping")
        return False

    def run(self):
        log_section("BOOTSTRAP - BUILDING ELO MODELS")
        self._build_football_elo()
        self._build_tennis_elo()
        self.elo_football.save()
        self.elo_tennis.save()
        CFG.BOOTSTRAP_FLAG.write_text(datetime.now(timezone.utc).isoformat())
        log_check("Bootstrap football ELO teams", len(self.elo_football.ratings))
        log_check("Bootstrap tennis ELO players", len(self.elo_tennis.ratings))
        logger.info("Bootstrap complete")

    def _download_csv(self, url: str, timeout: int = 30) -> Optional[pd.DataFrame]:
        try:
            res = requests.get(url, timeout=timeout)
            if res.status_code == 200:
                try:
                    return pd.read_csv(StringIO(res.text))
                except Exception:
                    return pd.read_csv(StringIO(res.text), encoding="latin-1")
            logger.warning("Download failed %d: %s", res.status_code, url)
        except Exception as e:
            logger.warning("Download error %s: %s", url, e)
        return None

    def _build_football_elo(self):
        log_section("Building Football ELO")
        total_matches = 0
        seasons = ["2122", "2223", "2324", "2425"]

        for league_code, league_name in self.FOOTBALL_LEAGUES:
            league_matches = 0
            for season in seasons:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
                df = self._download_csv(url)
                if df is None or df.empty:
                    continue
                required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
                if not required.issubset(df.columns):
                    continue
                df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
                for _, row in df.iterrows():
                    try:
                        home = str(row["HomeTeam"]).strip()
                        away = str(row["AwayTeam"]).strip()
                        ftr = str(row["FTR"]).strip().upper()
                        if ftr == "H":
                            score_home = 1.0
                        elif ftr == "A":
                            score_home = 0.0
                        elif ftr == "D":
                            score_home = 0.5
                        else:
                            continue
                        self.elo_football.update(home, away, score_home, is_home_a=True)
                        league_matches += 1
                    except Exception:
                        continue
                time.sleep(0.2)

            total_matches += league_matches
            if league_matches > 0:
                logger.info("Football ELO: %s -> %d matches", league_name, league_matches)

        log_check("Football ELO total matches processed", total_matches)
        log_check("Football ELO unique teams", len(self.elo_football.ratings))

    def _build_tennis_elo(self):
        log_section("Building Tennis ELO")
        total_matches = 0

        for filename in self.TENNIS_FILES:
            tour = "atp" if filename.startswith("atp") else "wta"
            url = (
                f"https://raw.githubusercontent.com/JeffSackmann/"
                f"tennis_{tour}/master/{filename}"
            )
            df = self._download_csv(url)
            if df is None or df.empty:
                continue
            required = {"winner_name", "loser_name", "surface"}
            if not required.issubset(df.columns):
                continue
            df = df.dropna(subset=["winner_name", "loser_name"])
            file_matches = 0
            for _, row in df.iterrows():
                try:
                    winner = str(row["winner_name"]).strip()
                    loser = str(row["loser_name"]).strip()
                    self.elo_tennis.update(winner, loser, score_a=1.0)
                    file_matches += 1
                except Exception:
                    continue
            total_matches += file_matches
            if file_matches > 0:
                logger.info("Tennis ELO: %s -> %d matches", filename, file_matches)
            time.sleep(0.3)

        log_check("Tennis ELO total matches processed", total_matches)
        log_check("Tennis ELO unique players", len(self.elo_tennis.ratings))

# =========================================================
# 9. UTILS
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
                    logger.warning("Timeout in %s attempt %d/%d",
                                   func.__name__, attempt + 1, max_retries)
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
    logger.error("JSON parse failed: %s", raw_text[:200])
    return None


def clean_team_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(name)).strip()


def normalize_sport_key(sport_title: str) -> str:
    football_kws = [
        "soccer", "football", "premier league", "la liga", "bundesliga",
        "serie a", "ligue 1", "champions league", "europa league",
        "mls", "eredivisie", "primeira liga", "championship",
        "league cup", "fa cup", "copa del rey", "dfb pokal",
        "super lig", "pro league",
    ]
    tennis_kws = ["tennis", "atp", "wta", "wimbledon", "roland garros",
                  "us open", "australian open"]
    tl = sport_title.lower()
    if any(kw in tl for kw in football_kws):
        return "football"
    if any(kw in tl for kw in tennis_kws):
        return "tennis"
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


def _in_window(ct: str, start: datetime, end: datetime) -> bool:
    try:
        return start <= datetime.fromisoformat(ct.replace("Z", "+00:00")) <= end
    except Exception:
        return False

# =========================================================
# 10. MATH ENGINE - SHARP EV + ELO COMBINED
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


def calculate_combined_ev(
    markets_data: dict,
    elo_prediction: Optional[dict],
    sport_key: str,
) -> list:
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
                        sharp_odds[name] = {
                            "price": price,
                            "bookmaker": entry["bookmaker"],
                        }

            for o in entry.get("outcomes", []):
                base_name = o["name"]
                point = o.get("point")
                name = f"{base_name} {point}" if point is not None else base_name
                price = float(o["price"])
                if price <= 1.0:
                    continue
                if name not in best_odds or price > best_odds[name]["price"]:
                    best_odds[name] = {
                        "price": price,
                        "bookmaker": entry["bookmaker"],
                    }

        if CFG.REQUIRE_SHARP_LINE and not has_real_sharp:
            logger.debug("No sharp line for %s - skipping (REQUIRE_SHARP_LINE=True)", market_key)
            continue

        if not sharp_odds and best_odds:
            sharp_odds = dict(best_odds)

        is_valid, reason = validate_sharp_odds(sharp_odds, market_key)
        if not is_valid:
            logger.debug("Skipping %s: %s", market_key, reason)
            continue

        implied_sum = sum(1.0 / v["price"] for v in sharp_odds.values())

        if market_key == "h2h":
            min_odds, min_ev = CFG.H2H_MIN_ODDS, CFG.H2H_MIN_EV
        else:
            min_odds, min_ev = CFG.TOTALS_MIN_ODDS, CFG.TOTALS_MIN_EV

        if not has_real_sharp:
            min_ev *= 2.0

        best_opp = None
        for outcome_name, sharp_data in sharp_odds.items():
            sharp_true_prob = (1.0 / sharp_data["price"]) / implied_sum

            # ترکیب ELO probability با sharp market probability
            elo_true_prob = None
            if elo_prediction and market_key == "h2h":
                elo_true_prob = _get_elo_prob_for_outcome(
                    outcome_name, elo_prediction, sport_key
                )

            if elo_true_prob is not None:
                # وزن‌دهی: ELO 40% + Sharp Market 60%
                true_prob = 0.6 * sharp_true_prob + 0.4 * elo_true_prob
                prob_source = "ELO+Sharp"
            else:
                true_prob = sharp_true_prob
                prob_source = "Sharp"

            best = best_odds.get(outcome_name, {})
            best_price = best.get("price", 0.0)
            best_bookie = best.get("bookmaker", "Unknown")

            if best_price <= 1.0:
                continue

            ev = (true_prob * best_price) - 1.0

            if ev > CFG.MAX_REALISTIC_EV:
                logger.warning(
                    "Rejected unrealistic EV=%.1f%% for %s (sharp=%s)",
                    ev * 100, outcome_name, has_real_sharp,
                )
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
                    "prob_source": prob_source,
                    "elo_prob": round(elo_true_prob, 4) if elo_true_prob else None,
                }
                if best_opp is None or opp["ev"] > best_opp["ev"]:
                    best_opp = opp

        if best_opp:
            best_per_market[market_key] = best_opp

    all_opps = list(best_per_market.values())
    all_opps.sort(key=lambda x: x["ev"], reverse=True)

    log_check("EV opportunities found", len(all_opps))
    for o in all_opps:
        log_check(
            f"  Opp: {o['pick'][:20]}",
            f"EV={o['edge_pct']:.1f}% odds={o['odds']} src={o['prob_source']}"
        )

    return all_opps[:1]


def _get_elo_prob_for_outcome(
    outcome_name: str,
    elo_pred: dict,
    sport_key: str,
) -> Optional[float]:
    name_lower = outcome_name.lower()
    if "draw" in name_lower or "tie" in name_lower:
        return elo_pred.get("draw_prob")
    if "over" in name_lower or "under" in name_lower:
        return None
    home_prob = elo_pred.get("home_prob", 0.5)
    away_prob = elo_pred.get("away_prob", 0.5)
    elo_diff = elo_pred.get("elo_diff", 0)
    if elo_diff > 0:
        return home_prob if "home" not in name_lower else home_prob
    return away_prob

# =========================================================
# 11. ODDS API - ONE REQUEST (h2h,totals combined)
# =========================================================
@retry_request(max_retries=3)
def fetch_odds_sync(now_utc: datetime) -> list:
    """
    یک request به جای دو - h2h و totals با هم
    این صرفه‌جویی در API credits انجام میده
    """
    end_window = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": CFG.ODDS_API_REGIONS,
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    res = requests.get(
        "https://api.the-odds-api.com/v4/sports/upcoming/odds",
        params=params,
        timeout=25,
    )
    res.raise_for_status()

    remaining = res.headers.get("x-requests-remaining", "?")
    used = res.headers.get("x-requests-used", "?")
    log_check("Odds API requests remaining", remaining)
    log_check("Odds API requests used", used)

    events = res.json()
    all_events: dict = {}

    for e in events:
        if not _in_window(e.get("commence_time", ""), now_utc, end_window):
            continue
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
    log_check("Odds API events in window", len(result))
    return result


@retry_request(max_retries=3)
def fetch_event_result(event_id: str, home: str, away: str) -> Optional[dict]:
    """
    دریافت نتیجه بازی از Odds API
    از scores endpoint استفاده میکنه - بدون هزینه request جداگانه
    """
    try:
        params = {
            "apiKey": ODDS_API_KEY,
            "daysFrom": 3,
            "dateFormat": "iso",
        }
        res = requests.get(
            "https://api.the-odds-api.com/v4/sports/upcoming/scores",
            params=params,
            timeout=15,
        )
        if res.status_code != 200:
            return None
        scores = res.json()
        for event in scores:
            if (event.get("home_team", "").lower() == home.lower() and
                    event.get("away_team", "").lower() == away.lower()):
                if event.get("completed"):
                    return event
        return None
    except Exception as e:
        logger.debug("Fetch result error: %s", e)
        return None

# =========================================================
# 12. SOFASCORE ASYNC
# =========================================================
def _sofa_headers() -> dict:
    return {
        "x-rapidapi-key": RAPIDAPI_KEY or "",
        "x-rapidapi-host": "sofascore.p.rapidapi.com",
    }


async def _sofa_get(
    session: aiohttp.ClientSession, url: str, params: dict
) -> Optional[dict]:
    try:
        async with session.get(
            url, headers=_sofa_headers(), params=params,
            timeout=aiohttp.ClientTimeout(total=12),
        ) as res:
            if res.status == 200:
                return await res.json()
            logger.debug("SofaScore %d for %s", res.status, url)
    except Exception as e:
        logger.debug("SofaScore error %s: %s", url, e)
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
            *[_sofa_get(session, url, params) for url in endpoints.values()],
            return_exceptions=True,
        )
    data = {}
    for key, result in zip(endpoints.keys(), results):
        if not isinstance(result, Exception) and result is not None:
            data[key] = result
    log_check(f"SofaScore data keys for match {match_id}", list(data.keys()))
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
                            logger.info("SofaScore: %s vs %s -> ID:%s", home, away, mid)
                            return mid
        except Exception as e:
            logger.debug("SofaScore search error: %s", e)
    return None

# =========================================================
# 13. FOOTBALL-DATA ADAPTER
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
        log_check("Football-Data calls today", self.call_count)

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
                "Football-Data limit (%d/%d)",
                self.call_count, CFG.FOOTBALL_DATA_DAILY_LIMIT,
            )
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
        logger.info("Searching team ID: '%s'", team_name)
        data = self._raw_get("/teams", {"name": clean_team_name(team_name)})
        team_id = None
        if data and data.get("teams"):
            team_id = data["teams"][0]["id"]
            logger.info("Team found: %s -> ID:%d", team_name, team_id)
        cache[key] = team_id
        CacheManager.save(CFG.TEAM_ID_CACHE_FILE, cache)
        return team_id

    def get_team_recent_form(self, team_id: int, team_name: str) -> dict:
        cache_key = f"form_{team_id}"
        if CacheManager.is_valid(self.daily_cache, cache_key, CFG.TTL_TEAM_FORM):
            cached = CacheManager.get(self.daily_cache, cache_key) or {}
            log_check(f"Form cache hit: {team_name[:20]}", cached.get("form_string", "N/A"))
            return cached
        logger.info("Fetching form: %s (id:%d)", team_name, team_id)
        data = self._raw_get(
            f"/teams/{team_id}/matches",
            {"status": "FINISHED", "limit": 5},
        )
        if not data:
            return {}
        form = self._parse_form(data, team_id)
        log_check(f"Form fetched: {team_name[:20]}", form.get("form_string", "N/A"))
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
        log_check(f"H2H {team1_id} vs {team2_id}", result.get("total_h2h", 0))
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
# 14. MATCH ID CACHE
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
# 15. STATS AGGREGATOR
# =========================================================
async def get_stats_async(
    home: str,
    away: str,
    sport_key: str,
    football_adapter: FootballDataAdapter,
    match_id_cache: MatchIDCache,
    elo_football: ELOSystem,
    elo_tennis: ELOSystem,
) -> dict:
    stats = {
        "home_form": {},
        "away_form": {},
        "h2h": {},
        "sofascore": {},
        "elo": {},
        "data_quality": "none",
    }

    # ELO prediction (instant - no API call)
    if sport_key == "football":
        elo_pred = elo_football.predict(home, away, apply_home_advantage=True)
    elif sport_key == "tennis":
        elo_pred = elo_tennis.predict(home, away, apply_home_advantage=False)
    else:
        elo_pred = None

    if elo_pred:
        stats["elo"] = elo_pred
        log_check(
            f"ELO for {home[:15]} vs {away[:15]}",
            f"H={elo_pred['home_prob']:.2%} A={elo_pred['away_prob']:.2%} "
            f"matches_h={elo_pred['home_matches']} matches_a={elo_pred['away_matches']}"
        )

    # SofaScore match ID
    cached_mid = match_id_cache.get(home, away)
    if cached_mid is not None:
        match_id = cached_mid if cached_mid != 0 else None
    else:
        match_id = await search_sofascore_match_async(home, away)
        match_id_cache.set(home, away, match_id if match_id else 0)

    log_check(f"SofaScore match_id {home[:15]}", match_id, warn_if_none=False)

    task_names = []
    coros = []

    if match_id:
        task_names.append("sofascore")
        coros.append(fetch_sofascore_stats_async(match_id))

    if sport_key == "football":
        loop = asyncio.get_running_loop()

        async def get_football_data():
            home_id = await loop.run_in_executor(
                None, football_adapter.find_team_id, home
            )
            away_id = await loop.run_in_executor(
                None, football_adapter.find_team_id, away
            )
            log_check(f"FD team_id {home[:15]}", home_id, warn_if_none=False)
            log_check(f"FD team_id {away[:15]}", away_id, warn_if_none=False)
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
                logger.warning("Stats fetch error (%s): %s", name, result)
                continue
            if name == "sofascore" and result:
                stats["sofascore"] = result
            elif name == "football" and isinstance(result, dict):
                stats.update(result)

    has_football_data = bool(stats.get("home_form") or stats.get("h2h"))
    has_sofascore = bool(stats.get("sofascore"))
    has_elo = bool(stats.get("elo") and stats["elo"].get("home_matches", 0) >= 5)

    if (has_football_data or has_elo) and has_sofascore:
        stats["data_quality"] = "high"
    elif has_football_data or has_elo or has_sofascore:
        stats["data_quality"] = "medium"

    log_check(
        f"Data quality {home[:15]} vs {away[:15]}",
        stats["data_quality"]
    )
    return stats, elo_pred

# =========================================================
# 16. DUAL-AI ANALYSIS
# =========================================================
def build_stats_summary(stats: dict, home: str, away: str) -> str:
    parts = []
    hf = stats.get("home_form", {})
    af = stats.get("away_form", {})
    h2h = stats.get("h2h", {})
    elo = stats.get("elo", {})

    if elo and elo.get("home_matches", 0) >= 3:
        parts.append(
            f"ELO RATINGS: Home={elo.get('home_elo', 1500):.0f} "
            f"Away={elo.get('away_elo', 1500):.0f} "
            f"Diff={elo.get('elo_diff', 0):.0f} | "
            f"ELO_Prob: Home={elo.get('home_prob', 0):.1%} "
            f"Draw={elo.get('draw_prob', 0):.1%} "
            f"Away={elo.get('away_prob', 0):.1%}"
        )
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
        parts.append(f"SOFASCORE: {json.dumps(ss, separators=(',', ':'))[:1000]}")

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
        content = res.choices[0].message.content
        log_check(f"Groq {model[:20]} response", content[:100] if content else None)
        return content
    except Exception as e:
        logger.error("Groq SDK error model=%s: %s", model, e)
        return None


def generate_dual_ai_analysis(
    home: str, away: str, sport: str,
    pick: str, market: str, ev_edge: float,
    stats: dict,
) -> dict:
    stats_summary = build_stats_summary(stats, home, away)
    data_quality = stats.get("data_quality", "none")

    log_check("AI input stats_summary length", len(stats_summary))
    log_check("AI input data_quality", data_quality)

    default_response = {
        "sport_emoji": "\U0001F3C6",
        "home_flag": get_flag_from_name(home),
        "away_flag": get_flag_from_name(away),
        "risk_level": "High",
        "confidence": 55,
        "logic": "Mathematical edge confirmed by Sharp Market Model and ELO rating system.",
    }

    sys1 = (
        "You are an Elite Quantitative Sports Analyst.\n"
        "TASKS:\n"
        "1. Write EXACTLY 2 sentences justifying the Pick.\n"
        "   Use ONLY data from STATISTICS. Do NOT invent numbers.\n"
        "   If ELO data is available, reference it.\n"
        "2. Choose correct sport_emoji.\n"
        "3. Determine EXACT country flag emoji for home_flag and away_flag.\n"
        "4. Assign risk_level AND confidence consistently:\n"
        "   Low risk = confidence 75-93\n"
        "   Medium risk = confidence 60-74\n"
        "   High risk = confidence 50-59\n\n"
        "OUTPUT: valid JSON only. No markdown.\n"
        '{"sport_emoji":"...","home_flag":"...","away_flag":"...",'
        '"risk_level":"Medium","confidence":65,"logic":"s1. s2."}'
    )
    u1 = (
        f"MATCH: {home} vs {away}\n"
        f"SPORT: {sport}\n"
        f"PICK: {pick} [{market}]\n"
        f"EV EDGE: +{ev_edge:.1%}\n"
        f"DATA QUALITY: {data_quality}\n\n"
        f"STATISTICS:\n{stats_summary}\n\n"
        "OUTPUT JSON ONLY:"
    )

    analysis_1 = None
    try:
        raw1 = call_groq_sdk(
            CFG.AI_MODEL_ANALYST,
            [{"role": "system", "content": sys1},
             {"role": "user", "content": u1}],
            temperature=0.1,
        )
        analysis_1 = robust_json_extractor(raw1)
        log_check("Model 1 analysis_1", "OK" if analysis_1 else "FAILED")
    except Exception as e:
        logger.warning("Model 1 failed: %s", e)

    time.sleep(1.5)

    initial_logic = (analysis_1 or {}).get("logic", "No initial analysis")
    initial_risk = (analysis_1 or {}).get("risk_level", "High")
    initial_conf = (analysis_1 or {}).get("confidence", 55)

    sys2 = (
        "You are a Senior Betting Risk Analyst.\n\n"
        "CONFIDENCE RUBRIC (base=50):\n"
        "  + Data quality: high=+15, medium=+8, none=0\n"
        "  + ELO available (5+ matches): +10\n"
        "  + EV edge: >5%=+10, 3-5%=+7, 1.5-3%=+4\n"
        "  + Form consistency (3+ same): +8\n"
        "  + H2H supports pick: +7, neutral: +2, against: -5\n"
        "  + Sharp line confirmed: +5\n"
        "  Cap: 93\n\n"
        "RISK RULE:\n"
        "  75-93 = Low | 60-74 = Medium | 50-59 = High\n\n"
        "Rewrite logic if vague. Never invent stats.\n\n"
        "OUTPUT: valid JSON only.\n"
        '{"confidence":72,"validated_logic":"logic.","risk_level":"Medium"}'
    )
    u2 = (
        f"MATCH: {home} vs {away} | PICK: {pick} [{market}] | EV: +{ev_edge:.1%}\n"
        f"DATA QUALITY: {data_quality}\n"
        f"STATS:\n{stats_summary[:1800]}\n\n"
        f"INITIAL LOGIC: {initial_logic}\n"
        f"INITIAL RISK: {initial_risk} | INITIAL CONF: {initial_conf}\n\n"
        "OUTPUT JSON ONLY:"
    )

    analysis_2 = None
    try:
        raw2 = call_groq_sdk(
            CFG.AI_MODEL_VALIDATOR,
            [{"role": "user", "content": sys2 + "\n\n" + u2}],
            temperature=0.2,
        )
        analysis_2 = robust_json_extractor(raw2)
        log_check("Model 2 analysis_2", "OK" if analysis_2 else "FAILED")
    except Exception as e:
        logger.warning("Model 2 failed: %s", e)

    result = dict(default_response)
    if analysis_1:
        for k, v in analysis_1.items():
            if v:
                result[k] = v

    result["home_flag"] = validate_flag(result.get("home_flag", ""), home)
    result["away_flag"] = validate_flag(result.get("away_flag", ""), away)

    if analysis_2:
        if analysis_2.get("validated_logic"):
            result["logic"] = analysis_2["validated_logic"]
        if analysis_2.get("confidence"):
            conf = max(50, min(93, int(analysis_2["confidence"])))
            result["confidence"] = conf
            if conf >= 75:
                result["risk_level"] = "Low"
            elif conf >= 60:
                result["risk_level"] = "Medium"
            else:
                result["risk_level"] = "High"

    if result["confidence"] == 55 and ev_edge > 0:
        conf = min(65, 55 + int(ev_edge * 200))
        result["confidence"] = conf
        result["risk_level"] = "Medium" if conf >= 60 else "High"

    log_check("Final confidence", result["confidence"])
    log_check("Final risk_level", result["risk_level"])
    log_check("Final logic preview", result["logic"][:80])

    return result

# =========================================================
# 17. RESULTS CHECKER
# =========================================================
def check_and_report_results(sent_history: SentHistory) -> Optional[str]:
    log_section("PHASE 1 - RESULTS CHECK")
    pending = sent_history.get_pending_results()
    log_check("Pending results to check", len(pending))

    if not pending:
        return None

    wins, losses, unknowns = [], [], []

    for key, entry in pending:
        home = entry.get("home", "")
        away = entry.get("away", "")
        pick = entry.get("pick", "")
        market = entry.get("market", "")
        odds = entry.get("odds", 0)

        logger.info("Checking result: %s vs %s | Pick: %s", home, away, pick)

        result_event = fetch_event_result("", home, away)
        log_check(f"Result event {home[:15]}", "found" if result_event else "not found",
                  warn_if_none=False)

        if not result_event:
            unknowns.append(entry)
            continue

        scores = result_event.get("scores", {})
        won = _determine_win(pick, market, scores, home, away)

        result_str = _format_score(scores, home, away)
        sent_history.mark_result_checked(key, result_str, won)

        if won is True:
            wins.append({**entry, "result": result_str})
        elif won is False:
            losses.append({**entry, "result": result_str})
        else:
            unknowns.append({**entry, "result": result_str})

    log_check("Results - Wins", len(wins))
    log_check("Results - Losses", len(losses))
    log_check("Results - Unknown", len(unknowns))

    if not wins and not losses:
        return None

    return _build_results_message(wins, losses)


def _format_score(scores: dict, home: str, away: str) -> str:
    try:
        home_score = scores.get(home, {}).get("score", "?")
        away_score = scores.get(away, {}).get("score", "?")
        return f"{home_score} - {away_score}"
    except Exception:
        return "? - ?"


def _determine_win(
    pick: str, market: str, scores: dict, home: str, away: str
) -> Optional[bool]:
    try:
        if market == "h2h":
            home_sc = int(scores.get(home, {}).get("score", -1))
            away_sc = int(scores.get(away, {}).get("score", -1))
            if home_sc < 0 or away_sc < 0:
                return None
            pick_lower = pick.lower()
            if home.lower() in pick_lower or "home" in pick_lower:
                return home_sc > away_sc
            if away.lower() in pick_lower or "away" in pick_lower:
                return away_sc > home_sc
            if "draw" in pick_lower or "tie" in pick_lower:
                return home_sc == away_sc

        elif market == "totals":
            home_sc = int(scores.get(home, {}).get("score", -1))
            away_sc = int(scores.get(away, {}).get("score", -1))
            if home_sc < 0 or away_sc < 0:
                return None
            total = home_sc + away_sc
            pick_lower = pick.lower()
            m = re.search(r"(over|under)\s*([\d.]+)", pick_lower)
            if m:
                direction = m.group(1)
                line = float(m.group(2))
                if direction == "over":
                    return total > line
                else:
                    return total < line
    except Exception as e:
        logger.debug("Win determination error: %s", e)
    return None


def _build_results_message(wins: list, losses: list) -> str:
    total = len(wins) + len(losses)
    win_rate = len(wins) / total if total > 0 else 0

    roi_parts = []
    for w in wins:
        roi_parts.append(w.get("odds", 1.0) - 1.0)
    for _ in losses:
        roi_parts.append(-1.0)
    roi = sum(roi_parts) / len(roi_parts) if roi_parts else 0

    lines = [
        "\U0001F4CA <b>RESULTS REPORT</b>\n",
    ]

    for w in wins:
        lines.append(
            f"\U00002705 <b>{html_lib.escape(w['home'])} vs {html_lib.escape(w['away'])}</b>\n"
            f"   Pick: {html_lib.escape(w['pick'])} @ <code>{w['odds']}</code>\n"
            f"   Result: {w.get('result', '?')} \u2014 <b>WIN</b>\n"
        )

    for l in losses:
        lines.append(
            f"\u274C <b>{html_lib.escape(l['home'])} vs {html_lib.escape(l['away'])}</b>\n"
            f"   Pick: {html_lib.escape(l['pick'])} @ <code>{l['odds']}</code>\n"
            f"   Result: {l.get('result', '?')} \u2014 <b>LOSS</b>\n"
        )

    lines.append(
        f"\n\U0001F3AF <b>Session:</b> "
        f"{len(wins)}W / {len(losses)}L | "
        f"Win Rate: {win_rate:.0%} | "
        f"ROI: {roi:+.1%}\n\n"
        f"\U0001F194 {CFG.TELEGRAM_ID}"
    )

    return "\n".join(lines)

# =========================================================
# 18. TELEGRAM
# =========================================================
@retry_request(max_retries=3)
def send_telegram(message_html: str) -> bool:
    res = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message_html,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=10,
    )
    res.raise_for_status()
    return True


def format_signal_message(
    home: str, away: str, sport: str,
    opp: dict, ai_data: dict, countdown_str: str,
) -> str:
    risk_raw = str(ai_data.get("risk_level", "High")).capitalize()
    risk_icon = {
        "Low": "\U0001F7E2",
        "Medium": "\U0001F7E0",
        "High": "\U0001F534",
    }.get(risk_raw, "\U0001F534")

    confidence = ai_data.get("confidence", 55)
    conf_icon = (
        "\U0001F525" if confidence >= 75
        else ("\U00002705" if confidence >= 60 else "\U000026A1")
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

    msg = (
        f"{sport_emoji} <b>{html_lib.escape(sport)}</b>\n\n"
        f"\u2694\uFE0F <b>{html_lib.escape(home)}</b> {home_flag}"
        f"  <b>vs</b>  "
        f"{away_flag} <b>{html_lib.escape(away)}</b>\n\n"
        f"\u23F3 <b>Starts in:</b> {countdown_str}\n\n"
        f"{pick_line}\n\n"
        f"{risk_icon} <b>Risk:</b> {risk_raw}"
        f"  |  {conf_icon} <b>Confidence: {confidence}%</b>\n\n"
        f"\U0001F4A1 <b>Analysis:</b>\n"
        f"<blockquote>{logic_escaped}</blockquote>\n\n"
        f"\U0001F194 <b>Channel:</b> {CFG.TELEGRAM_ID}"
    )
    return msg

# =========================================================
# 19. MAIN ASYNC PIPELINE
# =========================================================
async def async_main():
    log_section("ZBET90 ENTERPRISE ENGINE v3.0 STARTING")

    # ── Bootstrap ──────────────────────────────────────
    bootstrap = DataBootstrap()
    if bootstrap.should_run():
        bootstrap.run()

    elo_football = ELOSystem("football")
    elo_tennis = ELOSystem("tennis")

    log_check("ELO football teams loaded", len(elo_football.ratings))
    log_check("ELO tennis players loaded", len(elo_tennis.ratings))

    sent_history = SentHistory()
    football_adapter = FootballDataAdapter()
    match_id_cache = MatchIDCache()
    now_utc = datetime.now(timezone.utc)

    # ── Phase 1: Results Check ─────────────────────────
    log_section("PHASE 1 - RESULTS CHECK")
    results_msg = check_and_report_results(sent_history)
    if results_msg:
        if send_telegram(results_msg):
            logger.info("Results report sent to Telegram")
        await asyncio.sleep(2)
    else:
        logger.info("No results to report this run")

    # ── Phase 2: Fetch Odds (1 request) ────────────────
    log_section("PHASE 2 - FETCHING ODDS")
    events = fetch_odds_sync(now_utc) or []
    log_check("Events fetched from Odds API", len(events))

    if not events:
        logger.info("No events in 2-hour window.")
        return

    # ── Phase 3: Analyze & Signal ──────────────────────
    log_section("PHASE 3 - ANALYSIS & SIGNALS")
    total_sent = 0

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        sport = event.get("sport_title", "Unknown")
        sport_key = normalize_sport_key(sport)
        markets_data = event.get("_markets_data", {})
        commence_time = event.get("commence_time", "")
        countdown_str = get_countdown_str(commence_time, now_utc)

        if not home or not away:
            continue

        logger.info("--- Analyzing: %s vs %s [%s]", home, away, sport)

        # ELO prediction برای math engine
        if sport_key == "football":
            elo_pred = elo_football.predict(home, away)
        elif sport_key == "tennis":
            elo_pred = elo_tennis.predict(home, away, apply_home_advantage=False)
        else:
            elo_pred = None

        opportunities = calculate_combined_ev(markets_data, elo_pred, sport_key)

        if not opportunities:
            logger.debug("No EV opportunity: %s vs %s", home, away)
            continue

        opp = opportunities[0]

        if sent_history.was_sent(home, away, opp["market"]):
            logger.info("SKIP duplicate: %s vs %s [%s]", home, away, opp["market"])
            continue

        log_check(
            f"Signal: {home[:15]} vs {away[:15]}",
            f"pick={opp['pick']} ev={opp['edge_pct']:.1f}% odds={opp['odds']}"
        )

        # Stats async
        stats, _ = await get_stats_async(
            home, away, sport_key,
            football_adapter, match_id_cache,
            elo_football, elo_tennis,
        )

        # Dual-AI analysis
        ai_data = generate_dual_ai_analysis(
            home, away, sport,
            opp["pick"], opp["market"], opp["ev"], stats,
        )

        msg = format_signal_message(
            home, away, sport, opp, ai_data, countdown_str
        )

        if send_telegram(msg):
            sent_history.mark_sent(
                home, away, opp["pick"], opp["market"],
                opp["odds"], commence_time,
            )
            total_sent += 1
            logger.info("Sent: %s vs %s | %s", home, away, opp["pick"])
        else:
            logger.error("Telegram failed: %s vs %s", home, away)

        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    log_section("RUN COMPLETE")
    log_check("Total signals sent this run", total_sent)
    logger.info("=" * 55)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("SYSTEM FAILURE: %s", str(e), exc_info=True)
        sys.exit(1)
