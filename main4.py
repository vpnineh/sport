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
try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    cffi_requests = None
    logger.warning("curl_cffi not installed! Direct Sofascore scraping disabled.")
import pandas as pd
from io import StringIO
from groq import AsyncGroq
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
    CACHE_DIR:              Path = Path("api_cache")
    LOG_DIR:                Path = Path("log")
    MODELS_DIR:             Path = Path("api_cache/models")
    HISTORY_FILE:           Path = Path("api_cache/sent_history.json")
    TEAM_ID_CACHE_FILE:     Path = Path("api_cache/team_id_cache.json")
    MATCH_ID_CACHE_FILE:    Path = Path("api_cache/match_id_cache.json")
    DAILY_STATS_CACHE_FILE: Path = Path("api_cache/daily_stats_cache.json")
    DAILY_ODDS_CACHE_FILE:  Path = Path("api_cache/daily_odds.json")
    DAILY_RAPID_CACHE_FILE: Path = Path("api_cache/daily_rapid_stats.json")
    CLV_FILE:               Path = Path("api_cache/clv_tracker.json")
    PERFORMANCE_FILE:       Path = Path("api_cache/performance.json")
    LOG_FILE:               Path = Path("api_cache/execution_logs.log")
    ELO_FOOTBALL_FILE:      Path = Path("api_cache/models/elo_football.json")
    ELO_TENNIS_FILE:        Path = Path("api_cache/models/elo_tennis.json")
    BOOTSTRAP_FLAG:         Path = Path("api_cache/models/bootstrap_done.flag")
    KEY_STATUS_FILE:        Path = Path("api_cache/key_status.json")

    MATCH_WINDOW_HOURS:     float = 4.0
    RESULT_CHECK_HOURS:     float = 4.0
    TELEGRAM_SLEEP_BETWEEN: float = 3.0

    ODDS_API_MARKETS:      str = "h2h,totals"
    ODDS_API_REGIONS:      str = "eu,uk,us,au"
    ODDS_API_ODDS_FORMAT:  str = "decimal"
    ODDS_API_DATE_FORMAT:  str = "iso"
    MAX_SPORTS_PER_DAY:    int = 20

    TTL_SENT_HISTORY: float = 72.0
    TTL_MATCH_ID:     float = 24.0
    TTL_TEAM_FORM:    float = 6.0
    TTL_H2H:          float = 24.0

    H2H_MIN_ODDS:           float = 1.50
    H2H_MIN_EV:             float = 0.015
    TOTALS_MIN_ODDS:        float = 1.60
    TOTALS_MIN_EV:          float = 0.020
    MAX_REALISTIC_EV:       float = 0.12
    MAX_EV_WITHOUT_DATA:    float = 0.06
    MIN_CONFIDENCE_TO_SEND: int   = 62

    VALID_MARKETS: tuple = field(default_factory=lambda: ("h2h", "totals"))

    MARKET_EXPECTED_OUTCOMES: dict = field(default_factory=lambda: {
        "h2h":    {"min": 2, "max": 3},
        "totals": {"min": 2, "max": 2},
    })
    MAX_VALID_IMPLIED_SUM: float = 1.20
    MIN_VALID_IMPLIED_SUM: float = 0.80

    ELO_K_FACTOR_FOOTBALL: float = 32.0
    ELO_K_FACTOR_TENNIS:   float = 40.0
    ELO_HOME_ADVANTAGE:    float = 80.0
    ELO_DEFAULT:           float = 1500.0

    AI_MODEL_ANALYST:   str = "meta-llama/llama-4-scout-17b-16e-instruct"
    AI_MODEL_VALIDATOR: str = "llama-3.1-8b-instant"
    AI_MAX_TOKENS:      int = 1024

    TELEGRAM_ID: str = "@zBET90"

    SHARP_BOOKMAKERS: list = field(default_factory=lambda: [
        "pinnacle", "betfair_ex_eu", "matchbook", "betfair_ex_uk",
    ])

    MARKET_DISPLAY: dict = field(default_factory=lambda: {
        "h2h":    "1X2 Full Time",
        "totals": "Over / Under Goals",
    })

    PRIORITY_SPORT_KEYWORDS: list = field(default_factory=lambda: [
        "soccer", "tennis", "basketball", "baseball",
        "hockey", "football", "mma", "boxing",
    ])

    EXCLUDED_SPORT_KEYWORDS: list = field(default_factory=lambda: [
        "winner", "outrights", "futures", "specials",
    ])

    RAPID_SUPPORTED_SPORTS: set = field(
        default_factory=lambda: {"football", "tennis"}
    )

    FOOTBALL_DATA_DAILY_LIMIT: int = 80
    FD_COMPETITION_IDS: list = field(default_factory=lambda: [
        2021, 2014, 2002, 2019, 2015,
        2003, 2017, 2016, 2001,
    ])

    RAPID_REQUEST_DELAY:    float = 0.4
    RAPID_RATE_LIMIT_PAUSE: int   = 3

CFG = Config()

# =========================================================
# 2. LOGGING
# =========================================================
for _d in [CFG.CACHE_DIR, CFG.LOG_DIR, CFG.MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

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


def log_section(title: str) -> None:
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def log_check(label: str, value, warn_if_none: bool = True) -> None:
    if value is None or value in ({}, [], "", 0):
        if warn_if_none:
            logger.warning("CHECK | %-42s | EMPTY/NONE", label)
        else:
            logger.info("CHECK | %-42s | EMPTY (ok)", label)
    else:
        logger.info("CHECK | %-42s | OK | %s", label, str(value)[:100])

# =========================================================
# 3. API KEYS
# =========================================================
GROQ_API_KEY          = os.getenv("GROQ_API_KEY",          "").strip()
RAPIDAPI_KEY          = os.getenv("RAPIDAPI_KEY",          "").strip()
RAPIDAPI_KEY2         = os.getenv("RAPIDAPI_KEY2",         "").strip()
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN",    "").strip()
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID",      "").strip()
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
FORCE_BOOTSTRAP       = os.getenv("FORCE_BOOTSTRAP", "false").lower() == "true"

RAPIDAPI_KEYS: list[str] = [k for k in [RAPIDAPI_KEY, RAPIDAPI_KEY2] if k]

_RAW_ODDS_KEYS: list[str] = [
    os.getenv("ODDS_API_KEY",  "").strip(),
    os.getenv("ODDS_API_KEY2", "").strip(),
    os.getenv("ODDS_API_KEY3", "").strip(),
]
ODDS_API_KEYS: list[str] = [k for k in _RAW_ODDS_KEYS if k]

logger.info("━" * 60)
logger.info("  KEY STATUS")
logger.info("━" * 60)

for _i, _raw in enumerate(_RAW_ODDS_KEYS, 1):
    if _raw:
        logger.info(
            "KEY  | ODDS_API_KEY%-2d | SET  | len=%-3d | prefix=%s…",
            _i, len(_raw), _raw[:6],
        )
    else:
        logger.warning("KEY  | ODDS_API_KEY%-2d | MISSING", _i)

for _i, _raw in enumerate([RAPIDAPI_KEY, RAPIDAPI_KEY2], 1):
    _name = f"RAPIDAPI_KEY{'2' if _i == 2 else ''}"
    if _raw:
        logger.info(
            "KEY  | %-28s | SET  | len=%-3d | prefix=%s…",
            _name, len(_raw), _raw[:6],
        )
    else:
        logger.warning("KEY  | %-28s | MISSING", _name)

for _name, _val in [
    ("GROQ_API_KEY",          GROQ_API_KEY),
    ("TELEGRAM_BOT_TOKEN",    TELEGRAM_BOT_TOKEN),
    ("TELEGRAM_CHAT_ID",      TELEGRAM_CHAT_ID),
    ("FOOTBALL_DATA_API_KEY", FOOTBALL_DATA_API_KEY),
]:
    if _val:
        logger.info(
            "KEY  | %-28s | SET  | len=%-3d | prefix=%s…",
            _name, len(_val), _val[:4],
        )
    else:
        logger.warning("KEY  | %-28s | MISSING", _name)

if not ODDS_API_KEYS:
    logger.critical("FATAL: No ODDS_API_KEY found!")
    sys.exit(1)

if not all([GROQ_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing critical API key(s).")
    sys.exit(1)

if not RAPIDAPI_KEYS:
    logger.warning("WARNING: No RAPIDAPI_KEY — RapidAPI stats disabled")

logger.info(
    "Odds API keys: %d/3 | RapidAPI keys: %d/2",
    len(ODDS_API_KEYS), len(RAPIDAPI_KEYS),
)

groq_client = AsyncGroq(api_key=GROQ_API_KEY, max_retries=3)

# =========================================================
# 4. NATIONALITY FLAGS
# =========================================================
NATIONALITY_FLAGS: dict[str, str] = {
    "djokovic": "RS", "kecmanovic": "RS", "lajovic": "RS",
    "sinner": "IT", "berrettini": "IT", "musetti": "IT",
    "arnaldi": "IT", "sonego": "IT", "cobolli": "IT",
    "alcaraz": "ES", "nadal": "ES", "bautista agut": "ES",
    "davidovich fokina": "ES", "carreno busta": "ES",
    "munar": "ES", "baeza": "ES",
    "medvedev": "RU", "rublev": "RU", "khachanov": "RU",
    "safiullin": "RU", "karatsev": "RU",
    "zverev": "DE", "struff": "DE", "altmaier": "DE", "koepfer": "DE",
    "tsitsipas": "GR", "ruud": "NO", "rune": "DK", "hurkacz": "PL",
    "de minaur": "AU", "kyrgios": "AU", "thompson": "AU",
    "popyrin": "AU", "purcell": "AU",
    "fritz": "US", "paul": "US", "tiafoe": "US", "shelton": "US",
    "korda": "US", "nakashima": "US", "eubanks": "US", "michelsen": "US",
    "dimitrov": "BG",
    "auger-aliassime": "CA", "shapovalov": "CA", "diallo": "CA",
    "baez": "AR", "cerundolo": "AR", "etcheverry": "AR",
    "navone": "AR", "diaz acosta": "AR",
    "bublik": "KZ",
    "norrie": "GB", "draper": "GB", "murray": "GB", "evans": "GB",
    "humbert": "FR", "mannarino": "FR", "fils": "FR",
    "monfils": "FR", "cazaux": "FR", "mpetshi": "FR",
    "griekspoor": "NL", "brouwer": "NL",
    "lehecka": "CZ", "mensik": "CZ", "machac": "CZ",
    "wawrinka": "CH", "stricker": "CH",
    "swiatek": "PL", "linette": "PL", "frech": "PL",
    "sabalenka": "BY", "azarenka": "BY",
    "gauff": "US", "pegula": "US", "keys": "US",
    "navarro": "US", "collins": "US", "stephens": "US",
    "rybakina": "KZ", "putintseva": "KZ",
    "vondrousova": "CZ", "muchova": "CZ", "krejcikova": "CZ",
    "noskova": "CZ", "pliskova": "CZ", "bouzkova": "CZ",
    "zheng": "CN",
    "andreeva": "RU", "kasatkina": "RU", "samsonova": "RU",
    "alexandrova": "RU", "kudermetova": "RU",
    "potapova": "RU", "pavlyuchenkova": "RU",
    "sakkari": "GR", "jabeur": "TN", "ostapenko": "LV",
    "garcia": "FR", "cornet": "FR",
    "svitolina": "UA", "yastremska": "UA",
    "kostyuk": "UA", "kalinina": "UA",
    "haddad maia": "BR",
    "paolini": "IT", "giorgi": "IT",
    "teichmann": "CH", "bencic": "CH",
    "badosa": "ES", "sorribes": "ES",
    "boulter": "GB", "raducanu": "GB",
    "manchester city": "GB", "manchester united": "GB",
    "arsenal": "GB", "liverpool": "GB", "chelsea": "GB",
    "tottenham": "GB", "newcastle": "GB", "aston villa": "GB",
    "west ham": "GB", "brighton": "GB", "everton": "GB",
    "crystal palace": "GB", "bournemouth": "GB", "fulham": "GB",
    "brentford": "GB", "wolves": "GB", "nottingham": "GB",
    "leicester": "GB", "ipswich": "GB", "southampton": "GB",
    "real madrid": "ES", "barcelona": "ES", "atletico": "ES",
    "girona": "ES", "athletic bilbao": "ES", "real sociedad": "ES",
    "betis": "ES", "villarreal": "ES", "valencia": "ES",
    "osasuna": "ES", "sevilla": "ES", "celta": "ES",
    "mallorca": "ES", "las palmas": "ES", "rayo": "ES",
    "almeria": "ES", "almería": "ES", "granada": "ES",
    "cadiz": "ES", "cádiz": "ES", "valladolid": "ES",
    "inter": "IT", "milan": "IT", "juventus": "IT",
    "atalanta": "IT", "bologna": "IT", "roma": "IT",
    "lazio": "IT", "fiorentina": "IT", "napoli": "IT",
    "torino": "IT", "monza": "IT", "genoa": "IT",
    "udinese": "IT", "como": "IT", "parma": "IT", "venezia": "IT",
    "leverkusen": "DE", "stuttgart": "DE", "bayern": "DE",
    "leipzig": "DE", "dortmund": "DE", "frankfurt": "DE",
    "hoffenheim": "DE", "freiburg": "DE", "bremen": "DE",
    "wolfsburg": "DE", "gladbach": "DE", "augsburg": "DE",
    "mainz": "DE", "union berlin": "DE", "bochum": "DE",
    "psg": "FR", "monaco": "FR", "brest": "FR", "lille": "FR",
    "nice": "FR", "lyon": "FR", "lens": "FR", "marseille": "FR",
    "rennes": "FR", "reims": "FR", "nantes": "FR",
    "psv": "NL", "feyenoord": "NL", "ajax": "NL",
    "twente": "NL", "az alkmaar": "NL",
    "sporting": "PT", "benfica": "PT", "porto": "PT", "braga": "PT",
    "galatasaray": "TR", "fenerbahce": "TR",
    "besiktas": "TR", "trabzonspor": "TR",
    "celtic": "GB", "rangers": "GB",
    "anderlecht": "BE", "club brugge": "BE",
    "salzburg": "AT", "sturm graz": "AT",
    "shakhtar": "UA", "dynamo kyiv": "UA",
    "copenhagen": "DK", "midtjylland": "DK",
    "flamengo": "BR", "palmeiras": "BR",
    "atletico mineiro": "BR", "corinthians": "BR",
    "boca juniors": "AR", "river plate": "AR",
}


def _code_to_flag(code: str) -> str:
    code = code.upper().strip()
    if len(code) != 2:
        return "🏳️"
    offset = 0x1F1E6 - ord("A")
    return chr(ord(code[0]) + offset) + chr(ord(code[1]) + offset)


def get_flag_from_name(name: str) -> str:
    nl = name.lower()
    for kw, code in NATIONALITY_FLAGS.items():
        if kw in nl:
            return _code_to_flag(code)
    return "🏳️"


def validate_flag(flag: str, fallback_name: str) -> str:
    if not flag:
        return get_flag_from_name(fallback_name)
    BAD = {"🏳️", "🏁", "🚩", "🏳", ""}
    return (
        get_flag_from_name(fallback_name)
        if flag.strip() in BAD
        else flag.strip()
    )

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
            logger.warning("Cache load (%s): %s", filepath.name, e)
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
            logger.warning("Cache save (%s): %s", filepath.name, e)

    @staticmethod
    def is_valid(cache: dict, key: str, ttl_hours: float) -> bool:
        entry = cache.get(key)
        if not isinstance(entry, dict) or "timestamp" not in entry:
            return False
        try:
            ct = datetime.fromisoformat(entry["timestamp"])
            return datetime.now(timezone.utc) - ct < timedelta(hours=ttl_hours)
        except Exception:
            return False

    @staticmethod
    def set(cache: dict, key: str, value) -> dict:
        cache[key] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data":      value,
        }
        return cache

    @staticmethod
    def get(cache: dict, key: str):
        return cache.get(key, {}).get("data")

# =========================================================
# 6. DAILY CACHE
# =========================================================
class DailyCache:
    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def is_fresh(filepath: Path) -> bool:
        try:
            if not filepath.exists():
                return False
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return data.get("date") == DailyCache._today()
        except Exception:
            return False

    @staticmethod
    def save(filepath: Path, data) -> None:
        payload = {
            "date":     DailyCache._today(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "data":     data,
        }
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            tmp = filepath.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(filepath)
            logger.info("DailyCache saved → %s", filepath.name)
        except Exception as e:
            logger.warning("DailyCache save error (%s): %s", filepath.name, e)

    @staticmethod
    def load(filepath: Path):
        try:
            if not filepath.exists():
                return None
            payload = json.loads(filepath.read_text(encoding="utf-8"))
            if payload.get("date") != DailyCache._today():
                logger.info("DailyCache expired → %s", filepath.name)
                return None
            logger.info(
                "DailyCache HIT → %s (saved=%s)",
                filepath.name, payload.get("saved_at", "?")[:16],
            )
            return payload["data"]
        except Exception as e:
            logger.warning("DailyCache load error (%s): %s", filepath.name, e)
            return None

# =========================================================
# 7. PERFORMANCE TRACKER (رایگان — بدون API)
# بر اساس نتایج واقعی: win rate، ROI، CLV
# =========================================================
class PerformanceTracker:
    """
    ردیابی عملکرد سیگنال‌ها بدون هیچ API اضافه‌ای.
    CLV = Closing Line Value (مهم‌ترین KPI برای tipster)
    """

    def __init__(self) -> None:
        self._data = CacheManager.load(CFG.PERFORMANCE_FILE)
        if "bets" not in self._data:
            self._data["bets"] = []
        if "summary" not in self._data:
            self._data["summary"] = {}

    def record_bet(
        self,
        home:      str,
        away:      str,
        pick:      str,
        market:    str,
        our_odds:  float,
        ev:        float,
        conf:      int,
        sport_key: str,
    ) -> None:
        self._data["bets"].append({
            "id":          hashlib.md5(
                f"{home}|{away}|{market}|{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8],
            "home":        home,
            "away":        away,
            "pick":        pick,
            "market":      market,
            "our_odds":    our_odds,
            "ev_at_bet":   round(ev * 100, 2),
            "confidence":  conf,
            "sport_key":   sport_key,
            "bet_time":    datetime.now(timezone.utc).isoformat(),
            "result":      None,
            "won":         None,
            "closing_odds": None,
            "clv":         None,
            "profit":      None,
        })
        self._save()

    def record_result(
        self,
        home:         str,
        away:         str,
        market:       str,
        won:          Optional[bool],
        result_score: str,
        closing_odds: Optional[float] = None,
    ) -> None:
        for bet in self._data["bets"]:
            if (
                bet["home"].lower() == home.lower()
                and bet["away"].lower() == away.lower()
                and bet["market"] == market
                and bet["result"] is None
            ):
                bet["result"]  = result_score
                bet["won"]     = won
                bet["profit"]  = (
                    round(bet["our_odds"] - 1, 3) if won
                    else (-1.0 if won is False else 0.0)
                )
                if closing_odds and closing_odds > 1.0:
                    bet["closing_odds"] = closing_odds
                    bet["clv"] = round(
                        (bet["our_odds"] / closing_odds - 1) * 100, 2
                    )
                break
        self._update_summary()
        self._save()

    def _update_summary(self) -> None:
        completed = [
            b for b in self._data["bets"]
            if b["won"] is not None
        ]
        if not completed:
            return

        total  = len(completed)
        wins   = sum(1 for b in completed if b["won"])
        profit = sum(b.get("profit", 0) or 0 for b in completed)

        clv_bets = [b for b in completed if b.get("clv") is not None]
        avg_clv  = (
            round(sum(b["clv"] for b in clv_bets) / len(clv_bets), 2)
            if clv_bets else None
        )

        self._data["summary"] = {
            "total_bets":    total,
            "wins":          wins,
            "losses":        total - wins,
            "win_rate":      round(wins / total * 100, 1) if total else 0,
            "total_profit":  round(profit, 3),
            "roi":           round(profit / total * 100, 2) if total else 0,
            "avg_clv":       avg_clv,
            "clv_bets":      len(clv_bets),
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "Performance: %dW/%dL WR=%.0f%% ROI=%.1f%% CLV=%s%%",
            wins, total - wins,
            self._data["summary"]["win_rate"],
            self._data["summary"]["roi"],
            avg_clv if avg_clv is not None else "N/A",
        )

    def _save(self) -> None:
        CacheManager.save(CFG.PERFORMANCE_FILE, self._data)

    def get_summary(self) -> dict:
        return self._data.get("summary", {})

    def get_recent_form(self, n: int = 10) -> str:
        """آخرین N نتیجه به صورت W/L/P"""
        completed = [
            b for b in self._data["bets"]
            if b["won"] is not None
        ][-n:]
        return "".join(
            "W" if b["won"] else "L"
            for b in completed
        ) or "—"

    def format_summary_message(self) -> str:
        s = self.get_summary()
        if not s:
            return ""
        form = self.get_recent_form(10)
        clv_str = (
            f"{s['avg_clv']:+.1f}%"
            if s.get("avg_clv") is not None
            else "N/A"
        )
        return (
            f"📈 <b>Performance Stats</b>\n"
            f"├ Record: {s.get('wins',0)}W / {s.get('losses',0)}L\n"
            f"├ Win Rate: {s.get('win_rate',0):.1f}%\n"
            f"├ ROI: {s.get('roi',0):+.1f}%\n"
            f"├ Avg CLV: {clv_str}\n"
            f"└ Last 10: {form}"
        )

# =========================================================
# 8. ODDS API KEY MANAGER
# =========================================================
class OddsKeyManager:
    STATUS_OK        = "ok"
    STATUS_INVALID   = "invalid"
    STATUS_EXHAUSTED = "exhausted"
    STATUS_UNKNOWN   = "unknown"

    def __init__(self, keys: list[str]) -> None:
        self.keys    = keys
        self._status = CacheManager.load(CFG.KEY_STATUS_FILE)
        self._rr_idx = 0
        self._init_keys()
        self._log_all()

    @staticmethod
    def _kid(key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def _save(self) -> None:
        CacheManager.save(CFG.KEY_STATUS_FILE, self._status)

    def _init_keys(self) -> None:
        changed = False
        for k in self.keys:
            kid = self._kid(k)
            if kid not in self._status:
                self._status[kid] = {
                    "prefix":     k[:8] + "…",
                    "status":     self.STATUS_UNKNOWN,
                    "remaining":  None,
                    "used":       None,
                    "last_used":  None,
                    "last_error": None,
                }
                changed = True
        if changed:
            self._save()

    def _log_all(self) -> None:
        logger.info("OddsKeyManager status:")
        for k in self.keys:
            st = self._status.get(self._kid(k), {})
            logger.info(
                "  key=%-12s status=%-10s remaining=%-5s used=%s",
                st.get("prefix", "?"), st.get("status", "?"),
                st.get("remaining", "?"), st.get("used", "?"),
            )

    def _is_usable(self, key: str) -> bool:
        kid    = self._kid(key)
        st     = self._status.get(kid, {})
        status = st.get("status", self.STATUS_UNKNOWN)
        if status == self.STATUS_INVALID:
            return False
        if status == self.STATUS_EXHAUSTED:
            last = st.get("last_used", "")
            try:
                lt = datetime.fromisoformat(last)
                if datetime.now(timezone.utc).date() > lt.date():
                    self._status[kid]["status"]    = self.STATUS_UNKNOWN
                    self._status[kid]["remaining"] = None
                    self._save()
                    return True
                return False
            except Exception:
                return False
        return True

    def get_best_key(self) -> Optional[str]:
        usable = [k for k in self.keys if self._is_usable(k)]
        if not usable:
            logger.warning("OddsKeyManager: no usable keys!")
            return None
        key = usable[self._rr_idx % len(usable)]
        self._rr_idx += 1
        kid = self._kid(key)
        logger.debug(
            "OddsKeyManager: key=%s status=%s remaining=%s",
            self._status[kid].get("prefix", "?"),
            self._status[kid].get("status",    "?"),
            self._status[kid].get("remaining", "?"),
        )
        return key

    def mark_success(self, key: str, remaining: str, used: str) -> None:
        kid = self._kid(key)
        if kid not in self._status:
            return
        try:
            rem_int: Optional[int] = int(remaining)
        except (ValueError, TypeError):
            rem_int = None
        self._status[kid].update({
            "status": (
                self.STATUS_EXHAUSTED
                if rem_int is not None and rem_int <= 0
                else self.STATUS_OK
            ),
            "remaining":  rem_int,
            "used":       used,
            "last_used":  datetime.now(timezone.utc).isoformat(),
            "last_error": None,
        })
        self._save()

    def mark_invalid(self, key: str, reason: str) -> None:
        kid = self._kid(key)
        if kid not in self._status:
            return
        self._status[kid].update({
            "status":     self.STATUS_INVALID,
            "last_error": reason,
            "last_used":  datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        logger.error(
            "Key %s marked INVALID: %s",
            self._status[kid].get("prefix", "?"), reason,
        )

    def mark_exhausted(self, key: str) -> None:
        kid = self._kid(key)
        if kid not in self._status:
            return
        self._status[kid].update({
            "status":    self.STATUS_EXHAUSTED,
            "remaining": 0,
            "last_used": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        logger.warning(
            "Key %s marked EXHAUSTED",
            self._status[kid].get("prefix", "?"),
        )

    def get_summary(self) -> str:
        parts = []
        for k in self.keys:
            st = self._status.get(self._kid(k), {})
            parts.append(
                f"{st.get('prefix','?')}:{st.get('status','?')}"
                f"/rem={st.get('remaining','?')}"
            )
        return " | ".join(parts)

    async def validate_all_async(
        self, session: aiohttp.ClientSession
    ) -> None:
        log_section("VALIDATING ALL ODDS API KEYS")
        for key in self.keys:
            kid    = self._kid(key)
            prefix = self._status[kid].get("prefix", "?")
            if self._status[kid].get("status") == self.STATUS_INVALID:
                logger.info("Key %s: already INVALID — skip", prefix)
                continue
            try:
                async with session.get(
                    "https://api.the-odds-api.com/v4/sports",
                    params={"apiKey": key},
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as res:
                    remaining = res.headers.get("x-requests-remaining", "?")
                    used      = res.headers.get("x-requests-used",      "?")
                    if res.status == 200:
                        body = await res.json(content_type=None)
                        self.mark_success(key, remaining, used)
                        logger.info(
                            "Key %s ✅ VALID | sports=%d remaining=%s used=%s",
                            prefix, len(body), remaining, used,
                        )
                    elif res.status == 401:
                        self.mark_invalid(key, "HTTP 401 Unauthorized")
                    elif res.status == 422:
                        self.mark_invalid(key, "HTTP 422 Unprocessable")
                    elif res.status == 429:
                        self.mark_exhausted(key)
                    else:
                        logger.warning("Key %s: HTTP %d", prefix, res.status)
            except asyncio.TimeoutError:
                logger.warning("Key %s: timeout", prefix)
            except Exception as e:
                logger.warning("Key %s: error: %s", prefix, e)
        logger.info("Validation summary: %s", self.get_summary())

# =========================================================
# 9. RAPID API KEY MANAGER
# =========================================================
class RapidKeyManager:
    def __init__(self, keys: list[str]) -> None:
        self.keys           = keys
        self._current_idx   = 0
        self._blocked_until: dict[int, Optional[datetime]] = {
            i: None for i in range(len(keys))
        }
        self._req_counts: dict[int, int] = {
            i: 0 for i in range(len(keys))
        }
        if keys:
            logger.info(
                "RapidKeyManager: %d key(s) | primary=%s…",
                len(keys), keys[0][:8],
            )
        else:
            logger.warning("RapidKeyManager: NO KEYS!")

    def _is_available(self, idx: int) -> bool:
        bu = self._blocked_until.get(idx)
        if bu is None:
            return True
        if datetime.now(timezone.utc) > bu:
            self._blocked_until[idx] = None
            logger.info("RapidAPI key#%d unblocked", idx + 1)
            return True
        return False

    def get_current_key(self) -> Optional[str]:
        for offset in range(len(self.keys)):
            idx = (self._current_idx + offset) % len(self.keys)
            if self._is_available(idx):
                if offset > 0:
                    logger.info("RapidAPI switching to key#%d", idx + 1)
                    self._current_idx = idx
                return self.keys[idx]
        logger.warning("RapidAPI: all keys blocked!")
        return None

    def get_headers(self) -> Optional[dict]:
        key = self.get_current_key()
        if not key:
            return None
        return {
            "x-rapidapi-key":  key,
            "x-rapidapi-host": "sofascore6.p.rapidapi.com",
        }

    def mark_rate_limited(self) -> None:
        idx = self._current_idx
        self._blocked_until[idx] = (
            datetime.now(timezone.utc)
            + timedelta(minutes=CFG.RAPID_RATE_LIMIT_PAUSE)
        )
        logger.warning(
            "RapidAPI key#%d rate-limited → paused %dm",
            idx + 1, CFG.RAPID_RATE_LIMIT_PAUSE,
        )
        next_idx = (idx + 1) % max(len(self.keys), 1)
        if next_idx != idx:
            self._current_idx = next_idx
            logger.info("RapidAPI switched to key#%d", next_idx + 1)

    def mark_request(self) -> None:
        self._req_counts[self._current_idx] = (
            self._req_counts.get(self._current_idx, 0) + 1
        )

    def get_stats(self) -> str:
        parts = []
        for i, k in enumerate(self.keys):
            bu     = self._blocked_until.get(i)
            req    = self._req_counts.get(i, 0)
            status = (
                "blocked"
                if bu and datetime.now(timezone.utc) < bu
                else "ok"
            )
            parts.append(f"key#{i+1}({k[:6]}…):{status}/req={req}")
        return " | ".join(parts)

# =========================================================
# 10. SENT HISTORY
# =========================================================
class SentHistory:
    def __init__(self) -> None:
        self._lock   = asyncio.Lock()
        self.history = CacheManager.load(CFG.HISTORY_FILE)
        self._cleanup_old()

    def _cleanup_old(self) -> None:
        now    = datetime.now(timezone.utc)
        to_del = []
        for k, v in self.history.items():
            try:
                sa = v.get("sent_at", "2000-01-01T00:00:00+00:00")
                if now - datetime.fromisoformat(sa) > timedelta(
                    hours=CFG.TTL_SENT_HISTORY
                ):
                    to_del.append(k)
            except Exception:
                to_del.append(k)
        for k in to_del:
            del self.history[k]
        if to_del:
            logger.debug("SentHistory: cleaned %d old entries", len(to_del))

    @staticmethod
    def _key(home: str, away: str, market: str) -> str:
        return hashlib.md5(
            f"{home.lower()}|{away.lower()}|{market.lower()}".encode()
        ).hexdigest()

    def was_sent(self, home: str, away: str, market: str) -> bool:
        return self._key(home, away, market) in self.history

    async def mark_sent_async(
        self,
        home:          str,
        away:          str,
        pick:          str,
        market:        str,
        odds:          float,
        commence_time: str,
        sport_key:     str,
        sport_title:   str,
    ) -> None:
        k = self._key(home, away, market)
        async with self._lock:
            self.history[k] = {
                "home":           home,
                "away":           away,
                "pick":           pick,
                "market":         market,
                "odds":           odds,
                "commence_time":  commence_time,
                "sport_key":      sport_key,
                "sport_title":    sport_title,
                "sent_at":        datetime.now(timezone.utc).isoformat(),
                "result_checked": False,
            }
            CacheManager.save(CFG.HISTORY_FILE, self.history)

    def get_pending_results(self) -> list:
        now = datetime.now(timezone.utc)
        out = []
        for k, v in self.history.items():
            if v.get("result_checked"):
                continue
            try:
                mt = datetime.fromisoformat(
                    v.get("commence_time", "").replace("Z", "+00:00")
                )
                if (now - mt).total_seconds() / 3600 >= CFG.RESULT_CHECK_HOURS:
                    out.append((k, v))
            except Exception:
                continue
        return out

    async def mark_result_checked_async(
        self, key: str, result: str, won: Optional[bool]
    ) -> None:
        async with self._lock:
            if key in self.history:
                self.history[key].update({
                    "result_checked": True,
                    "result":         result,
                    "won":            won,
                })
                CacheManager.save(CFG.HISTORY_FILE, self.history)

# =========================================================
# 11. ELO SYSTEM
# =========================================================
def normalize_player_name(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return " ".join(parts[1:]).lower().strip()
    return name.lower().strip()


class ELOSystem:
    def __init__(self, sport: str = "football") -> None:
        self.sport = sport
        self.k = (
            CFG.ELO_K_FACTOR_FOOTBALL
            if sport == "football"
            else CFG.ELO_K_FACTOR_TENNIS
        )
        self.ratings:     dict = {}
        self.match_count: dict = {}
        fp = (
            CFG.ELO_FOOTBALL_FILE
            if sport == "football"
            else CFG.ELO_TENNIS_FILE
        )
        self._load(fp)

    def _load(self, fp: Path) -> None:
        data = CacheManager.load(fp)
        if data:
            self.ratings     = data.get("ratings",     {})
            self.match_count = data.get("match_count", {})
            log_check(
                f"ELO {self.sport} loaded",
                f"{len(self.ratings)} entities",
                warn_if_none=False,
            )
        else:
            logger.info("ELO %s: no data (bootstrap needed)", self.sport)

    def save(self) -> None:
        fp = (
            CFG.ELO_FOOTBALL_FILE
            if self.sport == "football"
            else CFG.ELO_TENNIS_FILE
        )
        CacheManager.save(fp, {
            "ratings":     self.ratings,
            "match_count": self.match_count,
            "updated_at":  datetime.now(timezone.utc).isoformat(),
        })

    def get_rating(self, name: str) -> float:
        key = name.lower().strip()
        if key in self.ratings:
            return self.ratings[key]
        if self.sport == "tennis":
            short = normalize_player_name(name)
            if short in self.ratings:
                return self.ratings[short]
        return CFG.ELO_DEFAULT

    def get_match_count(self, name: str) -> int:
        key = name.lower().strip()
        if key in self.match_count:
            return self.match_count[key]
        if self.sport == "tennis":
            short = normalize_player_name(name)
            if short in self.match_count:
                return self.match_count[short]
        return 0

    def expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def update(
        self, a: str, b: str, sa: float, is_home_a: bool = False
    ) -> None:
        ka = a.lower().strip()
        kb = b.lower().strip()
        ra = self.get_rating(a)
        rb = self.get_rating(b)
        ea = self.expected_score(
            ra + (CFG.ELO_HOME_ADVANTAGE if is_home_a else 0), rb
        )
        na   = self.get_match_count(a)
        nb   = self.get_match_count(b)
        kf_a = self.k * (1.5 if na < 20 else 1.0)
        kf_b = self.k * (1.5 if nb < 20 else 1.0)
        self.ratings[ka]     = ra + kf_a * (sa - ea)
        self.ratings[kb]     = rb + kf_b * ((1 - sa) - (1 - ea))
        self.match_count[ka] = na + 1
        self.match_count[kb] = nb + 1

    def predict(
        self, home: str, away: str, apply_home: bool = True
    ) -> dict:
        ra = self.get_rating(home)
        rb = self.get_rating(away)
        hp = self.expected_score(
            ra + (CFG.ELO_HOME_ADVANTAGE if apply_home else 0), rb
        )
        ap = 1.0 - hp
        dp = 0.0
        if self.sport == "football":
            df  = 0.22
            hp2 = hp * (1 - df)
            ap2 = ap * (1 - df)
            dp2 = df
            t   = hp2 + ap2 + dp2
            hp, ap, dp = hp2 / t, ap2 / t, dp2 / t
        hm = self.get_match_count(home)
        am = self.get_match_count(away)
        return {
            "home_prob":    round(hp, 4),
            "away_prob":    round(ap, 4),
            "draw_prob":    round(dp, 4),
            "home_elo":     round(ra, 1),
            "away_elo":     round(rb, 1),
            "elo_diff":     round(ra - rb, 1),
            "home_matches": hm,
            "away_matches": am,
        }

# =========================================================
# 12. BOOTSTRAP
# =========================================================
class DataBootstrap:
    FOOTBALL_LEAGUES = [
        ("E0",  "England PL"),    ("E1",  "England Championship"),
        ("SP1", "La Liga"),       ("D1",  "Bundesliga"),
        ("I1",  "Serie A"),       ("F1",  "Ligue 1"),
        ("N1",  "Eredivisie"),    ("P1",  "Liga Portugal"),
        ("B1",  "Belgium"),       ("T1",  "Turkey"),
        ("SC0", "Scotland PL"),   ("SP2", "La Liga 2"),
        ("D2",  "Bundesliga 2"),  ("I2",  "Serie B"),
    ]
    TENNIS_FILES = [
        "atp_matches_2022.csv", "atp_matches_2023.csv", "atp_matches_2024.csv",
        "wta_matches_2022.csv", "wta_matches_2023.csv", "wta_matches_2024.csv",
    ]

    def __init__(self) -> None:
        self.elo_football = ELOSystem("football")
        self.elo_tennis   = ELOSystem("tennis")

    def should_run(self) -> bool:
        if FORCE_BOOTSTRAP:
            return True
        if not CFG.BOOTSTRAP_FLAG.exists():
            return True
        try:
            ft = datetime.fromisoformat(
                CFG.BOOTSTRAP_FLAG.read_text().strip()
            )
            return (datetime.now(timezone.utc) - ft).days >= 7
        except Exception:
            return True

    def run(self) -> None:
        log_section("BOOTSTRAP — BUILDING ELO MODELS")
        self._build_football_elo()
        self._build_tennis_elo()
        self.elo_football.save()
        self.elo_tennis.save()
        CFG.BOOTSTRAP_FLAG.write_text(
            datetime.now(timezone.utc).isoformat()
        )
        log_check("Football teams", len(self.elo_football.ratings))
        log_check("Tennis players", len(self.elo_tennis.ratings))

    def _download_csv(self, url: str) -> Optional[pd.DataFrame]:
        try:
            res = requests.get(url, timeout=30)
            if res.status_code == 200:
                try:
                    return pd.read_csv(StringIO(res.text))
                except Exception:
                    return pd.read_csv(StringIO(res.text), encoding="latin-1")
        except Exception as e:
            logger.debug("CSV download error %s: %s", url, e)
        return None

    def _build_football_elo(self) -> None:
        log_section("Building Football ELO")
        total = 0
        for code, name in self.FOOTBALL_LEAGUES:
            cnt = 0
            for season in ["2223", "2324", "2425"]:
                url = (
                    f"https://www.football-data.co.uk"
                    f"/mmz4281/{season}/{code}.csv"
                )
                df = self._download_csv(url)
                if df is None or df.empty:
                    continue
                if not {"HomeTeam", "AwayTeam", "FTR"}.issubset(df.columns):
                    continue
                df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
                for row in df[["HomeTeam", "AwayTeam", "FTR"]].itertuples(
                    index=False
                ):
                    try:
                        ftr = str(row.FTR).strip().upper()
                        sc  = (
                            1.0 if ftr == "H"
                            else (0.0 if ftr == "A" else 0.5)
                        )
                        self.elo_football.update(
                            str(row.HomeTeam).strip(),
                            str(row.AwayTeam).strip(),
                            sc, is_home_a=True,
                        )
                        cnt += 1
                    except Exception:
                        continue
                del df
                time.sleep(0.1)
            total += cnt
            if cnt:
                logger.info("ELO football %-22s → %d matches", name, cnt)
        log_check("Football ELO total matches", total)

    def _build_tennis_elo(self) -> None:
        log_section("Building Tennis ELO")
        total = 0
        for fn in self.TENNIS_FILES:
            tour = "atp" if fn.startswith("atp") else "wta"
            url  = (
                f"https://raw.githubusercontent.com"
                f"/JeffSackmann/tennis_{tour}/master/{fn}"
            )
            df = self._download_csv(url)
            if df is None or df.empty:
                continue
            if not {"winner_name", "loser_name"}.issubset(df.columns):
                continue
            df  = df.dropna(subset=["winner_name", "loser_name"])
            cnt = 0
            for row in df[["winner_name", "loser_name"]].itertuples(
                index=False
            ):
                try:
                    winner = str(row.winner_name).strip()
                    loser  = str(row.loser_name).strip()
                    self.elo_tennis.update(winner, loser, 1.0)
                    wk = normalize_player_name(winner)
                    lk = normalize_player_name(loser)
                    if wk and wk not in self.elo_tennis.ratings:
                        self.elo_tennis.ratings[wk] = (
                            self.elo_tennis.get_rating(winner)
                        )
                    if lk and lk not in self.elo_tennis.ratings:
                        self.elo_tennis.ratings[lk] = (
                            self.elo_tennis.get_rating(loser)
                        )
                    cnt += 1
                except Exception:
                    continue
            total += cnt
            del df
            if cnt:
                logger.info("ELO tennis %-28s → %d matches", fn, cnt)
            time.sleep(0.15)
        log_check("Tennis ELO total matches", total)

# =========================================================
# 13. UTILS
# =========================================================
def retry_sync(max_retries: int = 3, delay: float = 2.0, backoff: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cd = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    st = (
                        e.response.status_code
                        if e.response is not None else 0
                    )
                    if st == 429:
                        wait = int(
                            e.response.headers.get("Retry-After", cd * 3)
                        )
                        logger.warning("429 %s — sleep %ds", func.__name__, wait)
                        time.sleep(wait)
                    elif st in [401, 403]:
                        logger.error("Auth %d in %s", st, func.__name__)
                        return None
                    else:
                        logger.warning(
                            "HTTP %d in %s (attempt %d/%d)",
                            st, func.__name__, attempt + 1, max_retries,
                        )
                        if attempt == max_retries - 1:
                            return None
                except (
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException,
                ) as e:
                    logger.warning(
                        "%s in %s (attempt %d/%d): %s",
                        type(e).__name__, func.__name__,
                        attempt + 1, max_retries, e,
                    )
                    if attempt == max_retries - 1:
                        return None
                time.sleep(cd)
                cd *= backoff
            return None
        return wrapper
    return decorator


def _find_json_objects(text: str) -> list[str]:
    results = []
    depth   = 0
    start   = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                results.append(text[start : i + 1])
    return results


def robust_json_extractor(raw: str) -> Optional[dict]:
    if not raw:
        return None
    clean = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE)
    clean = re.sub(r"<think>[\s\S]*", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"```(?:json)?", "", clean).strip()
    clean = clean.replace("```", "").strip()
    try:
        result = json.loads(clean)
        if isinstance(result, dict) and result:
            return result
    except json.JSONDecodeError:
        pass
    candidates = _find_json_objects(clean)
    for candidate in reversed(candidates):
        try:
            r = json.loads(candidate)
            if isinstance(r, dict) and r:
                return r
        except Exception:
            continue
    return None


def clean_team_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(name)).strip()


def normalize_sport_key(sport_title: str) -> str:
    tl = sport_title.lower()
    if any(k in tl for k in [
        "tennis", "atp", "wta", "wimbledon",
        "roland garros", "us open", "australian open", "french open",
    ]):
        return "tennis"
    if any(k in tl for k in [
        "soccer", "football", "premier", "liga", "bundesliga",
        "serie", "série", "ligue", "champions", "europa",
        "eredivisie", "fa cup", "copa del rey", "brasileirao",
        "süper lig", "primeira", "scottish", "mls",
    ]):
        return "football"
    return "other"


def get_countdown_str(ct: str, now: datetime) -> str:
    try:
        mt   = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        diff = (mt - now).total_seconds()
        if diff <= 0:
            return "⚡ Starting now"
        m = int(diff / 60)
        return f"{m // 60}h {m % 60:02d}m" if m > 60 else f"{m}m"
    except Exception:
        return "N/A"


def _flex_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_clean = a.lower().strip()
    b_clean = b.lower().strip()
    if a_clean == b_clean:
        return True
    if a_clean in b_clean or b_clean in a_clean:
        shorter = min(len(a_clean), len(b_clean))
        longer  = max(len(a_clean), len(b_clean))
        if shorter < 4 and longer > shorter * 2:
            return False
        return True
    stopwords = {
        "fc", "cf", "sc", "ac", "rc", "fk", "sk", "bk",
        "united", "city", "town", "athletic", "sport",
        "sporting", "club", "the", "de", "1.",
    }
    wa = {
        w for w in re.split(r"[\s\-_]+", a_clean)
        if w not in stopwords and len(w) > 2
    }
    wb = {
        w for w in re.split(r"[\s\-_]+", b_clean)
        if w not in stopwords and len(w) > 2
    }
    if not wa or not wb:
        return False
    common = wa & wb
    return len(common) >= max(1, min(len(wa), len(wb)) // 2)


def get_display_pick(raw: str, market: str, home: str, away: str) -> str:
    pl = raw.lower().strip()
    if market == "h2h":
        if "draw" in pl or "tie" in pl:
            return "Match to end in a Draw (X)"
        if _flex_match(home, raw):
            return f"{home} to Win (1)"
        if _flex_match(away, raw):
            return f"{away} to Win (2)"
        return f"{raw} to Win"
    if market == "totals":
        m = re.match(r"(over|under)\s*([\d.]+)", pl)
        if m:
            action = "Over" if m.group(1) == "over" else "Under"
            return f"{action} {m.group(2)} Total Goals"
        return raw.title()
    return raw.title()


def get_market_label(mk: str) -> str:
    return CFG.MARKET_DISPLAY.get(mk, mk.replace("_", " ").title())
    
# =========================================================
# 14. ODDS API — DAILY FETCH
# =========================================================
async def _fetch_one_sport(
    sport_key: str,
    api_key:   str,
    km:        OddsKeyManager,
    session:   aiohttp.ClientSession,
) -> tuple[list, int, str, str]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey":     api_key,
        "regions":    CFG.ODDS_API_REGIONS,
        "markets":    CFG.ODDS_API_MARKETS,
        "oddsFormat": CFG.ODDS_API_ODDS_FORMAT,
        "dateFormat": CFG.ODDS_API_DATE_FORMAT,
    }
    try:
        async with session.get(
            url, params=params,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as res:
            remaining = res.headers.get("x-requests-remaining", "?")
            used      = res.headers.get("x-requests-used",      "?")
            if res.status == 200:
                data = await res.json(content_type=None)
                km.mark_success(api_key, remaining, used)
                logger.info(
                    "OddsAPI ✅ %-35s | events=%d rem=%s",
                    sport_key, len(data), remaining,
                )
                return data, 200, remaining, used
            logger.warning(
                "OddsAPI %-35s | HTTP %d | rem=%s",
                sport_key, res.status, remaining,
            )
            return [], res.status, remaining, used
    except asyncio.TimeoutError:
        logger.warning("OddsAPI timeout: %s", sport_key)
        return [], 0, "?", "?"
    except Exception as e:
        logger.debug("OddsAPI error (%s): %s", sport_key, e)
        return [], 0, "?", "?"


async def fetch_all_odds_daily(
    now_utc: datetime,
    km:      OddsKeyManager,
    session: aiohttp.ClientSession,
) -> list:
    log_section("ODDS API — DAILY FETCH")

    cached = DailyCache.load(CFG.DAILY_ODDS_CACHE_FILE)
    if cached is not None:
        end_win  = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
        filtered = _filter_by_window(cached, now_utc, end_win)
        logger.info(
            "OddsAPI DailyCache HIT: total=%d | in_window=%d",
            len(cached), len(filtered),
        )
        return filtered

    api_key = km.get_best_key()
    if not api_key:
        logger.critical("All Odds API keys exhausted/invalid!")
        return []

    active_sports = await _get_active_sports(api_key, session)
    if not active_sports:
        logger.error("Failed to get active sports list!")
        return []

    target_sports = _select_priority_sports(active_sports)
    logger.info(
        "Sports: total=%d | selected=%d (max=%d)",
        len(active_sports), len(target_sports), CFG.MAX_SPORTS_PER_DAY,
    )

    all_events: dict[str, dict] = {}
    credits_used = 0

    for sport_key in target_sports:
        current_key = km.get_best_key()
        if not current_key:
            logger.warning("All keys exhausted after %d sports!", credits_used)
            break

        data, status, remaining, _ = await _fetch_one_sport(
            sport_key, current_key, km, session
        )
        credits_used += 1

        if status == 429:
            km.mark_exhausted(current_key)
            next_key = km.get_best_key()
            if next_key and next_key != current_key:
                data, status, remaining, _ = await _fetch_one_sport(
                    sport_key, next_key, km, session
                )
                credits_used += 1
            if status != 200:
                continue
        elif status in [401, 403, 422]:
            km.mark_invalid(current_key, f"HTTP {status}")
            continue
        elif status != 200:
            continue

        for e in data:
            _merge_event(all_events, e, sport_key)

        await asyncio.sleep(0.3)

    result_list = list(all_events.values())
    logger.info(
        "OddsAPI fetch complete: %d events | credits_used=%d | key_status=%s",
        len(result_list), credits_used, km.get_summary(),
    )

    if result_list:
        DailyCache.save(CFG.DAILY_ODDS_CACHE_FILE, result_list)

    end_win  = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    filtered = _filter_by_window(result_list, now_utc, end_win)
    logger.info(
        "Events in window (next %.1fh): %d / %d",
        CFG.MATCH_WINDOW_HOURS, len(filtered), len(result_list),
    )
    return filtered


async def _get_active_sports(
    api_key: str, session: aiohttp.ClientSession
) -> list:
    try:
        async with session.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": api_key},
            timeout=aiohttp.ClientTimeout(total=12),
        ) as res:
            if res.status == 200:
                sports = await res.json(content_type=None)
                active = [s for s in sports if s.get("active", False)]
                logger.info(
                    "Active sports: %d / %d total", len(active), len(sports)
                )
                return active
            logger.error("Sports list HTTP %d", res.status)
            return []
    except Exception as e:
        logger.error("Error getting sports list: %s", e)
        return []


def _select_priority_sports(sports: list) -> list[str]:
    excluded_kw = CFG.EXCLUDED_SPORT_KEYWORDS
    priority_kw = CFG.PRIORITY_SPORT_KEYWORDS
    excluded = []
    priority = []
    others   = []
    for s in sports:
        sk    = s.get("key",   "").lower()
        group = s.get("group", "").lower()
        if any(kw in sk for kw in excluded_kw) or any(
            kw in group for kw in excluded_kw
        ):
            excluded.append(sk)
            continue
        if any(kw in sk or kw in group for kw in priority_kw):
            priority.append(s["key"])
        else:
            others.append(s["key"])
    logger.info(
        "Sport selection: priority=%d others=%d excluded=%d",
        len(priority), len(others), len(excluded),
    )
    selected = priority + others
    return selected[: CFG.MAX_SPORTS_PER_DAY]


def _merge_event(all_events: dict, e: dict, sport_key: str) -> None:
    eid = e.get("id")
    if not eid:
        return
    if eid not in all_events:
        all_events[eid] = {
            "id":            eid,
            "home_team":     e.get("home_team",     ""),
            "away_team":     e.get("away_team",     ""),
            "sport_title":   e.get("sport_title",   ""),
            "sport_key":     e.get("sport_key",     sport_key),
            "commence_time": e.get("commence_time", ""),
            "_markets_data": {},
            "_source":       "odds_api",
        }
    md = all_events[eid]["_markets_data"]
    for bm in e.get("bookmakers", []):
        bm_title = bm.get("title", "")
        bm_key   = bm.get("key",   "")
        for m in bm.get("markets", []):
            mk = m.get("key", "")
            if mk not in CFG.VALID_MARKETS:
                continue
            if mk not in md:
                md[mk] = []
            md[mk].append({
                "bookmaker":     bm_title,
                "bookmaker_key": bm_key,
                "outcomes":      m.get("outcomes", []),
            })


def _filter_by_window(
    events: list, now_utc: datetime, end_win: datetime
) -> list:
    out = []
    for e in events:
        try:
            mt = datetime.fromisoformat(
                e.get("commence_time", "").replace("Z", "+00:00")
            )
            if now_utc <= mt <= end_win:
                out.append(e)
        except Exception:
            continue
    return out

# =========================================================
# 15. MATH ENGINE
# =========================================================
def calculate_combined_ev(
    markets_data:   dict,
    elo_prediction: Optional[dict],
    sport_key:      str,
    home_team:      str,
    away_team:      str,
    data_quality:   str = "none",
) -> list:
    best_per_market: dict = {}

    for market_key, market_data_list in markets_data.items():
        if market_key not in CFG.VALID_MARKETS:
            continue

        sharp_odds:     dict = {}
        best_odds:      dict = {}
        has_real_sharp: bool = False

        for entry in market_data_list:
            bk = entry.get("bookmaker_key", "")
            if bk in CFG.SHARP_BOOKMAKERS:
                has_real_sharp = True
            for o in entry.get("outcomes", []):
                base  = o.get("name",  "")
                point = o.get("point")
                name  = f"{base} {point}" if point is not None else base
                try:
                    price = float(o["price"])
                except (KeyError, TypeError, ValueError):
                    continue
                if price <= 1.0:
                    continue
                if bk in CFG.SHARP_BOOKMAKERS:
                    if (name not in sharp_odds
                            or price > sharp_odds[name]["price"]):
                        sharp_odds[name] = {
                            "price": price, "bookmaker": entry["bookmaker"],
                        }
                if name not in best_odds or price > best_odds[name]["price"]:
                    best_odds[name] = {
                        "price": price, "bookmaker": entry["bookmaker"],
                    }

        exp         = CFG.MARKET_EXPECTED_OUTCOMES.get(market_key, {"min": 2})
        valid_sharp = has_real_sharp and len(sharp_odds) >= exp["min"]
        baseline    = sharp_odds if valid_sharp else best_odds

        if not baseline or len(baseline) < exp["min"]:
            continue

        try:
            implied_sum = sum(1.0 / v["price"] for v in baseline.values())
        except ZeroDivisionError:
            continue

        if not (
            CFG.MIN_VALID_IMPLIED_SUM <= implied_sum <= CFG.MAX_VALID_IMPLIED_SUM
        ):
            logger.debug(
                "EV skip [%s]: implied_sum=%.3f out of range",
                market_key, implied_sum,
            )
            continue

        min_odds = (
            CFG.H2H_MIN_ODDS if market_key == "h2h" else CFG.TOTALS_MIN_ODDS
        )
        min_ev = (
            CFG.H2H_MIN_EV if market_key == "h2h" else CFG.TOTALS_MIN_EV
        )

        # =====================================================
        # DYNAMIC ELO WEIGHTING LOGIC
        # حالت بکاپ و پیش‌فرض (همان روش قبلی)
        # =====================================================
        w_sharp = 0.60
        w_elo   = 0.40
        
        if elo_prediction and market_key == "h2h":
            hm = elo_prediction.get("home_matches", 0)
            am = elo_prediction.get("away_matches", 0)
            min_m = min(hm, am)
            
            # 1. تنظیم بر اساس قابلیت اطمینان مدل (تعداد مسابقات)
            if min_m < 5:
                # دیتای ناکافی: اعتماد حداکثری به مارکت شارپ
                w_sharp, w_elo = 0.85, 0.15
            elif min_m < 10:
                # دیتای متوسط: کاهش جزئی تاثیر ELO
                w_sharp, w_elo = 0.70, 0.30
            elif min_m >= 30:
                # دیتای بسیار غنی: مدل بالغ شده و هم‌تراز با مارکت است
                w_sharp, w_elo = 0.50, 0.50

            # 2. تنظیم بر اساس کیفیت سایر داده‌ها (تایید متقاطع با SofaScore/FootballData)
            if data_quality == "high":
                # وقتی سایر آمارها هم مدل را تایید می‌کنند، وزن ELO را بالاتر می‌بریم
                w_elo += 0.05
                w_sharp -= 0.05
            elif data_quality == "none":
                # وقتی هیچ دیتای دیگری جز ELO نداریم، احتیاط کرده و به مارکت تکیه می‌کنیم
                w_elo -= 0.05
                w_sharp += 0.05
                
            # کلمپ کردن (Clamp) محدوده‌ها برای جلوگیری از اعداد غیرمنطقی
            w_sharp = max(0.50, min(0.95, w_sharp))
            w_elo   = round(1.0 - w_sharp, 2)
        # =====================================================

        best_opp = None
        for oname, sd in baseline.items():
            stp = (1.0 / sd["price"]) / implied_sum

            etp: Optional[float] = None
            if elo_prediction and market_key == "h2h":
                hm = elo_prediction.get("home_matches", 0)
                am = elo_prediction.get("away_matches", 0)
                nl = oname.lower()
                if hm >= 3 or am >= 3:
                    if "draw" in nl or "tie" in nl:
                        etp = elo_prediction.get("draw_prob")
                    elif _flex_match(home_team, oname):
                        etp = elo_prediction.get("home_prob")
                    elif _flex_match(away_team, oname):
                        etp = elo_prediction.get("away_prob")

            # اعمال وزن‌های داینامیک
            tp  = (w_sharp * stp) + (w_elo * etp) if etp is not None else stp
            
            bd  = best_odds.get(oname, {})
            bp  = float(bd.get("price", 0.0))
            bbk = bd.get("bookmaker", "Unknown")

            if bp <= 1.0:
                continue

            ev = (tp * bp) - 1.0

            if ev > CFG.MAX_REALISTIC_EV:
                logger.warning(
                    "EV rejected (too high=%.1f%%) for %s", ev * 100, oname,
                )
                continue

            if data_quality == "none" and ev > CFG.MAX_EV_WITHOUT_DATA:
                logger.warning(
                    "EV rejected (%.1f%% > %.1f%% max_no_data) for %s",
                    ev * 100, CFG.MAX_EV_WITHOUT_DATA * 100, oname,
                )
                continue

            if bp >= min_odds and ev > min_ev:
                opp = {
                    "pick":           oname,
                    "market":         market_key,
                    "market_label":   get_market_label(market_key),
                    "prob":           round(tp, 4),
                    "odds":           round(bp, 3),
                    "bookmaker":      bbk,
                    "ev":             round(ev, 4),
                    "edge_pct":       round(ev * 100, 2),
                    "has_sharp_line": valid_sharp,
                    "elo_used":       etp is not None,
                    "dyn_weight":     f"S:{w_sharp:.2f}/E:{w_elo:.2f}" if etp is not None else "N/A"
                }
                if best_opp is None or opp["ev"] > best_opp["ev"]:
                    best_opp = opp

        if best_opp:
            best_per_market[market_key] = best_opp
            logger.info(
                "EV ✅ [%-8s] pick='%s' ev=%.1f%% odds=%.2f "
                "bookie=%s elo=%s sharp=%s weights=%s",
                market_key, best_opp["pick"], best_opp["edge_pct"],
                best_opp["odds"], best_opp["bookmaker"],
                best_opp["elo_used"], valid_sharp, best_opp.get("dyn_weight", "N/A")
            )

    return sorted(
        best_per_market.values(), key=lambda x: x["ev"], reverse=True
    )[:1]

# =========================================================
# 16. SOFASCORE UNIFIED ENGINE (DIRECT + RAPID BACKUP)
# =========================================================

class SofascoreDirectFetcher:
    """گزینه اصلی: استخراج مستقیم، رایگان و بدون محدودیت با شبیه‌سازی مرورگر"""
    BASE_URL = "https://api.sofascore.com/api/v1"

    def __init__(self) -> None:
        self._total_requests = 0

    async def _get(self, session, endpoint: str, params: Optional[dict] = None, label: str = "Direct") -> Optional[dict]:
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {
            "Origin": "https://www.sofascore.com",
            "Referer": "https://www.sofascore.com/",
            "Accept": "application/json, text/plain, */*",
            "Cache-Control": "no-cache",
        }
        try:
            # استفاده از متدهای curl_cffi که json() را همگام برمی‌گرداند
            res = await session.get(url, params=params, headers=headers, timeout=15)
            self._total_requests += 1
            if res.status_code == 200:
                return res.json()
            if res.status_code in [403, 429]:
                logger.warning("⚠️  [%s] Blocked HTTP %d: %s", label, res.status_code, endpoint[:40])
                return None
            logger.debug("⚠️  [%s] HTTP %d: %s", label, res.status_code, endpoint[:40])
            return None
        except Exception as e:
            logger.debug("❓ [%s] Error: %s", label, e)
            return None

    async def find_event_id(self, home: str, away: str, session) -> Optional[int]:
        hl = clean_team_name(home).lower()
        al = clean_team_name(away).lower()
        
        # 1. جستجوی مستقیم
        for query in [f"{clean_team_name(home)} {clean_team_name(away)}", clean_team_name(home)]:
            await asyncio.sleep(1.0) # تاخیر برای جلوگیری از حساس شدن کلودفلر
            data = await self._get(session, "search/all", params={"q": query}, label="Direct-Search")
            if not data: continue
            
            for item in data.get("results", []):
                if item.get("type") != "event": continue
                e = item.get("entity", {})
                mid = e.get("id")
                if not mid: continue
                hn = e.get("homeTeam", {}).get("name", "").lower()
                an = e.get("awayTeam", {}).get("name", "").lower()
                if _flex_match(hl, hn) and _flex_match(al, an):
                    logger.info("Direct Sofascore found: %s vs %s → id=%d", home, away, mid)
                    return int(mid)

        # 2. جستجو در برنامه‌های امروز
        await asyncio.sleep(1.0)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for sport in ["football", "tennis"]:
            data = await self._get(session, f"sport/{sport}/scheduled-events/{today}", label=f"Direct-Sched-{sport}")
            if not data: continue
            for e in data.get("events", []):
                hn = e.get("homeTeam", {}).get("name", "").lower()
                an = e.get("awayTeam", {}).get("name", "").lower()
                if _flex_match(hl, hn) and _flex_match(al, an):
                    mid = e.get("id")
                    if mid:
                        logger.info("Direct Sofascore scheduled: %s vs %s → id=%d", home, away, mid)
                        return int(mid)
        return None

    async def fetch_stats(self, home: str, away: str) -> dict:
        if not cffi_requests:
            return {} # اگر پکیج نصب نبود، خروجی خالی بده تا سوییچ بشه رو بکاپ
            
        out: dict = {"_source": "sofascore_direct"}
        
        # باز کردن یک تونل امن با هویت کروم 120
        async with cffi_requests.AsyncSession(impersonate="chrome120") as session:
            event_id = await self.find_event_id(home, away, session)
            out["_event_id"] = event_id

            if event_id:
                await asyncio.sleep(1.0)
                # دریافت موازی داده‌ها
                form_d, h2h_d, lu_d, stats_d = await asyncio.gather(
                    self._get(session, f"event/{event_id}/pregame-form", label="Direct-Form"),
                    self._get(session, f"event/{event_id}/h2h/events", label="Direct-H2H"),
                    self._get(session, f"event/{event_id}/lineups", label="Direct-Lineups"),
                    self._get(session, f"event/{event_id}/statistics", label="Direct-Stats"),
                    return_exceptions=True,
                )

                # پارس کردن Form
                if isinstance(form_d, dict):
                    for side, key in [("homeTeam", "home_form"), ("awayTeam", "away_form")]:
                        fd = form_d.get(side, {})
                        if fd:
                            out[key] = {
                                "team": home if side == "homeTeam" else away,
                                "form": fd.get("value", ""),
                                "avg_rating": fd.get("avgRating"),
                                "position": fd.get("position"),
                            }

                # پارس کردن H2H
                events_list: list = []
                if isinstance(h2h_d, dict): events_list = h2h_d.get("events", [])
                elif isinstance(h2h_d, list): events_list = h2h_d

                if events_list:
                    hw = aw = d = 0
                    for m in events_list:
                        hs = m.get("homeScore", {}).get("current")
                        as_ = m.get("awayScore", {}).get("current")
                        if hs is None or as_ is None: continue
                        h_name = m.get("homeTeam", {}).get("name", "").lower()
                        if _flex_match(clean_team_name(home).lower(), h_name):
                            if hs > as_: hw += 1
                            elif as_ > hs: aw += 1
                            else: d += 1
                        else:
                            if as_ > hs: hw += 1
                            elif hs > as_: aw += 1
                            else: d += 1
                    out["h2h"] = {f"{home}_wins": hw, f"{away}_wins": aw, "draws": d, "total": hw + aw + d}

                # پارس کردن Lineups
                if isinstance(lu_d, dict) and lu_d:
                    out["lineups"] = {
                        "home_formation": lu_d.get("home", {}).get("formation", "N/A"),
                        "away_formation": lu_d.get("away", {}).get("formation", "N/A"),
                    }

                # پارس کردن Stats
                if isinstance(stats_d, dict):
                    groups = stats_d.get("statistics", [])
                    if groups:
                        wanted = {"Ball possession", "Total shots", "Shots on target", "Corner kicks", "Fouls", "Expected goals", "Big chances"}
                        match_stats: dict = {}
                        for group in groups:
                            for item in group.get("statisticsItems", []):
                                name = item.get("name", "")
                                if name in wanted:
                                    match_stats[name] = {"home": item.get("home"), "away": item.get("away")}
                        if match_stats: out["match_stats"] = match_stats

        return out if out.get("_event_id") else {}


class SofaScoreRapidFetcher:
    """گزینه بکاپ (کد قبلی شما): استفاده از RapidAPI در صورت مسدود شدن مستقیم"""
    BASE_URL = "https://sofascore6.p.rapidapi.com/api/sofascore/v1"

    def __init__(self, key_manager: RapidKeyManager) -> None:
        self.km = key_manager
        self._total_requests = 0

    async def _get(self, session: aiohttp.ClientSession, endpoint: str, params: Optional[dict] = None, label: str = "Rapid") -> Optional[dict]:
        headers = self.km.get_headers()
        if not headers: return None
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=12)) as res:
                self._total_requests += 1
                self.km.mark_request()
                if res.status == 200: return await res.json(content_type=None)
                if res.status == 429:
                    self.km.mark_rate_limited()
                    headers2 = self.km.get_headers()
                    if headers2:
                        await asyncio.sleep(1.5)
                        async with session.get(url, headers=headers2, params=params, timeout=12) as res2:
                            self._total_requests += 1
                            if res2.status == 200: return await res2.json(content_type=None)
                    return None
                if res.status in [401, 403]: return None
                return None
        except Exception: return None

    async def _find_event_id(self, home: str, away: str, session: aiohttp.ClientSession) -> Optional[int]:
        hl = clean_team_name(home).lower()
        al = clean_team_name(away).lower()
        for query in [f"{clean_team_name(home)} {clean_team_name(away)}", clean_team_name(home)]:
            await asyncio.sleep(CFG.RAPID_REQUEST_DELAY)
            data = await self._get(session, "search/multi-search", params={"query": query}, label="Rapid-Search")
            if not data: continue
            for item in data.get("results", []):
                if item.get("type") != "event": continue
                e = item.get("entity", {})
                mid = e.get("id")
                if not mid: continue
                hn = e.get("homeTeam", {}).get("name", "").lower()
                an = e.get("awayTeam", {}).get("name", "").lower()
                if _flex_match(hl, hn) and _flex_match(al, an):
                    return int(mid)
        return None

    async def fetch_stats(self, home: str, away: str, session: aiohttp.ClientSession) -> dict:
        if not self.km.get_current_key(): return {}
        out: dict = {"_source": "sofascore6_rapidapi"}
        event_id = await self._find_event_id(home, away, session)
        out["_event_id"] = event_id

        if event_id:
            await asyncio.sleep(CFG.RAPID_REQUEST_DELAY)
            form_d, h2h_d, lu_d, stats_d = await asyncio.gather(
                self._get(session, f"event/{event_id}/pregame-form"),
                self._get(session, f"event/{event_id}/h2h/events"),
                self._get(session, f"event/{event_id}/lineups"),
                self._get(session, f"event/{event_id}/statistics"),
                return_exceptions=True,
            )
            # استخراج دیتا (مشابه مستقیم)
            if isinstance(form_d, dict):
                for side, key in [("homeTeam", "home_form"), ("awayTeam", "away_form")]:
                    fd = form_d.get(side, {})
                    if fd: out[key] = {"team": home if side == "homeTeam" else away, "form": fd.get("value", "")}
            events_list: list = []
            if isinstance(h2h_d, dict): events_list = h2h_d.get("events", [])
            elif isinstance(h2h_d, list): events_list = h2h_d
            if events_list:
                hw = aw = d = 0
                for m in events_list:
                    hs = m.get("homeScore", {}).get("current")
                    as_ = m.get("awayScore", {}).get("current")
                    if hs is None or as_ is None: continue
                    h_name = m.get("homeTeam", {}).get("name", "").lower()
                    if _flex_match(clean_team_name(home).lower(), h_name):
                        if hs > as_: hw += 1
                        elif as_ > hs: aw += 1
                        else: d += 1
                    else:
                        if as_ > hs: hw += 1
                        elif hs > as_: aw += 1
                        else: d += 1
                out["h2h"] = {f"{home}_wins": hw, f"{away}_wins": aw, "draws": d, "total": hw + aw + d}
            if isinstance(lu_d, dict) and lu_d:
                out["lineups"] = {"home_formation": lu_d.get("home", {}).get("formation", "N/A"), "away_formation": lu_d.get("away", {}).get("formation", "N/A")}
            if isinstance(stats_d, dict) and stats_d.get("statistics"):
                match_stats = {}
                for group in stats_d["statistics"]:
                    for item in group.get("statisticsItems", []):
                        if item.get("name") in {"Ball possession", "Total shots", "Shots on target", "Corner kicks", "Fouls", "Expected goals", "Big chances"}:
                            match_stats[item["name"]] = {"home": item.get("home"), "away": item.get("away")}
                if match_stats: out["match_stats"] = match_stats
        return out if out.get("_event_id") else {}


class SofaScoreUnifiedFetcher:
    """این کلاس تصمیم می‌گیرد که از روش مستقیم استفاده کند یا در صورت شکست به RapidAPI سوییچ کند"""
    
    def __init__(self, rapid_km: RapidKeyManager) -> None:
        self.direct = SofascoreDirectFetcher()
        self.rapid = SofaScoreRapidFetcher(rapid_km)
        self._cache: dict = DailyCache.load(CFG.DAILY_RAPID_CACHE_FILE) or {}
        
    def _get_from_cache(self, home: str, away: str) -> Optional[dict]:
        k = hashlib.md5(f"{home.lower()}|{away.lower()}".encode()).hexdigest()
        if k in self._cache:
            logger.debug("Sofascore DailyCache HIT: %s vs %s", home, away)
            return self._cache[k]
        return None

    def _save_to_cache(self, home: str, away: str, stats: dict) -> None:
        k = hashlib.md5(f"{home.lower()}|{away.lower()}".encode()).hexdigest()
        self._cache[k] = stats
        DailyCache.save(CFG.DAILY_RAPID_CACHE_FILE, self._cache)

    async def fetch_stats(self, home: str, away: str, session: aiohttp.ClientSession, sport_key: str = "") -> dict:
        normalized_sk = normalize_sport_key(sport_key) if sport_key else ""
        if normalized_sk and normalized_sk not in CFG.RAPID_SUPPORTED_SPORTS:
            return {}

        cached = self._get_from_cache(home, away)
        if cached is not None: return cached

        logger.info("Sofascore Fetching: %s vs %s (Trying Direct First...)", home, away)
        
        # تلاش برای استخراج مستقیم و رایگان
        data = await self.direct.fetch_stats(home, away)
        
        # اگر مستقیم شکست خورد (خطای 403 یا پیدا نشدن)، سوییچ به بکاپ
        if not data or not data.get("_event_id"):
            logger.warning("Direct fetch failed for %s vs %s. Falling back to RapidAPI Backup!", home, away)
            data = await self.rapid.fetch_stats(home, away, session)
            
        if data:
            self._save_to_cache(home, away, data)
            logger.info("Sofascore cached: %s vs %s | src=%s", home, away, data.get("_source", "unknown"))
            
        return data

    async def prefetch_all(self, events: list, session: aiohttp.ClientSession) -> None:
        log_section("SOFASCORE — DAILY PREFETCH")
        to_fetch = [e for e in events if normalize_sport_key(e.get("sport_title", "")) in CFG.RAPID_SUPPORTED_SPORTS and e.get("home_team") and e.get("away_team")]
        logger.info("Prefetching %d events...", len(to_fetch))
        
        already = fetched = 0
        for event in to_fetch:
            home = event["home_team"]
            away = event["away_team"]
            sk   = event.get("sport_key", "")
            if self._get_from_cache(home, away) is not None:
                already += 1
                continue
            await self.fetch_stats(home, away, session, sport_key=sk)
            fetched += 1
            
        logger.info("Prefetch done: fetched=%d cached=%d", fetched, already)

# =========================================================
# 17. FOOTBALL-DATA ADAPTER
# =========================================================
class FootballDataAdapter:
    BASE_URL = "https://api.football-data.org/v4"
    COMP_MAP = {
        2021: "PL",  2014: "PD",  2002: "BL1",
        2019: "SA",  2015: "FL1", 2003: "DED",
        2017: "PPL", 2016: "ELC", 2001: "CL",
    }

    def __init__(self) -> None:
        self.headers = (
            {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
            if FOOTBALL_DATA_API_KEY else {}
        )
        self.daily_cache = CacheManager.load(CFG.DAILY_STATS_CACHE_FILE)
        self.call_count  = 0
        entry = self.daily_cache.get("_call_count_today", {})
        if isinstance(entry.get("data"), int):
            self.call_count = entry["data"]
        try:
            last = entry.get("timestamp", "2000-01-01T00:00:00+00:00")
            if (
                datetime.now(timezone.utc).date()
                > datetime.fromisoformat(last).date()
            ):
                self.call_count = 0
                logger.info("FD call counter reset (new day)")
        except Exception:
            self.call_count = 0
        log_check("FD calls today", self.call_count, warn_if_none=False)

    def _can_call(self) -> bool:
        return (
            self.call_count < CFG.FOOTBALL_DATA_DAILY_LIMIT
            and bool(FOOTBALL_DATA_API_KEY)
        )

    def _inc(self) -> None:
        self.call_count += 1
        self.daily_cache = CacheManager.set(
            self.daily_cache, "_call_count_today", self.call_count
        )
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)

    @retry_sync(max_retries=2, delay=3.0)
    def _get(self, ep: str, params: Optional[dict] = None) -> Optional[dict]:
        if not self._can_call():
            return None
        url = f"{self.BASE_URL}{ep}"
        res = requests.get(url, headers=self.headers, params=params, timeout=12)
        res.raise_for_status()
        self._inc()
        return res.json()

    def find_team_id(self, team_name: str) -> Optional[int]:
        cache = CacheManager.load(CFG.TEAM_ID_CACHE_FILE)
        key   = team_name.lower().strip()
        if key in cache:
            return cache[key]
        if not self._can_call():
            return None
        clean = clean_team_name(team_name).lower()
        tid: Optional[int] = None
        for cid in self.COMP_MAP:
            data = self._get(f"/competitions/{cid}/teams", {"season": "2024"})
            if not data or not data.get("teams"):
                continue
            for t in data["teams"]:
                tn = t.get("name",      "").lower()
                ts = t.get("shortName", "").lower()
                if (
                    clean == tn or clean == ts
                    or clean in tn or tn in clean or clean in ts
                ):
                    tid = t["id"]
                    logger.info("FD: '%s' → id=%d", team_name, tid)
                    break
            if tid:
                break
        if tid is None:
            logger.warning("FD: team '%s' NOT found", team_name)
        cache[key] = tid
        CacheManager.save(CFG.TEAM_ID_CACHE_FILE, cache)
        return tid

    def get_form(self, team_id: int, team_name: str) -> dict:
        ck = f"form_{team_id}"
        if CacheManager.is_valid(self.daily_cache, ck, CFG.TTL_TEAM_FORM):
            return CacheManager.get(self.daily_cache, ck) or {}
        data = self._get(
            f"/teams/{team_id}/matches/",
            {"status": "FINISHED", "limit": "5"},
        )
        if not data:
            return {}
        form = self._parse_form(data.get("matches", []), team_id, team_name)
        self.daily_cache = CacheManager.set(self.daily_cache, ck, form)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)
        return form

    def _parse_form(self, matches: list, tid: int, tname: str) -> dict:
        rs: list[str] = []
        gs: list[int] = []
        gc: list[int] = []
        for m in matches[-5:]:
            hid = m.get("homeTeam", {}).get("id")
            aid = m.get("awayTeam", {}).get("id")
            sc  = m.get("score", {}).get("fullTime", {})
            hg  = int(sc.get("home") or 0)
            ag  = int(sc.get("away") or 0)
            if hid == tid:
                s, c = hg, ag
                r = "W" if hg > ag else ("D" if hg == ag else "L")
            elif aid == tid:
                s, c = ag, hg
                r = "W" if ag > hg else ("D" if ag == hg else "L")
            else:
                continue
            rs.append(r); gs.append(s); gc.append(c)
        n = len(rs)
        if n == 0:
            return {}
        return {
            "form_string":        "".join(rs),
            "win_rate":           round(rs.count("W") / n, 2),
            "draw_rate":          round(rs.count("D") / n, 2),
            "avg_goals_scored":   round(sum(gs) / n, 2),
            "avg_goals_conceded": round(sum(gc) / n, 2),
            "btts_rate":          round(
                sum(1 for a, b in zip(gs, gc) if a > 0 and b > 0) / n, 2
            ),
            "over25_rate": round(
                sum(1 for a, b in zip(gs, gc) if a + b > 2.5) / n, 2
            ),
            "matches_analyzed": n,
        }

    def get_h2h(self, t1_id: int, t2_id: int, t1n: str, t2n: str) -> dict:
        ck = f"h2h_{min(t1_id, t2_id)}_{max(t1_id, t2_id)}"
        if CacheManager.is_valid(self.daily_cache, ck, CFG.TTL_H2H):
            return CacheManager.get(self.daily_cache, ck) or {}
        data = self._get(
            f"/teams/{t1_id}/matches/",
            {"status": "FINISHED", "limit": "20"},
        )
        if not data:
            return {}
        all_m  = data.get("matches", [])
        h2h_m  = [
            m for m in all_m
            if {m.get("homeTeam", {}).get("id"),
                m.get("awayTeam", {}).get("id")} == {t1_id, t2_id}
        ]
        result = self._parse_h2h(h2h_m, t1_id, t1n, t2n)
        self.daily_cache = CacheManager.set(self.daily_cache, ck, result)
        CacheManager.save(CFG.DAILY_STATS_CACHE_FILE, self.daily_cache)
        return result

    def _parse_h2h(self, matches: list, t1_id: int, t1: str, t2: str) -> dict:
        w1 = w2 = d = tg = bt = o25 = 0
        n  = len(matches)
        for m in matches:
            sc  = m.get("score", {}).get("fullTime", {})
            hg  = int(sc.get("home") or 0)
            ag  = int(sc.get("away") or 0)
            hid = m.get("homeTeam", {}).get("id")
            if hg > ag:
                if hid == t1_id: w1 += 1
                else:            w2 += 1
            elif ag > hg:
                if hid != t1_id: w1 += 1
                else:            w2 += 1
            else:
                d += 1
            tg  += hg + ag
            bt  += 1 if hg > 0 and ag > 0 else 0
            o25 += 1 if hg + ag > 2.5 else 0
        if n == 0:
            return {}
        return {
            "total_h2h":          n,
            f"{t1}_wins":         w1,
            f"{t2}_wins":         w2,
            "draws":              d,
            "avg_goals_per_game": round(tg / n, 2),
            "btts_rate":          round(bt  / n, 2),
            "over25_rate":        round(o25 / n, 2),
        }

# =========================================================
# 18. MATCH ID CACHE
# =========================================================
class MatchIDCache:
    def __init__(self) -> None:
        self.cache = CacheManager.load(CFG.MATCH_ID_CACHE_FILE)

    def get(self, home: str, away: str) -> Optional[int]:
        k = self._key(home, away)
        return (
            CacheManager.get(self.cache, k)
            if CacheManager.is_valid(self.cache, k, CFG.TTL_MATCH_ID)
            else None
        )

    def set(self, home: str, away: str, mid: Optional[int]) -> None:
        k          = self._key(home, away)
        self.cache = CacheManager.set(self.cache, k, mid)
        CacheManager.save(CFG.MATCH_ID_CACHE_FILE, self.cache)

    @staticmethod
    def _key(home: str, away: str) -> str:
        return hashlib.md5(
            f"{home.lower()}|{away.lower()}".encode()
        ).hexdigest()

# =========================================================
# 19. STATS AGGREGATOR
# =========================================================
async def get_stats_async(
    home:            str,
    away:            str,
    sport_key:       str,
    fd:              FootballDataAdapter,
    mic:             MatchIDCache,
    elo_f:           ELOSystem,
    elo_t:           ELOSystem,
    session:         aiohttp.ClientSession,
    rapid:           SofaScoreUnifiedFetcher,
    rapid_sport_key: str = "",
) -> tuple:
    log_section(f"STATS: {home} vs {away}")
    stats: dict = {
        "home_form":    {},
        "away_form":    {},
        "h2h":          {},
        "sofascore":    {},
        "elo":          {},
        "data_quality": "none",
        "_sources":     [],
    }

    elo_pred: Optional[dict] = None
    if sport_key == "football":
        elo_pred = elo_f.predict(home, away, apply_home=True)
    elif sport_key == "tennis":
        elo_pred = elo_t.predict(home, away, apply_home=False)

    if elo_pred and (
        elo_pred.get("home_matches", 0) >= 3
        or elo_pred.get("away_matches", 0) >= 3
    ):
        stats["elo"] = elo_pred
        logger.info(
            "ELO | %s vs %s | H=%.1f%% D=%.1f%% A=%.1f%%"
            " | hm=%d am=%d diff=%.0f",
            home, away,
            elo_pred["home_prob"] * 100,
            elo_pred["draw_prob"] * 100,
            elo_pred["away_prob"] * 100,
            elo_pred["home_matches"],
            elo_pred["away_matches"],
            elo_pred["elo_diff"],
        )
    else:
        logger.warning(
            "ELO insufficient: %s(hm=%d) %s(am=%d)",
            home, (elo_pred or {}).get("home_matches", 0),
            away, (elo_pred or {}).get("away_matches", 0),
        )

    rapid_data = await rapid.fetch_stats(
        home, away, session, sport_key=rapid_sport_key,
    )

    if rapid_data:
        for k in ["home_form", "away_form", "h2h", "lineups", "match_stats"]:
            if k in rapid_data and rapid_data[k]:
                stats[k] = rapid_data[k]
        stats["sofascore"] = {
            k: rapid_data[k]
            for k in ["home_form", "away_form", "h2h", "lineups", "match_stats"]
            if k in rapid_data and rapid_data[k]
        }
        if rapid_data.get("_event_id"):
            if "sofascore6_rapid" not in stats["_sources"]:
                stats["_sources"].append("sofascore6_rapid")

    if sport_key == "football":
        loop = asyncio.get_running_loop()

        async def _get_fd_data() -> dict:
            hid = await loop.run_in_executor(None, fd.find_team_id, home)
            aid = await loop.run_in_executor(None, fd.find_team_id, away)
            log_check(f"FD id '{home}'", hid)
            log_check(f"FD id '{away}'", aid)
            if not hid or not aid:
                return {}
            results = await asyncio.gather(
                loop.run_in_executor(None, fd.get_form, hid, home),
                loop.run_in_executor(None, fd.get_form, aid, away),
                loop.run_in_executor(None, fd.get_h2h, hid, aid, home, away),
                return_exceptions=True,
            )
            hf, af, h2h = results
            out: dict = {}
            if not isinstance(hf,  Exception) and hf:  out["home_form"] = hf
            if not isinstance(af,  Exception) and af:  out["away_form"] = af
            if not isinstance(h2h, Exception) and h2h: out["h2h"]       = h2h
            return out

        try:
            fd_data = await _get_fd_data()
            if fd_data.get("home_form"):
                stats["home_form"] = fd_data["home_form"]
                if "football_data" not in stats["_sources"]:
                    stats["_sources"].append("football_data")
            if fd_data.get("away_form"):
                stats["away_form"] = fd_data["away_form"]
            if fd_data.get("h2h"):
                stats["h2h"] = fd_data["h2h"]
        except Exception as e:
            logger.warning("FD gather error: %s", e)

    has_fb  = bool(stats.get("home_form") or stats.get("h2h"))
    has_ss  = bool(stats.get("sofascore"))
    has_elo = bool(stats.get("elo"))
    sources = stats.get("_sources", [])

    if (has_fb or has_elo) and has_ss:
        stats["data_quality"] = "high"
    elif has_fb or has_ss or has_elo:
        stats["data_quality"] = "medium"
    else:
        stats["data_quality"] = "none"

    logger.info(
        "DATA QUALITY | %s vs %s | %s (fb=%s ss=%s elo=%s src=%s)",
        home, away, stats["data_quality"].upper(),
        has_fb, has_ss, has_elo, sources,
    )
    return stats, elo_pred

# =========================================================
# 20. CONFIDENCE ENGINE
# =========================================================
def calculate_confidence(
    ev: float, stats: dict, market: str, has_sharp: bool,
) -> tuple[int, str]:
    score = 50
    dq    = stats.get("data_quality", "none")

    if dq == "high":     score += 15
    elif dq == "medium": score += 8

    ep = ev * 100
    if ep > 5.0:   score += 12
    elif ep > 3.0: score += 8
    elif ep > 1.5: score += 4

    elo = stats.get("elo", {})
    hm  = elo.get("home_matches", 0)
    am  = elo.get("away_matches", 0)
    if hm >= 10 and am >= 10: score += 10
    elif hm >= 5  and am >= 5:  score += 6
    elif hm >= 3  or  am >= 3:  score += 3

    if has_sharp:          score += 5
    if market == "totals": score += 3

    hf = stats.get("home_form", {})
    af = stats.get("away_form", {})
    if hf.get("form_string") and af.get("form_string"):
        if hf["form_string"].count("W") >= 3: score += 5
        if af["form_string"].count("L") >= 3: score += 3

    ss = stats.get("sofascore", {})
    if ss.get("home_form") and ss.get("away_form"): score += 4

    sources = stats.get("_sources", [])
    if len(sources) >= 2: score += 4
    elif len(sources) == 1: score += 2

    score = max(50, min(93, score))
    risk  = (
        "Low"    if score >= 75
        else ("Medium" if score >= 60
        else "High")
    )
    logger.info(
        "Confidence=%d risk=%s (dq=%s ev=%.1f%% hm=%d am=%d sharp=%s src=%s)",
        score, risk, dq, ep, hm, am, has_sharp, sources,
    )
    return score, risk

# =========================================================
# 21. DUAL-AI ANALYSIS
# =========================================================
def build_stats_summary(stats: dict, home: str, away: str) -> str:
    parts: list[str] = []
    elo = stats.get("elo",       {})
    hf  = stats.get("home_form", {})
    af  = stats.get("away_form", {})
    h2h = stats.get("h2h",       {})
    ss  = stats.get("sofascore", {})

    if elo and elo.get("home_matches", 0) >= 3:
        parts.append(
            f"[ELO MODEL]\n"
            f"  {home}: ELO={elo['home_elo']:.0f} ({elo['home_matches']} matches)\n"
            f"  {away}: ELO={elo['away_elo']:.0f} ({elo['away_matches']} matches)\n"
            f"  Win probs: {home}={elo['home_prob']:.1%} "
            f"Draw={elo['draw_prob']:.1%} {away}={elo['away_prob']:.1%}"
        )
    if hf:
        parts.append(
            f"[FORM — {home}]\n"
            f"  Last 5: {hf.get('form_string','N/A')} | "
            f"WR={hf.get('win_rate',0):.0%} | "
            f"GF={hf.get('avg_goals_scored',0)} | "
            f"GA={hf.get('avg_goals_conceded',0)} | "
            f"BTTS={hf.get('btts_rate',0):.0%} | "
            f"O2.5={hf.get('over25_rate',0):.0%}"
        )
    if af:
        parts.append(
            f"[FORM — {away}]\n"
            f"  Last 5: {af.get('form_string','N/A')} | "
            f"WR={af.get('win_rate',0):.0%} | "
            f"GF={af.get('avg_goals_scored',0)} | "
            f"GA={af.get('avg_goals_conceded',0)} | "
            f"BTTS={af.get('btts_rate',0):.0%} | "
            f"O2.5={af.get('over25_rate',0):.0%}"
        )
    if h2h and (h2h.get("total_h2h", 0) > 0 or h2h.get("total", 0) > 0):
        total = h2h.get("total_h2h", h2h.get("total", 0))
        w1    = h2h.get(f"{home}_wins", 0)
        w2    = h2h.get(f"{away}_wins", 0)
        parts.append(
            f"[HEAD TO HEAD — {total} games]\n"
            f"  {home}: {w1}W | {away}: {w2}W | "
            f"Draws: {h2h.get('draws',0)} | "
            f"AvgGoals={h2h.get('avg_goals_per_game',0)} | "
            f"BTTS={h2h.get('btts_rate',0):.0%} | "
            f"O2.5={h2h.get('over25_rate',0):.0%}"
        )
    if ss:
        shf = ss.get("home_form", {})
        saf = ss.get("away_form", {})
        if shf or saf:
            parts.append("[SOFASCORE PREGAME FORM]")
        if shf:
            parts.append(
                f"  {home}: form={shf.get('form','N/A')} "
                f"rating={shf.get('avg_rating','N/A')} "
                f"pos={shf.get('position','N/A')}"
            )
        if saf:
            parts.append(
                f"  {away}: form={saf.get('form','N/A')} "
                f"rating={saf.get('avg_rating','N/A')} "
                f"pos={saf.get('position','N/A')}"
            )
        ms = ss.get("match_stats", {})
        if ms:
            parts.append("[MATCH STATISTICS]")
            for sname, vals in ms.items():
                parts.append(
                    f"  {sname}: {home}={vals.get('home','?')} | "
                    f"{away}={vals.get('away','?')}"
                )
        sh2h = ss.get("h2h", {})
        if sh2h and sh2h.get("total", 0) > 0:
            parts.append(
                f"[SOFASCORE H2H — {sh2h['total']} games]\n"
                f"  {home}: {sh2h.get(f'{home}_wins','N/A')}W | "
                f"{away}: {sh2h.get(f'{away}_wins','N/A')}W | "
                f"Draws: {sh2h.get('draws','N/A')}"
            )
        lu = ss.get("lineups", {})
        if lu:
            parts.append(
                f"[LINEUPS] {home}={lu.get('home_formation','?')} "
                f"{away}={lu.get('away_formation','?')}"
            )

    return "\n\n".join(parts) if parts else "NO STATISTICAL DATA AVAILABLE"


async def call_groq_async(
    model: str, messages: list, temp: float = 0.1
) -> Optional[str]:
    SUPPORTS_JSON = ["llama-3", "llama3", "mixtral", "gemma", "llama-4", "scout"]
    use_json = any(k in model.lower() for k in SUPPORTS_JSON)
    kwargs: dict = {
        "model":       model,
        "messages":    messages,
        "temperature": temp,
        "max_tokens":  CFG.AI_MAX_TOKENS,
    }
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        res     = await groq_client.chat.completions.create(**kwargs)
        content = res.choices[0].message.content
        logger.info(
            "Groq %-32s | tokens=%s | out=%s",
            model, getattr(res.usage, "total_tokens", "?"),
            (content or "")[:80],
        )
        return content
    except Exception as e:
        logger.error("Groq error %s: %s", model, e)
        return None


async def generate_dual_ai_analysis_async(
    home:         str,
    away:         str,
    sport:        str,
    display_pick: str,
    market:       str,
    ev:           float,
    stats:        dict,
    confidence:   int,
    risk:         str,
) -> dict:
    summary   = build_stats_summary(stats, home, away)
    dq        = stats.get("data_quality", "none")
    has_stats = dq in ["medium", "high"]
    sources   = stats.get("_sources", [])

    default: dict = {
        "sport_emoji": "🏆",
        "home_flag":   get_flag_from_name(home),
        "away_flag":   get_flag_from_name(away),
        "risk_level":  risk,
        "confidence":  confidence,
        "logic":       "Sharp market lines show clear value on this selection.",
    }

    sys1 = (
        "You are an elite sports betting analyst.\n"
        "Write EXACTLY 2 punchy professional sentences justifying the pick.\n"
        "RULES:\n"
        "- Use ONLY the provided statistics. Never invent numbers.\n"
        "- Never mention EV, edge, models, algorithms, or data quality.\n"
        "- If no stats available: reference sharp market movement only.\n"
        "- Use exact country flag emoji for home_flag and away_flag.\n"
        "- Use the correct sport_emoji (⚽🎾🏀⚾🏒🏈 etc).\n"
        "OUTPUT: valid JSON only. No markdown, no extra text.\n"
        '{"sport_emoji":"...","home_flag":"...","away_flag":"...",'
        '"logic":"Sentence 1. Sentence 2."}'
    )
    u1 = (
        f"MATCH: {home} vs {away}\n"
        f"SPORT: {sport}\n"
        f"PICK: {display_pick}\n"
        f"MARKET: {get_market_label(market)}\n"
        f"DATA QUALITY: {dq} | SOURCES: {sources}\n\n"
        f"STATISTICS:\n{summary}\n\nOUTPUT JSON ONLY:"
    )

    a1: Optional[dict] = None
    try:
        r1 = await call_groq_async(
            CFG.AI_MODEL_ANALYST,
            [{"role": "system", "content": sys1},
             {"role": "user",   "content": u1}],
            temp=0.2,
        )
        a1 = robust_json_extractor(r1)
        log_check("AI analyst", "OK" if a1 else "FAILED")
    except Exception as e:
        logger.warning("AI analyst error: %s", e)

    logic: str = (a1 or {}).get("logic") or default["logic"]

    sys2 = (
        "You are a professional sports content editor.\n"
        "Review and polish the draft analysis to max 2 sentences.\n"
        "Maintain tipster tone. Remove any fabricated statistics.\n"
        "OUTPUT: valid JSON only.\n"
        '{"validated_logic":"..."}'
    )
    try:
        r2 = await call_groq_async(
            CFG.AI_MODEL_VALIDATOR,
            [
                {"role": "system", "content": sys2},
                {"role": "user",   "content": (
                    f"DRAFT: {logic}\nPICK: {display_pick}\n"
                    f"HAS_STATS: {has_stats}\nOUTPUT JSON ONLY:"
                )},
            ],
            temp=0.15,
        )
        a2 = robust_json_extractor(r2)
        if a2 and a2.get("validated_logic"):
            logic = a2["validated_logic"]
        log_check("AI validator", "OK" if a2 else "FAILED")
    except Exception as e:
        logger.warning("AI validator error: %s", e)

    result = dict(default)
    if a1:
        if a1.get("sport_emoji"): result["sport_emoji"] = a1["sport_emoji"]
        if a1.get("home_flag"):   result["home_flag"]   = validate_flag(a1["home_flag"], home)
        if a1.get("away_flag"):   result["away_flag"]   = validate_flag(a1["away_flag"], away)

    sl = str(logic).strip()
    result["logic"] = sl[:600] + "…" if len(sl) > 600 else sl
    logger.info(
        "AI final conf=%d risk=%s | '%s'",
        result["confidence"], result["risk_level"], result["logic"][:80],
    )
    return result

# =========================================================
# 22. RESULTS CHECKER
# =========================================================
async def fetch_event_result_async(
    home:      str,
    away:      str,
    sport_key: str,
    km:        OddsKeyManager,
    session:   aiohttp.ClientSession,
) -> Optional[dict]:
    key = km.get_best_key()
    if not key:
        return None

    sports_to_try = [sport_key] if sport_key else []
    for sk in sports_to_try:
        url    = f"https://api.the-odds-api.com/v4/sports/{sk}/scores"
        params = {"apiKey": key, "daysFrom": "3", "dateFormat": "iso"}
        try:
            async with session.get(
                url, params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as res:
                remaining = res.headers.get("x-requests-remaining", "?")
                used      = res.headers.get("x-requests-used",      "?")
                if res.status == 200:
                    km.mark_success(key, remaining, used)
                    events = await res.json(content_type=None)
                    for ev in events:
                        if (
                            _flex_match(home, ev.get("home_team", ""))
                            and _flex_match(away, ev.get("away_team", ""))
                            and ev.get("completed")
                        ):
                            logger.info(
                                "Result found: %s vs %s | scores=%s",
                                home, away, ev.get("scores"),
                            )
                            return ev
                elif res.status == 404:
                    continue
                elif res.status == 429:
                    km.mark_exhausted(key)
                    return None
        except Exception as e:
            logger.warning("fetch_event_result_async (%s): %s", sk, e)
    return None


def _determine_win(
    pick: str, market: str, scores, home: str, away: str,
) -> Optional[bool]:
    try:
        if isinstance(scores, list):
            sm = {s["name"]: s.get("score") for s in scores}
        elif isinstance(scores, dict):
            sm = scores
        else:
            return None

        hs  = None
        as_ = None
        for name, score in sm.items():
            if _flex_match(home, name):
                try:    hs = int(score)
                except: pass
            elif _flex_match(away, name):
                try:    as_ = int(score)
                except: pass

        if hs is None or as_ is None:
            return None

        pl = pick.lower()
        if market == "h2h":
            if "draw" in pl or "tie" in pl: return hs == as_
            if _flex_match(home, pick):     return hs > as_
            if _flex_match(away, pick):     return as_ > hs
            return None
        if market == "totals":
            total = hs + as_
            m     = re.search(r"(over|under)\s*([\d.]+)", pl)
            if m:
                line = float(m.group(2))
                return total > line if m.group(1) == "over" else total < line
    except Exception as e:
        logger.debug("Win check error: %s", e)
    return None


async def check_and_report_results_async(
    sent_history: SentHistory,
    km:           OddsKeyManager,
    session:      aiohttp.ClientSession,
    perf:         PerformanceTracker,
) -> Optional[str]:
    log_section("PHASE 1 — RESULTS CHECK")
    pending = sent_history.get_pending_results()
    log_check("Pending results", len(pending), warn_if_none=False)
    if not pending:
        return None

    wins:   list = []
    losses: list = []

    for key, entry in pending:
        ht     = entry.get("home",   "")
        at     = entry.get("away",   "")
        pick   = entry.get("pick",   "")
        market = entry.get("market", "")
        sk     = entry.get("sport_key", "")

        logger.info("Checking result: %s vs %s [%s]", ht, at, pick)
        rev = await fetch_event_result_async(ht, at, sk, km, session)

        if not rev:
            logger.info("No result yet: %s vs %s", ht, at)
            continue

        scores = rev.get("scores", [])
        won    = _determine_win(pick, market, scores, ht, at)

        try:
            if isinstance(scores, list):
                sm = {s["name"]: s.get("score", "?") for s in scores}
                rs = f"{sm.get(ht, '?')} - {sm.get(at, '?')}"
            else:
                rs = "? - ?"
        except Exception:
            rs = "? - ?"

        await sent_history.mark_result_checked_async(key, rs, won)

        # ← ثبت نتیجه در PerformanceTracker
        perf.record_result(ht, at, market, won, rs)

        logger.info("Result: %s vs %s | %s | won=%s", ht, at, rs, won)

        if won is True:
            wins.append({**entry, "result": rs})
        elif won is False:
            losses.append({**entry, "result": rs})

    if not wins and not losses:
        return None

    total = len(wins) + len(losses)
    wr    = len(wins) / total if total else 0
    roi_v = (
        [w.get("odds", 1.0) - 1.0 for w in wins]
        + [-1.0] * len(losses)
    )
    roi = sum(roi_v) / len(roi_v) if roi_v else 0

    lines: list[str] = ["📊 <b>RESULTS REPORT</b>\n"]
    for w in wins:
        lines.append(
            f"✅ <b>{html_lib.escape(w['home'])} vs "
            f"{html_lib.escape(w['away'])}</b>\n"
            f"   Pick: {html_lib.escape(w['pick'])} "
            f"@ <code>{w['odds']:.2f}</code>\n"
            f"   Score: {w.get('result','?')} — WIN ✅\n"
        )
    for lo in losses:
        lines.append(
            f"❌ <b>{html_lib.escape(lo['home'])} vs "
            f"{html_lib.escape(lo['away'])}</b>\n"
            f"   Pick: {html_lib.escape(lo['pick'])} "
            f"@ <code>{lo['odds']:.2f}</code>\n"
            f"   Score: {lo.get('result','?')} — LOSS ❌\n"
        )

    # ← اضافه کردن خلاصه performance
    perf_summary = perf.format_summary_message()
    lines.append(
        f"\n🎯 {len(wins)}W/{len(losses)}L | "
        f"WR={wr:.0%} | ROI={roi:+.1%}\n"
    )
    if perf_summary:
        lines.append(f"\n{perf_summary}\n")
    lines.append(f"\n🆔 {CFG.TELEGRAM_ID}")
    return "\n".join(lines)

# =========================================================
# 23. TELEGRAM
# =========================================================
async def send_telegram_async(
    message_html: str,
    session:      aiohttp.ClientSession,
) -> bool:
    MAX_LEN = 4000
    if len(message_html) <= MAX_LEN:
        chunks = [message_html]
    else:
        chunks: list[str] = []
        cur = ""
        for line in message_html.split("\n"):
            if len(cur) + len(line) + 1 > MAX_LEN:
                chunks.append(cur.strip())
                cur = line + "\n"
            else:
                cur += line + "\n"
        if cur:
            chunks.append(cur.strip())

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    ok  = True
    for chunk in chunks:
        try:
            async with session.post(
                url,
                json={
                    "chat_id":                  TELEGRAM_CHAT_ID,
                    "text":                     chunk,
                    "parse_mode":               "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as res:
                if res.status != 200:
                    body = await res.text()
                    logger.error("Telegram HTTP %d: %s", res.status, body[:150])
                    ok = False
        except Exception as e:
            logger.error("Telegram send error: %s", e)
            ok = False
    return ok

# =========================================================
# 24. MESSAGE BUILDER
# =========================================================
def build_telegram_message(
    sport:        str,
    home:         str,
    away:         str,
    ct:           str,
    now_utc:      datetime,
    opp:          dict,
    display_pick: str,
    conf:         int,
    risk:         str,
    ai:           dict,
    perf:         PerformanceTracker,
) -> str:
    ci = "🔥" if conf >= 75 else ("✅" if conf >= 60 else "⚡")
    ri = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(risk, "🟠")

    se  = ai.get("sport_emoji", "🏆")
    hf  = ai.get("home_flag",   "🏳️")
    af  = ai.get("away_flag",   "🏳️")
    lo  = html_lib.escape(
        str(ai.get("logic", "")).strip()
        .replace("<", "").replace(">", "")
    )
    ml  = get_market_label(opp["market"])
    bk  = opp.get("bookmaker", "Best Available")
    cd  = get_countdown_str(ct, now_utc)

    # ← خلاصه فرم اخیر
    form_str = perf.get_recent_form(5)
    form_line = f"📋 <b>Recent:</b> {form_str}\n\n" if form_str else ""

    return (
        f"{se} <b>{html_lib.escape(sport)}</b>\n"
        f"\n"
        f"{hf} <b>{html_lib.escape(home)}</b>"
        f"  vs  "
        f"<b>{html_lib.escape(away)}</b> {af}\n"
        f"⏱ <b>Kick-off in:</b> {cd}\n"
        f"\n"
        f"📌 <b>Market:</b> {html_lib.escape(ml)}\n"
        f"🎯 <b>Pick:</b> <code>{html_lib.escape(display_pick)}</code>\n"
        f"💰 <b>Odds:</b> <code>{opp['odds']:.2f}</code> "
        f"<i>({html_lib.escape(bk)})</i>\n"
        f"\n"
        f"{ri} <b>Risk:</b> {risk}  "
        f"{ci} <b>Confidence:</b> {conf}%\n"
        f"\n"
        f"💡 <b>Analysis:</b>\n"
        f"<blockquote>{lo}</blockquote>\n"
        f"\n"
        f"{form_line}"
        f"🆔 {CFG.TELEGRAM_ID}"
    )

# =========================================================
# 25. MAIN PIPELINE
# =========================================================
async def async_main() -> None:
    log_section("ZBET90 ENTERPRISE ENGINE v6.1 STARTING")

    connector = aiohttp.TCPConnector(ssl=False, limit=20, limit_per_host=5)
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": "ZBET90/6.1"},
    ) as session:

        km       = OddsKeyManager(ODDS_API_KEYS)
        rapid_km = RapidKeyManager(RAPIDAPI_KEYS)
        perf     = PerformanceTracker()

        await km.validate_all_async(session)

        if not km.get_best_key():
            logger.critical("NO VALID ODDS API KEY AVAILABLE!")
            sys.exit(1)

        bootstrap = DataBootstrap()
        if bootstrap.should_run():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, bootstrap.run)

        elo_football = ELOSystem("football")
        elo_tennis   = ELOSystem("tennis")
        log_check("ELO football teams", len(elo_football.ratings))
        log_check("ELO tennis players", len(elo_tennis.ratings))

        sent_history = SentHistory()
        fd           = FootballDataAdapter()
        mic          = MatchIDCache()
        rapid        = SofaScoreUnifiedFetcher(rapid_km) 
        now_utc      = datetime.now(timezone.utc)

        # ── فاز 1: بررسی نتایج ──────────────────────────
        results_msg = await check_and_report_results_async(
            sent_history, km, session, perf
        )
        if results_msg:
            if await send_telegram_async(results_msg, session):
                logger.info("Results report sent ✅")
            await asyncio.sleep(2)

        # ── فاز 2: دریافت odds ───────────────────────────
        log_section("PHASE 2 — ODDS FETCH (daily cache)")
        events = await fetch_all_odds_daily(now_utc, km, session)

        if not events:
            logger.error(
                "No events in window. Key status: %s", km.get_summary()
            )
            return

        log_check("Events in window", len(events))

        # ── فاز 2b: Prefetch RapidAPI ────────────────────
        if RAPIDAPI_KEYS:
            if not DailyCache.is_fresh(CFG.DAILY_RAPID_CACHE_FILE):
                log_section("RAPIDAPI — DAILY PREFETCH")
                all_today = DailyCache.load(CFG.DAILY_ODDS_CACHE_FILE) or events
                await rapid.prefetch_all(all_today, session)
            else:
                logger.info(
                    "RapidAPI DailyCache fresh ✅ — skip prefetch | %s",
                    rapid_km.get_stats(),
                )
        else:
            logger.warning("No RapidAPI keys — stats from FootballData only")

        # ── فاز 3: آنالیز و ارسال ───────────────────────
        log_section("PHASE 3 — ANALYSIS & SIGNALS")
        total_sent = 0

        for event in events:
            home      = event.get("home_team",     "")
            away      = event.get("away_team",     "")
            sport     = event.get("sport_title",   "Unknown")
            sk        = normalize_sport_key(sport)
            sport_key = event.get("sport_key",     "")
            ct        = event.get("commence_time", "")
            md        = event.get("_markets_data", {})

            if not home or not away:
                continue

            logger.info(
                "Processing: %-30s vs %-30s [%s]", home, away, sport,
            )

            elo_pred: Optional[dict] = None
            if sk == "football":
                elo_pred = elo_football.predict(home, away)
            elif sk == "tennis":
                elo_pred = elo_tennis.predict(home, away, apply_home=False)

            opps = calculate_combined_ev(
                md, elo_pred, sk, home, away,
                data_quality="none",
            )
            if not opps:
                logger.info("SKIP no value: %s vs %s", home, away)
                continue

            opp = opps[0]

            if sent_history.was_sent(home, away, opp["market"]):
                logger.info("SKIP duplicate: %s vs %s", home, away)
                continue

            stats, _ = await get_stats_async(
                home, away, sk,
                fd, mic, elo_football, elo_tennis,
                session, rapid,
                rapid_sport_key=sport_key,
            )

            real_dq = stats.get("data_quality", "none")
            if real_dq != "none":
                opps_v2 = calculate_combined_ev(
                    md, elo_pred, sk, home, away,
                    data_quality=real_dq,
                )
                if not opps_v2:
                    logger.info(
                        "SKIP after data check: %s vs %s", home, away
                    )
                    continue
                opp = opps_v2[0]

            conf, risk = calculate_confidence(
                opp["ev"], stats, opp["market"], opp["has_sharp_line"],
            )

            if conf < CFG.MIN_CONFIDENCE_TO_SEND:
                logger.info(
                    "SKIP low conf: %s vs %s conf=%d%% min=%d%%",
                    home, away, conf, CFG.MIN_CONFIDENCE_TO_SEND,
                )
                continue

            dp = get_display_pick(opp["pick"], opp["market"], home, away)

            ai = await generate_dual_ai_analysis_async(
                home, away, sport, dp, opp["market"],
                opp["ev"], stats, conf, risk,
            )

            msg = build_telegram_message(
                sport, home, away, ct, now_utc,
                opp, dp, conf, risk, ai, perf,
            )

            logger.info(
                "SIGNAL | %s vs %s | pick=%s odds=%.2f ev=%.1f%% conf=%d%%",
                home, away, dp, opp["odds"], opp["edge_pct"], conf,
            )

            if await send_telegram_async(msg, session):
                await sent_history.mark_sent_async(
                    home          = home,
                    away          = away,
                    pick          = opp["pick"],
                    market        = opp["market"],
                    odds          = opp["odds"],
                    commence_time = ct,
                    sport_key     = sport_key,
                    sport_title   = sport,
                )
                # ← ثبت bet در PerformanceTracker
                perf.record_bet(
                    home      = home,
                    away      = away,
                    pick      = opp["pick"],
                    market    = opp["market"],
                    our_odds  = opp["odds"],
                    ev        = opp["ev"],
                    conf      = conf,
                    sport_key = sport_key,
                )
                total_sent += 1
                logger.info("✅ Sent: %s vs %s", home, away)
            else:
                logger.error("❌ Send failed: %s vs %s", home, away)

            await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    log_section("RUN COMPLETE")
    log_check("Signals sent", total_sent, warn_if_none=False)
    logger.info("Final key status: %s", km.get_summary())
    if RAPIDAPI_KEYS:
        logger.info("RapidAPI stats: %s", rapid_km.get_stats())

    # ← لاگ خلاصه performance نهایی
    s = perf.get_summary()
    if s:
        logger.info(
            "Performance: %dW/%dL WR=%.0f%% ROI=%.1f%% CLV=%s%%",
            s.get("wins", 0), s.get("losses", 0),
            s.get("win_rate", 0), s.get("roi", 0),
            s.get("avg_clv", "N/A"),
        )


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.critical("SYSTEM FAILURE: %s", str(e), exc_info=True)
        sys.exit(1)
