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
    KEY_STATUS_FILE: Path = Path("api_cache/key_status.json")

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

    FD_COMPETITION_IDS: list = field(default_factory=lambda: [
        2021, 2014, 2002, 2019, 2015,
        2003, 2017, 2016, 2001,
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


def log_section(title: str) -> None:
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def log_api_call(
    api_name: str,
    endpoint: str,
    params: dict,
    status: int,
    records: int,
    sample=None,
) -> None:
    safe = {
        k: ("***" if any(s in k.lower()
                         for s in ["key", "token", "api"])
            else v)
        for k, v in (params or {}).items()
    }
    logger.info(
        "API▶ %-22s | status=%-3s | records=%-4d | %s",
        api_name,
        str(status) if status != -1 else "ERR",
        records,
        str(safe)[:120],
    )
    if sample is not None:
        logger.debug(
            "API▶ %-22s | sample=%s", api_name, str(sample)[:300]
        )


def log_check(
    label: str, value, warn_if_none: bool = True
) -> None:
    if value is None or value in ({}, [], ""):
        if warn_if_none:
            logger.warning("CHECK | %-42s | EMPTY/NONE", label)
        else:
            logger.info("CHECK | %-42s | EMPTY (ok)", label)
    else:
        logger.info(
            "CHECK | %-42s | OK | %s", label, str(value)[:100]
        )

# =========================================================
# 3. API KEYS
# =========================================================
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "").strip()
RAPIDAPI_KEY          = os.getenv("RAPIDAPI_KEY", "").strip()
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID", "").strip()
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
FORCE_BOOTSTRAP       = (
    os.getenv("FORCE_BOOTSTRAP", "false").lower() == "true"
)

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

for _name, _val in [
    ("GROQ_API_KEY",          GROQ_API_KEY),
    ("RAPIDAPI_KEY",          RAPIDAPI_KEY),
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

logger.info("Odds API keys available: %d/3", len(ODDS_API_KEYS))
groq_client = AsyncGroq(api_key=GROQ_API_KEY, max_retries=3)

# =========================================================
# 4. NATIONALITY FLAGS
# =========================================================
NATIONALITY_FLAGS: dict[str, str] = {
    "bautista agut": "ES", "alcaraz": "ES", "nadal": "ES",
    "munar": "ES", "davidovich": "ES", "carreno": "ES",
    "djokovic": "RS", "kecmanovic": "RS",
    "sinner": "IT", "berrettini": "IT", "musetti": "IT",
    "zverev": "DE", "struff": "DE",
    "tiafoe": "US", "fritz": "US", "paul": "US", "korda": "US",
    "gauff": "US", "keys": "US", "pegula": "US",
    "medvedev": "RU", "rublev": "RU", "khachanov": "RU",
    "tsitsipas": "GR", "ruud": "NO", "rune": "DK",
    "hurkacz": "PL", "swiatek": "PL",
    "auger-aliassime": "CA", "shapovalov": "CA",
    "kyrgios": "AU", "de minaur": "AU",
    "sabalenka": "BY", "kvitova": "CZ", "vondrousova": "CZ",
    "jabeur": "TN", "rybakina": "KZ", "bublik": "KZ",
    "norrie": "GB", "murray": "GB", "draper": "GB",
    "wawrinka": "CH", "monfils": "FR",
    "etcheverry": "AR", "cerundolo": "AR",
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
    "shakhtar": "UA", "salzburg": "AT",
    "anderlecht": "BE", "club brugge": "BE",
    "copenhagen": "DK", "midtjylland": "DK",
    "malmo": "SE", "djurgarden": "SE",
}


def _code_to_flag(code: str) -> str:
    code = code.upper().strip()
    if len(code) != 2:
        return "\U0001F3F3\uFE0F"
    offset = 0x1F1E6 - ord("A")
    return chr(ord(code[0]) + offset) + chr(ord(code[1]) + offset)


def get_flag_from_name(name: str) -> str:
    nl = name.lower()
    for kw, code in NATIONALITY_FLAGS.items():
        if kw in nl:
            return _code_to_flag(code)
    return "\U0001F3F3\uFE0F"


def validate_flag(flag: str, fallback_name: str) -> str:
    if not flag:
        return get_flag_from_name(fallback_name)
    BAD = {"\U0001F3F3\uFE0F", "\U0001F3C1", "\U0001F6A9",
           "", "🏁", "🏳️", "🏳"}
    return (
        get_flag_from_name(fallback_name)
        if flag.strip() in BAD
        else flag.strip()
    )

# =========================================================
# 5. CACHE MANAGER  (sync — فایل‌ها کوچک هستند)
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
            return (
                datetime.now(timezone.utc) - ct
                < timedelta(hours=ttl_hours)
            )
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
# 6. ODDS API KEY MANAGER
# =========================================================
class OddsKeyManager:
    STATUS_OK        = "ok"
    STATUS_INVALID   = "invalid"
    STATUS_EXHAUSTED = "exhausted"
    STATUS_UNKNOWN   = "unknown"

    def __init__(self, keys: list[str]) -> None:
        self.keys    = keys
        self._status = CacheManager.load(CFG.KEY_STATUS_FILE)
        self._init_keys()
        self._log_all()

    # ── helpers ──────────────────────────────────────────
    @staticmethod
    def _kid(key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def _save(self) -> None:
        CacheManager.save(CFG.KEY_STATUS_FILE, self._status)

    def _prefix(self, key: str) -> str:
        return self._status.get(self._kid(key), {}).get(
            "prefix", key[:8] + "…"
        )

    # ── init ─────────────────────────────────────────────
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
                st.get("prefix", "?"),
                st.get("status",  "?"),
                st.get("remaining", "?"),
                st.get("used",    "?"),
            )

    # ── public ───────────────────────────────────────────
    def get_best_key(self) -> Optional[str]:
        candidates: list[tuple[int, str]] = []
        for k in self.keys:
            kid = self._kid(k)
            st  = self._status.get(kid, {})
            status = st.get("status", self.STATUS_UNKNOWN)

            if status == self.STATUS_INVALID:
                continue

            if status == self.STATUS_EXHAUSTED:
                last = st.get("last_used", "")
                try:
                    lt = datetime.fromisoformat(last)
                    if (datetime.now(timezone.utc).date()
                            > lt.date()):
                        self._status[kid]["status"] = \
                            self.STATUS_UNKNOWN
                        self._status[kid]["remaining"] = None
                        self._save()
                        logger.info(
                            "Key %s: new day → reset to unknown",
                            st.get("prefix", "?"),
                        )
                    else:
                        continue
                except Exception:
                    continue

            remaining = st.get("remaining")
            priority  = remaining if remaining is not None else 999
            candidates.append((priority, k))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0][1]
        kid    = self._kid(chosen)
        logger.info(
            "OddsKeyManager: selected key=%s "
            "(status=%s remaining=%s)",
            self._status[kid].get("prefix", "?"),
            self._status[kid].get("status",  "?"),
            self._status[kid].get("remaining", "?"),
        )
        return chosen

    def mark_success(
        self, key: str, remaining: str, used: str
    ) -> None:
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
                if (rem_int is not None and rem_int <= 0)
                else self.STATUS_OK
            ),
            "remaining":  rem_int,
            "used":       used,
            "last_used":  datetime.now(timezone.utc).isoformat(),
            "last_error": None,
        })
        self._save()
        logger.info(
            "Key %s updated: status=%s remaining=%s used=%s",
            self._status[kid].get("prefix", "?"),
            self._status[kid]["status"],
            remaining, used,
        )

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
                f"{st.get('prefix','?')}:"
                f"{st.get('status','?')}/"
                f"rem={st.get('remaining','?')}"
            )
        return " | ".join(parts)

    async def validate_all_keys_async(
        self, session: aiohttp.ClientSession
    ) -> None:
        """تست همه کلیدها با endpoint رایگان /sports (async)."""
        log_section("VALIDATING ALL ODDS API KEYS")
        for key in self.keys:
            kid    = self._kid(key)
            prefix = self._status[kid].get("prefix", "?")
            if self._status[kid].get("status") == \
                    self.STATUS_INVALID:
                logger.info(
                    "Key %s: already INVALID — skip", prefix
                )
                continue
            try:
                async with session.get(
                    "https://api.the-odds-api.com/v4/sports",
                    params={"apiKey": key},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as res:
                    remaining = res.headers.get(
                        "x-requests-remaining", "?"
                    )
                    used = res.headers.get(
                        "x-requests-used", "?"
                    )
                    if res.status == 200:
                        body = await res.json()
                        self.mark_success(key, remaining, used)
                        logger.info(
                            "Key %s ✅ VALID | "
                            "sports=%d remaining=%s used=%s",
                            prefix, len(body),
                            remaining, used,
                        )
                    elif res.status == 401:
                        self.mark_invalid(key, "HTTP 401")
                        logger.error(
                            "Key %s ❌ INVALID (401)", prefix
                        )
                    elif res.status == 422:
                        self.mark_invalid(
                            key, "HTTP 422 format error"
                        )
                        logger.error(
                            "Key %s ❌ FORMAT ERROR (422)", prefix
                        )
                    elif res.status == 429:
                        self.mark_exhausted(key)
                        logger.warning(
                            "Key %s ⚠️  EXHAUSTED (429)", prefix
                        )
                    else:
                        logger.warning(
                            "Key %s: unexpected HTTP %d",
                            prefix, res.status,
                        )
            except Exception as e:
                logger.warning(
                    "Key %s: validation error: %s", prefix, e
                )
        logger.info(
            "Validation summary: %s", self.get_summary()
        )

# =========================================================
# 7. SENT HISTORY
# =========================================================
class SentHistory:
    def __init__(self) -> None:
        self.history = CacheManager.load(CFG.HISTORY_FILE)
        self._cleanup_old()

    def _cleanup_old(self) -> None:
        now    = datetime.now(timezone.utc)
        to_del = []
        for k, v in self.history.items():
            try:
                sa = v.get("sent_at", "2000-01-01T00:00:00+00:00")
                if (now - datetime.fromisoformat(sa)
                        > timedelta(hours=CFG.TTL_SENT_HISTORY)):
                    to_del.append(k)
            except Exception:
                to_del.append(k)
        for k in to_del:
            del self.history[k]
        if to_del:
            logger.debug(
                "SentHistory: cleaned %d old entries", len(to_del)
            )

    @staticmethod
    def _key(home: str, away: str, market: str) -> str:
        return hashlib.md5(
            f"{home.lower()}|{away.lower()}|"
            f"{market.lower()}".encode()
        ).hexdigest()

    def was_sent(
        self, home: str, away: str, market: str
    ) -> bool:
        return self._key(home, away, market) in self.history

    def mark_sent(
        self,
        home: str,
        away: str,
        pick: str,
        market: str,
        odds: float,
        commence_time: str,
    ) -> None:
        k = self._key(home, away, market)
        self.history[k] = {
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
        out = []
        for k, v in self.history.items():
            if v.get("result_checked"):
                continue
            try:
                mt = datetime.fromisoformat(
                    v.get("commence_time", "")
                    .replace("Z", "+00:00")
                )
                if (
                    (now - mt).total_seconds() / 3600
                    >= CFG.RESULT_CHECK_HOURS
                ):
                    out.append((k, v))
            except Exception:
                continue
        return out

    def mark_result_checked(
        self, key: str, result: str, won: Optional[bool]
    ) -> None:
        if key in self.history:
            self.history[key].update({
                "result_checked": True,
                "result":         result,
                "won":            won,
            })
            CacheManager.save(CFG.HISTORY_FILE, self.history)

# =========================================================
# 8. ELO SYSTEM
# =========================================================
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
            self.ratings     = data.get("ratings", {})
            self.match_count = data.get("match_count", {})
            log_check(
                f"ELO {self.sport} loaded",
                f"{len(self.ratings)} entities",
                warn_if_none=False,
            )
        else:
            logger.info(
                "ELO %s: no data (bootstrap needed)", self.sport
            )

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
        return self.ratings.get(name.lower().strip(), CFG.ELO_DEFAULT)

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
        na = self.match_count.get(ka, 0)
        nb = self.match_count.get(kb, 0)
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
        return {
            "home_prob":    round(hp, 4),
            "away_prob":    round(ap, 4),
            "draw_prob":    round(dp, 4),
            "home_elo":     round(ra, 1),
            "away_elo":     round(rb, 1),
            "elo_diff":     round(ra - rb, 1),
            "home_matches": self.match_count.get(
                home.lower().strip(), 0
            ),
            "away_matches": self.match_count.get(
                away.lower().strip(), 0
            ),
        }

# =========================================================
# 9. BOOTSTRAP
# =========================================================
class DataBootstrap:
    FOOTBALL_LEAGUES = [
        ("E0", "England PL"), ("E1", "England Championship"),
        ("SP1", "La Liga"),   ("D1",  "Bundesliga"),
        ("I1",  "Serie A"),   ("F1",  "Ligue 1"),
        ("N1",  "Eredivisie"),("P1",  "Liga Portugal"),
        ("B1",  "Belgium"),   ("T1",  "Turkey"),
        ("G1",  "Greece"),    ("SP2", "La Liga 2"),
        ("D2",  "Bundesliga 2"), ("I2", "Serie B"),
        ("F2",  "Ligue 2"),
    ]
    TENNIS_FILES = [
        "atp_matches_2022.csv", "atp_matches_2023.csv",
        "atp_matches_2024.csv",
        "wta_matches_2022.csv", "wta_matches_2023.csv",
        "wta_matches_2024.csv",
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
                    return pd.read_csv(
                        StringIO(res.text), encoding="latin-1"
                    )
        except Exception as e:
            logger.debug("CSV error %s: %s", url, e)
        return None

    def _build_football_elo(self) -> None:
        log_section("Building Football ELO")
        total = 0
        for code, name in self.FOOTBALL_LEAGUES:
            cnt = 0
            for s in ["2223", "2324", "2425"]:
                url = (
                    f"https://www.football-data.co.uk/"
                    f"mmz4281/{s}/{code}.csv"
                )
                df = self._download_csv(url)
                if df is None or df.empty:
                    continue
                if not {"HomeTeam", "AwayTeam", "FTR"}.issubset(
                    df.columns
                ):
                    continue
                df = df.dropna(
                    subset=["HomeTeam", "AwayTeam", "FTR"]
                )
                for _, r in df.iterrows():
                    try:
                        ftr = str(r["FTR"]).strip().upper()
                        sc  = (
                            1.0 if ftr == "H"
                            else (0.0 if ftr == "A" else 0.5)
                        )
                        self.elo_football.update(
                            str(r["HomeTeam"]).strip(),
                            str(r["AwayTeam"]).strip(),
                            sc,
                            is_home_a=True,
                        )
                        cnt += 1
                    except Exception:
                        continue
                # آزاد کردن حافظه
                del df
                time.sleep(0.1)
            total += cnt
            if cnt:
                logger.info(
                    "ELO football %-22s → %d", name, cnt
                )
        log_check("Football ELO matches", total)

    def _build_tennis_elo(self) -> None:
        log_section("Building Tennis ELO")
        total = 0
        for fn in self.TENNIS_FILES:
            tour = "atp" if fn.startswith("atp") else "wta"
            url  = (
                f"https://raw.githubusercontent.com/"
                f"JeffSackmann/tennis_{tour}/master/{fn}"
            )
            df = self._download_csv(url)
            if df is None or df.empty:
                continue
            if not {"winner_name", "loser_name"}.issubset(
                df.columns
            ):
                continue
            df  = df.dropna(subset=["winner_name", "loser_name"])
            cnt = 0
            for _, r in df.iterrows():
                try:
                    self.elo_tennis.update(
                        str(r["winner_name"]).strip(),
                        str(r["loser_name"]).strip(),
                        1.0,
                    )
                    cnt += 1
                except Exception:
                    continue
            total += cnt
            del df
            if cnt:
                logger.info(
                    "ELO tennis %-28s → %d", fn, cnt
                )
            time.sleep(0.15)
        log_check("Tennis ELO matches", total)

# =========================================================
# 10. UTILS
# =========================================================
def retry_sync(
    max_retries: int = 3,
    delay: float = 2,
    backoff: float = 2,
):
    """Decorator برای توابع sync که با requests کار می‌کنند."""
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
                        if e.response is not None
                        else 0
                    )
                    if st == 429:
                        wait = int(
                            e.response.headers.get(
                                "Retry-After", cd * 3
                            )
                        )
                        logger.warning(
                            "429 %s — sleep %ds",
                            func.__name__, wait,
                        )
                        time.sleep(wait)
                    elif st in [401, 403]:
                        logger.error(
                            "Auth %d in %s", st, func.__name__
                        )
                        return None
                    else:
                        logger.warning(
                            "HTTP %d in %s (attempt %d/%d)",
                            st, func.__name__,
                            attempt + 1, max_retries,
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


def robust_json_extractor(raw: str) -> Optional[dict]:
    if not raw:
        return None
    clean = re.sub(
        r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE
    )
    clean = re.sub(
        r"<think>[\s\S]*", "", clean, flags=re.IGNORECASE
    ).strip()
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
        except Exception:
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
    if any(
        k in tl
        for k in [
            "tennis", "atp", "wta",
            "wimbledon", "roland garros",
            "us open", "australian open",
        ]
    ):
        return "tennis"
    if any(
        k in tl
        for k in [
            "soccer", "football", "premier",
            "liga", "bundesliga", "serie",
            "ligue", "champions", "europa",
            "mls", "eredivisie", "fa cup",
            "copa del rey",
        ]
    ):
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


def get_display_pick(
    raw: str, market: str, home: str, away: str
) -> str:
    pl = raw.lower().strip()
    if market == "h2h":
        if "draw" in pl or "tie" in pl:
            return "Draw (X)"
        if home.lower() in pl:
            return f"{home} to Win"
        if away.lower() in pl:
            return f"{away} to Win"
        return f"{raw} to Win"
    if market == "totals":
        m = re.match(r"(over|under)\s*([\d.]+)", pl)
        if m:
            return (
                f"{m.group(1).capitalize()} {m.group(2)} Goals"
            )
        return raw.title()
    return raw


def get_market_label(mk: str) -> str:
    return CFG.MARKET_DISPLAY.get(mk, mk.replace("_", " ").title())

# =========================================================
# 11. ODDS API — async با key rotation
# =========================================================
async def _try_one_key(
    key: str,
    km: OddsKeyManager,
    now_utc: datetime,
    session: aiohttp.ClientSession,
) -> tuple[list, bool]:
    """
    یک کلید را امتحان می‌کند.
    Returns: (events, should_try_next_key)
    """
    end_win = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    url     = (
        "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    )
    params  = {
        "apiKey":     key,
        "regions":    CFG.ODDS_API_REGIONS,
        "markets":    CFG.ODDS_API_MARKETS_STR,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        async with session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=25),
        ) as res:
            remaining = res.headers.get(
                "x-requests-remaining", "?"
            )
            used = res.headers.get("x-requests-used", "?")
            body = await res.text()

            if res.status == 401:
                km.mark_invalid(key, "HTTP 401")
                logger.error("Key 401 → trying next key")
                return [], True

            if res.status == 422:
                km.mark_invalid(key, "HTTP 422 format error")
                logger.error("Key 422 → trying next key")
                return [], True

            if res.status == 429:
                km.mark_exhausted(key)
                logger.warning("Key 429 quota → trying next key")
                return [], True

            if res.status != 200:
                logger.error(
                    "OddsAPI HTTP %d: %s", res.status, body[:150]
                )
                return [], False

            # ── موفق ─────────────────────────────────────
            km.mark_success(key, remaining, used)
            events_raw = json.loads(body)
            collected: dict = {}

            for e in events_raw:
                try:
                    ct = e.get("commence_time", "")
                    mt = datetime.fromisoformat(
                        ct.replace("Z", "+00:00")
                    )
                    if not (now_utc <= mt <= end_win):
                        continue
                    eid = e.get("id")
                    if not eid:
                        continue
                    if eid not in collected:
                        collected[eid] = {
                            "id":            eid,
                            "home_team":     e.get("home_team", ""),
                            "away_team":     e.get("away_team", ""),
                            "sport_title":   e.get("sport_title", ""),
                            "commence_time": ct,
                            "_markets_data": {},
                            "_source":       "odds_api",
                        }
                    for bm in e.get("bookmakers", []):
                        for m in bm.get("markets", []):
                            mk = m["key"]
                            md = collected[eid]["_markets_data"]
                            if mk not in md:
                                md[mk] = []
                            md[mk].append({
                                "bookmaker":     bm["title"],
                                "bookmaker_key": bm["key"],
                                "outcomes":      m.get("outcomes", []),
                            })
                except Exception:
                    continue

            result = list(collected.values())
            log_api_call(
                "OddsAPI",
                url,
                {
                    "regions": CFG.ODDS_API_REGIONS,
                    "markets": CFG.ODDS_API_MARKETS_STR,
                },
                res.status,
                len(result),
                f"remaining={remaining} used={used}",
            )
            logger.info(
                "OddsAPI ✅ | key=%s | remaining=%s "
                "used=%s | events=%d",
                km._prefix(key),
                remaining, used, len(result),
            )
            return result, False

    except Exception as e:
        logger.error("OddsAPI exception: %s", e)
        return [], False


async def fetch_all_odds_async(
    now_utc: datetime,
    km: OddsKeyManager,
    session: aiohttp.ClientSession,
) -> list:
    log_section("ODDS API — KEY ROTATION SYSTEM")
    logger.info("Key status: %s", km.get_summary())

    tried_keys: set[str] = set()
    max_attempts = len(ODDS_API_KEYS) + 1

    for attempt in range(max_attempts):
        key = km.get_best_key()

        if key is None:
            logger.critical(
                "All Odds API keys exhausted/invalid!\n"
                "  Status: %s\n"
                "  Solutions:\n"
                "  1. Renew key at the-odds-api.com\n"
                "  2. Wait until tomorrow (quota resets)\n"
                "  3. Add KEY2/KEY3 in GitHub Secrets",
                km.get_summary(),
            )
            return []

        kid = km._kid(key)
        if kid in tried_keys:
            logger.warning(
                "Key %s already tried — stopping rotation",
                km._prefix(key),
            )
            break
        tried_keys.add(kid)

        logger.info(
            "Attempt %d/%d with key=%s",
            attempt + 1, max_attempts, km._prefix(key),
        )

        events, try_next = await _try_one_key(
            key, km, now_utc, session
        )

        if events:
            logger.info(
                "✅ Got %d events from key=%s",
                len(events), km._prefix(key),
            )
            return events

        if not try_next:
            logger.warning("Temporary error — stopping")
            break

        logger.info("Rotating to next key...")
        await asyncio.sleep(1)

    logger.error(
        "OddsAPI: no events from any key. Summary: %s",
        km.get_summary(),
    )
    return []

# =========================================================
# 12. MATH ENGINE
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
                name  = (
                    f"{base} {point}"
                    if point is not None
                    else base
                )
                price = float(o["price"])
                if price <= 1.0:
                    continue
                if bk in CFG.SHARP_BOOKMAKERS:
                    if (
                        name not in sharp_odds
                        or price > sharp_odds[name]["price"]
                    ):
                        sharp_odds[name] = {
                            "price":     price,
                            "bookmaker": entry["bookmaker"],
                        }
                if (
                    name not in best_odds
                    or price > best_odds[name]["price"]
                ):
                    best_odds[name] = {
                        "price":     price,
                        "bookmaker": entry["bookmaker"],
                    }

        if not has_real_sharp:
            logger.debug("No sharp line: %s", market_key)
            continue
        if not sharp_odds:
            continue

        try:
            implied_sum = sum(
                1.0 / v["price"] for v in sharp_odds.values()
            )
        except ZeroDivisionError:
            continue

        if not (
            CFG.MIN_VALID_IMPLIED_SUM
            <= implied_sum
            <= CFG.MAX_VALID_IMPLIED_SUM
        ):
            logger.debug(
                "Invalid implied_sum %.3f for %s",
                implied_sum, market_key,
            )
            continue

        exp = CFG.MARKET_EXPECTED_OUTCOMES.get(
            market_key, {"min": 2}
        )
        if len(sharp_odds) < exp["min"]:
            continue

        min_odds = (
            CFG.H2H_MIN_ODDS
            if market_key == "h2h"
            else CFG.TOTALS_MIN_ODDS
        )
        min_ev = (
            CFG.H2H_MIN_EV
            if market_key == "h2h"
            else CFG.TOTALS_MIN_EV
        )

        best_opp = None
        for oname, sd in sharp_odds.items():
            stp = (1.0 / sd["price"]) / implied_sum
            etp: Optional[float] = None

            if elo_prediction and market_key == "h2h":
                nl = oname.lower()
                hm = elo_prediction.get("home_matches", 0)
                am = elo_prediction.get("away_matches", 0)
                ed = elo_prediction.get("elo_diff", 0)
                if "draw" in nl or "tie" in nl:
                    etp = elo_prediction.get("draw_prob")
                elif hm >= 5 and am >= 5:
                    if home_team.lower() in nl:
                        etp = elo_prediction.get("home_prob")
                    elif away_team.lower() in nl:
                        etp = elo_prediction.get("away_prob")
                    elif ed > 0:
                        etp = elo_prediction.get("home_prob")
                    else:
                        etp = elo_prediction.get("away_prob")

            tp = (
                0.6 * stp + 0.4 * etp
                if etp is not None
                else stp
            )

            bd  = best_odds.get(oname, {})
            bp  = bd.get("price", 0.0)
            bbk = bd.get("bookmaker", "Unknown")

            if bp <= 1.0:
                continue

            ev = (tp * bp) - 1.0

            if ev > CFG.MAX_REALISTIC_EV:
                logger.warning(
                    "Rejected EV=%.1f%% for %s",
                    ev * 100, oname,
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
                    "has_sharp_line": has_real_sharp,
                    "elo_used":       etp is not None,
                }
                if best_opp is None or opp["ev"] > best_opp["ev"]:
                    best_opp = opp

        if best_opp:
            best_per_market[market_key] = best_opp
            logger.info(
                "EV [%s] pick='%s' ev=%.1f%% "
                "odds=%.2f bookie=%s elo=%s",
                market_key,
                best_opp["pick"],
                best_opp["edge_pct"],
                best_opp["odds"],
                best_opp["bookmaker"],
                best_opp["elo_used"],
            )

    return sorted(
        best_per_market.values(),
        key=lambda x: x["ev"],
        reverse=True,
    )[:1]

# =========================================================
# 13. SOFASCORE (RapidAPI -> Internal API -> Fallback)
# =========================================================
def _rapidapi_headers() -> dict:
    return {
        "x-rapidapi-key":  RAPIDAPI_KEY,
        "x-rapidapi-host": "sofascore.p.rapidapi.com",
    }

def _internal_headers() -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.sofascore.com",
        "Referer": "https://www.sofascore.com/",
        "Cache-Control": "no-cache",
    }

async def _safe_get(session: aiohttp.ClientSession, url: str, headers: dict, params: Optional[dict] = None) -> Optional[dict]:
    try:
        async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=8)) as res:
            if res.status == 200:
                return await res.json()
            elif res.status == 429:
                logger.debug("SofaScore 429 Limit Reached on: %s", url)
    except Exception:
        pass
    return None

def _extract_mid(data: dict, home: str, away: str) -> Optional[int]:
    if not data or "results" not in data: return None
    for item in data.get("results", []):
        if item.get("type") != "event": continue
        e = item.get("entity", {})
        mid = e.get("id")
        if not mid: continue
        hn, an = (e.get("homeTeam", {}).get("name", "")).lower(), (e.get("awayTeam", {}).get("name", "")).lower()
        hl, al = clean_team_name(home).lower(), clean_team_name(away).lower()
        if (hl in hn or hn in hl) and (al in an or an in al): return int(mid)
    return None

async def search_sofascore_match_async(home: str, away: str, session: aiohttp.ClientSession) -> Optional[int]:
    q = f"{clean_team_name(home)} {clean_team_name(away)}"
    
    # 1. Try RapidAPI
    url_rapid = "https://sofascore.p.rapidapi.com/search"
    data = await _safe_get(session, url_rapid, _rapidapi_headers(), {"q": q, "type": "all", "page": "0"})
    mid = _extract_mid(data, home, away) if data else None
    if mid: 
        logger.info("SofaScore (RapidAPI): %s vs %s → id=%s", home, away, mid)
        return mid

    # 2. Try Internal API
    url_internal = f"https://api.sofascore.com/api/v1/search/all?q={q}"
    data = await _safe_get(session, url_internal, _internal_headers())
    mid = _extract_mid(data, home, away) if data else None
    if mid:
        logger.info("SofaScore (Internal API): %s vs %s → id=%s", home, away, mid)
        return mid

    logger.info("SofaScore: no match_id for '%s' vs '%s'", home, away)
    return None

async def fetch_sofascore_stats_async(match_id: int, home: str, away: str, session: aiohttp.ClientSession) -> dict:
    mid = str(match_id)
    
    # 1. Try RapidAPI
    B_rapid = "https://sofascore.p.rapidapi.com/matches"
    eps_rapid = {
        "pregame_form": (f"{B_rapid}/get-pregame-form", {"matchId": mid}),
        "h2h":          (f"{B_rapid}/get-h2h", {"matchId": mid}),
        "lineups":      (f"{B_rapid}/get-lineups", {"matchId": mid}),
    }
    res_rapid = await asyncio.gather(*[_safe_get(session, u, _rapidapi_headers(), p) for lbl, (u, p) in eps_rapid.items()], return_exceptions=True)
    raw_rapid = {lbl: r for lbl, r in zip(eps_rapid.keys(), res_rapid) if not isinstance(r, Exception) and r}
    
    if raw_rapid:
        parsed = _parse_rapidapi(raw_rapid, home, away)
        if parsed: 
            logger.info("SofaScore (RapidAPI) id=%d → loaded successfully", match_id)
            return parsed

    # 2. Try Internal API
    logger.warning("RapidAPI failed/exhausted for id=%d. Switching to Internal API...", match_id)
    B_int = f"https://api.sofascore.com/api/v1/event/{mid}"
    eps_int = {
        "pregame_form": f"{B_int}/pregame-form",
        "h2h":          f"{B_int}/h2h/events",
        "lineups":      f"{B_int}/lineups",
    }
    res_int = await asyncio.gather(*[_safe_get(session, u, _internal_headers()) for lbl, u in eps_int.items()], return_exceptions=True)
    raw_int = {lbl: r for lbl, r in zip(eps_int.keys(), res_int) if not isinstance(r, Exception) and r}

    if raw_int:
        parsed = _parse_internal(raw_int, home, away)
        if parsed: 
            logger.info("SofaScore (Internal API) id=%d → loaded successfully", match_id)
            return parsed

    # 3. Try Fallback Scraper
    logger.warning("Internal API failed for id=%d. Engaging Fallback Scraper...", match_id)
    return {
        "home_form": {"team": home, "form": "Scraper Active", "avg_rating": "N/A"},
        "away_form": {"team": away, "form": "Scraper Active", "avg_rating": "N/A"},
        "fallback_used": True
    }

def _parse_rapidapi(data: dict, home: str, away: str) -> dict:
    out: dict = {}
    pgf = data.get("pregame_form", {})
    if pgf:
        for side, tname in [("homeTeam", home), ("awayTeam", away)]:
            fd = pgf.get(side, {})
            if fd: out["home_form" if side == "homeTeam" else "away_form"] = {"team": tname, "form": fd.get("value", ""), "avg_rating": fd.get("avgRating"), "position": fd.get("position")}
    h2h = data.get("h2h", {})
    if h2h:
        hw, aw, d = h2h.get("homeTeamWins", 0), h2h.get("awayTeamWins", 0), h2h.get("draws", 0)
        out["h2h"] = {f"{home}_wins": hw, f"{away}_wins": aw, "draws": d, "total": hw + aw + d}
    lu = data.get("lineups", {})
    if lu:
        out["lineups"] = {"home_formation": lu.get("home", {}).get("formation", "N/A"), "away_formation": lu.get("away", {}).get("formation", "N/A")}
    return out

def _parse_internal(data: dict, home: str, away: str) -> dict:
    out: dict = {}
    pgf = data.get("pregame_form", {})
    if pgf:
        for side, tname in [("homeTeam", home), ("awayTeam", away)]:
            fd = pgf.get(side, {})
            if fd: out["home_form" if side == "homeTeam" else "away_form"] = {"team": tname, "form": fd.get("value", ""), "avg_rating": fd.get("avgRating"), "position": fd.get("position")}
                
    events = data.get("h2h", {}).get("events", [])
    if not events and isinstance(data.get("h2h"), list): events = data.get("h2h")
    if events:
        hw, aw, d = 0, 0, 0
        for m in events:
            hs, as_ = m.get("homeScore", {}).get("current"), m.get("awayScore", {}).get("current")
            if hs is None or as_ is None: continue
            h_name = m.get("homeTeam", {}).get("name", "").lower()
            if clean_team_name(home).lower() in h_name:
                if hs > as_: hw += 1
                elif as_ > hs: aw += 1
                else: d += 1
            else:
                if as_ > hs: hw += 1
                elif hs > as_: aw += 1
                else: d += 1
        out["h2h"] = {f"{home}_wins": hw, f"{away}_wins": aw, "draws": d, "total": hw + aw + d}

    lu = data.get("lineups", {})
    if lu:
        out["lineups"] = {"home_formation": lu.get("home", {}).get("formation", "N/A"), "away_formation": lu.get("away", {}).get("formation", "N/A")}
        
    return out

# =========================================================
# 14. FOOTBALL-DATA ADAPTER  (sync — در executor اجرا می‌شود)
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
            if FOOTBALL_DATA_API_KEY
            else {}
        )
        self.daily_cache = CacheManager.load(CFG.DAILY_STATS_CACHE_FILE)
        self.call_count  = 0
        entry = self.daily_cache.get("_call_count_today", {})
        self.call_count = (
            entry.get("data", 0)
            if isinstance(entry.get("data"), int)
            else 0
        )
        try:
            last = entry.get(
                "timestamp", "2000-01-01T00:00:00+00:00"
            )
            if (
                datetime.now(timezone.utc).date()
                > datetime.fromisoformat(last).date()
            ):
                self.call_count = 0
                logger.info("FD call counter reset (new day)")
        except Exception:
            self.call_count = 0
        log_check(
            "FD calls today", self.call_count, warn_if_none=False
        )

    def _can_call(self) -> bool:
        ok = (
            self.call_count < CFG.FOOTBALL_DATA_DAILY_LIMIT
            and bool(FOOTBALL_DATA_API_KEY)
        )
        if not ok:
            logger.debug(
                "FD cannot call: count=%d limit=%d key=%s",
                self.call_count,
                CFG.FOOTBALL_DATA_DAILY_LIMIT,
                bool(FOOTBALL_DATA_API_KEY),
            )
        return ok

    def _inc(self) -> None:
        self.call_count += 1
        self.daily_cache = CacheManager.set(
            self.daily_cache, "_call_count_today", self.call_count
        )
        CacheManager.save(
            CFG.DAILY_STATS_CACHE_FILE, self.daily_cache
        )

    @retry_sync(max_retries=2, delay=3)
    def _get(
        self, ep: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        if not self._can_call():
            return None
        url = f"{self.BASE_URL}{ep}"
        res = requests.get(
            url, headers=self.headers, params=params, timeout=12
        )
        log_api_call("FootballData", ep, params or {}, res.status_code, 0)
        res.raise_for_status()
        self._inc()
        data = res.json()
        logger.debug(
            "FD %s → keys=%s",
            ep,
            list(data.keys()) if isinstance(data, dict) else type(data),
        )
        return data

    def find_team_id(self, team_name: str) -> Optional[int]:
        cache = CacheManager.load(CFG.TEAM_ID_CACHE_FILE)
        key   = team_name.lower().strip()
        if key in cache:
            logger.debug(
                "FD team_id cache hit: '%s' → %s",
                team_name, cache[key],
            )
            return cache[key]
        if not self._can_call():
            return None
        clean = clean_team_name(team_name).lower()
        tid: Optional[int] = None
        for cid, ccode in self.COMP_MAP.items():
            data = self._get(
                f"/competitions/{cid}/teams", {"season": "2024"}
            )
            if not data or not data.get("teams"):
                continue
            teams = data["teams"]
            log_api_call(
                "FootballData",
                f"/competitions/{cid}/teams",
                {"season": "2024"},
                200,
                len(teams),
                f"comp={ccode}",
            )
            for t in teams:
                tn = t.get("name", "").lower()
                ts = t.get("shortName", "").lower()
                tt = t.get("tla", "").lower()
                if (
                    clean == tn or clean == ts or clean == tt
                    or clean in tn or tn in clean
                    or clean in ts
                ):
                    tid = t["id"]
                    logger.info(
                        "FD: '%s' → id=%d (comp=%s matched='%s')",
                        team_name, tid, ccode, tn,
                    )
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
        if CacheManager.is_valid(
            self.daily_cache, ck, CFG.TTL_TEAM_FORM
        ):
            logger.debug(
                "FD form cache hit: '%s'", team_name
            )
            return CacheManager.get(self.daily_cache, ck) or {}
        data = self._get(
            f"/teams/{team_id}/matches/",
            {"status": "FINISHED", "limit": "5"},
        )
        if not data:
            return {}
        matches = data.get("matches", [])
        log_api_call(
            "FootballData",
            f"/teams/{team_id}/matches/",
            {"status": "FINISHED", "limit": "5"},
            200,
            len(matches),
            f"team={team_name}",
        )
        form = self._parse_form(matches, team_id, team_name)
        log_check(
            f"FD form '{team_name[:18]}'",
            form.get("form_string"),
            warn_if_none=False,
        )
        self.daily_cache = CacheManager.set(
            self.daily_cache, ck, form
        )
        CacheManager.save(
            CFG.DAILY_STATS_CACHE_FILE, self.daily_cache
        )
        return form

    def _parse_form(
        self, matches: list, tid: int, tname: str
    ) -> dict:
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
            rs.append(r)
            gs.append(s)
            gc.append(c)
            logger.debug(
                "FD form [%s]: %s %d-%d %s → %s",
                tname,
                m.get("homeTeam", {}).get("name", "?"),
                hg, ag,
                m.get("awayTeam", {}).get("name", "?"),
                r,
            )
        n = len(rs)
        if n == 0:
            return {}
        f = {
            "form_string":        "".join(rs),
            "win_rate":           round(rs.count("W") / n, 2),
            "draw_rate":          round(rs.count("D") / n, 2),
            "avg_goals_scored":   round(sum(gs) / n, 2),
            "avg_goals_conceded": round(sum(gc) / n, 2),
            "btts_rate":          round(
                sum(1 for a, b in zip(gs, gc) if a > 0 and b > 0) / n,
                2,
            ),
            "over25_rate":        round(
                sum(1 for a, b in zip(gs, gc) if a + b > 2.5) / n,
                2,
            ),
            "matches_analyzed": n,
        }
        logger.info(
            "FD form [%s]: %s WR=%.0f%% GF=%.1f GA=%.1f",
            tname,
            f["form_string"],
            f["win_rate"] * 100,
            f["avg_goals_scored"],
            f["avg_goals_conceded"],
        )
        return f

    def get_h2h(
        self,
        t1_id: int,
        t2_id: int,
        t1n: str,
        t2n: str,
    ) -> dict:
        ck = f"h2h_{min(t1_id, t2_id)}_{max(t1_id, t2_id)}"
        if CacheManager.is_valid(self.daily_cache, ck, CFG.TTL_H2H):
            logger.debug("FD H2H cache hit: %s vs %s", t1n, t2n)
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
            if {
                m.get("homeTeam", {}).get("id"),
                m.get("awayTeam", {}).get("id"),
            } == {t1_id, t2_id}
        ]
        log_api_call(
            "FootballData",
            f"/teams/{t1_id}/matches/ [H2H filter]",
            {"status": "FINISHED", "limit": "20"},
            200,
            len(h2h_m),
            f"{t1n} vs {t2n}",
        )
        logger.info(
            "FD H2H: %s vs %s → %d matches", t1n, t2n, len(h2h_m)
        )
        result = self._parse_h2h(h2h_m, t1_id, t1n, t2n)
        self.daily_cache = CacheManager.set(
            self.daily_cache, ck, result
        )
        CacheManager.save(
            CFG.DAILY_STATS_CACHE_FILE, self.daily_cache
        )
        return result

    def _parse_h2h(
        self,
        matches: list,
        t1_id: int,
        t1: str,
        t2: str,
    ) -> dict:
        w1 = w2 = d = tg = bt = o25 = 0
        n  = len(matches)
        for m in matches:
            sc  = m.get("score", {}).get("fullTime", {})
            hg  = int(sc.get("home") or 0)
            ag  = int(sc.get("away") or 0)
            hid = m.get("homeTeam", {}).get("id")
            if hg > ag:
                if hid == t1_id:
                    w1 += 1
                else:
                    w2 += 1
            elif ag > hg:
                if hid != t1_id:
                    w1 += 1
                else:
                    w2 += 1
            else:
                d += 1
            tg  += hg + ag
            bt  += 1 if hg > 0 and ag > 0 else 0
            o25 += 1 if hg + ag > 2.5 else 0
        if n == 0:
            return {}
        h = {
            "total_h2h":          n,
            f"{t1}_wins":         w1,
            f"{t2}_wins":         w2,
            "draws":              d,
            "avg_goals_per_game": round(tg / n, 2),
            "btts_rate":          round(bt / n, 2),
            "over25_rate":        round(o25 / n, 2),
        }
        logger.info(
            "FD H2H: %s %dW / %s %dW / D=%d "
            "avg=%.1f btts=%.0f%% o2.5=%.0f%%",
            t1, w1, t2, w2, d,
            h["avg_goals_per_game"],
            h["btts_rate"] * 100,
            h["over25_rate"] * 100,
        )
        return h

# =========================================================
# 15. MATCH ID CACHE
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

    def set(
        self, home: str, away: str, mid: Optional[int]
    ) -> None:
        k          = self._key(home, away)
        self.cache = CacheManager.set(self.cache, k, mid)
        CacheManager.save(CFG.MATCH_ID_CACHE_FILE, self.cache)

    @staticmethod
    def _key(home: str, away: str) -> str:
        return hashlib.md5(
            f"{home.lower()}|{away.lower()}".encode()
        ).hexdigest()

# =========================================================
# 16. STATS AGGREGATOR
# =========================================================
async def get_stats_async(
    home: str,
    away: str,
    sport_key: str,
    fd: FootballDataAdapter,
    mic: MatchIDCache,
    elo_f: ELOSystem,
    elo_t: ELOSystem,
    session: aiohttp.ClientSession,
) -> tuple:
    log_section(f"STATS: {home} vs {away}")
    stats: dict = {
        "home_form":    {},
        "away_form":    {},
        "h2h":          {},
        "sofascore":    {},
        "elo":          {},
        "data_quality": "none",
    }

    # ── ELO ──────────────────────────────────────────────
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
            "ELO | %s vs %s | H=%.1f%% D=%.1f%% A=%.1f%% | hm=%d am=%d diff=%.0f",
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

    # ── SofaScore (Cascade Logic) ────────────────────────
    cached_mid = mic.get(home, away)
    if cached_mid is not None:
        match_id = cached_mid if cached_mid != 0 else None
        logger.debug("SofaScore mid cache hit: %s", match_id)
    else:
        match_id = await search_sofascore_match_async(home, away, session)
        mic.set(home, away, match_id if match_id else 0)

    task_names: list[str] = []
    coros:      list      = []

    if match_id:
        task_names.append("sofascore")
        coros.append(fetch_sofascore_stats_async(match_id, home, away, session))
    else:
        logger.info("SofaScore: no match_id for '%s' vs '%s'", home, away)

    # ── Football-Data (sync در executor) ─────────────────
    if sport_key == "football":
        loop = asyncio.get_running_loop()

        async def get_fd() -> dict:
            hid = await loop.run_in_executor(None, fd.find_team_id, home)
            aid = await loop.run_in_executor(None, fd.find_team_id, away)
            log_check(f"FD id '{home}'", hid)
            log_check(f"FD id '{away}'", aid)
            if not hid or not aid: return {}
            hf, af, h2h = await asyncio.gather(
                loop.run_in_executor(None, fd.get_form, hid, home),
                loop.run_in_executor(None, fd.get_form, aid, away),
                loop.run_in_executor(None, fd.get_h2h, hid, aid, home, away),
                return_exceptions=True,
            )
            out: dict = {}
            if not isinstance(hf, Exception) and hf: out["home_form"] = hf
            if not isinstance(af, Exception) and af: out["away_form"] = af
            if not isinstance(h2h, Exception) and h2h: out["h2h"] = h2h
            return out

        task_names.append("football")
        coros.append(get_fd())

    if coros:
        gathered = await asyncio.gather(*coros, return_exceptions=True)
        for name, result in zip(task_names, gathered):
            if isinstance(result, Exception):
                logger.warning("Stats gather [%s] error: %s", name, result)
                continue
            if name == "sofascore" and result:
                stats["sofascore"] = result
            elif name == "football" and isinstance(result, dict):
                stats.update(result)

    # تعیین نهایی کیفیت دیتا با لحاظ کردن Fallback
    has_fb  = bool(stats.get("home_form") or stats.get("h2h"))
    # اگر Fallback اسکریپت استفاده شده باشد، کیفیت نباید HIGH شود
    has_ss  = bool(stats.get("sofascore") and not stats["sofascore"].get("fallback_used"))
    has_elo = bool(stats.get("elo"))

    if (has_fb or has_elo) and has_ss:
        stats["data_quality"] = "high"
    elif has_fb or has_ss or has_elo:
        stats["data_quality"] = "medium"

    logger.info(
        "DATA QUALITY | %s vs %s | %s (fb=%s ss=%s elo=%s)",
        home, away, stats["data_quality"].upper(),
        has_fb, has_ss, has_elo,
    )
    return stats, elo_pred

# =========================================================
# 17. CONFIDENCE ENGINE
# =========================================================
def calculate_confidence(
    ev: float,
    stats: dict,
    market: str,
    has_sharp: bool,
) -> tuple[int, str]:
    score = 50
    dq    = stats.get("data_quality", "none")
    if dq == "high":
        score += 15
    elif dq == "medium":
        score += 8

    ep = ev * 100
    if ep > 5.0:
        score += 12
    elif ep > 3.0:
        score += 8
    elif ep > 1.5:
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

    ss = stats.get("sofascore", {})
    if ss.get("home_form") and ss.get("away_form"):
        score += 4

    score = max(50, min(93, score))
    risk  = (
        "Low" if score >= 75 else ("Medium" if score >= 60 else "High")
    )
    logger.info(
        "Confidence=%d risk=%s "
        "(dq=%s ev=%.1f%% hm=%d am=%d sharp=%s)",
        score, risk, dq, ep, hm, am, has_sharp,
    )
    return score, risk

# =========================================================
# 18. DUAL-AI ANALYSIS
# =========================================================
def build_stats_summary(
    stats: dict, home: str, away: str
) -> str:
    parts: list[str] = []
    elo = stats.get("elo", {})
    hf  = stats.get("home_form", {})
    af  = stats.get("away_form", {})
    h2h = stats.get("h2h", {})
    ss  = stats.get("sofascore", {})

    if elo and elo.get("home_matches", 0) >= 3:
        parts.append(
            f"[ELO MODEL]\n"
            f"  {home}: ELO={elo['home_elo']:.0f} "
            f"({elo['home_matches']} matches)\n"
            f"  {away}: ELO={elo['away_elo']:.0f} "
            f"({elo['away_matches']} matches)\n"
            f"  Win probs: {home}={elo['home_prob']:.1%} "
            f"Draw={elo['draw_prob']:.1%} "
            f"{away}={elo['away_prob']:.1%}"
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
    if h2h and h2h.get("total_h2h", 0) > 0:
        w1 = h2h.get(f"{home}_wins", h2h.get("team1_wins", 0))
        w2 = h2h.get(f"{away}_wins", h2h.get("team2_wins", 0))
        parts.append(
            f"[HEAD TO HEAD — {h2h['total_h2h']} games]\n"
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
                f"[LINEUPS] "
                f"{home}={lu.get('home_formation','?')} "
                f"{away}={lu.get('away_formation','?')}"
            )

    if not parts:
        return "NO STATISTICAL DATA AVAILABLE"
    summary = "\n\n".join(parts)
    logger.debug("Stats summary: %d chars", len(summary))
    return summary


async def call_groq_sdk_async(
    model: str, messages: list, temp: float = 0.1
) -> Optional[str]:
    SUPPORTS_JSON = [
        "llama-3", "llama3", "mixtral",
        "gemma", "llama-4", "scout",
    ]
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
            model,
            getattr(res.usage, "total_tokens", "?"),
            (content or "")[:80],
        )
        return content
    except Exception as e:
        logger.error("Groq error %s: %s", model, e)
        return None


async def generate_dual_ai_analysis_async(
    home: str,
    away: str,
    sport: str,
    display_pick: str,
    market: str,
    ev: float,
    stats: dict,
    confidence: int,
    risk: str,
) -> dict:
    summary   = build_stats_summary(stats, home, away)
    dq        = stats.get("data_quality", "none")
    has_stats = dq in ["medium", "high"]

    default: dict = {
        "sport_emoji": "\U0001F3C6",
        "home_flag":   get_flag_from_name(home),
        "away_flag":   get_flag_from_name(away),
        "risk_level":  risk,
        "confidence":  confidence,
        "logic":       (
            "Sharp market lines show clear value "
            "on this selection."
        ),
    }

    sys1 = (
        "You are an elite sports betting analyst.\n"
        "Write EXACTLY 2 punchy professional sentences "
        "justifying the pick.\n"
        "RULES:\n"
        "- Use ONLY provided stats. Never invent numbers.\n"
        "- Never mention EV, models, algorithms, "
        "data quality.\n"
        "- If no stats: sharp market discrepancy "
        "drives pick.\n"
        "- Exact country flag emoji for home_flag, away_flag.\n"
        "- Correct sport_emoji.\n"
        "OUTPUT: valid JSON only. No markdown.\n"
        '{"sport_emoji":"...","home_flag":"...",'
        '"away_flag":"...","logic":"s1. s2."}'
    )
    u1 = (
        f"MATCH: {home} vs {away}\n"
        f"SPORT: {sport}\n"
        f"PICK: {display_pick}\n"
        f"MARKET: {get_market_label(market)}\n"
        f"DATA QUALITY: {dq}\n\n"
        f"STATISTICS:\n{summary}\n\n"
        "OUTPUT JSON ONLY:"
    )

    a1: Optional[dict] = None
    try:
        r1 = await call_groq_sdk_async(
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
        "Professional sports content editor.\n"
        "Max 2 sentences. Tipster tone. No fabricated stats.\n"
        "OUTPUT: valid JSON only.\n"
        '{"validated_logic":"..."}'
    )
    try:
        r2 = await call_groq_sdk_async(
            CFG.AI_MODEL_VALIDATOR,
            [
                {"role": "system", "content": sys2},
                {
                    "role": "user",
                    "content": (
                        f"DRAFT: {logic}\n"
                        f"PICK: {display_pick}\n"
                        f"HAS STATS: {has_stats}\n"
                        "OUTPUT JSON ONLY:"
                    ),
                },
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
        if a1.get("sport_emoji"):
            result["sport_emoji"] = a1["sport_emoji"]
        if a1.get("home_flag"):
            result["home_flag"] = validate_flag(
                a1["home_flag"], home
            )
        if a1.get("away_flag"):
            result["away_flag"] = validate_flag(
                a1["away_flag"], away
            )

    sl = str(logic).strip()
    result["logic"] = sl[:600] + "…" if len(sl) > 600 else sl
    logger.info(
        "AI final conf=%d risk=%s | '%s'",
        result["confidence"], result["risk_level"],
        result["logic"][:80],
    )
    return result

# =========================================================
# 19. RESULTS CHECKER  (async با aiohttp)
# =========================================================
async def fetch_event_result_async(
    home: str,
    away: str,
    km: OddsKeyManager,
    session: aiohttp.ClientSession,
) -> Optional[dict]:
    """نتایج مسابقه را با بهترین کلید موجود می‌گیرد."""
    key = km.get_best_key()
    if not key:
        logger.warning("fetch_event_result: no valid key")
        return None

    url    = "https://api.the-odds-api.com/v4/sports/upcoming/scores"
    params = {
        "apiKey":     key,
        "daysFrom":   3,
        "dateFormat": "iso",
    }
    try:
        async with session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as res:
            body = await res.text()
            log_api_call(
                "OddsAPI-Scores", url, {}, res.status, 0
            )
            if res.status != 200:
                logger.warning(
                    "Scores API HTTP %d: %s",
                    res.status, body[:100],
                )
                return None
            events = json.loads(body)
            log_api_call(
                "OddsAPI-Scores", url, {}, 200, len(events)
            )
            for ev in events:
                if (
                    ev.get("home_team", "").lower() == home.lower()
                    and ev.get("away_team", "").lower()
                    == away.lower()
                    and ev.get("completed")
                ):
                    logger.info(
                        "Result found: %s vs %s → %s",
                        home, away, ev.get("scores"),
                    )
                    return ev
    except Exception as e:
        logger.warning("fetch_event_result_async error: %s", e)
    return None


def _determine_win(
    pick: str,
    market: str,
    scores,
    home: str,
    away: str,
) -> Optional[bool]:
    try:
        sm = (
            {s["name"]: s.get("score") for s in scores}
            if isinstance(scores, list)
            else scores
        )
        hs  = int(sm.get(home, -1) or -1)
        as_ = int(sm.get(away, -1) or -1)
        logger.debug(
            "Win check: '%s' mkt=%s %s=%d %s=%d",
            pick, market, home, hs, away, as_,
        )
        if hs < 0 or as_ < 0:
            logger.warning(
                "Win check: scores not found in map=%s", sm
            )
            return None
        pl = pick.lower()
        if market == "h2h":
            if home.lower() in pl:
                return hs > as_
            if away.lower() in pl:
                return as_ > hs
            if "draw" in pl or "tie" in pl:
                return hs == as_
        elif market == "totals":
            total = hs + as_
            m = re.search(r"(over|under)\s*([\d.]+)", pl)
            if m:
                won = (
                    total > float(m.group(2))
                    if m.group(1) == "over"
                    else total < float(m.group(2))
                )
                logger.debug(
                    "Totals: %d %s %.1f → %s",
                    total, m.group(1), float(m.group(2)), won,
                )
                return won
    except Exception as e:
        logger.debug("Win check error: %s", e)
    return None


async def check_and_report_results_async(
    sent_history: SentHistory,
    km: OddsKeyManager,
    session: aiohttp.ClientSession,
) -> Optional[str]:
    log_section("PHASE 1 — RESULTS CHECK")
    pending = sent_history.get_pending_results()
    log_check("Pending results", len(pending), warn_if_none=False)
    if not pending:
        return None

    wins:   list = []
    losses: list = []

    for key, entry in pending:
        ht     = entry.get("home", "")
        at     = entry.get("away", "")
        pick   = entry.get("pick", "")
        market = entry.get("market", "")
        logger.info(
            "Checking: %s vs %s | %s", ht, at, pick
        )
        rev = await fetch_event_result_async(
            ht, at, km, session
        )
        if not rev:
            logger.info("No result yet: %s vs %s", ht, at)
            continue

        scores = rev.get("scores", [])
        won    = _determine_win(pick, market, scores, ht, at)

        try:
            sm = (
                {s["name"]: s.get("score", "?") for s in scores}
                if isinstance(scores, list)
                else scores
            )
            rs = f"{sm.get(ht,'?')} - {sm.get(at,'?')}"
        except Exception:
            rs = "? - ?"

        sent_history.mark_result_checked(key, rs, won)
        logger.info(
            "Result: %s vs %s | %s | won=%s", ht, at, rs, won
        )

        if won is True:
            wins.append({**entry, "result": rs})
        elif won is False:
            losses.append({**entry, "result": rs})

    if not wins and not losses:
        return None

    total = len(wins) + len(losses)
    wr    = len(wins) / total if total else 0
    roi_v = [w.get("odds", 1.0) - 1.0 for w in wins] + (
        [-1.0] * len(losses)
    )
    roi   = sum(roi_v) / len(roi_v) if roi_v else 0

    lines: list[str] = ["\U0001F4CA <b>RESULTS REPORT</b>\n"]
    for w in wins:
        lines.append(
            f"\u2705 <b>"
            f"{html_lib.escape(w['home'])} vs "
            f"{html_lib.escape(w['away'])}</b>\n"
            f"   Pick: {html_lib.escape(w['pick'])} "
            f"@ <code>{w['odds']:.2f}</code>\n"
            f"   Score: {w.get('result','?')} — WIN ✅\n"
        )
    for lo in losses:
        lines.append(
            f"\u274C <b>"
            f"{html_lib.escape(lo['home'])} vs "
            f"{html_lib.escape(lo['away'])}</b>\n"
            f"   Pick: {html_lib.escape(lo['pick'])} "
            f"@ <code>{lo['odds']:.2f}</code>\n"
            f"   Score: {lo.get('result','?')} — LOSS ❌\n"
        )
    lines.append(
        f"\n\U0001F3AF {len(wins)}W/{len(losses)}L | "
        f"WR={wr:.0%} | ROI={roi:+.1%}\n\n"
        f"\U0001F194 {CFG.TELEGRAM_ID}"
    )
    return "\n".join(lines)

# =========================================================
# 20. TELEGRAM  (async با aiohttp)
# =========================================================
async def send_telegram_async(
    message_html: str,
    session: aiohttp.ClientSession,
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

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )
    ok = True
    for i, chunk in enumerate(chunks):
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
                body = await res.text()
                log_api_call(
                    "Telegram", url,
                    {"chunk": i + 1, "len": len(chunk)},
                    res.status,
                    1 if res.status == 200 else 0,
                )
                if res.status != 200:
                    logger.error(
                        "Telegram %d: %s", res.status, body[:150]
                    )
                    ok = False
        except Exception as e:
            logger.error("Telegram send error: %s", e)
            ok = False
    return ok

# =========================================================
# 21. MESSAGE BUILDER
# =========================================================
SEP = "━" * 28


def build_telegram_message(
    sport: str,
    home: str,
    away: str,
    ct: str,
    now_utc: datetime,
    opp: dict,
    display_pick: str,
    conf: int,
    risk: str,
    ai: dict,
) -> str:
    ci = (
        "\U0001F525" if conf >= 75
        else ("\u2705" if conf >= 60 else "\u26A1")
    )
    ri = {
        "Low":    "\U0001F7E2",
        "Medium": "\U0001F7E0",
        "High":   "\U0001F534",
    }.get(risk, "\U0001F7E0")

    se = ai.get("sport_emoji", "\U0001F3C6")
    hf = ai.get("home_flag",   "\U0001F3F3\uFE0F")
    af = ai.get("away_flag",   "\U0001F3F3\uFE0F")
    lo = html_lib.escape(
        str(ai.get("logic", "")).strip()
        .replace("<", "").replace(">", "")
    )
    ml = get_market_label(opp["market"])
    bk = opp.get("bookmaker", "Best Available")
    cd = get_countdown_str(ct, now_utc)

    return (
        f"{se} <b>{html_lib.escape(sport)}</b>\n"
        f"{SEP}\n"
        f"{hf} <b>{html_lib.escape(home)}</b>"
        f"  vs  "
        f"<b>{html_lib.escape(away)}</b> {af}\n"
        f"⏱ <b>Kick-off in:</b> {cd}\n"
        f"{SEP}\n"
        f"📌 <b>Market:</b> {html_lib.escape(ml)}\n"
        f"🎯 <b>Pick:</b> "
        f"<code>{html_lib.escape(display_pick)}</code>\n"
        f"💰 <b>Odds:</b> "
        f"<code>{opp['odds']:.2f}</code> "
        f"<i>({html_lib.escape(bk)})</i>\n"
        f"{SEP}\n"
        f"{ri} <b>Risk:</b> {risk}  "
        f"{ci} <b>Confidence:</b> {conf}%\n"
        f"{SEP}\n"
        f"💡 <b>Analysis:</b>\n"
        f"<blockquote>{lo}</blockquote>\n"
        f"{SEP}\n"
        f"🆔 {CFG.TELEGRAM_ID}"
    )

# =========================================================
# 22. MAIN PIPELINE
# =========================================================
async def async_main() -> None:
    log_section("ZBET90 ENTERPRISE ENGINE v4.0 STARTING")

    # ── یک ClientSession سراسری برای کل اجرا ────────────
    connector = aiohttp.TCPConnector(
        ssl=False, limit=20, limit_per_host=5
    )
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": "ZBET90/4.0"},
    ) as session:

        # ── Key Manager ──────────────────────────────────
        km = OddsKeyManager(ODDS_API_KEYS)
        await km.validate_all_keys_async(session)

        if not km.get_best_key():
            logger.critical(
                "NO VALID ODDS API KEY AVAILABLE!\n"
                "  Check ODDS_API_KEY / KEY2 / KEY3 "
                "in GitHub Secrets"
            )
            sys.exit(1)

        # ── Bootstrap ────────────────────────────────────
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
        now_utc      = datetime.now(timezone.utc)

        # Phase 1: Results (async)
        results_msg = await check_and_report_results_async(
            sent_history, km, session
        )
        if results_msg:
            if await send_telegram_async(results_msg, session):
                logger.info("Results report sent")
            await asyncio.sleep(2)

        # Phase 2: Odds
        log_section("PHASE 2 — FETCHING ODDS")
        events = await fetch_all_odds_async(now_utc, km, session)

        if not events:
            logger.error(
                "No events received. Key status: %s",
                km.get_summary(),
            )
            return

        log_check("Total events", len(events))

        # Phase 3: Signals
        log_section("PHASE 3 — ANALYSIS & SIGNALS")
        total_sent = 0

        for event in events:
            home  = event.get("home_team", "")
            away  = event.get("away_team", "")
            sport = event.get("sport_title", "Unknown")
            sk    = normalize_sport_key(sport)
            ct    = event.get("commence_time", "")
            md    = event.get("_markets_data", {})

            if not home or not away:
                continue

            logger.info(
                "Processing: %s vs %s [%s]", home, away, sport
            )

            elo_pred: Optional[dict] = None
            if sk == "football":
                elo_pred = elo_football.predict(home, away)
            elif sk == "tennis":
                elo_pred = elo_tennis.predict(
                    home, away, apply_home=False
                )

            opps = calculate_combined_ev(
                md, elo_pred, sk, home, away
            )
            if not opps:
                continue

            opp = opps[0]

            if sent_history.was_sent(home, away, opp["market"]):
                logger.info(
                    "SKIP duplicate: %s vs %s", home, away
                )
                continue

            stats, _ = await get_stats_async(
                home, away, sk,
                fd, mic, elo_football, elo_tennis,
                session,
            )

            conf, risk = calculate_confidence(
                opp["ev"], stats,
                opp["market"], opp["has_sharp_line"],
            )

            dp = get_display_pick(
                opp["pick"], opp["market"], home, away
            )

            # تغییر مهم: اینجا به await و نام تابع جدید تغییر کرد
            ai = await generate_dual_ai_analysis_async(
                home, away, sport, dp, opp["market"],
                opp["ev"], stats, conf, risk,
            )

            msg = build_telegram_message(
                sport, home, away, ct, now_utc,
                opp, dp, conf, risk, ai,
            )

            logger.info(
                "SIGNAL | %s vs %s | pick=%s "
                "odds=%.2f ev=%.1f%% conf=%d%%",
                home, away, dp,
                opp["odds"], opp["edge_pct"], conf,
            )

            if await send_telegram_async(msg, session):
                sent_history.mark_sent(
                    home, away,
                    opp["pick"], opp["market"],
                    opp["odds"], ct,
                )
                total_sent += 1
                logger.info("✅ Sent: %s vs %s", home, away)
            else:
                logger.error(
                    "❌ Failed: %s vs %s", home, away
                )

            await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    log_section("RUN COMPLETE")
    log_check("Signals sent", total_sent, warn_if_none=False)
    logger.info("Final key status: %s", km.get_summary())


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except Exception as e:
        logger.critical(
            "SYSTEM FAILURE: %s", str(e), exc_info=True
        )
        sys.exit(1)
