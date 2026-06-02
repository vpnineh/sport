import os
import sys
import time
import json
import re
import random
import logging
import html as html_lib
import hashlib
import asyncio
import aiohttp
import requests
import numpy as np
import pandas as pd
import pickle
import warnings
import threading
from io import StringIO
from functools import wraps, lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict, deque

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── Google AI SDK (جایگزین requests خام) ──────────────
from google import genai
from google.genai import types

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
import scipy.stats as stats_scipy

# ── Optional: MLB Stats API ───────────────────────────
try:
    import statsapi as mlb_statsapi
    HAS_STATSAPI = True
except ImportError:
    HAS_STATSAPI = False


# =========================================================
# 1. CONFIGURATION
# =========================================================
@dataclass
class Config:
    CACHE_DIR: Path = Path("api_cache")
    LOG_DIR: Path = Path("log")
    HISTORICAL_DIR: Path = Path("api_cache/historical")
    ML_DIR: Path = Path("api_cache/ml_models")

    HISTORY_FILE: Path = Path("api_cache/sent_history.json")
    TEAM_ID_CACHE_FILE: Path = Path("api_cache/team_id_cache.json")
    MATCH_ID_CACHE_FILE: Path = Path("api_cache/match_id_cache.json")
    DAILY_STATS_CACHE_FILE: Path = Path("api_cache/daily_stats_cache.json")
    FOOTBALL_CACHE_FILE: Path = Path("api_cache/football_stats_cache.json")
    LOG_FILE: Path = Path("api_cache/execution_logs.log")
    ODDS_CACHE_FILE: Path = Path("api_cache/odds_cache.json")
    API_USAGE_FILE: Path = Path("api_cache/api_usage_tracker.json")
    PERFORMANCE_FILE: Path = Path("api_cache/performance_tracker.json")

    # ── Window & Timing ──────────────────────────────────
    MATCH_WINDOW_HOURS: float = 6.0
    TELEGRAM_SLEEP_BETWEEN: float = 3.0

    # ── Odds API ─────────────────────────────────────────
    ODDS_API_MARKETS: list = field(default_factory=lambda: ["h2h", "totals"])
    ODDS_API_REGIONS: str = "eu,us,uk,au"
    TTL_ODDS_CACHE_MINUTES: float = 6.0

    # ── Cache TTLs ───────────────────────────────────────
    TTL_SENT_HISTORY: float = 48.0
    TTL_MATCH_ID: float = 24.0
    TTL_TEAM_FORM: float = 6.0
    TTL_H2H: float = 24.0
    TTL_GITHUB_DATA: float = 12.0

    # ── EV Filters ───────────────────────────────────────
    H2H_MIN_ODDS: float = 1.50
    H2H_MIN_EV: float = 0.015
    TOTALS_MIN_ODDS: float = 1.60
    TOTALS_MIN_EV: float = 0.020
    MAX_REALISTIC_EV: float = 0.18

    # ── Market Validation ────────────────────────────────
    MARKET_EXPECTED_OUTCOMES: dict = field(default_factory=lambda: {
        "h2h": {"min": 2, "max": 3},
        "totals": {"min": 2, "max": 2}
    })
    MAX_VALID_IMPLIED_SUM: float = 1.25
    MIN_VALID_IMPLIED_SUM: float = 0.80

    # ── Kelly Criterion ──────────────────────────────────
    KELLY_FRACTION: float = 0.25
    MAX_KELLY_PCT: float = 5.0

    # ── Confidence Thresholds ────────────────────────────
    MIN_CONFIDENCE_TO_SEND: int = 58
    HIGH_CONFIDENCE: int = 75
    MEDIUM_CONFIDENCE: int = 65

    # ── AI Models ────────────────────────────────────────
    # FIX: مدل صحیح Gemini با SDK رسمی
    AI_MODEL_ANALYST: str = "gemini-flash-latest"
    AI_MAX_TOKENS: int = 2048
    AI_TEMPERATURE: float = 0.1

    TELEGRAM_ID: str = "@zBET90"

    # ── Sharp Bookmakers ─────────────────────────────────
    SHARP_BOOKMAKERS: list = field(default_factory=lambda: [
        "pinnacle", "betfair_ex_eu", "matchbook",
        "betfair_ex_uk", "sport888", "betsson",
    ])

    # ── Data Sources ─────────────────────────────────────
    GITHUB_SOURCES: dict = field(default_factory=lambda: {
        "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
        "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
        "atp_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv",
        "wta_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv",
        "football_eu": "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv",
        "club_elo": "http://api.clubelo.com/{team}",
        "club_elo_today": "http://api.clubelo.com/{date}",
    })

    FOOTBALL_DATA_UK_LEAGUES: dict = field(default_factory=lambda: {
        "E0": "Premier League", "E1": "Championship",
        "D1": "Bundesliga", "SP1": "La Liga",
        "I1": "Serie A", "F1": "Ligue 1",
        "N1": "Eredivisie", "P1": "Liga Portugal",
        "T1": "Super Lig", "B1": "Jupiler League",
    })

    FOOTBALL_DATA_UK_SEASONS: list = field(default_factory=lambda: [
        "2223", "2324", "2425"
    ])


CFG = Config()


# =========================================================
# 2. LOGGING
# =========================================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

for d in [CFG.CACHE_DIR, CFG.LOG_DIR, CFG.HISTORICAL_DIR, CFG.ML_DIR]:
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


# =========================================================
# 3. GOOGLE AI SDK MANAGER (Updated to google-genai)
# =========================================================
class GeminiManager:
    """
    مدیریت Google GenAI SDK با:
    - Connection pooling
    - Retry با exponential backoff
    - Thread-safe
    """
    
    _instance: Optional["GeminiManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # خواندن تمام کلیدهای تعریف شده
        raw_keys = [
            os.getenv("GEMINI", ""),
            os.getenv("GEMINI1", ""),
            os.getenv("GEMINI2", ""),
            os.getenv("GEMINI3", ""),
        ]
        # فیلتر کردن کلیدهای خالی
        valid_keys = [k.strip() for k in raw_keys if k.strip()]

        if not valid_keys:
            logger.critical("FATAL: No GEMINI API keys found!")
            sys.exit(1)

        # ساخت یک کلاینت مجزا برای هر کلید
        self.clients = [genai.Client(api_key=k) for k in valid_keys]

        # بقیه تنظیمات مثل قبل...
        self._safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        self._initialized = True
        logger.info("✅ [GEMINI SDK] Initialized with %d API keys for Load Balancing", len(self.clients))

    def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> Optional[dict]:
        
        target_model = model_name or CFG.AI_MODEL_ANALYST
        
        config_kwargs = {
            "temperature": temperature if temperature is not None else CFG.AI_TEMPERATURE,
            "max_output_tokens": CFG.AI_MAX_TOKENS,
            "response_mime_type": "application/json",
            "safety_settings": self._safety_settings,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
            
        gen_config = types.GenerateContentConfig(**config_kwargs)

        last_error = None
        for attempt in range(max_retries):
            try:
                # 🎯 انتخاب رندوم یک کلاینت (کلید) در هر تلاش
                client = random.choice(self.clients)
                
                response = client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=gen_config
                )
                
                if getattr(response, "prompt_feedback", None) and response.prompt_feedback.block_reason:
                    logger.warning("[GEMINI] Safety block!")
                    return None

                raw_text = response.text
                if not raw_text:
                    last_error = "Empty response"
                    continue

                # Parse JSON
                try:
                    result = json.loads(raw_text)
                    return result
                except json.JSONDecodeError:
                    # fallback به extractor شما
                    result = robust_json_extractor(raw_text)
                    if result:
                        return result
                    last_error = f"JSON parse failed: {raw_text[:100]}"

            except Exception as e:
                error_str = str(e)
                last_error = error_str
                
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = (attempt + 1) * 10
                    logger.warning(
                        "[GEMINI] Rate limited, waiting %ds (attempt %d/%d)",
                        wait_time, attempt + 1, max_retries
                    )
                    time.sleep(wait_time)
                    continue
                elif "400" in error_str or "invalid" in error_str.lower():
                    logger.error("[GEMINI] Invalid request: %s", error_str[:200])
                    return None
                else:
                    logger.warning(
                        "[GEMINI] Error (attempt %d/%d): %s",
                        attempt + 1, max_retries, error_str[:200]
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)

        logger.error("[GEMINI] All %d attempts failed. Last error: %s", max_retries, last_error)
        return None

# ─── Singleton ──────────────────────────────────────────
gemini_manager = GeminiManager()


# =========================================================
# 4. API KEY MANAGER
# =========================================================
class OddsAPIKeyManager:
    def __init__(self):
        self.keys: List[Dict] = []
        self._load_keys()
        self.usage = self._load_usage()

    def _load_keys(self):
        key_envs = [
            ("ODDS_API_KEY", "primary"),
            ("ODDS_API_KEY2", "backup_1"),
            ("ODDS_API_KEY3", "backup_2"),
        ]
        for env_name, label in key_envs:
            key = os.getenv(env_name, "").strip()
            if key:
                self.keys.append({
                    "key": key, "label": label, "env": env_name,
                    "failed": False, "fail_reason": None, "fail_time": None,
                })
                logger.info("🔑 [API KEY] %s (%s): Loaded ✓", label, env_name)

        if not self.keys:
            logger.critical("FATAL: No ODDS_API_KEY found!")
            sys.exit(1)

        logger.info("🔑 [API KEYS] %d key(s) available for fallback", len(self.keys))

    def _load_usage(self) -> dict:
        try:
            if CFG.API_USAGE_FILE.exists():
                with open(CFG.API_USAGE_FILE, "r") as f:
                    data = json.load(f)
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if data.get("date") != today:
                    return {"date": today, "keys": {}}
                return data
        except Exception:
            pass
        return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "keys": {}}

    def _save_usage(self):
        try:
            with open(CFG.API_USAGE_FILE, "w") as f:
                json.dump(self.usage, f, indent=2)
        except Exception:
            pass

    def record_usage(self, key_label: str, requests_used: int = 0, remaining: int = -1):
        if key_label not in self.usage["keys"]:
            self.usage["keys"][key_label] = {
                "calls": 0, "remaining": -1, "last_used": None
            }
        self.usage["keys"][key_label]["calls"] += 1
        self.usage["keys"][key_label]["last_used"] = datetime.now(timezone.utc).isoformat()
        if remaining >= 0:
            self.usage["keys"][key_label]["remaining"] = remaining
        self._save_usage()

    def mark_failed(self, key_index: int, reason: str):
        if 0 <= key_index < len(self.keys):
            self.keys[key_index]["failed"] = True
            self.keys[key_index]["fail_reason"] = reason
            self.keys[key_index]["fail_time"] = datetime.now(timezone.utc).isoformat()
            logger.warning(
                "🔑❌ [API KEY] %s marked FAILED: %s",
                self.keys[key_index]["label"], reason
            )

    def get_active_keys(self) -> List[Dict]:
        now = datetime.now(timezone.utc)
        active = []
        for k in self.keys:
            if not k["failed"]:
                active.append(k)
            else:
                if k.get("fail_time"):
                    try:
                        fail_dt = datetime.fromisoformat(k["fail_time"])
                        if fail_dt.tzinfo is None:
                            fail_dt = fail_dt.replace(tzinfo=timezone.utc)
                        if now - fail_dt > timedelta(minutes=30):
                            k["failed"] = False
                            k["fail_reason"] = None
                            active.append(k)
                            logger.info(
                                "🔑🔄 [API KEY] %s reset after cooldown", k["label"]
                            )
                    except Exception:
                        pass

        if not active:
            logger.warning("🔑⚠️ All keys failed! Resetting all...")
            for k in self.keys:
                k["failed"] = False
                k["fail_reason"] = None
            active = list(self.keys)

        return active

    def get_usage_summary(self) -> str:
        parts = []
        for k in self.keys:
            usage = self.usage.get("keys", {}).get(k["label"], {})
            calls = usage.get("calls", 0)
            remaining = usage.get("remaining", "?")
            status = "❌" if k["failed"] else "✅"
            parts.append(f"{status} {k['label']}: {calls} calls (rem: {remaining})")
        return " | ".join(parts)


# ── Globals ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()

if not all([GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical(
        "FATAL: Missing required env vars (GEMINI, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)"
    )
    sys.exit(1)

odds_key_manager = OddsAPIKeyManager()


# =========================================================
# 5. NATIONALITY FLAGS
# =========================================================
NATIONALITY_FLAGS: dict = {
    "bautista agut": "ES", "alcaraz": "ES", "nadal": "ES", "munar": "ES",
    "djokovic": "RS", "kecmanovic": "RS", "krajinovic": "RS",
    "sinner": "IT", "berrettini": "IT", "musetti": "IT", "sonego": "IT",
    "zverev": "DE", "struff": "DE", "koepfer": "DE", "altmaier": "DE",
    "tiafoe": "US", "fritz": "US", "paul": "US", "nakashima": "US",
    "sock": "US", "isner": "US", "korda": "US", "eubanks": "US",
    "medvedev": "RU", "rublev": "RU", "khachanov": "RU", "karatsev": "RU",
    "tsitsipas": "GR", "ruud": "NO", "rune": "DK",
    "hurkacz": "PL", "swiatek": "PL",
    "auger-aliassime": "CA", "shapovalov": "CA",
    "kyrgios": "AU", "de minaur": "AU", "thompson": "AU",
    "sabalenka": "BY", "gauff": "US", "keys": "US", "pegula": "US",
    "halep": "RO", "kvitova": "CZ", "vondrousova": "CZ",
    "jabeur": "TN", "badosa": "ES",
    "dimitrov": "BG", "norrie": "GB", "murray": "GB", "draper": "GB",
    "thiem": "AT", "wawrinka": "CH",
    "monfils": "FR", "bublik": "KZ", "rybakina": "KZ",
    "etcheverry": "AR", "cerundolo": "AR", "schwartzman": "AR",
    "real madrid": "ES", "barcelona": "ES", "atletico": "ES", "sevilla": "ES",
    "bayern": "DE", "dortmund": "DE", "leipzig": "DE", "leverkusen": "DE",
    "manchester united": "GB", "manchester city": "GB", "liverpool": "GB",
    "arsenal": "GB", "chelsea": "GB", "tottenham": "GB", "newcastle": "GB",
    "juventus": "IT", "milan": "IT", "inter": "IT", "napoli": "IT", "roma": "IT",
    "psg": "FR", "marseille": "FR", "lyon": "FR", "monaco": "FR",
    "ajax": "NL", "psv": "NL", "feyenoord": "NL",
    "porto": "PT", "benfica": "PT", "sporting": "PT",
    "lakers": "US", "celtics": "US", "warriors": "US",
    "cuiabá": "BR", "cruzeiro": "BR", "fluminense": "BR", "flamengo": "BR",
    "palmeiras": "BR", "santos": "BR", "corinthians": "BR", "atletico mineiro": "BR",
    "palestino": "CL", "audax italiano": "CL", "colo-colo": "CL",
    "river plate": "AR", "boca juniors": "AR", "independiente": "AR",
    "nacional": "UY", "penarol": "UY",
    "olimpia": "PY", "libertad": "PY",
}


def _code_to_flag(code: str) -> str:
    code = code.upper().strip()
    if len(code) != 2:
        return "🏳️"
    offset = 0x1F1E6 - ord("A")
    return chr(ord(code[0]) + offset) + chr(ord(code[1]) + offset)


def get_flag_from_name(name: str) -> str:
    name_lower = name.lower()
    for keyword, code in NATIONALITY_FLAGS.items():
        if keyword in name_lower:
            return _code_to_flag(code)
    return "🏳️"


def validate_flag(flag: str, fallback_name: str) -> str:
    if not flag:
        return get_flag_from_name(fallback_name)
    stripped = flag.strip()
    if stripped in ["🏳️", "🏁", "🚩", ""]:
        return get_flag_from_name(fallback_name)
    return stripped


# =========================================================
# 6. CACHE MANAGEMENT  ← Thread-safe با Lock
# =========================================================
_cache_lock = threading.Lock()


class CacheManager:

    @staticmethod
    def load(filepath: Path) -> dict:
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if DEBUG_MODE:
                logger.warning("Cache load error (%s): %s", filepath.name, e)
        return {}

    @staticmethod
    def save(filepath: Path, data: dict) -> None:
        """Atomic write با thread lock برای جلوگیری از race condition."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            tmp = filepath.with_suffix(".tmp")
            with _cache_lock:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                tmp.replace(filepath)
        except OSError as e:
            if DEBUG_MODE:
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
            if cached_time.tzinfo is None:
                cached_time = cached_time.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - cached_time < timedelta(hours=ttl_hours)
        except Exception:
            return False

    @staticmethod
    def is_valid_minutes(cache: dict, key: str, ttl_minutes: float) -> bool:
        if key not in cache:
            return False
        entry = cache[key]
        if not isinstance(entry, dict) or "timestamp" not in entry:
            return False
        try:
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if cached_time.tzinfo is None:
                cached_time = cached_time.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - cached_time < timedelta(minutes=ttl_minutes)
        except Exception:
            return False

    @staticmethod
    def set(cache: dict, key: str, value: Any) -> dict:
        cache[key] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": value,
        }
        return cache

    @staticmethod
    def get(cache: dict, key: str) -> Any:
        return cache.get(key, {}).get("data")


# =========================================================
# 7. PERFORMANCE TRACKER
# =========================================================
class PerformanceTracker:
    def __init__(self):
        self.data = CacheManager.load(CFG.PERFORMANCE_FILE)
        if "signals" not in self.data:
            self.data["signals"] = []
        if "summary" not in self.data:
            self.data["summary"] = {}

    def record_signal(
        self, home: str, away: str, pick: str, market: str,
        odds: float, ev: float, confidence: int, prob: float,
        sport: str = "other", api_sport_key: str = ""
    ):
        signal = {
            "id": hashlib.md5(
                f"{home}|{away}|{market}|{datetime.now(timezone.utc).date()}".encode()
            ).hexdigest()[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sport": sport,
            "api_sport_key": api_sport_key,
            "home": home, "away": away, "pick": pick, "market": market,
            "odds": odds, "ev": ev, "confidence": confidence,
            "implied_prob": prob,
            "outcome": None,
            "profit_loss": None,
        }
        self.data["signals"].append(signal)

        # فقط ۵۰۰ سیگنال آخر
        if len(self.data["signals"]) > 500:
            self.data["signals"] = self.data["signals"][-500:]

        self._update_summary()
        CacheManager.save(CFG.PERFORMANCE_FILE, self.data)

    def _update_summary(self):
        resolved = [s for s in self.data["signals"] if s.get("outcome") is not None]
        if not resolved:
            return

        wins = [s for s in resolved if s["outcome"] == "win"]
        win_rate = len(wins) / len(resolved)
        total_pl = sum(s.get("profit_loss", 0) or 0 for s in resolved)

        self.data["summary"] = {
            "total_signals": len(self.data["signals"]),
            "resolved": len(resolved),
            "win_rate": round(win_rate, 3),
            "total_profit_loss_units": round(total_pl, 2),
            "roi_pct": round(total_pl / len(resolved) * 100, 2) if resolved else 0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def get_recent_accuracy(self, n: int = 50) -> float:
        recent = [s for s in self.data["signals"][-n:] if s.get("outcome")]
        if len(recent) < 5:
            return 0.55
        return sum(1 for s in recent if s["outcome"] == "win") / len(recent)


performance_tracker = PerformanceTracker()


# =========================================================
# 8. SENT HISTORY
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
                dt = datetime.fromisoformat(sent_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if now - dt > timedelta(hours=CFG.TTL_SENT_HISTORY):
                    to_delete.append(key)
            except Exception:
                to_delete.append(key)

        for key in to_delete:
            del self.history[key]

    @staticmethod
    def _make_key(home: str, away: str, market: str) -> str:
        return hashlib.md5(
            f"{home.lower()}|{away.lower()}|{market.lower()}".encode()
        ).hexdigest()

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
# 9. FREE HISTORICAL DATA ENGINE
# =========================================================
class FreeDataEngine:
    def __init__(self):
        self.atp_matches: Optional[pd.DataFrame] = None
        self.wta_matches: Optional[pd.DataFrame] = None
        self.atp_rankings: Optional[pd.DataFrame] = None
        self.wta_rankings: Optional[pd.DataFrame] = None
        self.football_data: Dict[str, pd.DataFrame] = {}
        self.elo_cache: dict = CacheManager.load(CFG.CACHE_DIR / "elo_cache.json")
        self.american_sports_cache: dict = CacheManager.load(
            CFG.CACHE_DIR / "us_sports_cache.json"
        )
        self.years_to_fetch = [2022, 2023, 2024, 2025]

    def _download_csv(self, url: str, filepath: Path, timeout: int = 25) -> bool:
        if filepath.exists():
            age_hours = (time.time() - filepath.stat().st_mtime) / 3600
            if age_hours < CFG.TTL_GITHUB_DATA:
                return True

        logger.info("[FREE DATA] Downloading: %s", url.split("/")[-1])
        try:
            res = requests.get(
                url, timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ZBET90/6.5)"}
            )
            if res.status_code == 200 and len(res.content) > 100:
                with open(filepath, "wb") as f:
                    f.write(res.content)
                return True
            else:
                logger.debug("[FREE DATA] HTTP %d for %s", res.status_code, url)
        except requests.exceptions.Timeout:
            logger.warning("[FREE DATA] Timeout: %s", url.split("/")[-1])
        except requests.exceptions.RequestException as e:
            logger.warning("[FREE DATA] Error %s: %s", url.split("/")[-1], str(e)[:100])
        return False

    def load_tennis_data(self):
        atp_dfs, wta_dfs = [], []
        match_cols = [
            "tourney_date", "tourney_name", "surface", "draw_size", "tourney_level",
            "round", "winner_id", "winner_name", "winner_rank", "winner_rank_points",
            "winner_age", "winner_ht", "winner_ioc",
            "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
            "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
            "loser_id", "loser_name", "loser_rank", "loser_rank_points",
            "loser_age", "loser_ht", "loser_ioc",
            "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
            "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
            "score", "best_of", "minutes",
        ]

        for year in self.years_to_fetch:
            for tour, dfs_list, key in [("atp", atp_dfs, "atp"), ("wta", wta_dfs, "wta")]:
                url = CFG.GITHUB_SOURCES[key].format(year=year)
                path = CFG.HISTORICAL_DIR / f"{key}_{year}.csv"
                if self._download_csv(url, path):
                    try:
                        df = pd.read_csv(path, low_memory=False, encoding="utf-8",
                                         encoding_errors="replace")
                        available = [c for c in match_cols if c in df.columns]
                        sub = df[available].copy()
                        if "tourney_date" in sub.columns:
                            sub["tourney_date"] = pd.to_numeric(
                                sub["tourney_date"], errors="coerce"
                            )
                        dfs_list.append(sub)
                    except Exception as e:
                        logger.error("%s parse error %s: %s", tour.upper(), path, e)

        if atp_dfs:
            self.atp_matches = pd.concat(atp_dfs, ignore_index=True)
            if "tourney_date" in self.atp_matches.columns:
                self.atp_matches = self.atp_matches.sort_values(
                    "tourney_date", ascending=True
                ).reset_index(drop=True)
            logger.info("✅ [TENNIS] ATP loaded: %d matches", len(self.atp_matches))

        if wta_dfs:
            self.wta_matches = pd.concat(wta_dfs, ignore_index=True)
            if "tourney_date" in self.wta_matches.columns:
                self.wta_matches = self.wta_matches.sort_values(
                    "tourney_date", ascending=True
                ).reset_index(drop=True)
            logger.info("✅ [TENNIS] WTA loaded: %d matches", len(self.wta_matches))

        for tour, key in [("atp", "atp_rankings"), ("wta", "wta_rankings")]:
            url = CFG.GITHUB_SOURCES[key]
            path = CFG.HISTORICAL_DIR / f"{key}.csv"
            if self._download_csv(url, path):
                try:
                    df = pd.read_csv(path, low_memory=False)
                    if tour == "atp":
                        self.atp_rankings = df
                    else:
                        self.wta_rankings = df
                    logger.info(
                        "✅ [RANKINGS] %s loaded: %d entries", tour.upper(), len(df)
                    )
                except Exception as e:
                    logger.error("Rankings parse error: %s", e)

    def get_player_ranking(self, player_name: str, is_wta: bool = False) -> Optional[int]:
        df = self.wta_rankings if is_wta else self.atp_rankings
        if df is None or df.empty:
            return None

        clean = player_name.split()[-1].lower()
        try:
            name_col = next(
                (c for c in ["player", "name", "player_name"] if c in df.columns), None
            )
            if not name_col:
                return None
            matches = df[df[name_col].str.lower().str.contains(
                re.escape(clean), na=False
            )]
            if not matches.empty:
                rank_col = next(
                    (c for c in ["rank", "ranking", "player_rank"] if c in matches.columns),
                    None
                )
                if rank_col:
                    val = matches.iloc[0][rank_col]
                    return int(val) if pd.notna(val) else None
        except Exception:
            pass
        return None

    def _compute_player_rolling_stats(
        self, df: pd.DataFrame, player_clean: str, n_recent: int = 20
    ) -> dict:
        wins = df[df["winner_name"].str.lower().str.contains(
            re.escape(player_clean), na=False
        )].copy()
        losses = df[df["loser_name"].str.lower().str.contains(
            re.escape(player_clean), na=False
        )].copy()

        total = len(wins) + len(losses)
        if total == 0:
            return {}

        all_dates_results = []
        for _, r in wins.iterrows():
            all_dates_results.append((r.get("tourney_date", 0), "W", r))
        for _, r in losses.iterrows():
            all_dates_results.append((r.get("tourney_date", 0), "L", r))

        all_dates_results.sort(
            key=lambda x: x[0] if pd.notna(x[0]) else 0, reverse=True
        )
        recent = all_dates_results[:n_recent]

        result = {
            "total_matches": total,
            "win_rate_overall": round(len(wins) / total, 3),
            "matches_analyzed": len(recent),
        }

        if recent:
            rw = sum(1 for x in recent if x[1] == "W")
            result["recent_form"] = "".join(x[1] for x in recent[:10])
            result["recent_win_rate"] = round(rw / len(recent), 3)

            recent_wins = wins.tail(n_recent // 2)
            for stat, col in [
                ("aces_per_match", "w_ace"),
                ("df_per_match", "w_df"),
                ("svpt_per_match", "w_svpt"),
            ]:
                if col in recent_wins.columns:
                    valid = recent_wins[col].dropna()
                    if len(valid) > 0:
                        result[stat] = round(float(valid.mean()), 2)

            if all(c in recent_wins.columns for c in ["w_1stIn", "w_svpt"]):
                svpt_vals = recent_wins["w_svpt"].dropna()
                in1st_vals = recent_wins["w_1stIn"].dropna()
                if len(svpt_vals) > 0 and svpt_vals.mean() > 0:
                    result["first_serve_in_pct"] = round(
                        float(in1st_vals.mean() / svpt_vals.mean()), 3
                    )

            if all(c in recent_wins.columns for c in ["w_1stWon", "w_1stIn"]):
                in1st_vals = recent_wins["w_1stIn"].dropna()
                won1st_vals = recent_wins["w_1stWon"].dropna()
                if len(in1st_vals) > 0 and in1st_vals.mean() > 0:
                    result["first_serve_win_pct"] = round(
                        float(won1st_vals.mean() / in1st_vals.mean()), 3
                    )

            if all(c in recent_wins.columns for c in ["w_bpSaved", "w_bpFaced"]):
                bpf = recent_wins["w_bpFaced"].dropna()
                bps = recent_wins["w_bpSaved"].dropna()
                if len(bpf) > 0 and bpf.mean() > 0:
                    result["bp_saved_pct"] = round(
                        float(bps.mean() / bpf.mean()), 3
                    )

            surface_stats = {}
            for surface in ["Hard", "Clay", "Grass"]:
                sw = wins[wins["surface"].str.lower() == surface.lower()] \
                    if "surface" in wins.columns else pd.DataFrame()
                sl = losses[losses["surface"].str.lower() == surface.lower()] \
                    if "surface" in losses.columns else pd.DataFrame()
                st = len(sw) + len(sl)
                if st >= 5:
                    surface_stats[surface] = {
                        "win_rate": round(len(sw) / st, 3),
                        "matches": st,
                    }
            if surface_stats:
                result["surface_stats"] = surface_stats

        return result

    def get_tennis_stats(
        self, player_a: str, player_b: str, is_wta: bool = False
    ) -> dict:
        df = self.wta_matches if is_wta else self.atp_matches
        if df is None or df.empty:
            return {}

        def clean(n: str) -> str:
            return n.split()[-1].lower()

        pa, pb = clean(player_a), clean(player_b)
        stats = {
            "player_a": {"name": player_a},
            "player_b": {"name": player_b},
            "h2h": {},
        }

        for p_clean, key, p_full in [
            (pa, "player_a", player_a), (pb, "player_b", player_b)
        ]:
            p_stats = self._compute_player_rolling_stats(df, p_clean)
            if p_stats:
                stats[key].update(p_stats)
                ranking = self.get_player_ranking(p_full, is_wta)
                if ranking:
                    stats[key]["current_ranking"] = ranking

        h2h_a = df[
            df["winner_name"].str.lower().str.contains(pa, na=False)
            & df["loser_name"].str.lower().str.contains(pb, na=False)
        ]
        h2h_b = df[
            df["winner_name"].str.lower().str.contains(pb, na=False)
            & df["loser_name"].str.lower().str.contains(pa, na=False)
        ]
        total_h2h = len(h2h_a) + len(h2h_b)

        if total_h2h > 0:
            stats["h2h"] = {
                "total": total_h2h,
                f"{player_a}_wins": len(h2h_a),
                f"{player_b}_wins": len(h2h_b),
                "dominance": "balanced",
            }
            if len(h2h_a) > len(h2h_b) * 2:
                stats["h2h"]["dominance"] = f"{player_a}_dominant"
            elif len(h2h_b) > len(h2h_a) * 2:
                stats["h2h"]["dominance"] = f"{player_b}_dominant"

            if "surface" in h2h_a.columns:
                h2h_surfaces = {}
                for surface in ["Hard", "Clay", "Grass"]:
                    sa = h2h_a[h2h_a["surface"].str.lower() == surface.lower()]
                    sb = h2h_b[h2h_b["surface"].str.lower() == surface.lower()]
                    if len(sa) + len(sb) > 0:
                        h2h_surfaces[surface] = {
                            f"{player_a}_wins": len(sa),
                            f"{player_b}_wins": len(sb),
                        }
                if h2h_surfaces:
                    stats["h2h"]["by_surface"] = h2h_surfaces

            logger.info(
                "✅ [H2H] %s vs %s: %d H2H matches", player_a, player_b, total_h2h
            )

        return stats

    def load_football_data(self):
        football_cols = [
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST",
            "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
            "B365H", "B365D", "B365A", "BbMxH", "BbMxD", "BbMxA",
            "BbAvH", "BbAvD", "BbAvA",
            "BbMx>2.5", "BbAv>2.5", "BbMx<2.5", "BbAv<2.5",
        ]
        all_dfs = []
        for season in CFG.FOOTBALL_DATA_UK_SEASONS:
            for league_code in CFG.FOOTBALL_DATA_UK_LEAGUES:
                league_name = CFG.FOOTBALL_DATA_UK_LEAGUES[league_code]
                url = CFG.GITHUB_SOURCES["football_eu"].format(
                    season=season, league=league_code
                )
                path = CFG.HISTORICAL_DIR / f"football_{league_code}_{season}.csv"
                if self._download_csv(url, path):
                    try:
                        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
                        available = [c for c in football_cols if c in df.columns]
                        if len(available) >= 5:
                            sub = df[available].copy()
                            sub["League"] = league_name
                            sub["Season"] = season
                            if "Date" in sub.columns:
                                sub["Date"] = pd.to_datetime(
                                    sub["Date"], dayfirst=True, errors="coerce"
                                )
                            # فقط ردیف‌های معتبر
                            if "HomeTeam" in sub.columns:
                                sub = sub.dropna(subset=["HomeTeam", "AwayTeam"])
                            all_dfs.append(sub)
                    except Exception as e:
                        logger.debug("Football parse error %s: %s", path.name, e)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            if "Date" in combined.columns:
                combined = combined.sort_values(
                    "Date", ascending=True
                ).reset_index(drop=True)
            self.football_data["all"] = combined
            logger.info(
                "✅ [FOOTBALL] Loaded %d matches from %d leagues × %d seasons",
                len(combined),
                len(CFG.FOOTBALL_DATA_UK_LEAGUES),
                len(CFG.FOOTBALL_DATA_UK_SEASONS),
            )

    def _fuzzy_match(self, team: str, column: pd.Series) -> pd.Series:
        clean = team.lower().strip()
        mask = column.str.lower().str.strip() == clean
        if mask.any():
            return mask
        for part in clean.split():
            if len(part) > 3:
                m = column.str.lower().str.contains(re.escape(part), na=False)
                if m.any():
                    return m
        return pd.Series([False] * len(column), index=column.index)

    def get_football_stats(self, home_team: str, away_team: str) -> dict:
        df = self.football_data.get("all")
        if df is None or df.empty:
            return {}

        stats = {"home": {}, "away": {}, "h2h": {}}

        for team, key, is_home in [
            (home_team, "home", True),
            (away_team, "away", False),
        ]:
            home_mask = self._fuzzy_match(team, df["HomeTeam"])
            away_mask = self._fuzzy_match(team, df["AwayTeam"])
            team_home = df[home_mask].copy()
            team_away = df[away_mask].copy()

            all_results = []

            for _, row in team_home.iterrows():
                ftr = row.get("FTR", "")
                # FIX: مقادیر NaN را به 0 تبدیل
                hg = int(row.get("FTHG", 0) or 0) if pd.notna(row.get("FTHG")) else 0
                ag = int(row.get("FTAG", 0) or 0) if pd.notna(row.get("FTAG")) else 0
                all_results.append({
                    "date": row.get("Date"),
                    "result": "W" if ftr == "H" else ("D" if ftr == "D" else "L"),
                    "scored": hg, "conceded": ag, "venue": "home",
                    "shots": int(row["HS"]) if pd.notna(row.get("HS")) else 0,
                    "shots_target": int(row["HST"]) if pd.notna(row.get("HST")) else 0,
                    "corners": int(row["HC"]) if pd.notna(row.get("HC")) else 0,
                    "yellows": int(row["HY"]) if pd.notna(row.get("HY")) else 0,
                })

            for _, row in team_away.iterrows():
                ftr = row.get("FTR", "")
                hg = int(row.get("FTHG", 0) or 0) if pd.notna(row.get("FTHG")) else 0
                ag = int(row.get("FTAG", 0) or 0) if pd.notna(row.get("FTAG")) else 0
                all_results.append({
                    "date": row.get("Date"),
                    "result": "W" if ftr == "A" else ("D" if ftr == "D" else "L"),
                    "scored": ag, "conceded": hg, "venue": "away",
                    "shots": int(row["AS"]) if pd.notna(row.get("AS")) else 0,
                    "shots_target": int(row["AST"]) if pd.notna(row.get("AST")) else 0,
                    "corners": int(row["AC"]) if pd.notna(row.get("AC")) else 0,
                    "yellows": int(row["AY"]) if pd.notna(row.get("AY")) else 0,
                })

            all_results.sort(
                key=lambda x: (
                    x["date"] if isinstance(x["date"], pd.Timestamp)
                    else pd.Timestamp.min
                ),
                reverse=True,
            )
            recent = all_results[:10]

            if not recent:
                continue

            n = len(recent)
            scored_list = [r["scored"] for r in recent]
            conceded_list = [r["conceded"] for r in recent]
            shots_list = [r["shots"] for r in recent]

            # FIX: division by zero guard
            avg_shots = np.mean(shots_list) if shots_list else 1.0
            if avg_shots == 0:
                avg_shots = 1.0

            stats[key] = {
                "name": team,
                "form_string": "".join(r["result"] for r in recent),
                "win_rate": round(
                    sum(1 for r in recent if r["result"] == "W") / n, 3
                ),
                "draw_rate": round(
                    sum(1 for r in recent if r["result"] == "D") / n, 3
                ),
                "loss_rate": round(
                    sum(1 for r in recent if r["result"] == "L") / n, 3
                ),
                "avg_scored": round(float(np.mean(scored_list)), 2),
                "avg_conceded": round(float(np.mean(conceded_list)), 2),
                "std_scored": round(float(np.std(scored_list)), 2),
                "btts_rate": round(
                    sum(
                        1 for r in recent if r["scored"] > 0 and r["conceded"] > 0
                    ) / n,
                    3
                ),
                "over25_rate": round(
                    sum(
                        1 for r in recent if r["scored"] + r["conceded"] > 2.5
                    ) / n,
                    3
                ),
                "over35_rate": round(
                    sum(
                        1 for r in recent if r["scored"] + r["conceded"] > 3.5
                    ) / n,
                    3
                ),
                "clean_sheet_rate": round(
                    sum(1 for r in recent if r["conceded"] == 0) / n, 3
                ),
                "avg_shots": round(float(np.mean(shots_list)), 1),
                "avg_shots_target": round(
                    float(np.mean([r["shots_target"] for r in recent])), 1
                ),
                "avg_corners": round(
                    float(np.mean([r["corners"] for r in recent])), 1
                ),
                "shot_conversion": round(
                    float(np.mean(scored_list)) / avg_shots, 3
                ),
                "matches_analyzed": n,
                "total_historical": len(all_results),
            }

            # venue-specific
            venue_key_str = "home" if is_home else "away"
            venue_matches = [
                r for r in all_results[:20] if r["venue"] == venue_key_str
            ]
            if len(venue_matches) >= 3:
                vn = len(venue_matches)
                stats[key]["venue_win_rate"] = round(
                    sum(1 for r in venue_matches if r["result"] == "W") / vn, 3
                )
                stats[key]["venue_avg_goals"] = round(
                    float(np.mean([r["scored"] + r["conceded"] for r in venue_matches])),
                    2
                )
                stats[key]["venue_btts"] = round(
                    sum(
                        1 for r in venue_matches
                        if r["scored"] > 0 and r["conceded"] > 0
                    ) / vn,
                    3
                )

            # weighted form
            weights = np.array([1 / (i + 1) for i in range(n)])
            weights /= weights.sum()
            result_pts = np.array([
                3 if r["result"] == "W" else (1 if r["result"] == "D" else 0)
                for r in recent
            ], dtype=np.float64)
            stats[key]["weighted_form_points"] = round(
                float(np.dot(weights, result_pts)), 3
            )

        # H2H
        if "HomeTeam" in df.columns and "AwayTeam" in df.columns:
            hm = self._fuzzy_match(home_team, df["HomeTeam"])
            am = self._fuzzy_match(away_team, df["AwayTeam"])
            hm2 = self._fuzzy_match(away_team, df["HomeTeam"])
            am2 = self._fuzzy_match(home_team, df["AwayTeam"])
            h2h_df = df[(hm & am) | (hm2 & am2)]

            if len(h2h_df) > 0:
                h2h_results = []
                for _, row in h2h_df.iterrows():
                    hg = int(row.get("FTHG", 0) or 0) if pd.notna(row.get("FTHG")) else 0
                    ag = int(row.get("FTAG", 0) or 0) if pd.notna(row.get("FTAG")) else 0
                    h2h_results.append({
                        "total_goals": hg + ag,
                        "btts": hg > 0 and ag > 0,
                        "over25": hg + ag > 2.5,
                        "over35": hg + ag > 3.5,
                    })
                hn = len(h2h_results)
                goal_list = [r["total_goals"] for r in h2h_results]
                stats["h2h"] = {
                    "total_matches": hn,
                    "avg_goals": round(float(np.mean(goal_list)), 2),
                    "btts_rate": round(
                        sum(1 for r in h2h_results if r["btts"]) / hn, 3
                    ),
                    "over25_rate": round(
                        sum(1 for r in h2h_results if r["over25"]) / hn, 3
                    ),
                    "over35_rate": round(
                        sum(1 for r in h2h_results if r["over35"]) / hn, 3
                    ),
                    "std_goals": round(float(np.std(goal_list)), 2),
                }
                logger.info(
                    "✅ [FOOTBALL H2H] %s vs %s: %d matches",
                    home_team, away_team, hn
                )

        return stats

    def get_club_elo(self, team_name: str) -> Optional[float]:
        cache_key = f"elo_{team_name.lower()}"
        if CacheManager.is_valid(self.elo_cache, cache_key, CFG.TTL_TEAM_FORM):
            return CacheManager.get(self.elo_cache, cache_key)

        clean = re.sub(r"[^a-zA-Z]", "", team_name).replace("FC", "").strip()
        if not clean:
            return None

        try:
            url = CFG.GITHUB_SOURCES["club_elo"].format(team=clean)
            res = requests.get(url, timeout=8,
                               headers={"User-Agent": "Mozilla/5.0"})
            if res.status_code == 200 and res.text.strip():
                lines = [l for l in res.text.strip().split("\n") if l.strip()]
                if len(lines) > 1:
                    parts = lines[-1].split(",")
                    if len(parts) >= 5:
                        elo = float(parts[4])
                        self.elo_cache = CacheManager.set(
                            self.elo_cache, cache_key, elo
                        )
                        CacheManager.save(
                            CFG.CACHE_DIR / "elo_cache.json", self.elo_cache
                        )
                        return elo
        except (ValueError, requests.RequestException):
            pass
        return None

    def get_elo_delta(self, home_team: str, away_team: str) -> Optional[dict]:
        home_elo = self.get_club_elo(home_team)
        away_elo = self.get_club_elo(away_team)
        if home_elo and away_elo:
            delta = home_elo - away_elo
            home_prob = 1 / (1 + 10 ** (-delta / 400))
            ha_bonus = 0.03
            home_prob_adj = min(0.95, home_prob + ha_bonus)
            return {
                "home_elo": round(home_elo, 1),
                "away_elo": round(away_elo, 1),
                "delta": round(delta, 1),
                "home_win_prob_elo": round(home_prob_adj, 3),
                "away_win_prob_elo": round(1 - home_prob_adj, 3),
                "elo_confidence": (
                    "high" if abs(delta) > 150
                    else ("medium" if abs(delta) > 75 else "low")
                ),
            }
        return None

    def get_us_sports_stats(self, sport: str, team: str) -> dict:
        """آمار MLB با statsapi اگه موجود باشه."""
        if not HAS_STATSAPI:
            return {}

        cache_key = f"{sport}_{team.lower().replace(' ', '')}"
        if CacheManager.is_valid(self.american_sports_cache, cache_key, 12.0):
            cached = CacheManager.get(self.american_sports_cache, cache_key)
            return cached if cached else {}

        stats = {}
        try:
            if "baseball" in sport.lower() or "mlb" in sport.lower():
                teams = mlb_statsapi.lookup_team(team)
                if teams:
                    team_id = teams[0]["id"]
                    start_d = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
                    end_d = datetime.now().strftime("%Y-%m-%d")
                    sched = mlb_statsapi.schedule(
                        team=team_id, start_date=start_d, end_date=end_d
                    )

                    wins, losses = 0, 0
                    runs_scored, runs_allowed = 0, 0
                    finished_games = [
                        g for g in sched if g.get("status") == "Final"
                    ][-5:]

                    for game in finished_games:
                        is_home = game.get("home_id") == team_id
                        h_score = game.get("home_score", 0) or 0
                        a_score = game.get("away_score", 0) or 0
                        team_score = h_score if is_home else a_score
                        opp_score = a_score if is_home else h_score

                        if team_score > opp_score:
                            wins += 1
                        else:
                            losses += 1
                        runs_scored += team_score
                        runs_allowed += opp_score

                    total_g = wins + losses
                    if total_g > 0:
                        stats = {
                            "recent_form": f"{wins}W-{losses}L",
                            "avg_runs_scored": round(runs_scored / total_g, 1),
                            "avg_runs_allowed": round(runs_allowed / total_g, 1),
                        }
        except Exception as e:
            logger.debug("[US SPORTS] Error for %s: %s", team, str(e)[:100])

        if stats:
            self.american_sports_cache = CacheManager.set(
                self.american_sports_cache, cache_key, stats
            )
            CacheManager.save(
                CFG.CACHE_DIR / "us_sports_cache.json",
                self.american_sports_cache
            )

        return stats


# =========================================================
# 10. ML PREDICTION ENGINE
# =========================================================
class MLPredictionEngine:
    def __init__(self, data_engine: FreeDataEngine):
        self.data_engine = data_engine
        self.football_pipeline: Optional[dict] = None
        self.tennis_pipeline: Optional[dict] = None
        self.is_football_trained = False
        self.is_tennis_trained = False
        self._football_team_stats: dict = {}
        self._football_metrics: dict = {}
        self._tennis_metrics: dict = {}
        # FIX: ثابت برای reproducibility تنیس
        self._rng = np.random.RandomState(42)

    # ─────────────────────────────────────────────────────
    # Football
    # ─────────────────────────────────────────────────────
    def load_or_train_football_model(self):
        path = CFG.ML_DIR / "football_model_v6.pkl"
        if path.exists():
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            if age_hours < 24.0:
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    self.football_pipeline = data["pipeline"]
                    self._football_team_stats = data["stats"]
                    self.is_football_trained = True
                    logger.info("⚡ [ML FOOTBALL] Loaded pre-trained model from cache!")
                    return
                except Exception as e:
                    logger.warning("Football model cache corrupt: %s", e)

        self.train_football_model()
        if self.is_football_trained:
            try:
                with open(path, "wb") as f:
                    pickle.dump(
                        {
                            "pipeline": self.football_pipeline,
                            "stats": self._football_team_stats,
                        },
                        f,
                    )
                logger.info("💾 [ML FOOTBALL] Model saved to cache.")
            except Exception as e:
                logger.warning("Football model save error: %s", e)

    def _build_team_stats_rolling(
        self, df: pd.DataFrame
    ) -> Tuple[dict, dict]:
        team_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        result_lookup: Dict[Any, dict] = {}

        for idx, row in df.iterrows():
            ht = str(row.get("HomeTeam", "") or "")
            at = str(row.get("AwayTeam", "") or "")
            ftr = str(row.get("FTR", "") or "")

            if not ht or not at or ftr not in ["H", "D", "A"]:
                continue

            # FIX: safe numeric conversion
            try:
                hg = float(row.get("FTHG", 0) or 0)
                ag = float(row.get("FTAG", 0) or 0)
            except (ValueError, TypeError):
                continue

            def get_stats(team: str) -> Optional[dict]:
                hist = list(team_stats[team])
                if len(hist) < 3:
                    return None
                w = np.array([1 / (i + 1) for i in range(len(hist))][::-1])
                w /= w.sum()
                return {
                    "avg_gs": float(np.dot(w, [h["gs"] for h in hist])),
                    "avg_gc": float(np.dot(w, [h["gc"] for h in hist])),
                    "form_pts": float(np.dot(w, [h["pts"] for h in hist])),
                    "win_rate": sum(1 for h in hist if h["pts"] == 3) / len(hist),
                }

            hs = get_stats(ht)
            aws = get_stats(at)

            if hs is not None and aws is not None:
                result_lookup[idx] = {
                    "home_stats": hs,
                    "away_stats": aws,
                    "label": {"H": 0, "D": 1, "A": 2}[ftr],
                }

            # آپدیت stats AFTER feature capture (prevent lookahead)
            team_stats[ht].appendleft({
                "gs": hg, "gc": ag,
                "pts": 3 if ftr == "H" else (1 if ftr == "D" else 0),
            })
            team_stats[at].appendleft({
                "gs": ag, "gc": hg,
                "pts": 3 if ftr == "A" else (1 if ftr == "D" else 0),
            })

        return result_lookup, dict(team_stats)

    def _build_football_features(
        self, result_lookup: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        features, labels = [], []
        for idx, data in result_lookup.items():
            hs = data["home_stats"]
            aws = data["away_stats"]
            if not hs or not aws:
                continue
            features.append([
                hs.get("avg_gs", 0), hs.get("avg_gc", 0),
                hs.get("form_pts", 0), hs.get("win_rate", 0),
                aws.get("avg_gs", 0), aws.get("avg_gc", 0),
                aws.get("form_pts", 0), aws.get("win_rate", 0),
                hs.get("avg_gs", 0) - aws.get("avg_gc", 0),
                aws.get("avg_gs", 0) - hs.get("avg_gc", 0),
            ])
            labels.append(data["label"])

        if not features:
            return np.array([]), np.array([])

        return (
            np.nan_to_num(np.array(features, dtype=np.float64)),
            np.array(labels, dtype=np.int32),
        )

    def train_football_model(self):
        df = self.data_engine.football_data.get("all")
        if df is None or len(df) < 300:
            logger.warning("[ML FOOTBALL] Insufficient data (%d rows)", len(df) if df is not None else 0)
            return

        result_lookup, self._football_team_stats = self._build_team_stats_rolling(df)
        X, y = self._build_football_features(result_lookup)

        if len(X) < 200:
            logger.warning("[ML FOOTBALL] Too few features (%d)", len(X))
            return

        # FIX: بررسی unique classes - CalibratedClassifierCV نیاز به ≥2 class دارد
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning("[ML FOOTBALL] Only 1 class found, skipping training.")
            return

        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        )
        stacking = StackingClassifier(
            estimators=[("gb", gb), ("rf", rf)],
            final_estimator=LogisticRegression(
                max_iter=1000, C=0.1, random_state=42
            ),
            cv=3,
        )

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        final_model = CalibratedClassifierCV(stacking, cv=3, method="isotonic")
        try:
            final_model.fit(X_scaled, y)
            self.football_pipeline = {"model": final_model, "scaler": scaler}
            self.is_football_trained = True
            logger.info("✅ [ML FOOTBALL] Trained on %d samples", len(X))
        except Exception as e:
            logger.error("[ML FOOTBALL] Training failed: %s", e)

    def predict_football(
        self, home_team: str, away_team: str
    ) -> Optional[dict]:
        if not self.is_football_trained or not self.football_pipeline:
            return None

        def find_team(team: str) -> Optional[dict]:
            clean = team.lower().strip()
            best_match = next(
                (k for k in self._football_team_stats
                 if clean in k.lower() or k.lower() in clean),
                None
            )
            if not best_match:
                return None
            hist = list(self._football_team_stats[best_match])
            if len(hist) < 3:
                return None
            w = np.array([1 / (i + 1) for i in range(len(hist))][::-1])
            w /= w.sum()
            return {
                "avg_gs": float(np.dot(w, [h["gs"] for h in hist])),
                "avg_gc": float(np.dot(w, [h["gc"] for h in hist])),
                "form_pts": float(np.dot(w, [h["pts"] for h in hist])),
                "win_rate": sum(1 for h in hist if h["pts"] == 3) / len(hist),
            }

        hs = find_team(home_team)
        aws = find_team(away_team)
        if not hs or not aws:
            return None

        fv = [
            hs["avg_gs"], hs["avg_gc"], hs["form_pts"], hs["win_rate"],
            aws["avg_gs"], aws["avg_gc"], aws["form_pts"], aws["win_rate"],
            hs["avg_gs"] - aws["avg_gc"],
            aws["avg_gs"] - hs["avg_gc"],
        ]
        X = np.nan_to_num(np.array([fv], dtype=np.float64))
        X_scaled = self.football_pipeline["scaler"].transform(X)

        try:
            probs = self.football_pipeline["model"].predict_proba(X_scaled)[0]
            # FIX: استفاده از classes_ واقعی مدل
            classes = self.football_pipeline["model"].classes_
            label_map = {0: "home_win", 1: "draw", 2: "away_win"}
            return {
                label_map.get(int(c), f"class_{c}"): round(float(p), 4)
                for c, p in zip(classes, probs)
            }
        except Exception as e:
            logger.warning("[ML FOOTBALL] Predict error: %s", e)
            return None

    # ─────────────────────────────────────────────────────
    # Tennis  ← FIX: balanced labels + instance RNG
    # ─────────────────────────────────────────────────────
    def load_or_train_tennis_model(self, is_wta: bool = False):
        tour = "wta" if is_wta else "atp"
        path = CFG.ML_DIR / f"tennis_model_{tour}_v6.pkl"
        if path.exists():
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            if age_hours < 24.0:
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    self.tennis_pipeline = data["pipeline"]
                    self._tennis_metrics = data.get("metrics", {})
                    self.is_tennis_trained = True
                    logger.info(
                        "⚡ [ML TENNIS %s] Loaded pre-trained model from cache!",
                        tour.upper()
                    )
                    return
                except Exception as e:
                    logger.warning("Tennis model cache corrupt: %s", e)

        self.train_tennis_model(is_wta)
        if self.is_tennis_trained:
            try:
                with open(path, "wb") as f:
                    pickle.dump(
                        {
                            "pipeline": self.tennis_pipeline,
                            "metrics": self._tennis_metrics,
                        },
                        f,
                    )
                logger.info(
                    "💾 [ML TENNIS %s] Model saved to cache.", tour.upper()
                )
            except Exception as e:
                logger.warning("Tennis model save error: %s", e)

    def _build_tennis_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features, labels, weights = [], [], []

        for _, row in df.iterrows():
            wr = float(row.get("winner_rank", 0) or 0)
            lr = float(row.get("loser_rank", 0) or 0)
            if wr <= 0 or lr <= 0:
                continue

            surface = str(row.get("surface", "Hard") or "Hard").lower()
            best_of = float(row.get("best_of", 3) or 3)

            def safe_float(v, default=0.0) -> float:
                try:
                    return float(v or default)
                except (TypeError, ValueError):
                    return default

            w_stats = [
                safe_float(row.get("w_ace")), safe_float(row.get("w_df")),
                safe_float(row.get("w_svpt", 50)),
                safe_float(row.get("w_1stIn")), safe_float(row.get("w_1stWon")),
                safe_float(row.get("w_2ndWon")),
                safe_float(row.get("w_bpSaved")), safe_float(row.get("w_bpFaced")),
            ]
            l_stats = [
                safe_float(row.get("l_ace")), safe_float(row.get("l_df")),
                safe_float(row.get("l_svpt", 50)),
                safe_float(row.get("l_1stIn")), safe_float(row.get("l_1stWon")),
                safe_float(row.get("l_2ndWon")),
                safe_float(row.get("l_bpSaved")), safe_float(row.get("l_bpFaced")),
            ]

            def normalize_serve(s: list) -> list:
                svpt = max(s[2], 1.0)
                in1st = max(s[3], 1.0)
                bpf = max(s[7], 1.0)
                return [
                    s[0] / svpt,            # ace rate
                    s[1] / svpt,            # df rate
                    s[3] / svpt,            # 1st serve in
                    s[4] / in1st,           # 1st serve won
                    s[5] / max(svpt - s[3], 1.0),  # 2nd serve won
                    s[6] / bpf,             # bp saved
                ]

            wn = normalize_serve(w_stats)
            ln = normalize_serve(l_stats)
            w_age = safe_float(row.get("winner_age"), 25.0)
            l_age = safe_float(row.get("loser_age"), 25.0)

            # FIX: استفاده از instance RNG (نه global seed)
            is_p1_winner = self._rng.rand() > 0.5

            if is_p1_winner:
                p1_rank, p2_rank = wr, lr
                p1_age, p2_age = w_age, l_age
                p1_stats, p2_stats = wn, ln
                label = 1
            else:
                p1_rank, p2_rank = lr, wr
                p1_age, p2_age = l_age, w_age
                p1_stats, p2_stats = ln, wn
                label = 0

            rank_diff = p2_rank - p1_rank
            rank_ratio = p2_rank / max(p1_rank, 1.0)

            fv = [
                p1_rank, p2_rank, rank_diff, rank_ratio,
                p1_age, p2_age,
                1.0 if surface == "hard" else 0.0,
                1.0 if surface == "clay" else 0.0,
                1.0 if surface == "grass" else 0.0,
                best_of,
                *p1_stats, *p2_stats,
                p1_stats[0] - p2_stats[0],
                p1_stats[3] - p2_stats[3],
                p1_stats[5] - p2_stats[5],
            ]

            features.append(fv)
            labels.append(label)

            td = safe_float(row.get("tourney_date"), 20200101)
            weight = float(
                np.clip(
                    0.5 + 0.5 * (td - 20200101) / max(20260101 - 20200101, 1),
                    0.5, 1.0
                )
            )
            weights.append(weight)

        if not features:
            return np.array([]), np.array([]), np.array([])

        return (
            np.nan_to_num(np.array(features, dtype=np.float64)),
            np.array(labels, dtype=np.int32),
            np.array(weights, dtype=np.float64),
        )

    def train_tennis_model(self, is_wta: bool = False):
        df = self.data_engine.wta_matches if is_wta else self.data_engine.atp_matches
        tour = "WTA" if is_wta else "ATP"

        if df is None or len(df) < 500:
            logger.warning("[ML TENNIS %s] Insufficient data", tour)
            return

        X, y, sample_weights = self._build_tennis_features(df)
        if len(X) < 200:
            logger.warning("[ML TENNIS %s] Too few features (%d)", tour, len(X))
            return

        # FIX: بررسی class balance
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logger.warning(
                "[ML TENNIS %s] Only 1 class found in training data!", tour
            )
            return

        class_balance = dict(zip(unique.tolist(), counts.tolist()))
        logger.info("[ML TENNIS %s] Class distribution: %s", tour, class_balance)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42, subsample=0.8,
        )

        try:
            # sample_weight در fit
            calibrated = CalibratedClassifierCV(gb, cv=3, method="isotonic")
            calibrated.fit(X_scaled, y)

            self.tennis_pipeline = {"model": calibrated, "scaler": scaler}
            self.is_tennis_trained = True
            logger.info("✅ [ML TENNIS %s] Trained on %d samples", tour, len(X))
        except Exception as e:
            logger.error("[ML TENNIS %s] Training failed: %s", tour, e)

    def predict_tennis(
        self,
        player_a: str,
        player_b: str,
        stats: dict,
        surface: str = "hard",
    ) -> Optional[dict]:
        if not self.is_tennis_trained or not self.tennis_pipeline:
            return None

        pa_stats = stats.get("player_a", {})
        pb_stats = stats.get("player_b", {})
        rank_a = float(pa_stats.get("current_ranking", 100) or 100)
        rank_b = float(pb_stats.get("current_ranking", 100) or 100)

        def get_serve(p: dict) -> list:
            svpt = max(float(p.get("svpt_per_match", 50) or 50), 1.0)
            ace = float(p.get("aces_per_match", 5) or 5)
            df_val = float(p.get("df_per_match", 2) or 2)
            in1st = float(p.get("first_serve_in_pct", 0.6) or 0.6)
            won1st = float(p.get("first_serve_win_pct", 0.7) or 0.7)
            bpsaved = float(p.get("bp_saved_pct", 0.6) or 0.6)
            return [ace / svpt, df_val / svpt, in1st, won1st, 0.5, bpsaved]

        wa = get_serve(pa_stats)
        wb = get_serve(pb_stats)

        fv = [
            rank_a, rank_b, rank_b - rank_a, rank_b / max(rank_a, 1.0),
            25.0, 25.0,
            1.0 if surface == "hard" else 0.0,
            1.0 if surface == "clay" else 0.0,
            1.0 if surface == "grass" else 0.0,
            3.0,
            *wa, *wb,
            wa[0] - wb[0], wa[3] - wb[3], wa[5] - wb[5],
        ]

        try:
            X = np.nan_to_num(np.array([fv], dtype=np.float64))
            X_scaled = self.tennis_pipeline["scaler"].transform(X)
            probs = self.tennis_pipeline["model"].predict_proba(X_scaled)[0]

            # FIX: classes_ ممکن است [0] یا [1] باشد
            classes = self.tennis_pipeline["model"].classes_
            prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
            prob_a = prob_map.get(1, 0.5)

            return {
                f"{player_a}_win_prob": round(prob_a, 4),
                f"{player_b}_win_prob": round(1 - prob_a, 4),
            }
        except Exception as e:
            logger.warning("[ML TENNIS] Predict error: %s", e)
            return None


# =========================================================
# 11. DIXON-COLES ENGINE
# =========================================================
class PoissonEngine:

    @staticmethod
    def calculate_match_probabilities(
        home_team: str, away_team: str, df: Optional[pd.DataFrame]
    ) -> dict:
        if df is None or df.empty:
            return {}

        required_cols = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not required_cols.issubset(df.columns):
            return {}

        recent_df = df.dropna(subset=["FTHG", "FTAG"]).tail(1500).copy()
        if len(recent_df) < 50:
            return {}

        # FIX: safe mean calculation
        league_avg_home = recent_df["FTHG"].astype(float).mean()
        league_avg_away = recent_df["FTAG"].astype(float).mean()

        if pd.isna(league_avg_home) or league_avg_home == 0:
            return {}

        def _fuzzy(t: str, col: pd.Series) -> pd.Series:
            clean = t.lower().strip()
            mask = col.str.lower().str.strip() == clean
            if mask.any():
                return mask
            for part in clean.split():
                if len(part) > 3:
                    m = col.str.lower().str.contains(re.escape(part), na=False)
                    if m.any():
                        return m
            return pd.Series([False] * len(col), index=col.index)

        home_matches = recent_df[_fuzzy(home_team, recent_df["HomeTeam"])]
        if len(home_matches) < 5:
            return {}

        away_matches = recent_df[_fuzzy(away_team, recent_df["AwayTeam"])]
        if len(away_matches) < 5:
            return {}

        home_atk = home_matches["FTHG"].astype(float).mean() / league_avg_home
        home_def = home_matches["FTAG"].astype(float).mean() / league_avg_away
        away_atk = away_matches["FTAG"].astype(float).mean() / league_avg_away
        away_def = away_matches["FTHG"].astype(float).mean() / league_avg_home

        # FIX: NaN guard
        if any(pd.isna(v) or v == 0 for v in [home_atk, home_def, away_atk, away_def]):
            return {}

        home_xg = float(np.clip(home_atk * away_def * league_avg_home, 0.1, 8.0))
        away_xg = float(np.clip(away_atk * home_def * league_avg_away, 0.1, 8.0))

        max_goals = 6
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        rho = -0.1

        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                base = (
                    stats_scipy.poisson.pmf(x, home_xg)
                    * stats_scipy.poisson.pmf(y, away_xg)
                )
                if x == 0 and y == 0:
                    adj = max(0.0, 1 - home_xg * away_xg * rho)
                elif x == 0 and y == 1:
                    adj = max(0.0, 1 + home_xg * rho)
                elif x == 1 and y == 0:
                    adj = max(0.0, 1 + away_xg * rho)
                elif x == 1 and y == 1:
                    adj = max(0.0, 1 - rho)
                else:
                    adj = 1.0
                prob_matrix[x, y] = base * adj

        total = prob_matrix.sum()
        if total == 0:
            return {}
        prob_matrix /= total

        home_win = float(np.sum(np.tril(prob_matrix, -1)))
        draw = float(np.sum(np.diag(prob_matrix)))
        away_win = float(np.sum(np.triu(prob_matrix, 1)))

        return {
            "home_xg": round(home_xg, 2),
            "away_xg": round(away_xg, 2),
            "home_win_prob_poisson": round(home_win, 3),
            "draw_prob_poisson": round(draw, 3),
            "away_win_prob_poisson": round(away_win, 3),
        }


# =========================================================
# 12. EV ENGINE
# =========================================================
class EVEngine:

    @staticmethod
    def remove_vig_power(odds_list: List[float]) -> List[float]:
        implied = [1 / o for o in odds_list if o > 1.0]
        if not implied:
            return []

        total = sum(implied)
        if abs(total - 1.0) < 0.001:
            return implied

        def f(k: float) -> float:
            return sum(p ** k for p in implied) - 1.0

        try:
            from scipy.optimize import brentq
            # FIX: fallback اگه f(0.5) و f(3.0) هم‌علامت باشند
            if f(0.5) * f(3.0) >= 0:
                return [p / total for p in implied]
            k = brentq(f, 0.5, 3.0, xtol=1e-6)
            true_probs = [p ** k for p in implied]
            s = sum(true_probs)
            return [p / s for p in true_probs] if s > 0 else [p / total for p in implied]
        except Exception:
            return [p / total for p in implied]

    @staticmethod
    def remove_vig_shin(odds_list: List[float]) -> List[float]:
        implied = [1 / o for o in odds_list if o > 1.0]
        if not implied:
            return []

        n = len(implied)
        total = sum(implied)
        if total == 0:
            return implied

        z_est = max(0.0, min((total - 1) / max(total - 1 / n, 1e-10), 0.2))
        true_probs = []
        for p in implied:
            denom = 2 * (1 - n * z_est)
            if abs(denom) < 1e-10:
                true_probs.append(p / total)
                continue
            inner = z_est ** 2 + 4 * (1 - z_est) * (p ** 2 / total)
            if inner < 0:
                true_probs.append(p / total)
                continue
            numerator = -z_est + (inner ** 0.5)
            true_probs.append(numerator / denom)

        s = sum(true_probs)
        return [p / s for p in true_probs] if s > 0 else [p / total for p in implied]

    @staticmethod
    def kelly_criterion(
        prob: float, odds: float, fraction: float = CFG.KELLY_FRACTION
    ) -> float:
        b = odds - 1.0
        if b <= 0 or prob <= 0 or prob >= 1:
            return 0.0
        q = 1.0 - prob
        kelly = (prob * b - q) / b
        kelly = max(0.0, kelly)
        return round(min(kelly * fraction, CFG.MAX_KELLY_PCT / 100), 4)


def calculate_sharp_ev_advanced(
    markets_data: dict, bookmakers_raw: list
) -> list:
    best_per_market: dict = {}

    for market_key, market_data_list in markets_data.items():
        sharp_odds_all: Dict[str, List[float]] = defaultdict(list)
        soft_odds_all: Dict[str, List[float]] = defaultdict(list)
        best_market_odds: Dict[str, Tuple[float, str]] = {}

        for entry in market_data_list:
            bk = entry.get("bookmaker_key", "")
            is_sharp = bk in CFG.SHARP_BOOKMAKERS

            for o in entry.get("outcomes", []):
                name = (
                    f"{o['name']} {o.get('point')}"
                    if o.get("point") is not None
                    else o.get("name", "")
                )
                if not name:
                    continue
                try:
                    price = float(o["price"])
                except (KeyError, TypeError, ValueError):
                    continue

                if price <= 1.0:
                    continue

                if is_sharp:
                    sharp_odds_all[name].append(price)
                else:
                    soft_odds_all[name].append(price)

                if name not in best_market_odds or price > best_market_odds[name][0]:
                    best_market_odds[name] = (price, entry.get("bookmaker", "Unknown"))

        if not best_market_odds:
            continue

        sharp_best = {
            name: max(prices)
            for name, prices in sharp_odds_all.items()
            if prices
        }
        soft_best = {
            name: max(prices)
            for name, prices in soft_odds_all.items()
            if prices
        }

        if not sharp_best:
            sharp_best = dict(soft_best)

        if not sharp_best:
            continue

        has_real_sharp = bool(sharp_odds_all)
        outcomes = list(sharp_best.keys())
        odds_list = [sharp_best[o] for o in outcomes]

        # Validation
        implied_sum = sum(1 / o for o in odds_list if o > 0)
        if not (CFG.MIN_VALID_IMPLIED_SUM <= implied_sum <= CFG.MAX_VALID_IMPLIED_SUM):
            continue

        min_outcomes = CFG.MARKET_EXPECTED_OUTCOMES.get(market_key, {}).get("min", 2)
        if len(outcomes) < min_outcomes:
            continue

        # Triple devigging
        try:
            true_probs_power = EVEngine.remove_vig_power(odds_list)
            true_probs_shin = EVEngine.remove_vig_shin(odds_list)

            if len(true_probs_power) != len(outcomes) or \
               len(true_probs_shin) != len(outcomes):
                raise ValueError("Length mismatch")

            true_probs = {
                outcomes[i]: 0.6 * true_probs_power[i] + 0.4 * true_probs_shin[i]
                for i in range(len(outcomes))
            }
        except Exception:
            true_probs = {
                outcomes[i]: (1 / odds_list[i]) / max(implied_sum, 1e-10)
                for i in range(len(outcomes))
            }

        min_odds = CFG.H2H_MIN_ODDS if market_key == "h2h" else CFG.TOTALS_MIN_ODDS
        min_ev = CFG.H2H_MIN_EV if market_key == "h2h" else CFG.TOTALS_MIN_EV
        if not has_real_sharp:
            min_ev *= 1.8

        best_opp = None

        for outcome_name in outcomes:
            true_prob = true_probs.get(outcome_name, 0)
            if true_prob <= 0 or true_prob >= 1:
                continue

            best_price, best_bm = best_market_odds.get(outcome_name, (0, "Unknown"))
            if best_price <= 1.0:
                continue

            ev = (true_prob * best_price) - 1.0

            if ev < min_ev or ev > CFG.MAX_REALISTIC_EV:
                continue
            if best_price < min_odds:
                continue

            kelly_pct = EVEngine.kelly_criterion(true_prob, best_price)

            sharp_price = sharp_best.get(outcome_name, best_price)
            clv = (best_price / sharp_price - 1) * 100 if sharp_price > 0 else 0.0

            opp = {
                "pick": outcome_name,
                "market": market_key,
                "market_label": get_market_label(market_key),
                "prob": round(true_prob, 4),
                "odds": round(best_price, 3),
                "bookmaker": best_bm,
                "ev": round(ev, 4),
                "edge_pct": round(ev * 100, 2),
                "kelly_pct": round(kelly_pct * 100, 2),
                "clv_pct": round(clv, 2),
                "has_sharp_line": has_real_sharp,
                "devigging_method": "power_shin_weighted",
                "steam_pct": 0.0,
            }

            if best_opp is None or opp["ev"] > best_opp["ev"]:
                best_opp = opp

        if best_opp:
            best_per_market[market_key] = best_opp

    all_opps = sorted(best_per_market.values(), key=lambda x: x["ev"], reverse=True)
    return all_opps[:1]


# FIX: backward compat alias
def calculate_sharp_ev(markets_data: dict, bookmakers_raw: list) -> list:
    return calculate_sharp_ev_advanced(markets_data, bookmakers_raw)


# =========================================================
# 13. CONFIDENCE ENGINE
# =========================================================
class ConfidenceEngine:
    WEIGHTS = {
        "base": 40,
        "ev_high": 15, "ev_medium": 10, "ev_low": 5,
        "sharp_line": 10, "football_stats": 5,
        "elo_high": 10, "elo_medium": 5,
        "ml_strong": 10, "ml_medium": 5,
        "poisson_confirm": 5,
        "smart_money_steam": 10,
        "kelly_high": 5, "kelly_medium": 3,
    }

    @classmethod
    def calculate_math_score(
        cls, opp: dict, stats: dict, market: str,
        ml_prediction: Optional[dict] = None,
        poisson_prediction: Optional[dict] = None,
    ) -> int:
        score = cls.WEIGHTS["base"]

        ev_pct = opp.get("ev", 0) * 100
        if ev_pct > 5.0:
            score += cls.WEIGHTS["ev_high"]
        elif ev_pct > 3.0:
            score += cls.WEIGHTS["ev_medium"]
        elif ev_pct > 1.5:
            score += cls.WEIGHTS["ev_low"]

        if opp.get("has_sharp_line"):
            score += cls.WEIGHTS["sharp_line"]

        kelly = opp.get("kelly_pct", 0)
        if kelly > 2.0:
            score += cls.WEIGHTS["kelly_high"]
        elif kelly > 1.0:
            score += cls.WEIGHTS["kelly_medium"]

        if stats.get("football_stats"):
            score += cls.WEIGHTS["football_stats"]

        elo = stats.get("elo_data", {})
        if elo:
            delta = abs(elo.get("delta", 0))
            if delta > 150:
                score += cls.WEIGHTS["elo_high"]
            elif delta > 75:
                score += cls.WEIGHTS["elo_medium"]

        if ml_prediction:
            prob_values = [
                v for v in ml_prediction.values()
                if isinstance(v, float) and 0 < v <= 1.0
            ]
            if prob_values and max(prob_values) > 0.65:
                score += cls.WEIGHTS["ml_strong"]
            elif prob_values and max(prob_values) > 0.55:
                score += cls.WEIGHTS["ml_medium"]

        if poisson_prediction:
            score += cls.WEIGHTS["poisson_confirm"]

        steam_pct = opp.get("steam_pct", 0)
        if steam_pct >= 3.0:
            score += cls.WEIGHTS["smart_money_steam"]

        return int(np.clip(score, 0, 100))


# =========================================================
# 14. UTILITIES
# =========================================================
def retry_request(max_retries: int = 3, delay: float = 2, backoff: float = 2):
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
                        wait = int(
                            e.response.headers.get("Retry-After", current_delay * 3)
                        )
                        logger.warning("Rate limit 429, sleeping %ds", wait)
                        time.sleep(wait)
                    elif status in [401, 403]:
                        return None
                    else:
                        if attempt == max_retries - 1:
                            return None
                except (
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException,
                ):
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
    clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # جستجوی JSON در متن
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", clean))):
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict) and result:
                return result
        except json.JSONDecodeError:
            continue

    # تلاش نهایی با greedy match
    try:
        m = re.search(r"\{[\s\S]*\}", clean)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass

    return None


def clean_team_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(name or "")).strip()


def normalize_sport_key(sport_title: str) -> str:
    lower = (sport_title or "").lower()
    if any(k in lower for k in ["tennis", "atp", "wta"]):
        return "tennis"
    if any(k in lower for k in [
        "soccer", "football", "premier league", "la liga",
        "bundesliga", "serie a", "ligue 1", "champions",
        "brasileirao", "serie b", "primera division",
        "superliga", "liga mx",
    ]):
        return "football"
    if any(k in lower for k in ["basketball", "nba", "euroleague", "nbl"]):
        return "basketball"
    if any(k in lower for k in ["baseball", "mlb"]):
        return "baseball"
    return "other"


def get_countdown_str(commence_time_str: str, now_utc: datetime) -> str:
    try:
        match_time = datetime.fromisoformat(
            commence_time_str.replace("Z", "+00:00")
        )
        if match_time.tzinfo is None:
            match_time = match_time.replace(tzinfo=timezone.utc)
        minutes_left = int((match_time - now_utc).total_seconds() / 60)
        if minutes_left > 60:
            return f"{minutes_left // 60}h {minutes_left % 60}m"
        if minutes_left > 0:
            return f"{minutes_left}m"
        return "LIVE"
    except Exception:
        return "N/A"


def get_market_label(market_key: str) -> str:
    return {
        "h2h": "Match Winner",
        "totals": "Over/Under",
        "h2h_lay": "Lay (Betting Against)",
        "spreads": "Point Spread / Handicap",
    }.get(market_key, market_key.replace("_", " ").title())


def _get_sport_emoji(sport_key: str) -> str:
    return {
        "tennis": "🎾",
        "football": "⚽",
        "basketball": "🏀",
        "baseball": "⚾",
        "hockey": "🏒",
    }.get(sport_key, "🏆")


# =========================================================
# 15. SMART MONEY / LINE MOVEMENT
# =========================================================
class LineMovementTracker:
    def __init__(self):
        self.history_file = CFG.CACHE_DIR / "line_movement.json"
        self.data = CacheManager.load(self.history_file)
        self._cleanup_old_entries()

    def _cleanup_old_entries(self):
        """FIX: cleanup را به یک متد جداگانه تبدیل کردیم."""
        now = datetime.now(timezone.utc)
        to_delete = [
            k for k, v in self.data.items()
            if not isinstance(v, dict)
            or (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(
                    v.get("timestamp", "2000-01-01T00:00:00+00:00")
                ).replace(tzinfo=timezone.utc)
                > timedelta(hours=24)
            )
        ]
        for key in to_delete:
            self.data.pop(key, None)

    def record_and_get_movement(
        self,
        home: str, away: str, market: str, outcome: str, current_odds: float,
    ) -> float:
        if current_odds <= 1.0:
            return 0.0

        match_key = hashlib.md5(
            f"{home}|{away}|{market}|{outcome}".encode()
        ).hexdigest()
        now_str = datetime.now(timezone.utc).isoformat()

        if match_key not in self.data:
            self.data[match_key] = {
                "initial_odds": current_odds,
                "current_odds": current_odds,
                "timestamp": now_str,
            }
            CacheManager.save(self.history_file, self.data)
            return 0.0

        initial_odds = self.data[match_key].get("initial_odds", current_odds)
        self.data[match_key]["current_odds"] = current_odds
        self.data[match_key]["timestamp"] = now_str
        CacheManager.save(self.history_file, self.data)

        if initial_odds <= 0:
            return 0.0

        drop_pct = (initial_odds / current_odds - 1) * 100
        return round(drop_pct, 2)


line_movement_tracker = LineMovementTracker()


# =========================================================
# 16. DUAL AI ANALYSIS  ← بازنویسی با Google AI SDK
# =========================================================
def generate_dual_ai_analysis(
    home: str, away: str, sport: str, pick: str, market: str,
    opp: dict, stats: dict, has_real_stats: bool, math_score: int,
    ml_prediction: Optional[dict] = None,
    poisson_prediction: Optional[dict] = None,
    sport_key: str = "other",
) -> dict:

    default_response = {
        "sport_emoji": _get_sport_emoji(sport_key),
        "ai_confidence": math_score,
        "math_confidence": math_score,
        "final_confidence": math_score,
        "risk_level": "Medium",
        "logic": "Quantitative edge detected via mathematical variance analysis.",
    }

    # Build context string
    context_parts = []

    if stats.get("historical_data"):
        pa = stats["historical_data"].get("player_a", {})
        pb = stats["historical_data"].get("player_b", {})
        context_parts.append(
            f"RANK/FORM: {home}(R:{pa.get('current_ranking', '-')}, "
            f"WR:{pa.get('recent_win_rate', 0):.2f}) vs "
            f"{away}(R:{pb.get('current_ranking', '-')}, "
            f"WR:{pb.get('recent_win_rate', 0):.2f})"
        )

    if stats.get("football_stats"):
        hm = stats["football_stats"].get("home", {})
        aw = stats["football_stats"].get("away", {})
        if hm:
            context_parts.append(
                f"{home} METRICS: GF={hm.get('avg_scored', 0):.1f}, "
                f"GA={hm.get('avg_conceded', 0):.1f}, "
                f"Form={hm.get('form_string', 'N/A')}"
            )
        if aw:
            context_parts.append(
                f"{away} METRICS: GF={aw.get('avg_scored', 0):.1f}, "
                f"GA={aw.get('avg_conceded', 0):.1f}, "
                f"Form={aw.get('form_string', 'N/A')}"
            )
        if stats["football_stats"].get("elo_data"):
            elo = stats["football_stats"]["elo_data"]
            context_parts.append(
                f"ELO: {home}={elo.get('home_elo')}, "
                f"{away}={elo.get('away_elo')}, Delta={elo.get('delta')}"
            )

    if ml_prediction:
        context_parts.append(f"ML PROBS: {json.dumps(ml_prediction)}")
    if poisson_prediction:
        context_parts.append(f"POISSON xG: {json.dumps(poisson_prediction)}")

    context_parts.append(
        f"MARKET EDGE: EV={opp.get('edge_pct', 0):.2f}%, "
        f"Steam={opp.get('steam_pct', 0):.1f}%, "
        f"CLV={opp.get('clv_pct', 0):.1f}%, "
        f"Kelly={opp.get('kelly_pct', 0):.1f}%"
    )

    stats_str = "\n".join(context_parts) if context_parts else "NO EXTERNAL CONTEXT."

    system_instruction = (
        "ROLE: Elite Quantitative Sports Betting Analyst (Machine-like objectivity).\n"
        "GOAL: Evaluate structural betting data and output a strictly formatted JSON analysis.\n\n"
        "MANDATORY PROTOCOLS:\n"
        "1. ANTI-HALLUCINATION: Base logic 100% on the provided DATA CONTEXT only.\n"
        "2. TECHNICAL: Use sharp betting terminology (xG convergence, market resistance, CLV).\n"
        "3. LOGIC: Exactly 2 sentences maximum, highly condensed.\n"
        "4. AI CONFIDENCE: Strict integer 0-100 based ONLY on data context quality.\n"
        "5. OUTPUT: Valid JSON matching this exact schema:\n"
        '{"ai_confidence": int, "logic": "string", "sport_emoji": "string"}'
    )

    user_prompt = (
        f"MATCH: {home} vs {away}\n"
        f"SPORT: {sport}\n"
        f"TARGET PICK: {pick}\n"
        f"MARKET TYPE: {get_market_label(market)}\n\n"
        f"=== DATA CONTEXT ===\n{stats_str}\n==================\n\n"
        "Return a JSON object with keys: ai_confidence (int), logic (string), sport_emoji (string)"
    )

    # FIX: استفاده از Google AI SDK با system_instruction
    ai_data = gemini_manager.generate(
        prompt=user_prompt,
        system_instruction=system_instruction,
        temperature=CFG.AI_TEMPERATURE,
    )

    if not ai_data or not isinstance(ai_data, dict):
        logger.warning("[AI] Gemini returned no valid data, using math fallback")
        return default_response

    # Extract and validate fields
    try:
        ai_score_raw = ai_data.get("ai_confidence", math_score)
        ai_score = int(np.clip(float(ai_score_raw), 0, 100))
    except (ValueError, TypeError):
        ai_score = math_score

    final_conf = int(np.clip((math_score + ai_score) / 2, 0, 100))
    risk_level = (
        "Low" if final_conf >= CFG.HIGH_CONFIDENCE
        else ("Medium" if final_conf >= CFG.MEDIUM_CONFIDENCE else "High")
    )

    logic_raw = ai_data.get("logic", default_response["logic"])
    logic = str(logic_raw)[:600] if logic_raw else default_response["logic"]

    sport_emoji_raw = ai_data.get("sport_emoji", _get_sport_emoji(sport_key))
    sport_emoji = str(sport_emoji_raw) if sport_emoji_raw else _get_sport_emoji(sport_key)

    return {
        "sport_emoji": sport_emoji,
        "ai_confidence": ai_score,
        "math_confidence": math_score,
        "final_confidence": final_conf,
        "risk_level": risk_level,
        "logic": logic,
    }


# =========================================================
# 17. TELEGRAM  ← FIX: تمام فیلدها escape شده
# =========================================================
def send_telegram(message_html: str) -> bool:
    MAX_LEN = 4000
    chunks: List[str] = []

    if len(message_html) <= MAX_LEN:
        chunks.append(message_html)
    else:
        lines = message_html.split("\n")
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > MAX_LEN:
                if current:
                    chunks.append(current.strip())
                current = line + "\n"
            else:
                current += line + "\n"
        if current.strip():
            chunks.append(current.strip())

    success = True
    for chunk in chunks:
        try:
            res = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if not res.ok:
                logger.error(
                    "Telegram error [%d]: %s", res.status_code, res.text[:200]
                )
                success = False
        except requests.RequestException as e:
            logger.error("Telegram request failed: %s", e)
            success = False

    return success


def build_signal_message(
    home: str, away: str, sport: str, sport_key: str,
    opp: dict, ai_data: dict, stats: dict,
    math_score: int, ml_prediction: Optional[dict],
    poisson_prediction: Optional[dict],
    now_utc: datetime, commence_time: str,
) -> str:
    """
    ساخت پیام HTML با تمام escape های لازم.
    FIX: جداسازی منطق پیام از async_main برای خوانایی بهتر
    """
    final_confidence = ai_data["final_confidence"]

    conf_icon = (
        "🔥" if final_confidence >= CFG.HIGH_CONFIDENCE
        else ("✅" if final_confidence >= CFG.MEDIUM_CONFIDENCE else "⚡")
    )
    risk_icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(
        ai_data["risk_level"], "🟠"
    )

    # FIX: همه مقادیر string با html_lib.escape
    h_esc = html_lib.escape(home)
    a_esc = html_lib.escape(away)
    sport_esc = html_lib.escape(sport)
    pick_esc = html_lib.escape(str(opp["pick"]))
    market_label_esc = html_lib.escape(opp["market_label"])
    logic_esc = html_lib.escape(str(ai_data["logic"]))

    ev_line = f"📈 <b>Edge:</b> {opp['edge_pct']:.2f}%"
    kelly_line = f" | 💰 <b>Stake:</b> {opp.get('kelly_pct', 0):.1f}% bankroll"
    steam_line = (
        f" | 📉 <b>Steam:</b> -{opp['steam_pct']:.1f}%"
        if opp.get("steam_pct", 0) > 2.0 else ""
    )
    clv_line = (
        f" | 📊 <b>CLV:</b> {opp.get('clv_pct', 0):+.1f}%"
        if abs(opp.get("clv_pct", 0)) > 0.5 else ""
    )

    ml_line = (
        "\n🧠 <b>Math Models:</b> Verified"
        if ml_prediction or poisson_prediction else ""
    )

    data_badge = ""
    has_historical = bool(stats.get("historical_data"))
    has_football = bool(stats.get("football_stats"))
    has_elo = bool(stats.get("elo_data"))
    has_ml = bool(ml_prediction)
    has_us = bool(stats.get("us_sports"))

    if has_historical or has_football or has_elo or has_ml or has_us:
        badges = []
        if has_historical: badges.append("📚 Historical")
        if has_football:   badges.append("⚽ Match Data")
        if has_elo:        badges.append("📊 Elo")
        if has_ml:         badges.append("🧠 ML")
        if has_us:         badges.append("🇺🇸 US Stats")
        data_badge = "\n📋 <b>Sources:</b> " + " | ".join(badges)

    countdown = get_countdown_str(commence_time, now_utc)

    return (
        f"{ai_data.get('sport_emoji', '🏆')} <b>{sport_esc}</b>\n\n"
        f"⚔️ <b>{h_esc}</b> vs <b>{a_esc}</b>\n"
        f"⏳ <b>Starts in:</b> {countdown}\n\n"
        f"🎯 <b>PICK:</b> <code>{pick_esc}</code> @ <b>{opp['odds']}</b>\n"
        f"📊 <b>Market:</b> {market_label_esc}\n\n"
        f"{ev_line}{kelly_line}{steam_line}{clv_line}\n\n"
        f"{risk_icon} <b>Risk:</b> {ai_data['risk_level']}  |  "
        f"{conf_icon} <b>Final Confidence: {final_confidence}%</b>\n"
        f"⚙️ <i>(Math: {math_score} | AI: {ai_data['ai_confidence']})</i>"
        f"{ml_line}{data_badge}\n\n"
        f"💡 <b>AI ANALYSIS:</b>\n"
        f"<blockquote>{logic_esc}</blockquote>\n\n"
        f"🔍 <i>Curated by {html_lib.escape(CFG.TELEGRAM_ID)}</i>"
    )


# =========================================================
# 18. ODDS FETCHER
# =========================================================
class SmartOddsCache:
    def __init__(self):
        self.cache = CacheManager.load(CFG.ODDS_CACHE_FILE)

    def _make_key(self, markets: list, window_hours: float) -> str:
        now_rounded = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
        raw = f"{','.join(sorted(markets))}|{window_hours}|{now_rounded}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_cached(self, markets: list, window_hours: float) -> Optional[list]:
        key = self._make_key(markets, window_hours)
        if CacheManager.is_valid_minutes(self.cache, key, CFG.TTL_ODDS_CACHE_MINUTES):
            data = CacheManager.get(self.cache, key)
            if data:
                logger.info(
                    "💾 [ODDS CACHE] HIT! %d events (TTL=%.0fm)",
                    len(data), CFG.TTL_ODDS_CACHE_MINUTES
                )
                return data
        return None

    def save_cached(self, markets: list, window_hours: float, events: list):
        key = self._make_key(markets, window_hours)
        self.cache = CacheManager.set(self.cache, key, events)
        CacheManager.save(CFG.ODDS_CACHE_FILE, self.cache)
        logger.info("💾 [ODDS CACHE] SAVED %d events", len(events))

    def get_stale(
        self, markets: list, window_hours: float, max_ttl_hours: float = 2.0
    ) -> Optional[list]:
        key = self._make_key(markets, window_hours)
        if CacheManager.is_valid(self.cache, key, max_ttl_hours):
            return CacheManager.get(self.cache, key)
        return None


odds_cache = SmartOddsCache()


async def fetch_market_with_fallback(
    session: aiohttp.ClientSession,
    market: str,
    now_utc: datetime,
    api_key: str,
    key_label: str,
) -> Tuple[list, int, Optional[str]]:
    end_window = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    params = {
        "apiKey": api_key,
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
            remaining = int(res.headers.get("x-requests-remaining", -1))
            used = int(res.headers.get("x-requests-used", 0))

            if res.status == 200:
                events = await res.json(content_type=None)
                odds_key_manager.record_usage(key_label, used, remaining)
                logger.info(
                    "🔑 [%s] OK | Remaining: %d | Market: %s | Events: %d",
                    key_label, remaining, market, len(events)
                )

                valid = []
                for e in events:
                    if not isinstance(e, dict):
                        continue
                    try:
                        ct = e.get("commence_time", "")
                        mt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                        if mt.tzinfo is None:
                            mt = mt.replace(tzinfo=timezone.utc)
                        if now_utc <= mt <= end_window:
                            valid.append(e)
                    except Exception:
                        continue
                return valid, 200, None

            error_body = await res.text()
            reasons = {
                401: "Invalid API key",
                402: "Quota exhausted",
                429: "Rate limited",
                422: "Invalid params",
            }
            return (
                [], res.status,
                reasons.get(res.status, f"HTTP {res.status}: {error_body[:80]}")
            )

    except asyncio.TimeoutError:
        return [], 0, "Timeout"
    except aiohttp.ClientError as e:
        return [], 0, f"Connection: {str(e)[:80]}"
    except Exception as e:
        return [], 0, f"Error: {str(e)[:80]}"


async def fetch_all_odds_async() -> list:
    now_utc = datetime.now(timezone.utc)

    cached = odds_cache.get_cached(CFG.ODDS_API_MARKETS, CFG.MATCH_WINDOW_HOURS)
    if cached is not None:
        return cached

    logger.info("💾 [ODDS CACHE] MISS - Calling API...")
    active_keys = odds_key_manager.get_active_keys()
    all_events: Dict[str, dict] = {}
    success = False

    for key_info in active_keys:
        api_key = key_info["key"]
        key_label = key_info["label"]
        logger.info("🔑 [TRYING] %s...", key_label)

        connector = aiohttp.TCPConnector(limit=10, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                fetch_market_with_fallback(session, m, now_utc, api_key, key_label)
                for m in CFG.ODDS_API_MARKETS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        any_success = False
        hard_fail_status = None

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("🔑❌ [%s] Exception: %s", key_label, result)
                continue

            events, status, error = result

            if status == 200:
                any_success = True
                market_key = CFG.ODDS_API_MARKETS[i]
                for e in events:
                    eid = e.get("id")
                    if not eid:
                        continue
                    if eid not in all_events:
                        # کپی کامل event بدون bookmakers
                        all_events[eid] = {
                            k: v for k, v in e.items()
                            if k != "bookmakers"
                        }
                        all_events[eid]["_markets_data"] = {}

                    # FIX: ساختار صحیح _markets_data
                    for bm in e.get("bookmakers", []):
                        for m_data in bm.get("markets", []):
                            mk = m_data.get("key", "")
                            if not mk:
                                continue
                            if mk not in all_events[eid]["_markets_data"]:
                                all_events[eid]["_markets_data"][mk] = []
                            all_events[eid]["_markets_data"][mk].append({
                                "bookmaker": bm.get("title", ""),
                                "bookmaker_key": bm.get("key", ""),
                                "outcomes": m_data.get("outcomes", []),
                            })
            else:
                logger.warning(
                    "🔑⚠️ [%s] %s: %s",
                    key_label,
                    CFG.ODDS_API_MARKETS[i] if i < len(CFG.ODDS_API_MARKETS) else "?",
                    error,
                )
                if status in [401, 402, 429]:
                    hard_fail_status = status

        if any_success:
            success = True
            logger.info("✅ [%s] Fetched %d events", key_label, len(all_events))
            break
        else:
            key_index = next(
                (idx for idx, k in enumerate(odds_key_manager.keys)
                 if k["label"] == key_label),
                -1,
            )
            if key_index >= 0:
                reason = (
                    f"HTTP {hard_fail_status}"
                    if hard_fail_status else "All markets failed"
                )
                odds_key_manager.mark_failed(key_index, reason)
            logger.warning("🔑🔄 [%s] Failed → trying next key", key_label)

    if not success:
        logger.error("🔑❌ ALL KEYS FAILED!")
        stale = odds_cache.get_stale(
            CFG.ODDS_API_MARKETS, CFG.MATCH_WINDOW_HOURS, 2.0
        )
        if stale:
            logger.warning("💾 [STALE CACHE] Using %d cached events", len(stale))
            return stale
        return []

    final_events = list(all_events.values())
    odds_cache.save_cached(CFG.ODDS_API_MARKETS, CFG.MATCH_WINDOW_HOURS, final_events)
    logger.info("📊 [API USAGE] %s", odds_key_manager.get_usage_summary())
    return final_events


# =========================================================
# 19. MAIN PIPELINE
# =========================================================
async def async_main():
    logger.info("=" * 65)
    logger.info("  ZBET90 ENGINE v6.5 | Dual Confirmation (Math + Gemini 50/50)")
    logger.info("=" * 65)
    logger.info("🔑 [KEYS STATUS] %s", odds_key_manager.get_usage_summary())

    sent_history = SentHistory()
    now_utc = datetime.now(timezone.utc)

    # Phase 1: Load Data
    logger.info("📥 [PHASE 1] Loading data sources...")
    data_engine = FreeDataEngine()
    data_engine.load_tennis_data()
    data_engine.load_football_data()

    # Phase 2: Train ML
    logger.info("🧠 [PHASE 2] Initializing ML models...")
    ml_engine = MLPredictionEngine(data_engine)
    ml_engine.load_or_train_football_model()
    ml_engine.load_or_train_tennis_model(is_wta=False)

    # Phase 3: Fetch Odds
    logger.info(
        "📡 [PHASE 3] Fetching odds (window=%.1fh)...", CFG.MATCH_WINDOW_HOURS
    )
    events = await fetch_all_odds_async()

    if not events:
        logger.info("❌ No events in the %.1fh window.", CFG.MATCH_WINDOW_HOURS)
        logger.info("📊 [FINAL USAGE] %s", odds_key_manager.get_usage_summary())
        return

    logger.info("🔍 [PHASE 4] Analyzing %d events...", len(events))
    events.sort(key=lambda x: x.get("commence_time", ""))

    total_sent = 0
    total_analyzed = 0
    skipped_confidence = 0

    for event in events:
        home = clean_team_name(event.get("home_team", ""))
        away = clean_team_name(event.get("away_team", ""))
        sport = event.get("sport_title", "Unknown")
        sport_key = normalize_sport_key(sport)

        if not home or not away:
            continue

        opportunities = calculate_sharp_ev_advanced(
            event.get("_markets_data", {}),
            event.get("bookmakers", []),
        )
        if not opportunities:
            continue

        opp = opportunities[0]
        total_analyzed += 1

        if sent_history.was_sent(home, away, opp["market"]):
            continue

        # Smart Money
        steam_drop = line_movement_tracker.record_and_get_movement(
            home, away, opp["market"], opp["pick"], opp["odds"]
        )
        opp["steam_pct"] = steam_drop

        # Gather Stats
        stats: dict = {}
        ml_prediction: Optional[dict] = None
        poisson_prediction: Optional[dict] = None

        if sport_key == "tennis":
            is_wta = "wta" in sport.lower()
            tennis_stats = data_engine.get_tennis_stats(home, away, is_wta)
            if tennis_stats and (
                tennis_stats.get("player_a") or tennis_stats.get("player_b")
            ):
                stats["historical_data"] = tennis_stats
            if ml_engine.is_tennis_trained and tennis_stats:
                surface = "hard"
                ml_prediction = ml_engine.predict_tennis(
                    home, away, tennis_stats, surface
                )
                if ml_prediction:
                    stats["ml_prediction"] = ml_prediction

        elif sport_key == "football":
            fb_stats = data_engine.get_football_stats(home, away)
            if fb_stats and (fb_stats.get("home") or fb_stats.get("away")):
                stats["football_stats"] = fb_stats

            elo_data = data_engine.get_elo_delta(home, away)
            if elo_data:
                stats["elo_data"] = elo_data

            if ml_engine.is_football_trained:
                ml_prediction = ml_engine.predict_football(home, away)
                if ml_prediction:
                    stats["ml_prediction"] = ml_prediction

            poisson_prediction = PoissonEngine.calculate_match_probabilities(
                home, away, data_engine.football_data.get("all")
            )
            if poisson_prediction:
                stats["poisson_prediction"] = poisson_prediction

            # Dual confirmation filter
            if ml_prediction and poisson_prediction and opp["market"] == "h2h":
                pick_lower = opp["pick"].lower()
                home_lower = home.lower()
                away_lower = away.lower()

                is_home_pick = any(
                    part in pick_lower for part in home_lower.split() if len(part) > 2
                )
                is_away_pick = any(
                    part in pick_lower for part in away_lower.split() if len(part) > 2
                )

                ml_prob = (
                    ml_prediction.get("home_win", 0) if is_home_pick
                    else (ml_prediction.get("away_win", 0) if is_away_pick else 0.5)
                )
                poisson_prob = (
                    poisson_prediction.get("home_win_prob_poisson", 0) if is_home_pick
                    else (poisson_prediction.get("away_win_prob_poisson", 0) if is_away_pick else 0.5)
                )

                if ml_prob < 0.42 or poisson_prob < 0.42:
                    logger.info(
                        "⏭️ SKIP (Dual Filter): %s vs %s | ML=%.2f | Poisson=%.2f",
                        home, away, ml_prob, poisson_prob
                    )
                    continue

        elif sport_key in ["baseball", "basketball"]:
            us_stats = {
                "home": data_engine.get_us_sports_stats(sport, home),
                "away": data_engine.get_us_sports_stats(sport, away),
            }
            if us_stats["home"] or us_stats["away"]:
                stats["us_sports"] = us_stats

        # Math Score
        math_score = ConfidenceEngine.calculate_math_score(
            opp, stats, opp["market"], ml_prediction, poisson_prediction
        )

        # AI Analysis
        ai_data = generate_dual_ai_analysis(
            home, away, sport, opp["pick"], opp["market"],
            opp, stats,
            bool(stats.get("historical_data") or stats.get("football_stats")
                 or stats.get("elo_data") or ml_prediction or stats.get("us_sports")),
            math_score, ml_prediction, poisson_prediction, sport_key,
        )

        final_confidence = ai_data["final_confidence"]

        if final_confidence < CFG.MIN_CONFIDENCE_TO_SEND:
            skipped_confidence += 1
            logger.info(
                "⏭️ SKIP (Math:%d | AI:%d | Final:%d < %d): %s vs %s | %s",
                math_score, ai_data["ai_confidence"], final_confidence,
                CFG.MIN_CONFIDENCE_TO_SEND, home, away, opp["pick"],
            )
            continue

        logger.info(
            "✅ [APPROVED] Math:%d | AI:%d | EV=%.2f%% | Conf=%d%%",
            math_score, ai_data["ai_confidence"], opp["edge_pct"], final_confidence,
        )

        # Build and send message
        msg = build_signal_message(
            home, away, sport, sport_key, opp, ai_data, stats,
            math_score, ml_prediction, poisson_prediction,
            now_utc, event.get("commence_time", ""),
        )

        if send_telegram(msg):
            sent_history.mark_sent(home, away, opp["pick"], opp["market"])
            performance_tracker.record_signal(
                home, away, opp["pick"], opp["market"],
                opp["odds"], opp["ev"], final_confidence, opp["prob"],
                sport_key, event.get("sport_key", ""),
            )
            total_sent += 1
            logger.info(
                "📤 SENT: %s vs %s | EV=%.2f%% | Conf=%d%%",
                home, away, opp["edge_pct"], final_confidence,
            )
        else:
            logger.error("❌ Telegram failed: %s vs %s", home, away)

        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    # Summary
    logger.info("=" * 65)
    logger.info(
        "📊 SUMMARY | Analyzed: %d | Sent: %d | Skipped(conf): %d",
        total_analyzed, total_sent, skipped_confidence,
    )
    if total_sent == 0 and total_analyzed == 0:
        logger.info("ℹ️  No qualifying +EV opportunities in current window.")
    elif total_sent == 0:
        logger.info(
            "ℹ️  %d opportunities found but filtered by confidence threshold (%d).",
            total_analyzed, CFG.MIN_CONFIDENCE_TO_SEND,
        )

    logger.info("📊 [FINAL API USAGE] %s", odds_key_manager.get_usage_summary())

    perf = performance_tracker.data.get("summary", {})
    if perf.get("resolved", 0) > 0:
        logger.info(
            "📈 [PERFORMANCE] WR=%.1f%% | ROI=%.1f%% | Signals=%d",
            perf["win_rate"] * 100,
            perf["roi_pct"],
            perf["total_signals"],
        )
    logger.info("=" * 65)


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.critical("SYSTEM FAILURE: %s", str(e), exc_info=True)
        sys.exit(1)
