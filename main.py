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
import httpx
import numpy as np
import pandas as pd
import pickle
from io import StringIO
from groq import Groq
from functools import wraps
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV


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

    # 🔄 NEW: کش مخصوص Odds API
    ODDS_CACHE_FILE: Path = Path("api_cache/odds_cache.json")
    API_USAGE_FILE: Path = Path("api_cache/api_usage_tracker.json")

    MATCH_WINDOW_HOURS: float = 2.0
    TELEGRAM_SLEEP_BETWEEN: float = 3.0

    ODDS_API_MARKETS: list = field(default_factory=lambda: ["h2h", "totals"])
    ODDS_API_REGIONS: str = "eu,us,uk,au"

    # 🔄 NEW: TTL برای کش Odds (دقیقه) - odds هر ۸ دقیقه آپدیت میشن
    TTL_ODDS_CACHE_MINUTES: float = 6.0

    TTL_SENT_HISTORY: float = 48.0
    TTL_MATCH_ID: float = 24.0
    TTL_TEAM_FORM: float = 6.0
    TTL_H2H: float = 24.0
    TTL_GITHUB_DATA: float = 12.0

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

    GITHUB_SOURCES: dict = field(default_factory=lambda: {
        "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
        "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
        "atp_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv",
        "wta_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv",
        "football_eu": "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv",
        "openfootball_cl": "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/cl.json",
        "nba_games": "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_team.csv",
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
        "2324", "2425"
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
# 3. API KEYS - سیستم Fallback هوشمند  🔄 NEW
# =========================================================
class OddsAPIKeyManager:
    """
    مدیریت هوشمند ۳ کلید API:
    - اول از کش استفاده می‌کنه
    - اگه کش نبود، Key 1 رو امتحان می‌کنه
    - اگه Key 1 خطا داد (429/402/exhausted)، Key 2
    - اگه Key 2 هم خطا داد، Key 3
    - مصرف هر Key رو ترک می‌کنه
    """

    def __init__(self):
        self.keys: List[Dict] = []
        self._load_keys()
        self.usage = self._load_usage()

    def _load_keys(self):
        """بارگذاری کلیدها به ترتیب اولویت."""
        key_envs = [
            ("ODDS_API_KEY", "primary"),
            ("ODDS_API_KEY2", "backup_1"),
            ("ODDS_API_KEY3", "backup_2"),
        ]
        for env_name, label in key_envs:
            key = os.getenv(env_name, "").strip()
            if key:
                self.keys.append({
                    "key": key,
                    "label": label,
                    "env": env_name,
                    "failed": False,
                    "fail_reason": None,
                    "fail_time": None,
                })
                logger.info("🔑 [API KEY] %s (%s): Loaded ✓", label, env_name)
            else:
                logger.debug("🔑 [API KEY] %s (%s): Not set", label, env_name)

        if not self.keys:
            logger.critical("FATAL: No ODDS_API_KEY found!")
            sys.exit(1)

        logger.info("🔑 [API KEYS] %d key(s) available for fallback", len(self.keys))

    def _load_usage(self) -> dict:
        """بارگذاری آمار مصرف."""
        try:
            if CFG.API_USAGE_FILE.exists():
                with open(CFG.API_USAGE_FILE, "r") as f:
                    data = json.load(f)
                # ریست روزانه
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if data.get("date") != today:
                    return {"date": today, "keys": {}}
                return data
        except Exception:
            pass
        return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "keys": {}}

    def _save_usage(self):
        """ذخیره آمار مصرف."""
        try:
            with open(CFG.API_USAGE_FILE, "w") as f:
                json.dump(self.usage, f, indent=2)
        except Exception:
            pass

    def record_usage(self, key_label: str, requests_used: int = 0, remaining: int = -1):
        """ثبت مصرف یک Key."""
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
        """علامت‌گذاری Key به عنوان خراب (موقت)."""
        if 0 <= key_index < len(self.keys):
            self.keys[key_index]["failed"] = True
            self.keys[key_index]["fail_reason"] = reason
            self.keys[key_index]["fail_time"] = datetime.now(timezone.utc).isoformat()
            logger.warning(
                "🔑❌ [API KEY] %s marked FAILED: %s",
                self.keys[key_index]["label"], reason
            )

    def get_active_keys(self) -> List[Dict]:
        """
        لیست Key‌های فعال (که fail نشدن).
        اگه همه fail شدن، یکبار دیگه همه رو امتحان می‌کنه.
        """
        now = datetime.now(timezone.utc)
        active = []
        for k in self.keys:
            if not k["failed"]:
                active.append(k)
            else:
                # بعد از ۳۰ دقیقه دوباره امتحان کن
                if k.get("fail_time"):
                    try:
                        fail_dt = datetime.fromisoformat(k["fail_time"])
                        if now - fail_dt > timedelta(minutes=30):
                            k["failed"] = False
                            k["fail_reason"] = None
                            active.append(k)
                            logger.info(
                                "🔑🔄 [API KEY] %s reset after cooldown",
                                k["label"]
                            )
                    except Exception:
                        pass

        if not active:
            # همه fail شدن → ریست همه
            logger.warning("🔑⚠️ All keys failed! Resetting all...")
            for k in self.keys:
                k["failed"] = False
                k["fail_reason"] = None
            active = list(self.keys)

        return active

    def get_usage_summary(self) -> str:
        """خلاصه مصرف برای لاگ."""
        parts = []
        for k in self.keys:
            usage = self.usage.get("keys", {}).get(k["label"], {})
            calls = usage.get("calls", 0)
            remaining = usage.get("remaining", "?")
            status = "❌" if k["failed"] else "✅"
            parts.append(f"{status} {k['label']}: {calls} calls (rem: {remaining})")
        return " | ".join(parts)


# بارگذاری کلیدها
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")

# 🔄 NEW: مدیر کلیدها
odds_key_manager = OddsAPIKeyManager()

if not all([GROQ_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing GROQ_API_KEY, TELEGRAM_BOT_TOKEN, or TELEGRAM_CHAT_ID")
    sys.exit(1)

timeout_settings = httpx.Timeout(25.0, connect=10.0)
groq_client = Groq(api_key=GROQ_API_KEY, max_retries=3, timeout=timeout_settings)


# =========================================================
# 4. NATIONALITY FLAGS (بدون تغییر)
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
# 5. CACHE MANAGEMENT (بدون تغییر)
# =========================================================
class CacheManager:
    @staticmethod
    def load(filepath: Path) -> dict:
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            if DEBUG_MODE:
                logger.warning("Cache load error (%s): %s", filepath.name, e)
        return {}

    @staticmethod
    def save(filepath: Path, data: dict) -> None:
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
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
            return datetime.now(timezone.utc) - cached_time < timedelta(hours=ttl_hours)
        except Exception:
            return False

    @staticmethod
    def is_valid_minutes(cache: dict, key: str, ttl_minutes: float) -> bool:
        """🔄 NEW: بررسی TTL بر حسب دقیقه."""
        if key not in cache:
            return False
        entry = cache[key]
        if not isinstance(entry, dict) or "timestamp" not in entry:
            return False
        try:
            cached_time = datetime.fromisoformat(entry["timestamp"])
            return datetime.now(timezone.utc) - cached_time < timedelta(minutes=ttl_minutes)
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
# 6. SENT HISTORY (بدون تغییر)
# =========================================================
class SentHistory:
    def __init__(self):
        self.history = CacheManager.load(CFG.HISTORY_FILE)
        self._cleanup_old()

    def _cleanup_old(self):
        now = datetime.now(timezone.utc)
        to_delete = [
            key for key, value in self.history.items()
            if now - datetime.fromisoformat(
                value.get("sent_at", "2000-01-01T00:00:00+00:00")
            ) > timedelta(hours=CFG.TTL_SENT_HISTORY)
        ]
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
# 7. FREE HISTORICAL DATA ENGINE (بدون تغییر)
# =========================================================
class FreeDataEngine:
    def __init__(self):
        self.atp_matches: Optional[pd.DataFrame] = None
        self.wta_matches: Optional[pd.DataFrame] = None
        self.atp_rankings: Optional[pd.DataFrame] = None
        self.wta_rankings: Optional[pd.DataFrame] = None
        self.football_data: Dict[str, pd.DataFrame] = {}
        self.elo_cache: dict = CacheManager.load(CFG.CACHE_DIR / "elo_cache.json")
        self.years_to_fetch = [2023, 2024, 2025]

    def _download_csv(self, url: str, filepath: Path, timeout: int = 20) -> bool:
        if filepath.exists():
            age_hours = (time.time() - filepath.stat().st_mtime) / 3600
            if age_hours < CFG.TTL_GITHUB_DATA:
                return True

        logger.info("[FREE DATA] Downloading: %s", url.split("/")[-1])
        try:
            res = requests.get(url, timeout=timeout)
            if res.status_code == 200 and len(res.text) > 100:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(res.text)
                return True
        except Exception as e:
            logger.warning("[FREE DATA] Download error %s: %s", url, e)
        return False

    def load_tennis_data(self):
        atp_dfs, wta_dfs = [], []
        match_cols = [
            "tourney_date", "surface", "round",
            "winner_name", "winner_rank", "winner_age",
            "winner_ht", "w_ace", "w_df", "w_svpt",
            "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
            "loser_name", "loser_rank", "loser_age",
            "l_ace", "l_df", "l_svpt",
            "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
            "score", "best_of",
        ]

        for year in self.years_to_fetch:
            atp_url = CFG.GITHUB_SOURCES["atp"].format(year=year)
            atp_path = CFG.HISTORICAL_DIR / f"atp_{year}.csv"
            if self._download_csv(atp_url, atp_path):
                try:
                    df = pd.read_csv(atp_path, low_memory=False)
                    available = [c for c in match_cols if c in df.columns]
                    atp_dfs.append(df[available])
                except Exception as e:
                    logger.error("ATP parse error %s: %s", atp_path, e)

            wta_url = CFG.GITHUB_SOURCES["wta"].format(year=year)
            wta_path = CFG.HISTORICAL_DIR / f"wta_{year}.csv"
            if self._download_csv(wta_url, wta_path):
                try:
                    df = pd.read_csv(wta_path, low_memory=False)
                    available = [c for c in match_cols if c in df.columns]
                    wta_dfs.append(df[available])
                except Exception as e:
                    logger.error("WTA parse error %s: %s", wta_path, e)

        if atp_dfs:
            self.atp_matches = pd.concat(atp_dfs, ignore_index=True)
            logger.info("✅ [TENNIS] ATP loaded: %d matches", len(self.atp_matches))
        if wta_dfs:
            self.wta_matches = pd.concat(wta_dfs, ignore_index=True)
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
                    logger.info("✅ [RANKINGS] %s loaded: %d entries", tour.upper(), len(df))
                except Exception as e:
                    logger.error("Rankings parse error: %s", e)

    def get_player_ranking(self, player_name: str, is_wta: bool = False) -> Optional[int]:
        df = self.wta_rankings if is_wta else self.atp_rankings
        if df is None or df.empty:
            return None
        clean = player_name.split()[-1].lower()
        try:
            name_col = None
            for col in ["player", "name", "player_name"]:
                if col in df.columns:
                    name_col = col
                    break
            if not name_col:
                return None
            matches = df[df[name_col].str.lower().str.contains(clean, na=False)]
            if not matches.empty:
                rank_col = None
                for col in ["rank", "ranking", "player_rank"]:
                    if col in matches.columns:
                        rank_col = col
                        break
                if rank_col:
                    return int(matches.iloc[0][rank_col])
        except Exception:
            pass
        return None

    def get_tennis_stats(self, player_a: str, player_b: str, is_wta: bool = False) -> dict:
        df = self.wta_matches if is_wta else self.atp_matches
        if df is None or df.empty:
            return {}

        def clean(n):
            return n.split()[-1].lower()

        pa, pb = clean(player_a), clean(player_b)
        stats = {"player_a": {"name": player_a}, "player_b": {"name": player_b}, "h2h": {}}

        for p_clean, key in [(pa, "player_a"), (pb, "player_b")]:
            wins = df[df["winner_name"].str.lower().str.contains(p_clean, na=False)]
            losses = df[df["loser_name"].str.lower().str.contains(p_clean, na=False)]
            total = len(wins) + len(losses)
            if total == 0:
                continue

            stats[key]["win_rate"] = round(len(wins) / total, 3)
            stats[key]["total_matches"] = total

            all_matches = []
            for _, row in wins.iterrows():
                all_matches.append((row.get("tourney_date", 0), "W"))
            for _, row in losses.iterrows():
                all_matches.append((row.get("tourney_date", 0), "L"))
            all_matches.sort(key=lambda x: x[0], reverse=True)
            recent = all_matches[:10]
            if recent:
                stats[key]["recent_form"] = "".join(r[1] for r in recent)
                stats[key]["recent_win_rate"] = round(
                    sum(1 for r in recent if r[1] == "W") / len(recent), 3
                )

            surface_stats = {}
            for surface in ["Hard", "Clay", "Grass"]:
                sw = wins[wins["surface"].str.lower() == surface.lower()] if "surface" in wins.columns else pd.DataFrame()
                sl = losses[losses["surface"].str.lower() == surface.lower()] if "surface" in losses.columns else pd.DataFrame()
                st = len(sw) + len(sl)
                if st > 0:
                    surface_stats[surface] = {"win_rate": round(len(sw) / st, 3), "matches": st}
            if surface_stats:
                stats[key]["surface_stats"] = surface_stats

            for stat_name, col in {"aces_per_match": "w_ace", "df_per_match": "w_df"}.items():
                if col in wins.columns:
                    valid = wins[col].dropna()
                    if len(valid) > 0:
                        stats[key][stat_name] = round(valid.mean(), 1)

            if "w_1stIn" in wins.columns and "w_1stWon" in wins.columns:
                w1i = wins["w_1stIn"].dropna()
                w1w = wins["w_1stWon"].dropna()
                if len(w1i) > 0 and w1i.mean() > 0:
                    stats[key]["first_serve_win_pct"] = round(w1w.mean() / w1i.mean(), 3)

            if "w_bpSaved" in wins.columns and "w_bpFaced" in wins.columns:
                bps = wins["w_bpSaved"].dropna()
                bpf = wins["w_bpFaced"].dropna()
                if len(bpf) > 0 and bpf.mean() > 0:
                    stats[key]["bp_saved_pct"] = round(bps.mean() / bpf.mean(), 3)

            ranking = self.get_player_ranking(
                player_a if key == "player_a" else player_b, is_wta
            )
            if ranking:
                stats[key]["current_ranking"] = ranking

        h2h_a = df[
            (df["winner_name"].str.lower().str.contains(pa, na=False))
            & (df["loser_name"].str.lower().str.contains(pb, na=False))
        ]
        h2h_b = df[
            (df["winner_name"].str.lower().str.contains(pb, na=False))
            & (df["loser_name"].str.lower().str.contains(pa, na=False))
        ]
        total_h2h = len(h2h_a) + len(h2h_b)
        if total_h2h > 0:
            stats["h2h"] = {
                "total": total_h2h,
                f"{player_a}_wins": len(h2h_a),
                f"{player_b}_wins": len(h2h_b),
            }
            h2h_surfaces = {}
            for surface in ["Hard", "Clay", "Grass"]:
                sa = h2h_a[h2h_a["surface"].str.lower() == surface.lower()] if "surface" in h2h_a.columns else pd.DataFrame()
                sb = h2h_b[h2h_b["surface"].str.lower() == surface.lower()] if "surface" in h2h_b.columns else pd.DataFrame()
                if len(sa) + len(sb) > 0:
                    h2h_surfaces[surface] = {
                        f"{player_a}_wins": len(sa),
                        f"{player_b}_wins": len(sb),
                    }
            if h2h_surfaces:
                stats["h2h"]["by_surface"] = h2h_surfaces
            logger.info("✅ [H2H] %s vs %s: %d matches found", player_a, player_b, total_h2h)

        return stats

    def load_football_data(self):
        football_cols = [
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST",
            "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
            "B365H", "B365D", "B365A", "BbMxH", "BbMxD", "BbMxA",
            "BbAvH", "BbAvD", "BbAvA", "BbOU", "BbMx>2.5", "BbAv>2.5",
            "BbMx<2.5", "BbAv<2.5",
        ]
        all_dfs = []
        for season in CFG.FOOTBALL_DATA_UK_SEASONS:
            for league_code, league_name in CFG.FOOTBALL_DATA_UK_LEAGUES.items():
                url = CFG.GITHUB_SOURCES["football_eu"].format(season=season, league=league_code)
                path = CFG.HISTORICAL_DIR / f"football_{league_code}_{season}.csv"
                if self._download_csv(url, path):
                    try:
                        df = pd.read_csv(path, low_memory=False)
                        available = [c for c in football_cols if c in df.columns]
                        if available:
                            sub = df[available].copy()
                            sub["League"] = league_name
                            sub["Season"] = season
                            all_dfs.append(sub)
                    except Exception:
                        pass

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            self.football_data["all"] = combined
            logger.info("✅ [FOOTBALL] Loaded %d matches from %d leagues", len(combined), len(CFG.FOOTBALL_DATA_UK_LEAGUES))

    def get_football_stats(self, home_team: str, away_team: str) -> dict:
        df = self.football_data.get("all")
        if df is None or df.empty:
            return {}

        def fuzzy_match(team, column):
            clean = team.lower().strip()
            mask = column.str.lower().str.strip() == clean
            if mask.any():
                return mask
            parts = clean.split()
            for part in parts:
                if len(part) > 3:
                    mask = column.str.lower().str.contains(part, na=False)
                    if mask.any():
                        return mask
            return pd.Series([False] * len(column))

        stats = {"home": {}, "away": {}, "h2h": {}}

        for team, key, is_home in [(home_team, "home", True), (away_team, "away", False)]:
            home_mask = fuzzy_match(team, df["HomeTeam"])
            away_mask = fuzzy_match(team, df["AwayTeam"])
            team_home = df[home_mask].copy()
            team_away = df[away_mask].copy()

            if len(team_home) + len(team_away) == 0:
                continue

            all_results = []
            for _, row in team_home.iterrows():
                ftr = row.get("FTR", "")
                hg = row.get("FTHG", 0) or 0
                ag = row.get("FTAG", 0) or 0
                all_results.append({
                    "date": str(row.get("Date", "")),
                    "result": "W" if ftr == "H" else ("D" if ftr == "D" else "L"),
                    "scored": int(hg), "conceded": int(ag), "venue": "home",
                    "shots": int(row.get("HS", 0) or 0),
                    "shots_target": int(row.get("HST", 0) or 0),
                    "corners": int(row.get("HC", 0) or 0),
                })
            for _, row in team_away.iterrows():
                ftr = row.get("FTR", "")
                hg = row.get("FTHG", 0) or 0
                ag = row.get("FTAG", 0) or 0
                all_results.append({
                    "date": str(row.get("Date", "")),
                    "result": "W" if ftr == "A" else ("D" if ftr == "D" else "L"),
                    "scored": int(ag), "conceded": int(hg), "venue": "away",
                    "shots": int(row.get("AS", 0) or 0),
                    "shots_target": int(row.get("AST", 0) or 0),
                    "corners": int(row.get("AC", 0) or 0),
                })

            all_results.sort(key=lambda x: x["date"], reverse=True)
            recent = all_results[:6]
            if not recent:
                continue

            n = len(recent)
            stats[key] = {
                "name": team,
                "form_string": "".join(r["result"] for r in recent),
                "win_rate": round(sum(1 for r in recent if r["result"] == "W") / n, 3),
                "draw_rate": round(sum(1 for r in recent if r["result"] == "D") / n, 3),
                "avg_scored": round(sum(r["scored"] for r in recent) / n, 2),
                "avg_conceded": round(sum(r["conceded"] for r in recent) / n, 2),
                "btts_rate": round(sum(1 for r in recent if r["scored"] > 0 and r["conceded"] > 0) / n, 3),
                "over25_rate": round(sum(1 for r in recent if r["scored"] + r["conceded"] > 2.5) / n, 3),
                "avg_shots": round(sum(r["shots"] for r in recent) / n, 1),
                "avg_shots_target": round(sum(r["shots_target"] for r in recent) / n, 1),
                "avg_corners": round(sum(r["corners"] for r in recent) / n, 1),
                "matches_analyzed": n,
                "total_historical": len(all_results),
            }

            venue_matches = [r for r in all_results if r["venue"] == ("home" if is_home else "away")]
            if venue_matches:
                vn = len(venue_matches)
                stats[key]["venue_win_rate"] = round(sum(1 for r in venue_matches if r["result"] == "W") / vn, 3)
                stats[key]["venue_avg_goals"] = round(sum(r["scored"] + r["conceded"] for r in venue_matches) / vn, 2)

        h2h_mask = (
            (fuzzy_match(home_team, df["HomeTeam"]) & fuzzy_match(away_team, df["AwayTeam"]))
            | (fuzzy_match(away_team, df["HomeTeam"]) & fuzzy_match(home_team, df["AwayTeam"]))
        )
        h2h_df = df[h2h_mask]
        if len(h2h_df) > 0:
            h2h_results = []
            for _, row in h2h_df.iterrows():
                hg = int(row.get("FTHG", 0) or 0)
                ag = int(row.get("FTAG", 0) or 0)
                h2h_results.append({
                    "home_goals": hg, "away_goals": ag,
                    "total_goals": hg + ag, "btts": hg > 0 and ag > 0,
                    "over25": hg + ag > 2.5,
                })
            hn = len(h2h_results)
            stats["h2h"] = {
                "total_matches": hn,
                "avg_goals": round(sum(r["total_goals"] for r in h2h_results) / hn, 2),
                "btts_rate": round(sum(1 for r in h2h_results if r["btts"]) / hn, 3),
                "over25_rate": round(sum(1 for r in h2h_results if r["over25"]) / hn, 3),
            }
            logger.info("✅ [FOOTBALL H2H] %s vs %s: %d matches", home_team, away_team, hn)

        return stats

    def get_club_elo(self, team_name: str) -> Optional[float]:
        cache_key = f"elo_{team_name.lower()}"
        if CacheManager.is_valid(self.elo_cache, cache_key, CFG.TTL_TEAM_FORM):
            return CacheManager.get(self.elo_cache, cache_key)

        clean = team_name.replace(" ", "").replace("FC", "").strip()
        try:
            url = CFG.GITHUB_SOURCES["club_elo"].format(team=clean)
            res = requests.get(url, timeout=8)
            if res.status_code == 200 and res.text.strip():
                lines = res.text.strip().split("\n")
                if len(lines) > 1:
                    parts = lines[-1].split(",")
                    if len(parts) >= 4:
                        elo = float(parts[4])
                        self.elo_cache = CacheManager.set(self.elo_cache, cache_key, elo)
                        CacheManager.save(CFG.CACHE_DIR / "elo_cache.json", self.elo_cache)
                        return elo
        except Exception:
            pass
        return None

    def get_elo_delta(self, home_team: str, away_team: str) -> Optional[dict]:
        home_elo = self.get_club_elo(home_team)
        away_elo = self.get_club_elo(away_team)
        if home_elo and away_elo:
            delta = home_elo - away_elo
            home_prob = 1 / (1 + 10 ** (-delta / 400))
            return {
                "home_elo": round(home_elo, 1), "away_elo": round(away_elo, 1),
                "delta": round(delta, 1),
                "home_win_prob_elo": round(home_prob, 3),
                "away_win_prob_elo": round(1 - home_prob, 3),
            }
        return None


# =========================================================
# 8. ML PREDICTION ENGINE (بدون تغییر)
# =========================================================
class MLPredictionEngine:
    def __init__(self, data_engine: FreeDataEngine):
        self.data_engine = data_engine
        self.football_model = None
        self.tennis_model = None
        self.football_scaler = StandardScaler()
        self.tennis_scaler = StandardScaler()
        self.is_football_trained = False
        self.is_tennis_trained = False

    def train_football_model(self):
        df = self.data_engine.football_data.get("all")
        if df is None or len(df) < 200:
            logger.warning("[ML] Not enough football data to train (%s rows)", len(df) if df is not None else 0)
            return

        logger.info("[ML] Training football model on %d matches...", len(df))
        features, labels = [], []

        teams = set()
        if "HomeTeam" in df.columns:
            teams.update(df["HomeTeam"].dropna().unique())
        if "AwayTeam" in df.columns:
            teams.update(df["AwayTeam"].dropna().unique())

        team_stats = {}
        for team in teams:
            home_matches = df[df["HomeTeam"] == team]
            away_matches = df[df["AwayTeam"] == team]
            goals_scored, goals_conceded, results = [], [], []

            for _, row in home_matches.iterrows():
                hg = row.get("FTHG", 0) or 0
                ag = row.get("FTAG", 0) or 0
                goals_scored.append(float(hg))
                goals_conceded.append(float(ag))
                ftr = row.get("FTR", "")
                results.append(1 if ftr == "H" else (0.5 if ftr == "D" else 0))

            for _, row in away_matches.iterrows():
                hg = row.get("FTHG", 0) or 0
                ag = row.get("FTAG", 0) or 0
                goals_scored.append(float(ag))
                goals_conceded.append(float(hg))
                ftr = row.get("FTR", "")
                results.append(1 if ftr == "A" else (0.5 if ftr == "D" else 0))

            if results:
                team_stats[team] = {
                    "avg_scored": np.mean(goals_scored[-10:]),
                    "avg_conceded": np.mean(goals_conceded[-10:]),
                    "win_rate": np.mean([1 for r in results[-10:] if r == 1]) / max(len(results[-10:]), 1),
                    "form_points": np.mean(results[-5:]),
                }

        for _, row in df.iterrows():
            ht = row.get("HomeTeam", "")
            at = row.get("AwayTeam", "")
            ftr = row.get("FTR", "")
            if ht not in team_stats or at not in team_stats or ftr not in ["H", "D", "A"]:
                continue

            hs = team_stats[ht]
            aws = team_stats[at]
            feature_vec = [
                hs["avg_scored"], hs["avg_conceded"], hs["win_rate"], hs["form_points"],
                aws["avg_scored"], aws["avg_conceded"], aws["win_rate"], aws["form_points"],
                hs["avg_scored"] - aws["avg_conceded"],
                aws["avg_scored"] - hs["avg_conceded"],
                hs["form_points"] - aws["form_points"],
            ]
            for col in ["B365H", "B365D", "B365A"]:
                val = row.get(col, 0) or 0
                try:
                    feature_vec.append(float(val))
                except (ValueError, TypeError):
                    feature_vec.append(0.0)

            features.append(feature_vec)
            label = {"H": 0, "D": 1, "A": 2}.get(ftr, -1)
            if label >= 0:
                labels.append(label)
            else:
                features.pop()

        if len(features) < 100:
            return

        X = np.nan_to_num(np.array(features, dtype=np.float64))
        y = np.array(labels)
        X_scaled = self.football_scaler.fit_transform(X)

        base_model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
        try:
            cv_scores = cross_val_score(base_model, X_scaled, y, cv=5, scoring="accuracy")
            logger.info("✅ [ML FOOTBALL] CV Accuracy: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())
        except Exception:
            pass

        self.football_model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
        self.football_model.fit(X_scaled, y)
        self.is_football_trained = True
        self._team_stats = team_stats

        try:
            model_path = CFG.ML_DIR / "football_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"model": self.football_model, "scaler": self.football_scaler, "team_stats": team_stats}, f)
            logger.info("✅ [ML] Football model saved")
        except Exception:
            pass

    def predict_football(self, home_team: str, away_team: str) -> Optional[dict]:
        if not self.is_football_trained:
            model_path = CFG.ML_DIR / "football_model.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        saved = pickle.load(f)
                    self.football_model = saved["model"]
                    self.football_scaler = saved["scaler"]
                    self._team_stats = saved["team_stats"]
                    self.is_football_trained = True
                except Exception:
                    return None
            else:
                return None

        def find_stats(team):
            clean = team.lower().strip()
            for key, val in self._team_stats.items():
                if clean in key.lower() or key.lower() in clean:
                    return val
            parts = clean.split()
            for part in parts:
                if len(part) > 3:
                    for key, val in self._team_stats.items():
                        if part in key.lower():
                            return val
            return None

        hs = find_stats(home_team)
        aws = find_stats(away_team)
        if not hs or not aws:
            return None

        feature_vec = [
            hs["avg_scored"], hs["avg_conceded"], hs["win_rate"], hs["form_points"],
            aws["avg_scored"], aws["avg_conceded"], aws["win_rate"], aws["form_points"],
            hs["avg_scored"] - aws["avg_conceded"],
            aws["avg_scored"] - hs["avg_conceded"],
            hs["form_points"] - aws["form_points"],
            0.0, 0.0, 0.0,
        ]

        X = np.nan_to_num(np.array([feature_vec], dtype=np.float64))
        X_scaled = self.football_scaler.transform(X)
        probs = self.football_model.predict_proba(X_scaled)[0]
        classes = self.football_model.classes_

        result = {}
        label_map = {0: "home_win", 1: "draw", 2: "away_win"}
        for cls, prob in zip(classes, probs):
            result[label_map.get(cls, f"class_{cls}")] = round(float(prob), 4)
        result["model_type"] = "GradientBoosting_Calibrated"
        return result

    def train_tennis_model(self, is_wta: bool = False):
        df = self.data_engine.wta_matches if is_wta else self.data_engine.atp_matches
        if df is None or len(df) < 200:
            return

        tour = "WTA" if is_wta else "ATP"
        logger.info("[ML] Training %s tennis model on %d matches...", tour, len(df))
        features, labels = [], []

        for _, row in df.iterrows():
            wr = row.get("winner_rank", 0) or 0
            lr = row.get("loser_rank", 0) or 0
            if wr == 0 or lr == 0:
                continue

            feature_vec = [
                float(wr), float(lr), float(lr) - float(wr),
                float(row.get("winner_age", 25) or 25),
                float(row.get("loser_age", 25) or 25),
            ]
            surface = str(row.get("surface", "Hard")).lower()
            feature_vec.extend([
                1.0 if surface == "hard" else 0.0,
                1.0 if surface == "clay" else 0.0,
                1.0 if surface == "grass" else 0.0,
            ])
            for col in ["w_ace", "w_df", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
                         "l_ace", "l_df", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced"]:
                val = row.get(col, 0) or 0
                try:
                    feature_vec.append(float(val))
                except (ValueError, TypeError):
                    feature_vec.append(0.0)
            feature_vec.append(float(row.get("best_of", 3) or 3))

            features.append(feature_vec)
            labels.append(1)

            flipped = feature_vec.copy()
            flipped[0], flipped[1] = flipped[1], flipped[0]
            flipped[2] = -flipped[2]
            flipped[3], flipped[4] = flipped[4], flipped[3]
            for i in range(7):
                flipped[8 + i], flipped[15 + i] = flipped[15 + i], flipped[8 + i]
            features.append(flipped)
            labels.append(0)

        if len(features) < 200:
            return

        X = np.nan_to_num(np.array(features, dtype=np.float64))
        y = np.array(labels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        base = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
        try:
            cv_scores = cross_val_score(base, X_scaled, y, cv=5, scoring="accuracy")
            logger.info("✅ [ML TENNIS %s] CV Accuracy: %.3f ± %.3f", tour, cv_scores.mean(), cv_scores.std())
        except Exception:
            pass

        model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        model.fit(X_scaled, y)
        self.tennis_model = model
        self.tennis_scaler = scaler
        self.is_tennis_trained = True
        logger.info("✅ [ML TENNIS %s] Model trained", tour)

    def predict_tennis(self, player_a: str, player_b: str, stats: dict, surface: str = "hard") -> Optional[dict]:
        if not self.is_tennis_trained:
            self.train_tennis_model()
            if not self.is_tennis_trained:
                return None

        pa_stats = stats.get("player_a", {})
        pb_stats = stats.get("player_b", {})
        rank_a = pa_stats.get("current_ranking", 100)
        rank_b = pb_stats.get("current_ranking", 100)

        feature_vec = [
            float(rank_a), float(rank_b), float(rank_b) - float(rank_a),
            25.0, 25.0,
            1.0 if surface.lower() == "hard" else 0.0,
            1.0 if surface.lower() == "clay" else 0.0,
            1.0 if surface.lower() == "grass" else 0.0,
            pa_stats.get("aces_per_match", 5.0), pa_stats.get("df_per_match", 2.0),
            0.0, 0.0, 0.0, 0.0, 0.0,
            pb_stats.get("aces_per_match", 5.0), pb_stats.get("df_per_match", 2.0),
            0.0, 0.0, 0.0, 0.0, 0.0,
            3.0,
        ]

        X = np.nan_to_num(np.array([feature_vec], dtype=np.float64))
        X_scaled = self.tennis_scaler.transform(X)
        probs = self.tennis_model.predict_proba(X_scaled)[0]

        return {
            f"{player_a}_win_prob": round(float(probs[1]), 4),
            f"{player_b}_win_prob": round(float(probs[0]), 4),
            "model_type": "GradientBoosting_Calibrated_Tennis",
        }


# =========================================================
# 9. UTILS & MATH ENGINE (بدون تغییر)
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
                        logger.warning("Rate limit 429, sleeping %ds", wait)
                        time.sleep(wait)
                    elif status in [401, 403]:
                        return None
                    else:
                        if attempt == max_retries - 1:
                            return None
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        return None
                except requests.exceptions.RequestException:
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
    if any(k in lower_title for k in ["tennis", "atp", "wta"]):
        return "tennis"
    if any(kw in lower_title for kw in ["soccer", "football", "premier league", "la liga",
                                         "bundesliga", "serie a", "ligue 1", "champions league"]):
        return "football"
    if any(k in lower_title for k in ["basketball", "nba", "euroleague"]):
        return "basketball"
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


def get_market_label(market_key: str) -> str:
    mapping = {
        "h2h": "Match Winner", "totals": "Over/Under",
        "h2h_lay": "Lay (Betting Against)", "spreads": "Point Spread / Handicap",
    }
    return mapping.get(market_key, market_key.replace("_", " ").title())


def calculate_sharp_ev(markets_data: dict, bookmakers_raw: list) -> list:
    best_per_market: dict = {}
    for market_key, market_data_list in markets_data.items():
        sharp_odds, best_odds = {}, {}
        has_real_sharp = False

        for entry in market_data_list:
            bk = entry.get("bookmaker_key", "")
            if bk in CFG.SHARP_BOOKMAKERS:
                has_real_sharp = True

            for o in entry.get("outcomes", []):
                name = f"{o['name']} {o.get('point')}" if o.get("point") is not None else o["name"]
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
                    "pick": outcome_name, "market": market_key,
                    "market_label": get_market_label(market_key),
                    "prob": round(true_prob, 4), "odds": round(best_price, 3),
                    "bookmaker": best_odds[outcome_name]["bookmaker"],
                    "ev": round(ev, 4), "edge_pct": round(ev * 100, 2),
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
# 10. ASYNC ODDS API - با Fallback و کش هوشمند  🔄 COMPLETELY NEW
# =========================================================
class SmartOddsCache:
    """
    سیستم کش هوشمند برای Odds API:
    - ذخیره نتایج با TTL دقیقه‌ای
    - کلید کش = hash(markets + regions + window)
    - اگه کش معتبر بود، اصلاً API زده نمیشه
    """

    def __init__(self):
        self.cache = CacheManager.load(CFG.ODDS_CACHE_FILE)

    def _make_key(self, markets: list, window_hours: float) -> str:
        """کلید منحصر به فرد بر اساس پارامترها."""
        now_rounded = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
        raw = f"{','.join(sorted(markets))}|{window_hours}|{now_rounded}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_cached(self, markets: list, window_hours: float) -> Optional[list]:
        """اگه کش معتبر بود، برگردون."""
        key = self._make_key(markets, window_hours)
        if CacheManager.is_valid_minutes(self.cache, key, CFG.TTL_ODDS_CACHE_MINUTES):
            data = CacheManager.get(self.cache, key)
            if data:
                logger.info(
                    "💾 [ODDS CACHE] HIT! Using cached odds (%d events, TTL=%.0fm)",
                    len(data), CFG.TTL_ODDS_CACHE_MINUTES
                )
                return data
        return None

    def save_cached(self, markets: list, window_hours: float, events: list):
        """ذخیره نتایج جدید."""
        key = self._make_key(markets, window_hours)
        self.cache = CacheManager.set(self.cache, key, events)
        CacheManager.save(CFG.ODDS_CACHE_FILE, self.cache)
        logger.info("💾 [ODDS CACHE] SAVED %d events (TTL=%.0fm)", len(events), CFG.TTL_ODDS_CACHE_MINUTES)

    def invalidate(self):
        """پاک کردن کش (مثلاً اگه همه Key‌ها عوض بشن)."""
        self.cache = {}
        CacheManager.save(CFG.ODDS_CACHE_FILE, self.cache)


# 🔄 NEW: Global cache instance
odds_cache = SmartOddsCache()


async def fetch_market_with_fallback(
    session: aiohttp.ClientSession,
    market: str,
    now_utc: datetime,
    api_key: str,
    key_label: str,
) -> Tuple[list, int, Optional[str]]:
    """
    🔄 NEW: فچ یک مارکت با یک Key خاص.
    Returns: (events, status_code, error_reason)
    """
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
            # 🔄 NEW: خواندن هدرهای مصرف
            remaining = int(res.headers.get("x-requests-remaining", -1))
            used = int(res.headers.get("x-requests-used", 0))

            if res.status == 200:
                events = await res.json()

                # ثبت مصرف
                odds_key_manager.record_usage(key_label, used, remaining)

                if remaining >= 0:
                    logger.info(
                        "🔑 [%s] API call OK | Remaining: %d | Used: %d | Market: %s",
                        key_label, remaining, used, market
                    )

                # فیلتر بر اساس زمان
                valid = []
                for e in events:
                    try:
                        mt = datetime.fromisoformat(e.get("commence_time", "").replace("Z", "+00:00"))
                        if now_utc <= mt <= end_window:
                            valid.append(e)
                    except Exception:
                        continue

                return valid, 200, None

            elif res.status == 401:
                return [], 401, "Invalid API key"
            elif res.status == 429:
                return [], 429, "Rate limited"
            elif res.status == 402:
                return [], 402, "Quota exhausted"
            elif res.status == 422:
                return [], 422, "Invalid params"
            else:
                body = await res.text()
                return [], res.status, f"HTTP {res.status}: {body[:100]}"

    except asyncio.TimeoutError:
        return [], 0, "Timeout"
    except aiohttp.ClientError as e:
        return [], 0, f"Connection error: {str(e)[:100]}"
    except Exception as e:
        return [], 0, f"Unknown error: {str(e)[:100]}"


async def fetch_all_odds_async() -> list:
    """
    🔄 COMPLETELY REWRITTEN: سیستم فچ با Fallback + کش.

    فلو:
    1. اول از کش چک کن
    2. اگه کش نبود → Key 1 رو امتحان کن
    3. اگه Key 1 خطا داد → Key 2
    4. اگه Key 2 هم خطا داد → Key 3
    5. نتیجه رو کش کن
    """
    now_utc = datetime.now(timezone.utc)

    # ============================
    # مرحله ۱: چک کش
    # ============================
    cached = odds_cache.get_cached(CFG.ODDS_API_MARKETS, CFG.MATCH_WINDOW_HOURS)
    if cached is not None:
        return cached

    logger.info("💾 [ODDS CACHE] MISS - Need fresh API call")

    # ============================
    # مرحله ۲: Fallback بین Key‌ها
    # ============================
    active_keys = odds_key_manager.get_active_keys()
    logger.info("🔑 [KEYS] %d active key(s): %s",
                len(active_keys),
                ", ".join(k["label"] for k in active_keys))

    all_events: dict = {}
    success = False

    for key_info in active_keys:
        api_key = key_info["key"]
        key_label = key_info["label"]

        logger.info("🔑 [TRYING] %s (%s)...", key_label, key_info["env"])

        connector = aiohttp.TCPConnector(limit=10, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            # فچ همه مارکت‌ها با این Key
            tasks = [
                fetch_market_with_fallback(session, m, now_utc, api_key, key_label)
                for m in CFG.ODDS_API_MARKETS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # بررسی نتایج
        any_success = False
        should_fallback = False
        fallback_reason = ""

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("🔑❌ [%s] Exception for %s: %s",
                             key_label, CFG.ODDS_API_MARKETS[i], result)
                should_fallback = True
                fallback_reason = str(result)
                continue

            events, status, error = result

            if status == 200:
                any_success = True
                for e in events:
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
            else:
                logger.warning(
                    "🔑⚠️ [%s] Market '%s' failed: %s (status=%d)",
                    key_label, CFG.ODDS_API_MARKETS[i], error, status
                )
                if status in [401, 402, 429]:
                    should_fallback = True
                    fallback_reason = error or f"HTTP {status}"

        if any_success:
            success = True
            logger.info(
                "✅ [%s] Successfully fetched %d events",
                key_label, len(all_events)
            )
            break  # موفق شد → دیگه Key بعدی رو نزن
        else:
            # این Key کار نکرد → علامت بزن و برو سراغ بعدی
            key_index = next(
                (i for i, k in enumerate(odds_key_manager.keys) if k["label"] == key_label),
                -1
            )
            if key_index >= 0:
                odds_key_manager.mark_failed(key_index, fallback_reason)
            logger.warning("🔑🔄 [%s] Failed, trying next key... Reason: %s", key_label, fallback_reason)

    if not success:
        logger.error("🔑❌ ALL API KEYS FAILED! No odds data available.")
        # آخرین تلاش: شاید کش قدیمی داشته باشیم
        old_cached = odds_cache.get_cached(CFG.ODDS_API_MARKETS, CFG.MATCH_WINDOW_HOURS * 10)
        if old_cached:
            logger.warning("💾 [FALLBACK] Using stale cache (%d events)", len(old_cached))
            return old_cached
        return []

    # ============================
    # مرحله ۳: ذخیره در کش
    # ============================
    final_events = list(all_events.values())
    odds_cache.save_cached(CFG.ODDS_API_MARKETS, CFG.MATCH_WINDOW_HOURS, final_events)

    # لاگ خلاصه مصرف
    logger.info("📊 [API USAGE] %s", odds_key_manager.get_usage_summary())

    return final_events


# =========================================================
# 11. CONFIDENCE & AI (بدون تغییر)
# =========================================================
def calculate_system_confidence(ev_edge, stats, market, ml_prediction=None):
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

    hist = stats.get("historical_data", {})
    if hist:
        pa_wr = hist.get("player_a", {}).get("recent_win_rate", 0)
        pb_wr = hist.get("player_b", {}).get("recent_win_rate", 0)
        if pa_wr >= 0.65 or pb_wr >= 0.65:
            score += 5
        if hist.get("h2h", {}).get("total", 0) > 0:
            score += 4

    fb = stats.get("football_stats", {})
    if fb:
        if fb.get("home", {}).get("win_rate", 0) > 0.6 or fb.get("away", {}).get("win_rate", 0) > 0.6:
            score += 4
        if fb.get("h2h", {}).get("total_matches", 0) > 0:
            score += 3

    elo = stats.get("elo_data", {})
    if elo and abs(elo.get("delta", 0)) > 100:
        score += 4

    if ml_prediction:
        max_prob = max(ml_prediction.values()) if ml_prediction else 0
        if isinstance(max_prob, (int, float)):
            if max_prob > 0.65:
                score += 8
            elif max_prob > 0.55:
                score += 4

    if market.lower() == "totals":
        score += 3

    score = max(50, min(93, int(score)))
    risk = "Low" if score >= 75 else ("Medium" if score >= 60 else "High")
    return score, risk


def call_groq_sdk(model, messages, temperature=0.1):
    kwargs = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": CFG.AI_MAX_TOKENS,
    }
    if "qwen" not in model.lower() and "gpt-oss" not in model.lower():
        kwargs["response_format"] = {"type": "json_object"}
    try:
        res = groq_client.chat.completions.create(**kwargs)
        return res.choices[0].message.content
    except Exception as e:
        logger.error("Groq SDK error model=%s: %s", model, e)
        return None


def generate_dual_ai_analysis(home, away, sport, pick, market, ev_edge, stats, has_real_stats, ml_prediction=None):
    calc_conf, calc_risk = calculate_system_confidence(ev_edge, stats, market, ml_prediction)

    default_response = {
        "sport_emoji": "🏆",
        "home_flag": get_flag_from_name(home),
        "away_flag": get_flag_from_name(away),
        "risk_level": calc_risk,
        "confidence": calc_conf,
        "logic": "Sharp market analysis reveals significant value on this selection based on quantitative edge detection.",
    }

    context_parts = []
    if stats.get("historical_data"):
        context_parts.append(f"HISTORICAL: {json.dumps(stats['historical_data'], indent=1)[:500]}")
    if stats.get("football_stats"):
        context_parts.append(f"FOOTBALL STATS: {json.dumps(stats['football_stats'], indent=1)[:500]}")
    if stats.get("elo_data"):
        context_parts.append(f"ELO: {json.dumps(stats['elo_data'])}")
    if ml_prediction:
        context_parts.append(f"ML PREDICTION: {json.dumps(ml_prediction)}")

    stats_str = "\n".join(context_parts) if context_parts else "No external data available."

    sys_analyst = (
        "You are an elite sports betting analyst for a VIP syndicate.\n"
        "Write EXACTLY two punchy, professional sentences justifying the pick.\n"
        "STRICT RULES:\n"
        "- NEVER mention 'Expected Value', 'EV', 'data quality', models.\n"
        "- Determine the EXACT country flag emoji for home_flag and away_flag.\n"
    )
    if has_real_stats:
        sys_analyst += "- Use the provided statistics to highlight dominance.\n"
    else:
        sys_analyst += "- NO statistics available. State pick is driven by quantitative edge in sharp betting market. DO NOT invent data.\n"
    sys_analyst += 'OUTPUT JSON ONLY: {"logic": "...", "sport_emoji": "...", "home_flag": "...", "away_flag": "..."}'

    u1 = f"MATCH: {home} vs {away}\nSPORT: {sport}\nPICK: {pick}\nMARKET: {get_market_label(market)}\n\nDATA:\n{stats_str}\n\nOUTPUT JSON ONLY:"

    analysis_1 = None
    try:
        raw1 = call_groq_sdk(CFG.AI_MODEL_ANALYST, [{"role": "system", "content": sys_analyst}, {"role": "user", "content": u1}], temperature=0.3)
        analysis_1 = robust_json_extractor(raw1)
    except Exception as e:
        logger.warning("Analyst model failed: %s", e)

    initial_logic = (analysis_1 or {}).get("logic", default_response["logic"])

    sys_editor = (
        "You are the Chief Editor for a VIP sports platform.\n"
        "Rewrite if it hallucinates or sounds robotic.\n"
        "Keep under 3 sentences. Professional tone.\n"
        'OUTPUT JSON ONLY: {"validated_logic": "..."}'
    )

    try:
        raw2 = call_groq_sdk(CFG.AI_MODEL_VALIDATOR, [
            {"role": "system", "content": sys_editor},
            {"role": "user", "content": f"DRAFT: {initial_logic}\nPICK: {pick}\nOUTPUT JSON ONLY:"},
        ], temperature=0.2)
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
# 12. TELEGRAM (بدون تغییر)
# =========================================================
def send_telegram(message_html: str) -> bool:
    MAX_LEN = 4000
    chunks = []
    if len(message_html) <= MAX_LEN:
        chunks.append(message_html)
    else:
        lines = message_html.split("\n")
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
            json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
        if not res.ok:
            logger.error("Telegram error: %s", res.text)
            success = False
    return success


# =========================================================
# 13. MAIN PIPELINE  🔄 UPDATED
# =========================================================
async def async_main():
    logger.info("=" * 60)
    logger.info("ZBET90 ENGINE v4.1 (FREE + ML + SMART FALLBACK) STARTING")
    logger.info("=" * 60)

    # 🔄 NEW: نمایش وضعیت Key‌ها
    logger.info("🔑 [KEYS STATUS] %s", odds_key_manager.get_usage_summary())

    sent_history = SentHistory()
    now_utc = datetime.now(timezone.utc)

    # Phase 1: Load Data
    data_engine = FreeDataEngine()
    logger.info("📥 [PHASE 1] Loading FREE data sources...")
    data_engine.load_tennis_data()
    data_engine.load_football_data()

    # Phase 2: Train ML
    logger.info("🧠 [PHASE 2] Training ML models...")
    ml_engine = MLPredictionEngine(data_engine)
    ml_engine.train_football_model()
    ml_engine.train_tennis_model(is_wta=False)

    # Phase 3: Fetch Odds (با کش + Fallback)
    logger.info("📡 [PHASE 3] Fetching live odds (cache-first + fallback)...")
    events = await fetch_all_odds_async()

    if not events:
        logger.info("No events in the %sh window.", CFG.MATCH_WINDOW_HOURS)
        # 🔄 NEW: نمایش نهایی وضعیت
        logger.info("📊 [FINAL USAGE] %s", odds_key_manager.get_usage_summary())
        return

    logger.info("🔍 [PHASE 4] Analyzing %d events...", len(events))
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

        # Gather stats
        stats = {"data_quality": "none"}
        ml_prediction = None

        if sport_key == "tennis":
            is_wta = "wta" in sport.lower()
            tennis_stats = data_engine.get_tennis_stats(home, away, is_wta)
            if tennis_stats and (tennis_stats.get("player_a") or tennis_stats.get("player_b")):
                stats["historical_data"] = tennis_stats
                stats["data_quality"] = "high"
            if ml_engine.is_tennis_trained and tennis_stats:
                ml_prediction = ml_engine.predict_tennis(home, away, tennis_stats)
                if ml_prediction:
                    stats["ml_prediction"] = ml_prediction

        elif sport_key == "football":
            fb_stats = data_engine.get_football_stats(home, away)
            if fb_stats and (fb_stats.get("home") or fb_stats.get("away")):
                stats["football_stats"] = fb_stats
                stats["data_quality"] = "medium"
            elo_data = data_engine.get_elo_delta(home, away)
            if elo_data:
                stats["elo_data"] = elo_data
                stats["data_quality"] = "high"
            if ml_engine.is_football_trained:
                ml_prediction = ml_engine.predict_football(home, away)
                if ml_prediction:
                    stats["ml_prediction"] = ml_prediction

        has_historical = bool(stats.get("historical_data"))
        has_football = bool(stats.get("football_stats") and (stats["football_stats"].get("home") or stats["football_stats"].get("away")))
        has_elo = bool(stats.get("elo_data"))
        has_ml = bool(ml_prediction)
        has_real_stats = has_historical or has_football or has_elo or has_ml

        if has_real_stats:
            sources = []
            if has_historical: sources.append("GitHub-Tennis")
            if has_football: sources.append("Football-Data.co.uk")
            if has_elo: sources.append("ClubElo")
            if has_ml: sources.append("ML-Model")
            logger.info("✅ [VERIFIED] Data from: %s | %s vs %s", ", ".join(sources), home, away)

        # AI Analysis
        ai_data = generate_dual_ai_analysis(home, away, sport, opp["pick"], opp["market"], opp["ev"], stats, has_real_stats, ml_prediction)

        # Telegram
        conf_icon = "🔥" if ai_data["confidence"] >= 75 else ("✅" if ai_data["confidence"] >= 65 else "⚡")
        risk_icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(ai_data["risk_level"], "🟠")

        ml_line = "\n🧠 <b>ML Model:</b> Verified ✓" if ml_prediction else ""
        data_badge = ""
        if has_real_stats:
            badges = []
            if has_historical: badges.append("📚 Historical")
            if has_football: badges.append("⚽ Match Data")
            if has_elo: badges.append("📊 Elo Ratings")
            if has_ml: badges.append("🧠 ML")
            data_badge = "\n📋 <b>Sources:</b> " + " | ".join(badges)

        msg = (
            f"{ai_data.get('sport_emoji', '🏆')} <b>{html_lib.escape(sport)}</b>\n\n"
            f"⚔️ <b>{html_lib.escape(home)}</b> {ai_data.get('home_flag', '🏳️')}  vs  "
            f"{ai_data.get('away_flag', '🏳️')} <b>{html_lib.escape(away)}</b>\n"
            f"⏳ <b>Starts in:</b> {get_countdown_str(event.get('commence_time', ''), now_utc)}\n\n"
            f"🎯 <b>PICK: {html_lib.escape(opp['pick'])}</b> @ <code>{opp['odds']}</code>\n\n"
            f"📊 <b>MARKET:</b> {html_lib.escape(opp['market_label'])}\n"
            f"{risk_icon} <b>Risk:</b> {ai_data['risk_level']}  |  "
            f"{conf_icon} <b>Confidence: {ai_data['confidence']}%</b>"
            f"{ml_line}{data_badge}\n\n"
            f"💡 <b>EXPERT ANALYSIS:</b>\n"
            f"<blockquote>{html_lib.escape(ai_data['logic'])}</blockquote>\n\n"
            f"🔍 <i>Curated by {CFG.TELEGRAM_ID}</i>"
        )

        if send_telegram(msg):
            sent_history.mark_sent(home, away, opp["pick"], opp["market"])
            total_sent += 1
            logger.info("📤 Sent: %s vs %s | %s | Conf: %d%%", home, away, opp["pick"], ai_data["confidence"])
        else:
            logger.error("❌ Telegram failed: %s vs %s", home, away)

        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    logger.info("=" * 60)
    if total_sent > 0:
        logger.info("✅ Done! %d signal(s) sent.", total_sent)
    else:
        logger.info("No qualifying +EV opportunities found.")
    # 🔄 NEW: گزارش نهایی مصرف
    logger.info("📊 [FINAL API USAGE] %s", odds_key_manager.get_usage_summary())
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except Exception as e:
        logger.critical("SYSTEM FAILURE: %s", str(e), exc_info=True)
        sys.exit(1)
