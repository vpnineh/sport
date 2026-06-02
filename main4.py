# =========================================================
# ZBET90 ENGINE v7.0 | AI Judge Mode | Multi-Sport
# =========================================================
import os, sys, time, json, re, random, logging, html as html_lib
import hashlib, asyncio, aiohttp, requests, numpy as np, pandas as pd
import pickle, warnings, threading
from io import StringIO
from functools import wraps
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict, deque

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from google import genai
from google.genai import types
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import scipy.stats as stats_scipy
from scipy.optimize import brentq

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
    ODDS_CACHE_FILE: Path = Path("api_cache/odds_cache.json")
    API_USAGE_FILE: Path = Path("api_cache/api_usage_tracker.json")
    PERFORMANCE_FILE: Path = Path("api_cache/performance_tracker.json")
    LOG_FILE: Path = Path("api_cache/execution_logs.log")

    MATCH_WINDOW_HOURS: float = 6.0
    TELEGRAM_SLEEP_BETWEEN: float = 3.0
    ODDS_API_MARKETS: list = field(default_factory=lambda: ["h2h", "totals"])
    ODDS_API_REGIONS: str = "eu,us,uk,au"
    TTL_ODDS_CACHE_MINUTES: float = 6.0
    TTL_SENT_HISTORY: float = 48.0
    TTL_TEAM_FORM: float = 6.0
    TTL_GITHUB_DATA: float = 12.0

    # EV Filters
    H2H_MIN_ODDS: float = 1.40
    H2H_MIN_EV: float = 0.010
    TOTALS_MIN_ODDS: float = 1.50
    TOTALS_MIN_EV: float = 0.012
    MAX_REALISTIC_EV: float = 0.25
    MATH_MIN_EV_TO_ANALYZE: float = 0.005          # FIX: نام صحیح
    MARKET_EXPECTED_OUTCOMES: dict = field(default_factory=lambda: {
        "h2h": {"min": 2, "max": 3}, "totals": {"min": 2, "max": 2}
    })
    MAX_VALID_IMPLIED_SUM: float = 1.30
    MIN_VALID_IMPLIED_SUM: float = 0.75

    KELLY_FRACTION: float = 0.25
    MAX_KELLY_PCT: float = 5.0

    # Confidence
    MIN_MATH_SCORE_TO_CALL_AI: int = 38
    MIN_CONFIDENCE_TO_SEND: int = 58
    HIGH_CONFIDENCE: int = 75
    MEDIUM_CONFIDENCE: int = 62

    # AI Judge
    AI_IS_FINAL_JUDGE: bool = True
    AI_WEIGHT: float = 0.70
    MATH_WEIGHT: float = 0.30
    MAX_AI_BOOST: int = 12
    MAX_AI_PENALTY: int = 8
    AI_MODEL_ANALYST: str = "gemini-2.0-flash"
    AI_MAX_TOKENS: int = 3000
    AI_TEMPERATURE: float = 0.05

    TELEGRAM_ID: str = "@zBET90"
    SHARP_BOOKMAKERS: list = field(default_factory=lambda: [
        "pinnacle", "betfair_ex_eu", "matchbook",
        "betfair_ex_uk", "sport888", "betsson",
    ])

    GITHUB_SOURCES: dict = field(default_factory=lambda: {
        "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
        "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
        "atp_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv",
        "wta_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv",
        "football_eu": "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv",
        "club_elo": "http://api.clubelo.com/{team}",
        # FIX: منبع واقعی NBA
        "nba_games": "https://raw.githubusercontent.com/swar/nba_api/master/docs/table_of_contents.md",
        # MLB via statsapi (داخلی)
        # Cricket - cricsheet
        "cricket_t20": "https://cricsheet.org/downloads/t20s_csv2.zip",
        "cricket_odi": "https://cricsheet.org/downloads/odis_csv2.zip",
    })

    FOOTBALL_DATA_UK_LEAGUES: dict = field(default_factory=lambda: {
        "E0": "Premier League", "E1": "Championship",
        "D1": "Bundesliga", "SP1": "La Liga",
        "I1": "Serie A", "F1": "Ligue 1",
        "N1": "Eredivisie", "P1": "Liga Portugal",
        "T1": "Super Lig", "B1": "Jupiler League",
    })
    FOOTBALL_DATA_UK_SEASONS: list = field(default_factory=lambda: ["2223", "2324", "2425"])


CFG = Config()

# =========================================================
# 2. LOGGING
# =========================================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
for d in [CFG.CACHE_DIR, CFG.LOG_DIR, CFG.HISTORICAL_DIR, CFG.ML_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ZBET90_ENGINE")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
_ch = logging.StreamHandler(sys.stdout); _ch.setFormatter(_fmt); logger.addHandler(_ch)
_fh = logging.FileHandler(CFG.LOG_FILE, mode="a", encoding="utf-8"); _fh.setFormatter(_fmt); logger.addHandler(_fh)

# =========================================================
# 3. GEMINI MANAGER
# =========================================================
class GeminiManager:
    _instance: Optional["GeminiManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized: return
        keys = [k.strip() for k in [
            os.getenv("GEMINI",""), os.getenv("GEMINI1",""),
            os.getenv("GEMINI2",""), os.getenv("GEMINI3",""),
        ] if k.strip()]
        if not keys: logger.critical("FATAL: No GEMINI API keys!"); sys.exit(1)
        self.clients = [genai.Client(api_key=k) for k in keys]
        self._safety = [
            types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            for c in [
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ]
        self._initialized = True
        logger.info("✅ [GEMINI] %d keys loaded", len(self.clients))

    def generate(self, prompt: str, system_instruction: str = None,
                 temperature: float = None, max_retries: int = 3) -> Optional[dict]:
        cfg_kw = dict(
            temperature=temperature or CFG.AI_TEMPERATURE,
            max_output_tokens=CFG.AI_MAX_TOKENS,
            response_mime_type="application/json",
            safety_settings=self._safety,
        )
        if system_instruction: cfg_kw["system_instruction"] = system_instruction
        gen_cfg = types.GenerateContentConfig(**cfg_kw)

        for attempt in range(max_retries):
            try:
                resp = random.choice(self.clients).models.generate_content(
                    model=CFG.AI_MODEL_ANALYST, contents=prompt, config=gen_cfg)
                if getattr(resp, "prompt_feedback", None) and resp.prompt_feedback.block_reason:
                    return None
                raw = resp.text
                if not raw: continue
                try: return json.loads(raw)
                except json.JSONDecodeError: return robust_json_extractor(raw)
            except Exception as e:
                es = str(e)
                if "429" in es or "quota" in es.lower():
                    time.sleep((attempt+1)*10); continue
                if "400" in es: return None
                if attempt < max_retries-1: time.sleep(2**attempt)
        return None

gemini_manager = GeminiManager()

# =========================================================
# 4. API KEY MANAGER
# =========================================================
class OddsAPIKeyManager:
    def __init__(self):
        self.keys: List[Dict] = []
        self._lock = threading.Lock()
        for env, label in [("ODDS_API_KEY","primary"),("ODDS_API_KEY2","backup_1"),("ODDS_API_KEY3","backup_2")]:
            k = os.getenv(env,"").strip()
            if k:
                self.keys.append({"key":k,"label":label,"env":env,"failed":False,"fail_reason":None,"fail_time":None})
                logger.info("🔑 [API KEY] %s: Loaded ✓", label)
        if not self.keys: logger.critical("FATAL: No ODDS_API_KEY!"); sys.exit(1)
        self.usage = self._load_usage()

    def _load_usage(self) -> dict:
        try:
            if CFG.API_USAGE_FILE.exists():
                d = json.loads(CFG.API_USAGE_FILE.read_text())
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if d.get("date") == today: return d
        except Exception: pass
        return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "keys": {}}

    def _save_usage(self):
        try: CFG.API_USAGE_FILE.write_text(json.dumps(self.usage, indent=2))
        except Exception: pass

    def record_usage(self, label: str, used: int = 0, remaining: int = -1):
        with self._lock:
            self.usage["keys"].setdefault(label, {"calls":0,"remaining":-1,"last_used":None})
            self.usage["keys"][label]["calls"] += 1
            self.usage["keys"][label]["last_used"] = datetime.now(timezone.utc).isoformat()
            if remaining >= 0: self.usage["keys"][label]["remaining"] = remaining
            self._save_usage()

    def mark_failed(self, idx: int, reason: str):
        with self._lock:
            if 0 <= idx < len(self.keys):
                self.keys[idx].update({"failed":True,"fail_reason":reason,
                                       "fail_time":datetime.now(timezone.utc).isoformat()})
                logger.warning("🔑❌ %s FAILED: %s", self.keys[idx]["label"], reason)

    def get_active_keys(self) -> List[Dict]:
        now = datetime.now(timezone.utc)
        active = []
        for k in self.keys:
            if not k["failed"]:
                active.append(k)
            elif k.get("fail_time"):
                try:
                    ft = datetime.fromisoformat(k["fail_time"])
                    if ft.tzinfo is None: ft = ft.replace(tzinfo=timezone.utc)
                    if now - ft > timedelta(minutes=30):
                        k["failed"] = False; active.append(k)
                except Exception: pass
        if not active:
            for k in self.keys: k["failed"] = False
            active = list(self.keys)
        return active

    def get_usage_summary(self) -> str:
        parts = []
        for k in self.keys:
            u = self.usage.get("keys",{}).get(k["label"],{})
            parts.append(f"{'❌' if k['failed'] else '✅'} {k['label']}: {u.get('calls',0)} calls (rem:{u.get('remaining','?')})")
        return " | ".join(parts)

GEMINI_API_KEY = os.getenv("GEMINI","").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","").strip()
if not all([GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing env vars"); sys.exit(1)
odds_key_manager = OddsAPIKeyManager()

# =========================================================
# 5. NATIONALITY FLAGS
# =========================================================
NATIONALITY_FLAGS: dict = {
    "alcaraz":"ES","nadal":"ES","djokovic":"RS","sinner":"IT","zverev":"DE",
    "tiafoe":"US","fritz":"US","paul":"US","medvedev":"RU","rublev":"RU",
    "tsitsipas":"GR","ruud":"NO","rune":"DK","hurkacz":"PL","swiatek":"PL",
    "auger-aliassime":"CA","shapovalov":"CA","kyrgios":"AU","de minaur":"AU",
    "sabalenka":"BY","gauff":"US","rybakina":"KZ","jabeur":"TN",
    "real madrid":"ES","barcelona":"ES","bayern":"DE","dortmund":"DE",
    "manchester city":"GB","liverpool":"GB","arsenal":"GB","chelsea":"GB",
    "juventus":"IT","milan":"IT","inter":"IT","napoli":"IT",
    "psg":"FR","ajax":"NL","porto":"PT","benfica":"PT",
    "lakers":"US","celtics":"US","warriors":"US","bulls":"US",
    "flamengo":"BR","palmeiras":"BR","river plate":"AR","boca juniors":"AR",
}

def _code_to_flag(code: str) -> str:
    code = code.upper()
    return chr(ord(code[0])+0x1F1E6-ord('A')) + chr(ord(code[1])+0x1F1E6-ord('A'))

def get_flag_from_name(name: str) -> str:
    nl = name.lower()
    for kw, code in NATIONALITY_FLAGS.items():
        if kw in nl: return _code_to_flag(code)
    return "🏳️"

# =========================================================
# 6. CACHE MANAGER
# =========================================================
_cache_lock = threading.Lock()

class CacheManager:
    @staticmethod
    def load(fp: Path) -> dict:
        try:
            if fp.exists(): return json.loads(fp.read_text(encoding="utf-8"))
        except Exception: pass
        return {}

    @staticmethod
    def save(fp: Path, data: dict):
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_suffix(".tmp")
            with _cache_lock:
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                tmp.replace(fp)
        except Exception: pass

    @staticmethod
    def is_valid(cache: dict, key: str, ttl_hours: float) -> bool:
        e = cache.get(key)
        if not isinstance(e, dict) or "timestamp" not in e: return False
        try:
            t = datetime.fromisoformat(e["timestamp"])
            if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - t < timedelta(hours=ttl_hours)
        except Exception: return False

    @staticmethod
    def is_valid_minutes(cache: dict, key: str, ttl_min: float) -> bool:
        return CacheManager.is_valid(cache, key, ttl_min/60)

    @staticmethod
    def set(cache: dict, key: str, value: Any) -> dict:
        cache[key] = {"timestamp": datetime.now(timezone.utc).isoformat(), "data": value}
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
        self.data.setdefault("signals", [])
        self.data.setdefault("summary", {})

    def record_signal(self, home, away, pick, market, odds, ev, confidence, prob, sport="other", api_sport_key=""):
        sig = {
            "id": hashlib.md5(f"{home}|{away}|{market}|{datetime.now(timezone.utc).date()}".encode()).hexdigest()[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sport": sport, "api_sport_key": api_sport_key,
            "home": home, "away": away, "pick": pick, "market": market,
            "odds": odds, "ev": ev, "confidence": confidence,
            "implied_prob": prob, "outcome": None, "profit_loss": None,
        }
        self.data["signals"].append(sig)
        if len(self.data["signals"]) > 500: self.data["signals"] = self.data["signals"][-500:]
        self._update_summary()
        CacheManager.save(CFG.PERFORMANCE_FILE, self.data)

    def _update_summary(self):
        res = [s for s in self.data["signals"] if s.get("outcome")]
        if not res: return
        wins = [s for s in res if s["outcome"] == "win"]
        pl = sum(s.get("profit_loss",0) or 0 for s in res)
        self.data["summary"] = {
            "total_signals": len(self.data["signals"]), "resolved": len(res),
            "win_rate": round(len(wins)/len(res),3),
            "total_profit_loss_units": round(pl,2),
            "roi_pct": round(pl/len(res)*100,2),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

performance_tracker = PerformanceTracker()

# =========================================================
# 8. SENT HISTORY
# =========================================================
class SentHistory:
    def __init__(self):
        self.history = CacheManager.load(CFG.HISTORY_FILE)
        self._cleanup()

    def _cleanup(self):
        now = datetime.now(timezone.utc)
        to_del = []
        for k, v in self.history.items():
            try:
                t = datetime.fromisoformat(v.get("sent_at","2000-01-01T00:00:00+00:00"))
                if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
                if now - t > timedelta(hours=CFG.TTL_SENT_HISTORY): to_del.append(k)
            except Exception: to_del.append(k)
        for k in to_del: del self.history[k]

    @staticmethod
    def _key(home, away, market) -> str:
        return hashlib.md5(f"{home.lower()}|{away.lower()}|{market.lower()}".encode()).hexdigest()

    def was_sent(self, home, away, market) -> bool:
        return self._key(home, away, market) in self.history

    def mark_sent(self, home, away, pick, market):
        self.history[self._key(home, away, market)] = {
            "match": f"{home} vs {away}", "pick": pick, "market": market,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        CacheManager.save(CFG.HISTORY_FILE, self.history)

# =========================================================
# 9. FREE DATA ENGINE (Football + Tennis + NBA + NHL + MLB + Cricket)
# =========================================================
class FreeDataEngine:
    def __init__(self):
        self.atp_matches: Optional[pd.DataFrame] = None
        self.wta_matches: Optional[pd.DataFrame] = None
        self.atp_rankings: Optional[pd.DataFrame] = None
        self.wta_rankings: Optional[pd.DataFrame] = None
        self.football_data: Dict[str, pd.DataFrame] = {}
        self.nba_data: Optional[pd.DataFrame] = None
        self.nhl_data: Optional[pd.DataFrame] = None
        self.mlb_data: Optional[pd.DataFrame] = None
        self.cricket_data: Optional[pd.DataFrame] = None
        self.elo_cache: dict = CacheManager.load(CFG.CACHE_DIR / "elo_cache.json")
        self.us_cache: dict = CacheManager.load(CFG.CACHE_DIR / "us_sports_cache.json")
        self.years = [2022, 2023, 2024, 2025]

    def _download_csv(self, url: str, path: Path, timeout: int = 25) -> bool:
        if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < CFG.TTL_GITHUB_DATA:
            return True
        logger.info("[FREE DATA] Downloading: %s", path.name)
        try:
            r = requests.get(
                url, timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ZBET90/7.0)"}
            )
            if r.status_code == 200 and len(r.content) > 100:
                path.write_bytes(r.content)
                return True
            logger.debug("[FREE DATA] HTTP %d for %s", r.status_code, url)
        except requests.exceptions.Timeout:
            logger.warning("[FREE DATA] Timeout: %s", path.name)
        except Exception as e:
            logger.warning("[FREE DATA] %s: %s", path.name, str(e)[:80])
        return False

    # ──────────────────────────────────────────────────────
    # TENNIS
    # ──────────────────────────────────────────────────────
    def load_tennis_data(self):
        COLS = [
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
        atp_dfs, wta_dfs = [], []

        for year in self.years:
            for tour, lst, key in [("atp", atp_dfs, "atp"), ("wta", wta_dfs, "wta")]:
                url = CFG.GITHUB_SOURCES[key].format(year=year)
                path = CFG.HISTORICAL_DIR / f"{key}_{year}.csv"
                if self._download_csv(url, path):
                    try:
                        df = pd.read_csv(
                            path, low_memory=False,
                            encoding="utf-8", encoding_errors="replace"
                        )
                        sub = df[[c for c in COLS if c in df.columns]].copy()
                        if "tourney_date" in sub.columns:
                            sub["tourney_date"] = pd.to_numeric(
                                sub["tourney_date"], errors="coerce"
                            )
                        lst.append(sub)
                    except Exception as e:
                        logger.error("[TENNIS] %s %s: %s", tour.upper(), year, e)

        for lst, attr, name in [
            (atp_dfs, "atp_matches", "ATP"),
            (wta_dfs, "wta_matches", "WTA"),
        ]:
            if lst:
                df = pd.concat(lst, ignore_index=True)
                if "tourney_date" in df.columns:
                    df = df.sort_values("tourney_date").reset_index(drop=True)
                setattr(self, attr, df)
                logger.info("✅ [TENNIS] %s: %d matches", name, len(df))

        for tour, key, attr in [
            ("atp", "atp_rankings", "atp_rankings"),
            ("wta", "wta_rankings", "wta_rankings"),
        ]:
            path = CFG.HISTORICAL_DIR / f"{key}.csv"
            if self._download_csv(CFG.GITHUB_SOURCES[key], path):
                try:
                    setattr(self, attr, pd.read_csv(path, low_memory=False))
                    logger.info("✅ [RANKINGS] %s loaded", tour.upper())
                except Exception as e:
                    logger.error("[RANKINGS] %s: %s", tour, e)

    def get_player_ranking(self, name: str, is_wta: bool = False) -> Optional[int]:
        df = self.wta_rankings if is_wta else self.atp_rankings
        if df is None or df.empty:
            return None
        clean = name.split()[-1].lower()
        nc = next((c for c in ["player", "name", "player_name"] if c in df.columns), None)
        if not nc:
            return None
        m = df[df[nc].str.lower().str.contains(re.escape(clean), na=False)]
        if not m.empty:
            rc = next((c for c in ["rank", "ranking", "player_rank"] if c in m.columns), None)
            if rc:
                v = m.iloc[0][rc]
                return int(v) if pd.notna(v) else None
        return None

    def _player_rolling(self, df: pd.DataFrame, clean: str, n: int = 20) -> dict:
        wins = df[df["winner_name"].str.lower().str.contains(re.escape(clean), na=False)]
        losses = df[df["loser_name"].str.lower().str.contains(re.escape(clean), na=False)]
        total = len(wins) + len(losses)
        if total == 0:
            return {}

        all_r = (
            [(r.get("tourney_date", 0), "W", r) for _, r in wins.iterrows()] +
            [(r.get("tourney_date", 0), "L", r) for _, r in losses.iterrows()]
        )
        all_r.sort(key=lambda x: x[0] if pd.notna(x[0]) else 0, reverse=True)
        recent = all_r[:n]
        rw = sum(1 for x in recent if x[1] == "W")

        result = {
            "total_matches": total,
            "win_rate_overall": round(len(wins) / total, 3),
            "recent_form": "".join(x[1] for x in recent[:10]),
            "recent_win_rate": round(rw / len(recent), 3) if recent else 0,
        }

        rw_df = wins.tail(n // 2)
        for stat, col in [
            ("aces_per_match", "w_ace"),
            ("df_per_match", "w_df"),
            ("svpt_per_match", "w_svpt"),
        ]:
            if col in rw_df.columns:
                v = rw_df[col].dropna()
                if len(v):
                    result[stat] = round(float(v.mean()), 2)

        if all(c in rw_df.columns for c in ["w_1stIn", "w_svpt"]):
            sv = rw_df["w_svpt"].dropna().mean()
            if sv:
                result["first_serve_in_pct"] = round(
                    float(rw_df["w_1stIn"].dropna().mean() / sv), 3
                )

        if all(c in rw_df.columns for c in ["w_1stWon", "w_1stIn"]):
            i1 = rw_df["w_1stIn"].dropna().mean()
            if i1:
                result["first_serve_win_pct"] = round(
                    float(rw_df["w_1stWon"].dropna().mean() / i1), 3
                )

        if all(c in rw_df.columns for c in ["w_bpSaved", "w_bpFaced"]):
            bpf = rw_df["w_bpFaced"].dropna().mean()
            if bpf:
                result["bp_saved_pct"] = round(
                    float(rw_df["w_bpSaved"].dropna().mean() / bpf), 3
                )

        ss = {}
        for surf in ["Hard", "Clay", "Grass"]:
            if "surface" in wins.columns:
                sw = wins[wins["surface"].str.lower() == surf.lower()]
                sl = (
                    losses[losses["surface"].str.lower() == surf.lower()]
                    if "surface" in losses.columns
                    else pd.DataFrame()
                )
                st = len(sw) + len(sl)
                if st >= 5:
                    ss[surf] = {"win_rate": round(len(sw) / st, 3), "matches": st}
        if ss:
            result["surface_stats"] = ss

        return result

    def get_tennis_stats(self, pa: str, pb: str, is_wta: bool = False) -> dict:
        df = self.wta_matches if is_wta else self.atp_matches
        if df is None or df.empty:
            return {}

        def cl(n):
            return n.split()[-1].lower()

        ca, cb = cl(pa), cl(pb)
        stats = {"player_a": {"name": pa}, "player_b": {"name": pb}, "h2h": {}}

        for p_c, key, p_f in [(ca, "player_a", pa), (cb, "player_b", pb)]:
            s = self._player_rolling(df, p_c)
            if s:
                stats[key].update(s)
                r = self.get_player_ranking(p_f, is_wta)
                if r:
                    stats[key]["current_ranking"] = r

        h2h_a = df[
            df["winner_name"].str.lower().str.contains(ca, na=False) &
            df["loser_name"].str.lower().str.contains(cb, na=False)
        ]
        h2h_b = df[
            df["winner_name"].str.lower().str.contains(cb, na=False) &
            df["loser_name"].str.lower().str.contains(ca, na=False)
        ]
        t = len(h2h_a) + len(h2h_b)

        if t:
            stats["h2h"] = {
                "total": t,
                f"{pa}_wins": len(h2h_a),
                f"{pb}_wins": len(h2h_b),
                "dominance": (
                    f"{pa}_dominant" if len(h2h_a) > len(h2h_b) * 2
                    else f"{pb}_dominant" if len(h2h_b) > len(h2h_a) * 2
                    else "balanced"
                ),
            }
            if "surface" in h2h_a.columns:
                bs = {}
                for surf in ["Hard", "Clay", "Grass"]:
                    sa = h2h_a[h2h_a["surface"].str.lower() == surf.lower()]
                    sb = h2h_b[h2h_b["surface"].str.lower() == surf.lower()]
                    if len(sa) + len(sb):
                        bs[surf] = {f"{pa}_wins": len(sa), f"{pb}_wins": len(sb)}
                if bs:
                    stats["h2h"]["by_surface"] = bs
            logger.info("✅ [H2H TENNIS] %s vs %s: %d matches", pa, pb, t)

        return stats

    # ──────────────────────────────────────────────────────
    # FOOTBALL
    # ──────────────────────────────────────────────────────
    def load_football_data(self):
        COLS = [
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST",
            "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
            "B365H", "B365D", "B365A", "BbMxH", "BbMxD", "BbMxA",
            "BbAvH", "BbAvD", "BbAvA",
            "BbMx>2.5", "BbAv>2.5", "BbMx<2.5", "BbAv<2.5",
        ]
        all_dfs = []
        for season in CFG.FOOTBALL_DATA_UK_SEASONS:
            for code, name in CFG.FOOTBALL_DATA_UK_LEAGUES.items():
                url = CFG.GITHUB_SOURCES["football_eu"].format(
                    season=season, league=code
                )
                path = CFG.HISTORICAL_DIR / f"football_{code}_{season}.csv"
                if self._download_csv(url, path):
                    try:
                        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
                        avail = [c for c in COLS if c in df.columns]
                        if len(avail) < 5:
                            continue
                        sub = df[avail].copy()
                        sub["League"] = name
                        sub["Season"] = season
                        if "Date" in sub.columns:
                            sub["Date"] = pd.to_datetime(
                                sub["Date"], dayfirst=True, errors="coerce"
                            )
                        if "HomeTeam" in sub.columns:
                            sub = sub.dropna(subset=["HomeTeam", "AwayTeam"])
                        all_dfs.append(sub)
                    except Exception as e:
                        logger.debug("[FOOTBALL] %s: %s", path.name, e)

        if all_dfs:
            comb = pd.concat(all_dfs, ignore_index=True)
            if "Date" in comb.columns:
                comb = comb.sort_values("Date").reset_index(drop=True)
            self.football_data["all"] = comb
            logger.info(
                "✅ [FOOTBALL] %d matches from %d leagues × %d seasons",
                len(comb),
                len(CFG.FOOTBALL_DATA_UK_LEAGUES),
                len(CFG.FOOTBALL_DATA_UK_SEASONS),
            )

    def _fuzzy(self, team: str, col: pd.Series) -> pd.Series:
        clean = team.lower().strip()
        m = col.str.lower().str.strip() == clean
        if m.any():
            return m
        for p in clean.split():
            if len(p) > 3:
                m2 = col.str.lower().str.contains(re.escape(p), na=False)
                if m2.any():
                    return m2
        return pd.Series([False] * len(col), index=col.index)

    def get_football_stats(self, home: str, away: str) -> dict:
        df = self.football_data.get("all")
        if df is None or df.empty:
            return {}

        stats: dict = {"home": {}, "away": {}, "h2h": {}}

        for team, key, is_home in [(home, "home", True), (away, "away", False)]:
            hm = self._fuzzy(team, df["HomeTeam"])
            am = self._fuzzy(team, df["AwayTeam"])
            th = df[hm]
            ta = df[am]
            all_r = []

            for _, row in th.iterrows():
                hg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                ag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                ftr = row.get("FTR", "")
                all_r.append({
                    "date": row.get("Date"),
                    "result": "W" if ftr == "H" else ("D" if ftr == "D" else "L"),
                    "scored": hg, "conceded": ag, "venue": "home",
                    "shots": int(row["HS"]) if pd.notna(row.get("HS")) else 0,
                    "shots_target": int(row["HST"]) if pd.notna(row.get("HST")) else 0,
                    "corners": int(row["HC"]) if pd.notna(row.get("HC")) else 0,
                    "yellows": int(row["HY"]) if pd.notna(row.get("HY")) else 0,
                })

            for _, row in ta.iterrows():
                hg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                ag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                ftr = row.get("FTR", "")
                all_r.append({
                    "date": row.get("Date"),
                    "result": "W" if ftr == "A" else ("D" if ftr == "D" else "L"),
                    "scored": ag, "conceded": hg, "venue": "away",
                    "shots": int(row["AS"]) if pd.notna(row.get("AS")) else 0,
                    "shots_target": int(row["AST"]) if pd.notna(row.get("AST")) else 0,
                    "corners": int(row["AC"]) if pd.notna(row.get("AC")) else 0,
                    "yellows": int(row["AY"]) if pd.notna(row.get("AY")) else 0,
                })

            all_r.sort(
                key=lambda x: (
                    x["date"] if isinstance(x["date"], pd.Timestamp)
                    else pd.Timestamp.min
                ),
                reverse=True,
            )
            recent = all_r[:10]
            if not recent:
                continue

            n = len(recent)
            sc = [r["scored"] for r in recent]
            cn = [r["conceded"] for r in recent]
            sh = [r["shots"] for r in recent]
            avg_sh = float(np.mean(sh)) if sh else 1.0
            if avg_sh == 0:
                avg_sh = 1.0

            wts = np.array([1 / (i + 1) for i in range(n)])
            wts /= wts.sum()
            rpts = np.array(
                [3 if r["result"] == "W" else (1 if r["result"] == "D" else 0)
                 for r in recent],
                dtype=np.float64,
            )

            stats[key] = {
                "name": team,
                "form_string": "".join(r["result"] for r in recent),
                "win_rate": round(sum(1 for r in recent if r["result"] == "W") / n, 3),
                "draw_rate": round(sum(1 for r in recent if r["result"] == "D") / n, 3),
                "loss_rate": round(sum(1 for r in recent if r["result"] == "L") / n, 3),
                "avg_scored": round(float(np.mean(sc)), 2),
                "avg_conceded": round(float(np.mean(cn)), 2),
                "std_scored": round(float(np.std(sc)), 2),
                "btts_rate": round(
                    sum(1 for r in recent if r["scored"] > 0 and r["conceded"] > 0) / n, 3
                ),
                "over25_rate": round(
                    sum(1 for r in recent if r["scored"] + r["conceded"] > 2.5) / n, 3
                ),
                "over35_rate": round(
                    sum(1 for r in recent if r["scored"] + r["conceded"] > 3.5) / n, 3
                ),
                "clean_sheet_rate": round(
                    sum(1 for r in recent if r["conceded"] == 0) / n, 3
                ),
                "avg_shots": round(float(np.mean(sh)), 1),
                "avg_shots_target": round(
                    float(np.mean([r["shots_target"] for r in recent])), 1
                ),
                "avg_corners": round(
                    float(np.mean([r["corners"] for r in recent])), 1
                ),
                "shot_conversion": round(float(np.mean(sc)) / avg_sh, 3),
                "weighted_form_points": round(float(np.dot(wts, rpts)), 3),
                "matches_analyzed": n,
                "total_historical": len(all_r),
            }

            vk = "home" if is_home else "away"
            vm = [r for r in all_r[:20] if r["venue"] == vk]
            if len(vm) >= 3:
                vn = len(vm)
                stats[key]["venue_win_rate"] = round(
                    sum(1 for r in vm if r["result"] == "W") / vn, 3
                )
                stats[key]["venue_avg_goals"] = round(
                    float(np.mean([r["scored"] + r["conceded"] for r in vm])), 2
                )
                stats[key]["venue_btts"] = round(
                    sum(1 for r in vm if r["scored"] > 0 and r["conceded"] > 0) / vn, 3
                )

        # H2H Football
        if "HomeTeam" in df.columns:
            hm2 = self._fuzzy(home, df["HomeTeam"])
            am2 = self._fuzzy(away, df["AwayTeam"])
            hm3 = self._fuzzy(away, df["HomeTeam"])
            am3 = self._fuzzy(home, df["AwayTeam"])
            h2h_df = df[(hm2 & am2) | (hm3 & am3)]

            if len(h2h_df):
                h2hr = []
                for _, row in h2h_df.iterrows():
                    hg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                    ag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                    h2hr.append({
                        "total_goals": hg + ag,
                        "btts": hg > 0 and ag > 0,
                        "over25": hg + ag > 2.5,
                        "over35": hg + ag > 3.5,
                    })
                hn = len(h2hr)
                gl = [r["total_goals"] for r in h2hr]
                stats["h2h"] = {
                    "total_matches": hn,
                    "avg_goals": round(float(np.mean(gl)), 2),
                    "btts_rate": round(sum(1 for r in h2hr if r["btts"]) / hn, 3),
                    "over25_rate": round(sum(1 for r in h2hr if r["over25"]) / hn, 3),
                    "over35_rate": round(sum(1 for r in h2hr if r["over35"]) / hn, 3),
                    "std_goals": round(float(np.std(gl)), 2),
                }
                logger.info(
                    "✅ [FOOTBALL H2H] %s vs %s: %d matches", home, away, hn
                )

        return stats

    def get_club_elo(self, team: str) -> Optional[float]:
        ck = f"elo_{team.lower()}"
        if CacheManager.is_valid(self.elo_cache, ck, CFG.TTL_TEAM_FORM):
            return CacheManager.get(self.elo_cache, ck)
        clean = re.sub(r"[^a-zA-Z]", "", team).replace("FC", "").strip()
        if not clean:
            return None
        try:
            r = requests.get(
                CFG.GITHUB_SOURCES["club_elo"].format(team=clean),
                timeout=8, headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code == 200 and r.text.strip():
                lines = [l for l in r.text.strip().split("\n") if l.strip()]
                if len(lines) > 1:
                    parts = lines[-1].split(",")
                    if len(parts) >= 5:
                        elo = float(parts[4])
                        self.elo_cache = CacheManager.set(self.elo_cache, ck, elo)
                        CacheManager.save(CFG.CACHE_DIR / "elo_cache.json", self.elo_cache)
                        return elo
        except Exception:
            pass
        return None

    def get_elo_delta(self, home: str, away: str) -> Optional[dict]:
        he = self.get_club_elo(home)
        ae = self.get_club_elo(away)
        if not (he and ae):
            return None
        delta = he - ae
        hp = min(0.95, 1 / (1 + 10 ** (-delta / 400)) + 0.03)
        return {
            "home_elo": round(he, 1),
            "away_elo": round(ae, 1),
            "delta": round(delta, 1),
            "home_win_prob_elo": round(hp, 3),
            "away_win_prob_elo": round(1 - hp, 3),
            "elo_confidence": (
                "high" if abs(delta) > 150
                else "medium" if abs(delta) > 75
                else "low"
            ),
        }

    # ──────────────────────────────────────────────────────
    # NBA
    # ──────────────────────────────────────────────────────
    def load_nba_data(self):
    """
    منبع: nba_api - پکیج رسمی Python برای NBA Stats
    pip install nba_api
    """
    cache_path = CFG.HISTORICAL_DIR / "nba_standings.json"

    # چک cache اول
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) / 3600 < 12:
        try:
            data = json.loads(cache_path.read_text())
            if data:
                self.nba_data = pd.DataFrame(data)
                logger.info("✅ [NBA] %d teams from cache", len(self.nba_data))
                return
        except Exception:
            pass

    try:
        from nba_api.stats.endpoints import leaguestandings
        from nba_api.stats.static import teams as nba_teams_static

        standings = leaguestandings.LeagueStandings(
            season="2024-25",
            season_type="Regular Season",
            league_id="00",
        )
        df = standings.get_data_frames()[0]

        if df is not None and not df.empty:
            self.nba_data = df
            cache_path.write_text(
                json.dumps(df.to_dict(orient="records"), indent=2)
            )
            logger.info("✅ [NBA] %d teams via nba_api", len(df))
            return

    except ImportError:
        logger.warning("[NBA] nba_api not installed → pip install nba_api")
    except Exception as e:
        logger.warning("[NBA] nba_api error: %s", str(e)[:80])

    # Fallback: cache قدیمی
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if data:
                self.nba_data = pd.DataFrame(data)
                logger.info("✅ [NBA] %d teams from stale cache", len(self.nba_data))
                return
        except Exception as e:
            logger.error("[NBA] Cache error: %s", e)

    logger.warning("[NBA] No data available")
    self.nba_data = None


def get_nba_stats(self, team: str) -> dict:
    if self.nba_data is None or self.nba_data.empty:
        return {}

    df = self.nba_data
    clean = team.lower().strip()

    # جستجو در ستون‌های مختلف
    m = pd.DataFrame()
    for col in ["TeamName", "TEAM_NAME", "TeamCity", "TEAM_CITY"]:
        if col in df.columns:
            found = df[
                df[col].astype(str).str.lower().str.contains(
                    re.escape(clean), na=False
                )
            ]
            if not found.empty:
                m = found
                break

    if m.empty:
        return {}

    row = m.iloc[0]

    def safe_int(*col_names):
        for c in col_names:
            if c in row.index:
                try: return int(row[c] or 0)
                except (ValueError, TypeError): return 0
        return 0

    def safe_float(*col_names):
        for c in col_names:
            if c in row.index:
                try: return float(row[c] or 0)
                except (ValueError, TypeError): return 0.0
        return 0.0

    wins    = safe_int("WINS", "W")
    losses  = safe_int("LOSSES", "L")
    gp      = max(wins + losses, 1)
    win_pct = safe_float("WinPCT", "WIN_PCT", "PCT")
    pts_pg  = safe_float("PointsPG", "PTS_PG")
    opp_pts = safe_float("OppPointsPG", "OPP_PTS_PG")
    l10     = str(row.get("L10", row.get("LAST_TEN", "")))
    streak  = str(row.get("strCurrentStreak", row.get("CurrentStreak", "")))

    return {
        "season_record":    f"{wins}W-{losses}L",
        "win_pct":          round(win_pct, 3),
        "avg_pts_scored":   round(pts_pg, 1),
        "avg_pts_allowed":  round(opp_pts, 1),
        "pt_diff":          round(pts_pg - opp_pts, 1),
        "last_10":          l10,
        "streak":           streak,
        "games_played":     gp,
        "source":           "nba_api",
    }


def get_nba_matchup(self, home: str, away: str) -> dict:
    hs = self.get_nba_stats(home)
    aw = self.get_nba_stats(away)
    if not hs or not aw:
        return {}

    h_str = hs.get("win_pct", 0.5) * 0.6 + max(
        min(hs.get("pt_diff", 0) / 20, 0.3), -0.3
    )
    a_str = aw.get("win_pct", 0.5) * 0.6 + max(
        min(aw.get("pt_diff", 0) / 20, 0.3), -0.3
    )
    total = h_str + a_str
    home_prob = (
        min(0.85, max(0.15, (h_str / total) + 0.02))
        if total > 0 else 0.52
    )
    return {
        "home":              hs,
        "away":              aw,
        "elo_home_win_prob": round(home_prob, 3),
        "elo_away_win_prob": round(1 - home_prob, 3),
    }

    # ──────────────────────────────────────────────────────
    # NHL  (Official NHL API - رایگان، بدون key)
    # ──────────────────────────────────────────────────────
    def load_nhl_data(self):
        """
        NHL Official API: https://api-web.nhle.com/v1/standings/now
        رایگان، بدون API key، همیشه آپدیت
        """
        url = "https://api-web.nhle.com/v1/standings/now"
        cache_path = CFG.HISTORICAL_DIR / "nhl_standings.json"

        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                data = r.json()
                standings = data.get("standings", [])
                if standings:
                    rows = []
                    for team in standings:
                        rows.append({
                            "team": team.get("teamName", {}).get("default", ""),
                            "teamAbbrev": team.get("teamAbbrev", {}).get("default", ""),
                            "wins": team.get("wins", 0),
                            "losses": team.get("losses", 0),
                            "otLosses": team.get("otLosses", 0),
                            "points": team.get("points", 0),
                            "goalsFor": team.get("goalFor", 0),
                            "goalsAgainst": team.get("goalAgainst", 0),
                            "goalsForPctg": team.get("goalsForPctg", 0.0),
                            "home_wins": team.get("homeWins", 0),
                            "home_losses": team.get("homeLosses", 0),
                            "road_wins": team.get("roadWins", 0),
                            "road_losses": team.get("roadLosses", 0),
                            "l10Wins": team.get("l10Wins", 0),
                            "l10Losses": team.get("l10Losses", 0),
                            "streakCode": team.get("streakCode", ""),
                            "streakCount": team.get("streakCount", 0),
                        })
                    self.nhl_data = pd.DataFrame(rows)
                    cache_path.write_text(json.dumps(data, indent=2))
                    logger.info(
                        "✅ [NHL] %d teams loaded from official API", len(rows)
                    )
                    return
        except Exception as e:
            logger.warning("[NHL] Live API error: %s", str(e)[:80])

        # Fallback: cache قبلی
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                standings = data.get("standings", [])
                if standings:
                    rows = []
                    for t in standings:
                        rows.append({
                            "team": t.get("teamName", {}).get("default", ""),
                            "teamAbbrev": t.get("teamAbbrev", {}).get("default", ""),
                            "wins": t.get("wins", 0),
                            "losses": t.get("losses", 0),
                            "otLosses": t.get("otLosses", 0),
                            "points": t.get("points", 0),
                            "goalsFor": t.get("goalFor", 0),
                            "goalsAgainst": t.get("goalAgainst", 0),
                            "l10Wins": t.get("l10Wins", 0),
                            "l10Losses": t.get("l10Losses", 0),
                            "streakCode": t.get("streakCode", ""),
                            "streakCount": t.get("streakCount", 0),
                            "home_wins": t.get("homeWins", 0),
                            "home_losses": t.get("homeLosses", 0),
                            "road_wins": t.get("roadWins", 0),
                            "road_losses": t.get("roadLosses", 0),
                        })
                    self.nhl_data = pd.DataFrame(rows)
                    logger.info("✅ [NHL] %d teams from cache", len(rows))
                    return
            except Exception as e:
                logger.error("[NHL] Cache fallback error: %s", e)

        logger.warning("[NHL] No data available")
        self.nhl_data = None

    def get_nhl_stats(self, team: str) -> dict:
        if self.nhl_data is None or self.nhl_data.empty:
            return {}
        clean = team.lower().strip()
        m = self.nhl_data[
            self.nhl_data["team"].str.lower().str.contains(
                re.escape(clean), na=False
            ) |
            self.nhl_data["teamAbbrev"].str.lower().str.contains(
                re.escape(clean), na=False
            )
        ]
        if m.empty:
            return {}
        row = m.iloc[0]
        gp = max(
            int(row.get("wins", 0)) +
            int(row.get("losses", 0)) +
            int(row.get("otLosses", 0)),
            1,
        )
        gf = int(row.get("goalsFor", 0))
        ga = int(row.get("goalsAgainst", 0))
        l10w = int(row.get("l10Wins", 0))
        l10l = int(row.get("l10Losses", 0))
        return {
            "wins": int(row.get("wins", 0)),
            "losses": int(row.get("losses", 0)),
            "ot_losses": int(row.get("otLosses", 0)),
            "points": int(row.get("points", 0)),
            "games_played": gp,
            "win_pct": round(int(row.get("wins", 0)) / gp, 3),
            "avg_goals_for": round(gf / gp, 2),
            "avg_goals_against": round(ga / gp, 2),
            "goal_diff_per_game": round((gf - ga) / gp, 2),
            "last_10": f"{l10w}W-{l10l}L",
            "streak": f"{row.get('streakCode', '?')}{row.get('streakCount', 0)}",
            "home_wins": int(row.get("home_wins", 0)),
            "home_losses": int(row.get("home_losses", 0)),
            "road_wins": int(row.get("road_wins", 0)),
            "road_losses": int(row.get("road_losses", 0)),
            "source": "nhl_official_api",
        }

    # ──────────────────────────────────────────────────────
    # MLB
    # ──────────────────────────────────────────────────────
    def load_mlb_data(self):
    """
    MLB Stats API رسمی - رایگان، بدون key، همیشه آپدیت
    https://statsapi.mlb.com
    """
    url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025&standingsTypes=regularSeason"
    cache_path = CFG.HISTORICAL_DIR / "mlb_standings.json"

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            data = r.json()
            rows = []
            for record in data.get("records", []):
                for tr in record.get("teamRecords", []):
                    team_info = tr.get("team", {})
                    rows.append({
                        "team": team_info.get("name", ""),
                        "team_id": team_info.get("id", 0),
                        "wins": tr.get("wins", 0),
                        "losses": tr.get("losses", 0),
                        "win_pct": float(tr.get("winningPercentage", 0) or 0),
                        "runs_scored": tr.get("runsScored", 0),
                        "runs_allowed": tr.get("runsAllowed", 0),
                        "home_wins": tr.get("records", {}).get("splitRecords", [{}])[0].get("wins", 0) if tr.get("records", {}).get("splitRecords") else 0,
                        "away_wins": tr.get("records", {}).get("splitRecords", [{}])[1].get("wins", 0) if tr.get("records", {}).get("splitRecords") and len(tr.get("records", {}).get("splitRecords", [])) > 1 else 0,
                        "last10_wins": next((s.get("wins", 0) for s in tr.get("records", {}).get("splitRecords", []) if s.get("type") == "lastTen"), 0),
                        "streak": tr.get("streak", {}).get("streakCode", ""),
                    })
            if rows:
                self.mlb_data = pd.DataFrame(rows)
                cache_path.write_text(json.dumps(data, indent=2))
                logger.info("✅ [MLB] %d teams from official API", len(rows))
                return
    except Exception as e:
        logger.warning("[MLB] Live API error: %s", str(e)[:80])

    # Fallback: cache
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            rows = []
            for record in data.get("records", []):
                for tr in record.get("teamRecords", []):
                    team_info = tr.get("team", {})
                    rows.append({
                        "team": team_info.get("name", ""),
                        "team_id": team_info.get("id", 0),
                        "wins": tr.get("wins", 0),
                        "losses": tr.get("losses", 0),
                        "win_pct": float(tr.get("winningPercentage", 0) or 0),
                        "runs_scored": tr.get("runsScored", 0),
                        "runs_allowed": tr.get("runsAllowed", 0),
                        "streak": tr.get("streak", {}).get("streakCode", ""),
                    })
            if rows:
                self.mlb_data = pd.DataFrame(rows)
                logger.info("✅ [MLB] %d teams from cache", len(rows))
                return
        except Exception as e:
            logger.error("[MLB] Cache error: %s", e)

    logger.warning("[MLB] No data available")
    self.mlb_data = None


def get_mlb_stats(self, team: str) -> dict:
    # اول statsapi پکیج (real-time game logs)
    if HAS_STATSAPI:
        try:
            teams = mlb_statsapi.lookup_team(team)
            if teams:
                tid = teams[0]["id"]
                sd = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
                ed = datetime.now().strftime("%Y-%m-%d")
                sched = mlb_statsapi.schedule(team=tid, start_date=sd, end_date=ed)
                finished = [g for g in sched if g.get("status") == "Final"][-7:]
                if finished:
                    wins = losses = rs = ra = 0
                    for g in finished:
                        ih = g.get("home_id") == tid
                        hs = g.get("home_score", 0) or 0
                        as_ = g.get("away_score", 0) or 0
                        ts = hs if ih else as_
                        os_ = as_ if ih else hs
                        if ts > os_:
                            wins += 1
                        else:
                            losses += 1
                        rs += ts
                        ra += os_
                    tg = max(wins + losses, 1)
                    return {
                        "recent_form": f"{wins}W-{losses}L",
                        "avg_runs_scored": round(rs / tg, 1),
                        "avg_runs_allowed": round(ra / tg, 1),
                        "run_diff": round((rs - ra) / tg, 1),
                        "source": "statsapi_live",
                    }
        except Exception as e:
            logger.debug("[MLB STATSAPI] %s: %s", team, str(e)[:80])

    # Fallback: standings data
    if self.mlb_data is None or self.mlb_data.empty:
        return {}
    clean = team.lower().strip()
    m = self.mlb_data[
        self.mlb_data["team"].str.lower().str.contains(re.escape(clean), na=False)
    ]
    if m.empty:
        return {}
    row = m.iloc[0]
    w = int(row.get("wins", 0))
    l = int(row.get("losses", 0))
    gp = max(w + l, 1)
    rs = int(row.get("runs_scored", 0))
    ra = int(row.get("runs_allowed", 0))
    return {
        "season_record": f"{w}W-{l}L",
        "win_pct": round(float(row.get("win_pct", 0)), 3),
        "avg_runs_scored": round(rs / gp, 1),
        "avg_runs_allowed": round(ra / gp, 1),
        "run_diff_per_game": round((rs - ra) / gp, 2),
        "streak": str(row.get("streak", "")),
        "source": "mlb_official_api",
    }

    # ──────────────────────────────────────────────────────
    # CRICKET
    # ──────────────────────────────────────────────────────
    def load_cricket_data(self):
        """
        منبع: cricsheet.org - T20 internationals
        https://cricsheet.org/downloads/t20s_csv2.zip
        """
        import zipfile
        import io

        zip_path = CFG.HISTORICAL_DIR / "cricket_t20.zip"
        extracted = CFG.HISTORICAL_DIR / "cricket_t20_matches.csv"

        if extracted.exists() and (time.time() - extracted.stat().st_mtime) / 3600 < CFG.TTL_GITHUB_DATA:
            try:
                self.cricket_data = pd.read_csv(extracted, low_memory=False)
                logger.info("✅ [CRICKET] %d records from cache", len(self.cricket_data))
                return
            except Exception:
                pass

        url = "https://cricsheet.org/downloads/t20s_csv2.zip"
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.content) > 1000:
                zip_path.write_bytes(r.content)
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    # فایل اصلی match info
                    csv_files = sorted(
                        [f for f in z.namelist() if f.endswith(".csv")],
                        key=lambda x: len(x),
                    )
                    if csv_files:
                        with z.open(csv_files[0]) as f:
                            extracted.write_bytes(f.read())
                        self.cricket_data = pd.read_csv(extracted, low_memory=False)
                        if "date" in self.cricket_data.columns:
                            self.cricket_data["date"] = pd.to_datetime(
                                self.cricket_data["date"], errors="coerce"
                            )
                        logger.info(
                            "✅ [CRICKET] %d records loaded", len(self.cricket_data)
                        )
                        return
        except Exception as e:
            logger.debug("[CRICKET] Download error: %s", str(e)[:80])

        logger.warning("[CRICKET] No data available")
        self.cricket_data = None

    def get_cricket_stats(self, team: str) -> dict:
        if self.cricket_data is None or self.cricket_data.empty:
            return {}
        df = self.cricket_data
        tc = next(
            (c for c in ["batting_team", "team1", "team"] if c in df.columns), None
        )
        if not tc:
            return {}
        clean = team.lower()
        m = df[df[tc].str.lower().str.contains(re.escape(clean), na=False)]
        if len(m) < 5:
            return {}
        recent = m.tail(10)
        run_col = next(
            (c for c in ["runs_off_bat", "total_runs", "runs"] if c in recent.columns),
            None,
        )
        result = {"matches_found": len(m), "form_sample": len(recent)}
        if run_col:
            result["avg_runs_recent"] = round(
                float(recent[run_col].dropna().mean()), 1
            )
        return result

    # ──────────────────────────────────────────────────────
    # US SPORTS - Combined dispatcher
    # ──────────────────────────────────────────────────────
    def get_us_sports_stats(self, sport: str, team: str) -> dict:
        sport_lower = sport.lower()
        ck = f"{sport_lower}_{team.lower().replace(' ', '')}"

        if CacheManager.is_valid(self.us_cache, ck, 12.0):
            cached = CacheManager.get(self.us_cache, ck)
            if cached:
                return cached

        result = {}
        if "basketball" in sport_lower or "nba" in sport_lower:
            result = self.get_nba_stats(team)
        elif "baseball" in sport_lower or "mlb" in sport_lower:
            result = self.get_mlb_stats(team)
        elif "hockey" in sport_lower or "nhl" in sport_lower:
            result = self.get_nhl_stats(team)
        elif "cricket" in sport_lower or "ipl" in sport_lower or "t20" in sport_lower:
            result = self.get_cricket_stats(team)

        if result:
            self.us_cache = CacheManager.set(self.us_cache, ck, result)
            CacheManager.save(CFG.CACHE_DIR / "us_sports_cache.json", self.us_cache)

        return result

# =========================================================
# 10. ML ENGINE
# =========================================================
class MLPredictionEngine:
    def __init__(self, de: FreeDataEngine):
        self.de = de
        self.football_pipeline: Optional[dict] = None
        self.tennis_pipeline: Optional[dict] = None
        self.nba_pipeline: Optional[dict] = None
        self.is_football_trained = self.is_tennis_trained = self.is_nba_trained = False
        self._football_team_stats: dict = {}
        self._rng = np.random.RandomState(42)

    # ── Football ─────────────────────────────────────────
    def load_or_train_football_model(self):
        path = CFG.ML_DIR/"football_model_v7.pkl"
        if path.exists() and (time.time()-path.stat().st_mtime)/3600 < 24:
            try:
                d=pickle.loads(path.read_bytes())
                self.football_pipeline=d["pipeline"]; self._football_team_stats=d["stats"]
                self.is_football_trained=True; logger.info("⚡ [ML FOOTBALL] Loaded from cache")
                return
            except Exception: pass
        self._train_football()
        if self.is_football_trained:
            try: path.write_bytes(pickle.dumps({"pipeline":self.football_pipeline,"stats":self._football_team_stats})); logger.info("💾 [ML FOOTBALL] Saved")
            except Exception: pass

    def _train_football(self):
        df = self.de.football_data.get("all")
        if df is None or len(df)<300: return
        result_lookup, self._football_team_stats = self._build_fb_rolling(df)
        X, y = self._build_fb_features(result_lookup)
        if len(X)<200 or len(np.unique(y))<2: return
        scaler=RobustScaler(); Xs=scaler.fit_transform(X)
        # FIX: استفاده از Pipeline داخلی برای جلوگیری از data leakage
        model=CalibratedClassifierCV(
            StackingClassifier(
                estimators=[("gb",GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42)),
                             ("rf",RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,n_jobs=-1))],
                final_estimator=LogisticRegression(max_iter=1000,C=0.1,random_state=42), cv=3),
            cv=3, method="isotonic")
        try:
            model.fit(Xs, y)
            self.football_pipeline={"model":model,"scaler":scaler}
            self.is_football_trained=True
            logger.info("✅ [ML FOOTBALL] Trained on %d samples", len(X))
        except Exception as e: logger.error("[ML FOOTBALL] %s", e)

    def _build_fb_rolling(self, df: pd.DataFrame) -> Tuple[dict,dict]:
        ts: Dict[str,deque] = defaultdict(lambda: deque(maxlen=10))
        lookup = {}
        for idx, row in df.iterrows():
            ht=str(row.get("HomeTeam","") or ""); at=str(row.get("AwayTeam","") or "")
            ftr=str(row.get("FTR","") or "")
            if not ht or not at or ftr not in ["H","D","A"]: continue
            try: hg=float(row.get("FTHG",0) or 0); ag=float(row.get("FTAG",0) or 0)
            except (ValueError,TypeError): continue
            def gs(team):
                h=list(ts[team])
                if len(h)<3: return None
                w=np.array([1/(i+1) for i in range(len(h))][::-1]); w/=w.sum()
                return {"avg_gs":float(np.dot(w,[x["gs"] for x in h])),"avg_gc":float(np.dot(w,[x["gc"] for x in h])),
                        "form_pts":float(np.dot(w,[x["pts"] for x in h])),"win_rate":sum(1 for x in h if x["pts"]==3)/len(h)}
            hs=gs(ht); aws=gs(at)
            if hs and aws:
                lookup[idx]={"home_stats":hs,"away_stats":aws,"label":{"H":0,"D":1,"A":2}[ftr]}
            ts[ht].appendleft({"gs":hg,"gc":ag,"pts":3 if ftr=="H" else(1 if ftr=="D" else 0)})
            ts[at].appendleft({"gs":ag,"gc":hg,"pts":3 if ftr=="A" else(1 if ftr=="D" else 0)})
        return lookup, dict(ts)

    def _build_fb_features(self, lookup: dict) -> Tuple[np.ndarray,np.ndarray]:
        feats,labels=[],[]
        for _,d in lookup.items():
            hs=d["home_stats"]; aws=d["away_stats"]
            if not hs or not aws: continue
            feats.append([hs.get("avg_gs",0),hs.get("avg_gc",0),hs.get("form_pts",0),hs.get("win_rate",0),
                          aws.get("avg_gs",0),aws.get("avg_gc",0),aws.get("form_pts",0),aws.get("win_rate",0),
                          hs.get("avg_gs",0)-aws.get("avg_gc",0),aws.get("avg_gs",0)-hs.get("avg_gc",0)])
            labels.append(d["label"])
        if not feats: return np.array([]),np.array([])
        return np.nan_to_num(np.array(feats,dtype=np.float64)), np.array(labels,dtype=np.int32)

    def predict_football(self, home: str, away: str) -> Optional[dict]:
        if not self.is_football_trained: return None
        def ft(team):
            cl=team.lower().strip()
            bm=next((k for k in self._football_team_stats if cl in k.lower() or k.lower() in cl),None)
            if not bm: return None
            h=list(self._football_team_stats[bm])
            if len(h)<3: return None
            w=np.array([1/(i+1) for i in range(len(h))][::-1]); w/=w.sum()
            return {"avg_gs":float(np.dot(w,[x["gs"] for x in h])),"avg_gc":float(np.dot(w,[x["gc"] for x in h])),
                    "form_pts":float(np.dot(w,[x["pts"] for x in h])),"win_rate":sum(1 for x in h if x["pts"]==3)/len(h)}
        hs=ft(home); aws=ft(away)
        if not hs or not aws: return None
        fv=[hs["avg_gs"],hs["avg_gc"],hs["form_pts"],hs["win_rate"],
            aws["avg_gs"],aws["avg_gc"],aws["form_pts"],aws["win_rate"],
            hs["avg_gs"]-aws["avg_gc"],aws["avg_gs"]-hs["avg_gc"]]
        X=np.nan_to_num(np.array([fv],dtype=np.float64))
        Xs=self.football_pipeline["scaler"].transform(X)
        try:
            probs=self.football_pipeline["model"].predict_proba(Xs)[0]
            classes=self.football_pipeline["model"].classes_
            lm={0:"home_win",1:"draw",2:"away_win"}
            return {lm.get(int(c),f"c{c}"):round(float(p),4) for c,p in zip(classes,probs)}
        except Exception as e: logger.warning("[ML FOOTBALL] %s",e); return None

    # ── Tennis ───────────────────────────────────────────
    def load_or_train_tennis_model(self, is_wta: bool = False):
        tour="wta" if is_wta else "atp"
        path=CFG.ML_DIR/f"tennis_model_{tour}_v7.pkl"
        if path.exists() and (time.time()-path.stat().st_mtime)/3600<24:
            try:
                d=pickle.loads(path.read_bytes())
                self.tennis_pipeline=d["pipeline"]; self.is_tennis_trained=True
                logger.info("⚡ [ML TENNIS %s] Loaded from cache", tour.upper()); return
            except Exception: pass
        self._train_tennis(is_wta)
        if self.is_tennis_trained:
            try: path.write_bytes(pickle.dumps({"pipeline":self.tennis_pipeline})); logger.info("💾 [ML TENNIS] Saved")
            except Exception: pass

    def _build_tennis_features(self, df: pd.DataFrame) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        feats,labels,wts=[],[],[]
        for _,row in df.iterrows():
            wr=float(row.get("winner_rank",0) or 0); lr=float(row.get("loser_rank",0) or 0)
            if wr<=0 or lr<=0: continue
            surf=str(row.get("surface","Hard") or "Hard").lower()
            bo=float(row.get("best_of",3) or 3)
            def sf(v,d=0.0):
                try: return float(v or d)
                except: return d
            ws=[sf(row.get("w_ace")),sf(row.get("w_df")),sf(row.get("w_svpt",50)),
                sf(row.get("w_1stIn")),sf(row.get("w_1stWon")),sf(row.get("w_2ndWon")),
                sf(row.get("w_bpSaved")),sf(row.get("w_bpFaced"))]
            ls=[sf(row.get("l_ace")),sf(row.get("l_df")),sf(row.get("l_svpt",50)),
                sf(row.get("l_1stIn")),sf(row.get("l_1stWon")),sf(row.get("l_2ndWon")),
                sf(row.get("l_bpSaved")),sf(row.get("l_bpFaced"))]
            def ns(s):
                sv=max(s[2],1.); i1=max(s[3],1.); bpf=max(s[7],1.)
                return [s[0]/sv,s[1]/sv,s[3]/sv,s[4]/i1,s[5]/max(sv-s[3],1.),s[6]/bpf]
            wn=ns(ws); ln=ns(ls)
            is_p1=self._rng.rand()>0.5
            p1r,p2r=( wr,lr) if is_p1 else (lr,wr)
            p1s,p2s=(wn,ln) if is_p1 else (ln,wn)
            label=1 if is_p1 else 0
            fv=[p1r,p2r,p2r-p1r,p2r/max(p1r,1.),25.,25.,
                1. if surf=="hard" else 0.,1. if surf=="clay" else 0.,1. if surf=="grass" else 0.,bo,
                *p1s,*p2s,p1s[0]-p2s[0],p1s[3]-p2s[3],p1s[5]-p2s[5]]
            feats.append(fv); labels.append(label)
            td=sf(row.get("tourney_date"),20200101)
            wts.append(float(np.clip(0.5+0.5*(td-20200101)/max(20260101-20200101,1),0.5,1.0)))
        if not feats: return np.array([]),np.array([]),np.array([])
        return np.nan_to_num(np.array(feats,dtype=np.float64)),np.array(labels,dtype=np.int32),np.array(wts,dtype=np.float64)

    def _train_tennis(self, is_wta: bool = False):
        df=self.de.wta_matches if is_wta else self.de.atp_matches
        tour="WTA" if is_wta else "ATP"
        if df is None or len(df)<500: logger.warning("[ML TENNIS %s] Insufficient data",tour); return
        X,y,sw=self._build_tennis_features(df)
        if len(X)<200 or len(np.unique(y))<2: return
        scaler=RobustScaler(); Xs=scaler.fit_transform(X)
        # FIX: cv="prefit" بعد از fit با sample_weight - جلوگیری از data leakage
        gb=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42,subsample=0.8)
        try:
            gb.fit(Xs,y,sample_weight=sw)
            # FIX: برای calibration از داده validation جداگانه استفاده کن
            from sklearn.model_selection import train_test_split
            X_cal,X_val,y_cal,y_val=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
            gb2=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42,subsample=0.8)
            sw_cal=sw[:len(X_cal)] if len(sw)>=len(X_cal) else sw
            gb2.fit(X_cal,y_cal,sample_weight=sw_cal[:len(X_cal)])
            cal=CalibratedClassifierCV(gb2,cv="prefit",method="isotonic")
            cal.fit(X_val,y_val)
            self.tennis_pipeline={"model":cal,"scaler":scaler}
            self.is_tennis_trained=True
            logger.info("✅ [ML TENNIS %s] Trained on %d samples",tour,len(X))
        except Exception as e: logger.error("[ML TENNIS %s] %s",tour,e)

    def predict_tennis(self, pa: str, pb: str, stats: dict, surface: str = "hard") -> Optional[dict]:
        if not self.is_tennis_trained: return None
        pas=stats.get("player_a",{}); pbs=stats.get("player_b",{})
        ra=float(pas.get("current_ranking",100) or 100); rb=float(pbs.get("current_ranking",100) or 100)
        def gs(p):
            sv=max(float(p.get("svpt_per_match",50) or 50),1.)
            return [float(p.get("aces_per_match",5) or 5)/sv,
                    float(p.get("df_per_match",2) or 2)/sv,
                    float(p.get("first_serve_in_pct",0.6) or 0.6),
                    float(p.get("first_serve_win_pct",0.7) or 0.7),0.5,
                    float(p.get("bp_saved_pct",0.6) or 0.6)]
        wa=gs(pas); wb=gs(pbs)
        fv=[ra,rb,rb-ra,rb/max(ra,1.),25.,25.,
            1. if surface=="hard" else 0.,1. if surface=="clay" else 0.,1. if surface=="grass" else 0.,3.,
            *wa,*wb,wa[0]-wb[0],wa[3]-wb[3],wa[5]-wb[5]]
        try:
            X=np.nan_to_num(np.array([fv],dtype=np.float64))
            Xs=self.tennis_pipeline["scaler"].transform(X)
            probs=self.tennis_pipeline["model"].predict_proba(Xs)[0]
            classes=self.tennis_pipeline["model"].classes_
            pm={int(c):float(p) for c,p in zip(classes,probs)}
            pa_p=pm.get(1,0.5)
            return {f"{pa}_win_prob":round(pa_p,4),f"{pb}_win_prob":round(1-pa_p,4)}
        except Exception as e: logger.warning("[ML TENNIS] %s",e); return None

    # ── NBA ML ───────────────────────────────────────────
    def load_or_train_nba_model(self):
        path = CFG.ML_DIR/"nba_model_v7.pkl"
        if path.exists() and (time.time()-path.stat().st_mtime)/3600<24:
            try:
                d=pickle.loads(path.read_bytes())
                self.nba_pipeline=d["pipeline"]; self.is_nba_trained=True
                logger.info("⚡ [ML NBA] Loaded from cache"); return
            except Exception: pass
        self._train_nba()
        if self.is_nba_trained:
            try: path.write_bytes(pickle.dumps({"pipeline":self.nba_pipeline}))
            except Exception: pass

    def _train_nba(self):
        df = self.de.nba_data
        if df is None or len(df) < 200: logger.warning("[ML NBA] Insufficient data"); return
        # ستون‌های مورد نیاز: elo_i, opp_elo_i, pts, opp_pts
        need = ["elo_i","opp_elo_i","pts","opp_pts"]
        if not all(c in df.columns for c in need): logger.warning("[ML NBA] Missing columns"); return
        sub = df[need].dropna()
        if len(sub)<100: return
        X = sub[["elo_i","opp_elo_i"]].values
        X = np.hstack([X, (X[:,0]-X[:,1]).reshape(-1,1)])  # elo diff
        y = (sub["pts"].values > sub["opp_pts"].values).astype(int)
        if len(np.unique(y))<2: return
        scaler=RobustScaler(); Xs=scaler.fit_transform(X)
        model=CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=42),
            cv=3, method="isotonic")
        try:
            model.fit(Xs,y)
            self.nba_pipeline={"model":model,"scaler":scaler}
            self.is_nba_trained=True
            logger.info("✅ [ML NBA] Trained on %d samples",len(X))
        except Exception as e: logger.error("[ML NBA] %s",e)

    def predict_nba(self, home_stats: dict, away_stats: dict) -> Optional[dict]:
        if not self.is_nba_trained: return None
        he=float(home_stats.get("elo_rating",1500)); ae=float(away_stats.get("elo_rating",1500))
        fv=[[he,ae,he-ae]]
        try:
            X=np.nan_to_num(np.array(fv,dtype=np.float64))
            Xs=self.nba_pipeline["scaler"].transform(X)
            probs=self.nba_pipeline["model"].predict_proba(Xs)[0]
            classes=self.nba_pipeline["model"].classes_
            pm={int(c):float(p) for c,p in zip(classes,probs)}
            hp=pm.get(1,0.5)
            return {"home_win_prob":round(hp,4),"away_win_prob":round(1-hp,4)}
        except Exception as e: logger.warning("[ML NBA] %s",e); return None

# =========================================================
# 11. POISSON ENGINE
# =========================================================
class PoissonEngine:
    @staticmethod
    def calculate_match_probabilities(home: str, away: str, df: Optional[pd.DataFrame]) -> dict:
        if df is None or df.empty: return {}
        req={"HomeTeam","AwayTeam","FTHG","FTAG"}
        if not req.issubset(df.columns): return {}
        rec=df.dropna(subset=["FTHG","FTAG"]).tail(1500).copy()
        if len(rec)<50: return {}
        la_home=rec["FTHG"].astype(float).mean(); la_away=rec["FTAG"].astype(float).mean()
        if pd.isna(la_home) or la_home==0: return {}
        def fz(t,col):
            cl=t.lower().strip(); m=col.str.lower().str.strip()==cl
            if m.any(): return m
            for p in cl.split():
                if len(p)>3:
                    m2=col.str.lower().str.contains(re.escape(p),na=False)
                    if m2.any(): return m2
            return pd.Series([False]*len(col),index=col.index)
        hm=rec[fz(home,rec["HomeTeam"])]; am=rec[fz(away,rec["AwayTeam"])]
        if len(hm)<5 or len(am)<5: return {}
        ha=hm["FTHG"].astype(float).mean()/la_home; hd=hm["FTAG"].astype(float).mean()/la_away
        aa=am["FTAG"].astype(float).mean()/la_away; ad=am["FTHG"].astype(float).mean()/la_home
        if any(pd.isna(v) or v==0 for v in [ha,hd,aa,ad]): return {}
        hxg=float(np.clip(ha*ad*la_home,0.1,8.)); axg=float(np.clip(aa*hd*la_away,0.1,8.))
        mg=6; pm=np.zeros((mg+1,mg+1)); rho=-0.1
        for x in range(mg+1):
            for y in range(mg+1):
                base=stats_scipy.poisson.pmf(x,hxg)*stats_scipy.poisson.pmf(y,axg)
                adj=(1-hxg*axg*rho if x==0 and y==0 else(1+hxg*rho if x==0 and y==1 else(1+axg*rho if x==1 and y==0 else(1-rho if x==1 and y==1 else 1.))))
                pm[x,y]=base*max(0.,adj)
        t=pm.sum();
        if t==0: return {}
        pm/=t
        return {"home_xg":round(hxg,2),"away_xg":round(axg,2),
                "home_win_prob_poisson":round(float(np.sum(np.tril(pm,-1))),3),
                "draw_prob_poisson":round(float(np.sum(np.diag(pm))),3),
                "away_win_prob_poisson":round(float(np.sum(np.triu(pm,1))),3)}

# =========================================================
# 12. EV ENGINE
# =========================================================
class EVEngine:
    @staticmethod
    def remove_vig_power(odds_list: List[float]) -> List[float]:
        implied=[1/o for o in odds_list if o>1.]
        if not implied: return []
        total=sum(implied)
        if abs(total-1.)<0.001: return implied
        def f(k): return sum(p**k for p in implied)-1.
        try:
            fa,fb=f(0.5),f(3.0)
            if fa*fb>=0: return [p/total for p in implied]
            k=brentq(f,0.5,3.0,xtol=1e-6)
            tp=[p**k for p in implied]; s=sum(tp)
            return [p/s for p in tp] if s>0 else [p/total for p in implied]
        except Exception: return [p/total for p in implied]

    @staticmethod
    def remove_vig_shin(odds_list: List[float]) -> List[float]:
        implied=[1/o for o in odds_list if o>1.]
        if not implied: return []
        n=len(implied); total=sum(implied)
        if total==0: return implied
        z=max(0.,min((total-1)/max(total-1/n,1e-10),0.2))
        tp=[]
        for p in implied:
            denom=2*(1-n*z)
            if abs(denom)<1e-10: tp.append(p/total); continue
            inner=z**2+4*(1-z)*(p**2/total)
            if inner<0: tp.append(p/total); continue
            tp.append((-z+inner**0.5)/denom)
        s=sum(tp)
        return [p/s for p in tp] if s>0 else [p/total for p in implied]

    @staticmethod
    def kelly(prob: float, odds: float) -> float:
        b=odds-1.
        if b<=0 or prob<=0 or prob>=1: return 0.
        k=max(0.,(prob*b-(1-prob))/b)
        return round(min(k*CFG.KELLY_FRACTION, CFG.MAX_KELLY_PCT/100),4)

def calculate_sharp_ev_advanced(markets_data: dict) -> list:  # FIX: حذف پارامتر بلااستفاده
    best_per_market: dict = {}
    for mk, ml in markets_data.items():
        if not isinstance(ml,list): continue
        sharp_all: Dict[str,List[float]]=defaultdict(list)
        soft_all: Dict[str,List[float]]=defaultdict(list)
        best_mkt: Dict[str,Tuple[float,str]]={}
        for entry in ml:
            if not isinstance(entry,dict): continue
            bk=entry.get("bookmaker_key",""); bk_name=entry.get("bookmaker",bk)
            is_sharp=bk in CFG.SHARP_BOOKMAKERS
            for o in entry.get("outcomes",[]):
                if not isinstance(o,dict): continue
                name=(f"{o['name']} {o.get('point')}" if o.get("point") is not None else o.get("name",""))
                if not name: continue
                try: price=float(o["price"])
                except (KeyError,TypeError,ValueError): continue
                if price<=1.: continue
                (sharp_all if is_sharp else soft_all)[name].append(price)
                if name not in best_mkt or price>best_mkt[name][0]: best_mkt[name]=(price,bk_name)
        if not best_mkt: continue
        sharp_best={n:max(p) for n,p in sharp_all.items() if p}
        has_sharp=bool(sharp_best)
        if not sharp_best: sharp_best={n:max(p) for n,p in soft_all.items() if p}
        if not sharp_best: continue
        outcomes=list(sharp_best.keys()); odds_list=[sharp_best[o] for o in outcomes]
        impl_sum=sum(1/o for o in odds_list if o>0)
        if not (CFG.MIN_VALID_IMPLIED_SUM<=impl_sum<=CFG.MAX_VALID_IMPLIED_SUM): continue
        if len(outcomes)<CFG.MARKET_EXPECTED_OUTCOMES.get(mk,{}).get("min",2): continue
        try:
            tp_pw=EVEngine.remove_vig_power(odds_list); tp_sh=EVEngine.remove_vig_shin(odds_list)
            if len(tp_pw)!=len(outcomes) or len(tp_sh)!=len(outcomes): raise ValueError()
            tp={outcomes[i]:0.6*tp_pw[i]+0.4*tp_sh[i] for i in range(len(outcomes))}
        except Exception: tp={outcomes[i]:(1/odds_list[i])/max(impl_sum,1e-10) for i in range(len(outcomes))}
        min_odds=CFG.H2H_MIN_ODDS if mk=="h2h" else CFG.TOTALS_MIN_ODDS
        min_ev=(CFG.H2H_MIN_EV if mk=="h2h" else CFG.TOTALS_MIN_EV)*(1. if has_sharp else 1.5)
        best_opp=None
        for on in outcomes:
            true_p=tp.get(on,0)
            if true_p<=0 or true_p>=1: continue
            bp,bbm=best_mkt.get(on,(0,"?"))
            if bp<=1.: continue
            ev=true_p*bp-1.
            if ev<min_ev or ev>CFG.MAX_REALISTIC_EV or bp<min_odds: continue
            kelly_p=EVEngine.kelly(true_p,bp)
            sp=sharp_best.get(on,bp)
            clv=(bp/sp-1)*100 if sp>0 else 0.
            opp={"pick":on,"market":mk,"market_label":get_market_label(mk),
                 "prob":round(true_p,4),"odds":round(bp,3),"bookmaker":bbm,
                 "ev":round(ev,4),"edge_pct":round(ev*100,2),"kelly_pct":round(kelly_p*100,2),
                 "clv_pct":round(clv,2),"has_sharp_line":has_sharp,
                 "devigging_method":"power_shin_weighted","steam_pct":0.}
            if best_opp is None or opp["ev"]>best_opp["ev"]: best_opp=opp
        if best_opp: best_per_market[mk]=best_opp
    return sorted(best_per_market.values(),key=lambda x:x["ev"],reverse=True)

# =========================================================
# 13. CONFIDENCE ENGINE
# =========================================================
class ConfidenceEngine:
    W={"base":42,"ev_high":15,"ev_medium":10,"ev_low":5,"sharp_line":8,
       "football_stats":5,"elo_high":8,"elo_medium":4,"ml_strong":8,
       "ml_medium":4,"poisson_confirm":5,"smart_money":8,"kelly_high":5,"kelly_medium":3}

    @classmethod
    def calculate_math_score(cls, opp: dict, stats: dict, market: str,
                              ml_pred: Optional[dict]=None, poisson_pred: Optional[dict]=None) -> int:
        s=cls.W["base"]
        ev=opp.get("ev",0)*100
        s+=(cls.W["ev_high"] if ev>5 else cls.W["ev_medium"] if ev>3 else cls.W["ev_low"] if ev>1 else 0)
        if opp.get("has_sharp_line"): s+=cls.W["sharp_line"]
        kelly=opp.get("kelly_pct",0)
        s+=(cls.W["kelly_high"] if kelly>2 else cls.W["kelly_medium"] if kelly>1 else 0)
        if stats.get("football_stats"): s+=cls.W["football_stats"]
        delta=abs(stats.get("elo_data",{}).get("delta",0))
        s+=(cls.W["elo_high"] if delta>150 else cls.W["elo_medium"] if delta>75 else 0)
        if ml_pred:
            mx=max((v for v in ml_pred.values() if isinstance(v,float) and 0<v<=1),default=0)
            s+=(cls.W["ml_strong"] if mx>0.65 else cls.W["ml_medium"] if mx>0.55 else 0)
        if poisson_pred: s+=cls.W["poisson_confirm"]
        if opp.get("steam_pct",0)>=3: s+=cls.W["smart_money"]
        return int(np.clip(s,0,100))

# =========================================================
# 14. UTILITIES
# =========================================================
def robust_json_extractor(raw: str) -> Optional[dict]:
    if not raw: return None
    clean=re.sub(r"<think>[\s\S]*?</think>","",raw,flags=re.IGNORECASE)
    clean=re.sub(r"```(?:json)?","",clean).strip().rstrip("`").strip()
    try: return json.loads(clean)
    except Exception: pass
    for m in reversed(list(re.finditer(r"\{[^{}]*\}",clean))):
        try:
            r=json.loads(m.group(0))
            if isinstance(r,dict) and r: return r
        except Exception: continue
    try:
        m=re.search(r"\{[\s\S]*\}",clean)
        if m: return json.loads(m.group(0))
    except Exception: pass
    return None

def clean_team_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)","",str(name or "")).strip()

def normalize_sport_key(sport_title: str) -> str:
    lower=(sport_title or "").lower()
    if any(k in lower for k in ["tennis","atp","wta"]): return "tennis"
    if any(k in lower for k in ["soccer","football","premier league","la liga","bundesliga","serie a","ligue 1","champions","brasileirao","liga mx"]): return "football"
    if any(k in lower for k in ["basketball","nba","euroleague"]): return "basketball"
    if any(k in lower for k in ["baseball","mlb"]): return "baseball"
    if any(k in lower for k in ["hockey","nhl"]): return "hockey"
    if any(k in lower for k in ["cricket","ipl","t20","odi"]): return "cricket"
    return "other"

def get_countdown_str(ct: str, now: datetime) -> str:
    try:
        mt=datetime.fromisoformat(ct.replace("Z","+00:00"))
        if mt.tzinfo is None: mt=mt.replace(tzinfo=timezone.utc)
        mins=int((mt-now).total_seconds()/60)
        if mins>60: return f"{mins//60}h {mins%60}m"
        if mins>0: return f"{mins}m"
        return "LIVE"
    except Exception: return "N/A"

def get_market_label(mk: str) -> str:
    return {"h2h":"Match Winner","totals":"Over/Under","spreads":"Point Spread"}.get(mk,mk.replace("_"," ").title())

def _get_sport_emoji(sk: str) -> str:
    return {"tennis":"🎾","football":"⚽","basketball":"🏀","baseball":"⚾","hockey":"🏒","cricket":"🏏"}.get(sk,"🏆")

# =========================================================
# 15. LINE MOVEMENT TRACKER
# =========================================================
class LineMovementTracker:
    def __init__(self):
        self._path=CFG.CACHE_DIR/"line_movement.json"
        self._lock=threading.Lock()
        self.data=CacheManager.load(self._path)
        self._cleanup()

    def _cleanup(self):
        now=datetime.now(timezone.utc)
        to_del=[k for k,v in self.data.items() if not isinstance(v,dict) or
                (now-datetime.fromisoformat(v.get("timestamp","2000-01-01T00:00:00+00:00")).replace(tzinfo=timezone.utc)
                 if datetime.fromisoformat(v.get("timestamp","2000-01-01T00:00:00+00:00")).tzinfo is None
                 else now-datetime.fromisoformat(v.get("timestamp","2000-01-01T00:00:00+00:00")))>timedelta(hours=24)]
        for k in to_del: self.data.pop(k,None)

    def record_and_get_movement(self, home: str, away: str, market: str, outcome: str, odds: float) -> float:
        if odds<=1.: return 0.
        mk=hashlib.md5(f"{home}|{away}|{market}|{outcome}".encode()).hexdigest()
        with self._lock:  # FIX: thread-safe
            now=datetime.now(timezone.utc).isoformat()
            if mk not in self.data:
                self.data[mk]={"initial_odds":odds,"current_odds":odds,"timestamp":now}
                CacheManager.save(self._path,self.data); return 0.
            init=self.data[mk].get("initial_odds",odds)
            self.data[mk].update({"current_odds":odds,"timestamp":now})
            CacheManager.save(self._path,self.data)
        return round((init/odds-1)*100,2) if init>0 else 0.

line_movement_tracker=LineMovementTracker()

# =========================================================
# 16. AI DECISION ENGINE
# =========================================================
def generate_ai_decision(home: str, away: str, sport: str, sport_key: str,
                          opp: dict, stats: dict, math_score: int,
                          ml_pred: Optional[dict]=None, poisson_pred: Optional[dict]=None) -> dict:
    default={
        "sport_emoji":_get_sport_emoji(sport_key),"decision":"SKIP",
        "ai_confidence":math_score,"math_confidence":math_score,
        "final_confidence":math_score,"risk_level":"High",
        "logic":"Insufficient data for AI decision.","key_factors":[],"red_flags":[],
    }
    if math_score < CFG.MIN_MATH_SCORE_TO_CALL_AI:
        return {**default,"logic":f"Math score {math_score} below threshold."}

    parts=[]
    parts.append(f"=== MARKET ===\nPick:{opp['pick']} | Market:{opp['market_label']} | "
                 f"Odds:{opp['odds']} | TrueProb:{opp['prob']*100:.1f}% | EV:{opp['edge_pct']:+.2f}% | "
                 f"Kelly:{opp.get('kelly_pct',0):.1f}% | SharpLine:{opp.get('has_sharp_line',False)} | "
                 f"CLV:{opp.get('clv_pct',0):+.1f}% | Steam:{opp.get('steam_pct',0):.1f}% | MathScore:{math_score}/100")

    if stats.get("historical_data"):
        pa=stats["historical_data"].get("player_a",{}); pb=stats["historical_data"].get("player_b",{})
        h2h=stats["historical_data"].get("h2h",{})
        parts.append(f"=== TENNIS ===\n{home}: Rank={pa.get('current_ranking','N/A')} Form={pa.get('recent_form','N/A')} "
                     f"WR={pa.get('recent_win_rate',0)*100:.1f}% Ace={pa.get('aces_per_match','N/A')} "
                     f"1stIn={pa.get('first_serve_in_pct',0)*100:.1f}% BP%={pa.get('bp_saved_pct',0)*100:.1f}%\n"
                     f"{away}: Rank={pb.get('current_ranking','N/A')} Form={pb.get('recent_form','N/A')} "
                     f"WR={pb.get('recent_win_rate',0)*100:.1f}% Ace={pb.get('aces_per_match','N/A')}\n"
                     f"H2H:{h2h.get('total',0)} matches Dominance:{h2h.get('dominance','balanced')} "
                     f"Surface:{json.dumps(h2h.get('by_surface',{}))}")

    if stats.get("football_stats"):
        hm=stats["football_stats"].get("home",{}); aw=stats["football_stats"].get("away",{})
        h2h=stats["football_stats"].get("h2h",{})
        parts.append(f"=== FOOTBALL ===\n{home}(H): Form={hm.get('form_string','N/A')} "
                     f"GS={hm.get('avg_scored',0):.2f} GC={hm.get('avg_conceded',0):.2f} "
                     f"WR={hm.get('win_rate',0)*100:.1f}% Over25={hm.get('over25_rate',0)*100:.1f}% "
                     f"BTTS={hm.get('btts_rate',0)*100:.1f}% HomeWR={hm.get('venue_win_rate',0)*100:.1f}% "
                     f"WFP={hm.get('weighted_form_points',0):.2f}\n"
                     f"{away}(A): Form={aw.get('form_string','N/A')} "
                     f"GS={aw.get('avg_scored',0):.2f} GC={aw.get('avg_conceded',0):.2f} "
                     f"WR={aw.get('win_rate',0)*100:.1f}% Over25={aw.get('over25_rate',0)*100:.1f}%\n"
                     f"H2H({h2h.get('total_matches',0)}): AvgG={h2h.get('avg_goals',0):.2f} "
                     f"Over25={h2h.get('over25_rate',0)*100:.1f}% BTTS={h2h.get('btts_rate',0)*100:.1f}%")

    if stats.get("elo_data"):
        e=stats["elo_data"]
        parts.append(f"=== ELO ===\n{home}:{e.get('home_elo')} {away}:{e.get('away_elo')} "
                     f"Delta:{e.get('delta')}({e.get('elo_confidence','?')}) "
                     f"WinProb→{home}:{e.get('home_win_prob_elo',0)*100:.1f}% {away}:{e.get('away_win_prob_elo',0)*100:.1f}%")

    if ml_pred: parts.append(f"=== ML MODEL ===\n{json.dumps(ml_pred)}")
    if poisson_pred:
        parts.append(f"=== POISSON ===\nhome_xg:{poisson_pred.get('home_xg')} away_xg:{poisson_pred.get('away_xg')} "
                     f"H:{poisson_pred.get('home_win_prob_poisson',0)*100:.1f}% "
                     f"D:{poisson_pred.get('draw_prob_poisson',0)*100:.1f}% "
                     f"A:{poisson_pred.get('away_win_prob_poisson',0)*100:.1f}%")

    if stats.get("us_sports"):
        us=stats["us_sports"]
        parts.append(f"=== US SPORTS ===\n{home}:{json.dumps(us.get('home',{}))} {away}:{json.dumps(us.get('away',{}))}")

    if stats.get("nba_matchup"):
        nb=stats["nba_matchup"]
        parts.append(f"=== NBA ELO ===\n{home}:{nb.get('home',{}).get('elo_rating','?')} "
                     f"{away}:{nb.get('away',{}).get('elo_rating','?')} "
                     f"HomeWinProb:{nb.get('elo_home_win_prob',0)*100:.1f}%")

    sys_inst=(
        "You are an elite sports betting analyst. Analyze ALL data and make a BET/SKIP decision.\n\n"
        "BET when: EV>1.5% AND (sharp line OR strong historical edge) AND models agree\n"
        "SKIP when: Conflicting signals OR EV only from soft books OR insufficient data\n\n"
        "Confidence: 75-100=Strong BET | 62-74=Moderate BET | 50-61=Weak BET | 0-49=NO BET\n\n"
        'Return ONLY: {"decision":"BET"/"SKIP","confidence":0-100,"sport_emoji":"<emoji>",'
        '"risk_level":"Low"/"Medium"/"High","key_factors":["..."],"logic":"2-3 sentences","red_flags":[]}\n\n'
        "RULES: Base on data only | EV<1.5% no sharp→SKIP | ML/Poisson disagree→lower conf | Be conservative"
    )
    prompt=f"MATCH:{home} vs {away} | SPORT:{sport} | PICK:{opp['pick']} | MARKET:{opp['market_label']}\n\n"+"\n\n".join(parts)+"\n\nReturn BET/SKIP decision as JSON."

    ai_data=gemini_manager.generate(prompt=prompt,system_instruction=sys_inst,temperature=CFG.AI_TEMPERATURE)

    if not ai_data or not isinstance(ai_data,dict):
        logger.warning("[AI JUDGE] No response → math fallback")
        return {**default,"decision":"BET" if math_score>=55 else "SKIP","logic":"AI unavailable - math models only."}

    decision=str(ai_data.get("decision","SKIP")).upper().strip()
    if decision not in ["BET","SKIP"]: decision="SKIP"
    try: ai_conf=int(np.clip(float(ai_data.get("confidence",math_score)),0,100))
    except (ValueError,TypeError): ai_conf=math_score

    hybrid=ai_conf*CFG.AI_WEIGHT+math_score*CFG.MATH_WEIGHT
    ai_delta=hybrid-math_score
    if ai_delta>CFG.MAX_AI_BOOST: hybrid=math_score+CFG.MAX_AI_BOOST
    elif ai_delta<-CFG.MAX_AI_PENALTY: hybrid=math_score-CFG.MAX_AI_PENALTY
    final=int(np.clip(hybrid,0,100))

    if decision=="BET" and ai_conf<50: decision="SKIP"; logger.warning("[AI] Inconsistent BET+conf%d→SKIP",ai_conf)

    kf_raw=ai_data.get("key_factors",[]); kf=[str(f)[:120] for f in kf_raw[:5]] if isinstance(kf_raw,list) else []
    rf_raw=ai_data.get("red_flags",[]); rf=[str(f)[:120] for f in rf_raw[:3]] if isinstance(rf_raw,list) else []
    rl=str(ai_data.get("risk_level","Medium")); rl=rl if rl in ["Low","Medium","High"] else "Medium"
    se_raw=ai_data.get("sport_emoji",""); se=str(se_raw).strip() if se_raw else _get_sport_emoji(sport_key)
    logic=str(ai_data.get("logic",default["logic"]))[:600]

    logger.info("[AI JUDGE] %s vs %s | %s | AI:%d Math:%d Final:%d | Flags:%s",
                home,away,decision,ai_conf,math_score,final,rf if rf else "none")
    return {"sport_emoji":se,"decision":decision,"ai_confidence":ai_conf,
            "math_confidence":math_score,"final_confidence":final,
            "risk_level":rl,"logic":logic,"key_factors":kf,"red_flags":rf}

# =========================================================
# 17. TELEGRAM
# =========================================================
def send_telegram(msg: str) -> bool:
    MAX=4000; chunks=[]
    if len(msg)<=MAX: chunks=[msg]
    else:
        cur=""
        for line in msg.split("\n"):
            if len(cur)+len(line)+1>MAX:
                if cur: chunks.append(cur.strip())
                cur=line+"\n"
            else: cur+=line+"\n"
        if cur.strip(): chunks.append(cur.strip())
    ok=True
    for chunk in chunks:
        try:
            r=requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                            json={"chat_id":TELEGRAM_CHAT_ID,"text":chunk,"parse_mode":"HTML","disable_web_page_preview":True},timeout=15)
            if not r.ok: logger.error("Telegram [%d]: %s",r.status_code,r.text[:200]); ok=False
        except requests.RequestException as e: logger.error("Telegram: %s",e); ok=False
    return ok

def build_signal_message(home,away,sport,sport_key,opp,ai_data,stats,math_score,ml_pred,poisson_pred,now_utc,commence_time) -> str:
    fc=ai_data["final_confidence"]
    ci="🔥" if fc>=CFG.HIGH_CONFIDENCE else("✅" if fc>=CFG.MEDIUM_CONFIDENCE else "⚡")
    ri={"Low":"🟢","Medium":"🟠","High":"🔴"}.get(ai_data["risk_level"],"🟠")
    he=html_lib.escape(home); ae=html_lib.escape(away)
    se=html_lib.escape(sport); pe=html_lib.escape(str(opp["pick"]))
    me=html_lib.escape(opp["market_label"]); le=html_lib.escape(str(ai_data["logic"]))
    extras=""
    if opp.get("steam_pct",0)>2.: extras+=f" | 📉 <b>Steam:</b>-{opp['steam_pct']:.1f}%"
    if abs(opp.get("clv_pct",0))>0.5: extras+=f" | 📊 <b>CLV:</b>{opp.get('clv_pct',0):+.1f}%"
    ml_line=("\n🧠 <b>Models:</b> ML+Poisson" if ml_pred and poisson_pred else
             "\n🧠 <b>Models:</b> ML" if ml_pred else
             "\n🧠 <b>Models:</b> Poisson" if poisson_pred else "")
    badges=[]
    if stats.get("historical_data"): badges.append("📚 Historical")
    if stats.get("football_stats"): badges.append("⚽ Match Data")
    if stats.get("elo_data"): badges.append("📊 Elo")
    if ml_pred: badges.append("🤖 ML")
    if poisson_pred: badges.append("📐 Poisson")
    if stats.get("nba_matchup"): badges.append("🏀 NBA-Elo")
    if stats.get("us_sports"): badges.append("🇺🇸 US Stats")
    data_line="\n📋 <b>Sources:</b> "+" | ".join(badges) if badges else ""
    kf=ai_data.get("key_factors",[])
    kf_line=f"\n\n🔑 <b>Key Factors:</b>\n  • "+"  \n• ".join(html_lib.escape(str(f)) for f in kf[:3]) if kf else ""
    rf=ai_data.get("red_flags",[])
    rf_line=f"\n⚠️ <b>Monitored:</b> <i>"+" | ".join(html_lib.escape(str(f)) for f in rf)+"</i>" if rf else ""
    return (f"{ai_data.get('sport_emoji','🏆')} <b>{se}</b>\n\n"
            f"⚔️ <b>{he}</b> vs <b>{ae}</b>\n⏳ <b>Starts in:</b> {get_countdown_str(commence_time,now_utc)}\n\n"
            f"🎯 <b>PICK:</b> <code>{pe}</code> @ <b>{opp['odds']}</b>\n📊 <b>Market:</b> {me}\n\n"
            f"📈 <b>Edge:</b> {opp['edge_pct']:.2f}% | 💰 <b>Stake:</b> {opp.get('kelly_pct',0):.1f}% bankroll{extras}\n\n"
            f"{ri} <b>Risk:</b> {ai_data['risk_level']}  |  {ci} <b>Confidence: {fc}%</b>\n"
            f"⚙️ <i>(Math:{math_score} | AI:{ai_data['ai_confidence']})</i>"
            f"{ml_line}{data_line}{kf_line}{rf_line}\n\n"
            f"💡 <b>AI ANALYSIS:</b>\n<blockquote>{le}</blockquote>\n\n"
            f"🔍 <i>Curated by {html_lib.escape(CFG.TELEGRAM_ID)}</i>")

# =========================================================
# 18. ODDS FETCHER
# =========================================================
class SmartOddsCache:
    def __init__(self):
        self.cache=CacheManager.load(CFG.ODDS_CACHE_FILE)

    def _key(self,markets,wh):
        raw=f"{','.join(sorted(markets))}|{wh}|{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H')}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_cached(self,markets,wh):
        k=self._key(markets,wh)
        if CacheManager.is_valid_minutes(self.cache,k,CFG.TTL_ODDS_CACHE_MINUTES):
            d=CacheManager.get(self.cache,k)
            if d: logger.info("💾 [ODDS CACHE] HIT %d events",len(d)); return d
        return None

    def save_cached(self,markets,wh,events):
        k=self._key(markets,wh)
        self.cache=CacheManager.set(self.cache,k,events)
        CacheManager.save(CFG.ODDS_CACHE_FILE,self.cache)
        logger.info("💾 [ODDS CACHE] SAVED %d events",len(events))

    def get_stale(self,markets,wh,max_ttl=2.):
        k=self._key(markets,wh)
        if CacheManager.is_valid(self.cache,k,max_ttl): return CacheManager.get(self.cache,k)
        return None

odds_cache=SmartOddsCache()

async def fetch_market(session,market,now_utc,api_key,label):
    end=now_utc+timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    params={"apiKey":api_key,"regions":CFG.ODDS_API_REGIONS,"markets":market,"oddsFormat":"decimal","dateFormat":"iso"}
    try:
        async with session.get("https://api.the-odds-api.com/v4/sports/upcoming/odds",params=params,
                               timeout=aiohttp.ClientTimeout(total=20)) as r:
            rem=int(r.headers.get("x-requests-remaining",-1)); used=int(r.headers.get("x-requests-used",0))
            if r.status==200:
                events=await r.json(content_type=None)
                odds_key_manager.record_usage(label,used,rem)
                valid=[]
                for e in events:
                    if not isinstance(e,dict): continue
                    try:
                        mt=datetime.fromisoformat(e.get("commence_time","").replace("Z","+00:00"))
                        if mt.tzinfo is None: mt=mt.replace(tzinfo=timezone.utc)
                        if now_utc<=mt<=end: valid.append(e)
                    except Exception: continue
                logger.info("🔑 [%s] OK rem:%d market:%s events:%d",label,rem,market,len(valid))
                return valid,200,None
            err=await r.text()
            reasons={401:"Invalid key",402:"Quota exhausted",429:"Rate limited",422:"Invalid params"}
            return [],r.status,reasons.get(r.status,f"HTTP {r.status}:{err[:80]}")
    except asyncio.TimeoutError: return [],0,"Timeout"
    except Exception as e: return [],0,str(e)[:80]

async def fetch_all_odds_async() -> list:
    now=datetime.now(timezone.utc)
    cached=odds_cache.get_cached(CFG.ODDS_API_MARKETS,CFG.MATCH_WINDOW_HOURS)
    if cached: return cached
    logger.info("💾 [ODDS CACHE] MISS - Calling API...")
    all_events: Dict[str,dict]={}
    success=False
    for ki in odds_key_manager.get_active_keys():
        ak=ki["key"]; label=ki["label"]
        logger.info("🔑 [TRYING] %s...",label)
        conn=aiohttp.TCPConnector(limit=10,ssl=False)
        async with aiohttp.ClientSession(connector=conn) as sess:
            tasks=[fetch_market(sess,m,now,ak,label) for m in CFG.ODDS_API_MARKETS]
            results=await asyncio.gather(*tasks,return_exceptions=True)
        any_ok=False; hard_fail=None
        for i,res in enumerate(results):
            if isinstance(res,Exception): logger.error("🔑❌ [%s] %s",label,res); continue
            events,status,err=res
            if status==200:
                any_ok=True
                for e in events:
                    eid=e.get("id")
                    if not eid: continue
                    if eid not in all_events:
                        all_events[eid]={"id":eid,"sport_key":e.get("sport_key",""),
                                         "sport_title":e.get("sport_title",""),
                                         "commence_time":e.get("commence_time",""),
                                         "home_team":e.get("home_team",""),
                                         "away_team":e.get("away_team",""),"_markets_data":{}}
                    # FIX: ساختار صحیح _markets_data
                    for bm in e.get("bookmakers",[]):
                        bk=bm.get("key",""); bt=bm.get("title",bk)
                        for md in bm.get("markets",[]):
                            mk=md.get("key","")
                            if not mk: continue
                            all_events[eid]["_markets_data"].setdefault(mk,[]).append(
                                {"bookmaker":bt,"bookmaker_key":bk,"outcomes":md.get("outcomes",[])})
            else:
                logger.warning("🔑⚠️ [%s] %s: %s",label,CFG.ODDS_API_MARKETS[i] if i<len(CFG.ODDS_API_MARKETS) else "?",err)
                if status in [401,402,429]: hard_fail=status
        if any_ok:
            success=True; logger.info("✅ [%s] %d events",label,len(all_events)); break
        else:
            idx=next((i for i,k in enumerate(odds_key_manager.keys) if k["label"]==label),-1)
            if idx>=0: odds_key_manager.mark_failed(idx,f"HTTP {hard_fail}" if hard_fail else "All failed")
    if not success:
        logger.error("🔑❌ ALL KEYS FAILED!")
        stale=odds_cache.get_stale(CFG.ODDS_API_MARKETS,CFG.MATCH_WINDOW_HOURS,2.)
        if stale: logger.warning("💾 [STALE] %d events",len(stale)); return stale
        return []
    final=list(all_events.values())
    odds_cache.save_cached(CFG.ODDS_API_MARKETS,CFG.MATCH_WINDOW_HOURS,final)
    logger.info("📊 [API USAGE] %s",odds_key_manager.get_usage_summary())
    return final

# =========================================================
# 19. MAIN PIPELINE
# =========================================================
async def async_main():
    logger.info("="*65)
    logger.info("  ZBET90 ENGINE v7.0 | Multi-Sport | AI 70%% + Math 30%%")
    logger.info("="*65)
    logger.info("🔑 %s",odds_key_manager.get_usage_summary())
    sent=SentHistory(); now=datetime.now(timezone.utc)

    # Phase 1: Load Data
    logger.info("📥 [PHASE 1] Loading data...")
    de=FreeDataEngine()
    de.load_tennis_data(); de.load_football_data()
    de.load_nba_data(); de.load_nhl_data(); de.load_mlb_data(); de.load_cricket_data()

    # Phase 2: Train ML
    logger.info("🧠 [PHASE 2] ML models...")
    ml=MLPredictionEngine(de)
    ml.load_or_train_football_model()
    ml.load_or_train_tennis_model(is_wta=False)
    ml.load_or_train_nba_model()

    # Phase 3: Fetch Odds
    logger.info("📡 [PHASE 3] Fetching odds (%.1fh window)...",CFG.MATCH_WINDOW_HOURS)
    events=await fetch_all_odds_async()
    if not events:
        logger.info("❌ No events in window."); logger.info("📊 %s",odds_key_manager.get_usage_summary()); return

    logger.info("🔍 [PHASE 4] Analyzing %d events...",len(events))
    events.sort(key=lambda x:x.get("commence_time",""))
    total_sent=total_analyzed=skip_math=skip_ai=skip_conf=0

    for event in events:
        home=clean_team_name(event.get("home_team","")); away=clean_team_name(event.get("away_team",""))
        sport=event.get("sport_title","Unknown"); sport_key=normalize_sport_key(sport)
        if not home or not away: continue
        markets_data=event.get("_markets_data",{})
        opps=calculate_sharp_ev_advanced(markets_data)  # FIX: یک آرگومان
        if not opps: continue
        opp=opps[0]; total_analyzed+=1
        if sent.was_sent(home,away,opp["market"]): continue
        if opp["ev"]<CFG.MATH_MIN_EV_TO_ANALYZE: skip_math+=1; continue  # FIX: نام صحیح

        opp["steam_pct"]=line_movement_tracker.record_and_get_movement(home,away,opp["market"],opp["pick"],opp["odds"])

        stats: dict={}; ml_pred=None; poisson_pred=None

        if sport_key=="tennis":
            is_wta="wta" in sport.lower()
            ts=de.get_tennis_stats(home,away,is_wta)
            if ts: stats["historical_data"]=ts
            if ml.is_tennis_trained and ts:
                surf="grass" if "wimbledon" in sport.lower() else("hard" if any(k in sport.lower() for k in ["us open","hard"]) else "clay" if "clay" in sport.lower() else "hard")
                ml_pred=ml.predict_tennis(home,away,ts,surf)
                if ml_pred: stats["ml_prediction"]=ml_pred

        elif sport_key=="football":
            fs=de.get_football_stats(home,away)
            if fs: stats["football_stats"]=fs
            ed=de.get_elo_delta(home,away)
            if ed: stats["elo_data"]=ed
            if ml.is_football_trained:
                ml_pred=ml.predict_football(home,away)
                if ml_pred: stats["ml_prediction"]=ml_pred
            poisson_pred=PoissonEngine.calculate_match_probabilities(home,away,de.football_data.get("all"))
            if poisson_pred: stats["poisson_prediction"]=poisson_pred

        elif sport_key=="basketball":
            hs=de.get_nba_stats(home); aws=de.get_nba_stats(away)
            nb_matchup=de.get_nba_matchup(home,away)
            if nb_matchup: stats["nba_matchup"]=nb_matchup
            us={"home":hs,"away":aws}
            if hs or aws: stats["us_sports"]=us
            if ml.is_nba_trained and hs and aws:
                ml_pred=ml.predict_nba(hs,aws)
                if ml_pred: stats["ml_prediction"]=ml_pred

        elif sport_key in ["baseball","hockey","cricket"]:
            hs=de.get_us_sports_stats(sport,home); aws=de.get_us_sports_stats(sport,away)
            if hs or aws: stats["us_sports"]={"home":hs,"away":aws}

        math_score=ConfidenceEngine.calculate_math_score(opp,stats,opp["market"],ml_pred,poisson_pred)
        if math_score<CFG.MIN_MATH_SCORE_TO_CALL_AI:
            skip_math+=1
            logger.info("⏭️ SKIP(math:%d<%d) %s vs %s EV=%.2f%%",math_score,CFG.MIN_MATH_SCORE_TO_CALL_AI,home,away,opp["edge_pct"])
            continue

        ai=generate_ai_decision(home,away,sport,sport_key,opp,stats,math_score,ml_pred,poisson_pred)
        fc=ai["final_confidence"]
        if ai.get("decision")=="SKIP":
            skip_ai+=1
            logger.info("⏭️ AI SKIP: %s vs %s Math:%d AI:%d Final:%d Flags:%s",home,away,math_score,ai["ai_confidence"],fc,ai.get("red_flags",[]))
            continue
        if fc<CFG.MIN_CONFIDENCE_TO_SEND:
            skip_conf+=1; logger.info("⏭️ SKIP(conf:%d<%d) %s vs %s",fc,CFG.MIN_CONFIDENCE_TO_SEND,home,away); continue

        logger.info("✅ APPROVED %s vs %s | %s | Math:%d AI:%d Final:%d EV=%.2f%%",
                    home,away,ai["decision"],math_score,ai["ai_confidence"],fc,opp["edge_pct"])
        msg=build_signal_message(home,away,sport,sport_key,opp,ai,stats,math_score,ml_pred,poisson_pred,now,event.get("commence_time",""))
        if send_telegram(msg):
            sent.mark_sent(home,away,opp["pick"],opp["market"])
            performance_tracker.record_signal(home,away,opp["pick"],opp["market"],opp["odds"],opp["ev"],fc,opp["prob"],sport_key,event.get("sport_key",""))
            total_sent+=1
            logger.info("📤 SENT: %s vs %s EV=%.2f%% Conf=%d%%",home,away,opp["edge_pct"],fc)
        else: logger.error("❌ Telegram failed: %s vs %s",home,away)
        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    logger.info("="*65)
    logger.info("📊 SUMMARY | Analyzed:%d Sent:%d Skip(math):%d Skip(AI):%d Skip(conf):%d",
                total_analyzed,total_sent,skip_math,skip_ai,skip_conf)
    if total_sent==0 and total_analyzed>0:
        logger.info("ℹ️  %d analyzed → filtered. Math:%d AI:%d Conf:%d",total_analyzed,skip_math,skip_ai,skip_conf)
    logger.info("📊 %s",odds_key_manager.get_usage_summary())
    perf=performance_tracker.data.get("summary",{})
    if perf.get("resolved",0)>0:
        logger.info("📈 WR=%.1f%% ROI=%.1f%% Signals=%d",perf["win_rate"]*100,perf["roi_pct"],perf["total_signals"])
    logger.info("="*65)

if __name__=="__main__":
    try: asyncio.run(async_main())
    except KeyboardInterrupt: logger.info("Stopped.")
    except Exception as e: logger.critical("SYSTEM FAILURE: %s",e,exc_info=True); sys.exit(1)
