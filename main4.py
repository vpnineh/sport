# =========================================================
# ZBET90 ENGINE v9.1 | Production Grade | Refactored
# =========================================================
import os, sys, time, json, re, random, logging, html as html_lib
import hashlib, asyncio, aiohttp, requests, numpy as np, pandas as pd
import pickle, warnings, threading, difflib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict, deque
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import scipy.stats as stats_scipy
from scipy.optimize import brentq

# =========================================================
# 1. CONFIG
# =========================================================
@dataclass
class Config:
    CACHE_DIR:        Path = Path("api_cache")
    LOG_DIR:          Path = Path("log")
    HISTORICAL_DIR:   Path = Path("api_cache/historical")
    ML_DIR:           Path = Path("api_cache/ml_models")
    HISTORY_FILE:     Path = Path("api_cache/sent_history.json")
    ODDS_CACHE_FILE:  Path = Path("api_cache/odds_cache.json")
    PERFORMANCE_FILE: Path = Path("api_cache/performance_tracker.json")
    LOG_FILE:         Path = Path("api_cache/execution_logs.log")

    MATCH_WINDOW_HOURS:   float = 10.0
    ODDS_API_MARKETS:     List[str] = field(default_factory=lambda: ["h2h", "totals"])
    ODDS_API_REGIONS:     str = "eu,us,uk,au"
    TTL_ODDS_CACHE_MINUTES: float = 10.0
    TTL_SENT_HISTORY:     float = 48.0
    TTL_GITHUB_DATA:      float = 12.0

    H2H_MIN_ODDS:    float = 1.30;  H2H_MIN_EV:     float = 0.010
    TOTALS_MIN_ODDS: float = 1.40;  TOTALS_MIN_EV:  float = 0.012
    MAX_REALISTIC_EV:       float = 0.18
    MATH_MIN_EV_TO_ANALYZE: float = 0.008
    MAX_VALID_IMPLIED_SUM:  float = 1.20
    MIN_VALID_IMPLIED_SUM:  float = 0.65

    KELLY_FRACTION: float = 0.25
    MAX_KELLY_PCT:  float = 5.0

    MIN_MATH_SCORE_TO_CALL_AI: int = 32
    MIN_CONFIDENCE_TO_SEND:    int = 60
    HIGH_CONFIDENCE:           int = 75

    AI_WEIGHT:    float = 0.50;  MATH_WEIGHT:  float = 0.50
    MAX_AI_BOOST: int   = 18;    MAX_AI_PENALTY: int = 12
    AI_MODEL:         str   = "gemini-2.0-flash"
    AI_MAX_TOKENS:    int   = 2000
    AI_TEMPERATURE:   float = 0.05

    SHARP_BOOKMAKERS: List[str] = field(default_factory=lambda: [
        "pinnacle","betfair_ex_eu","matchbook","betfair_ex_uk",
        "sport888","betsson","nordicbet","unibet_eu"])

    API_FOOTBALL_KEY:        str   = ""
    API_FOOTBALL_TTL:        float = 6.0
    API_FOOTBALL_MAX_CALLS:  int   = 95
    FOOTBALL_DATA_ORG_KEY:   str   = ""
    FOOTBALL_DATA_ORG_TTL:   float = 6.0
    TSDB_API_KEY:            str   = "123"

    MARKET_EXPECTED_OUTCOMES: Dict = field(default_factory=lambda: {
        "h2h":    {"min": 2, "max": 3},
        "totals": {"min": 2, "max": 2}})

    TELEGRAM_ID:            str   = "@zBET90"
    TELEGRAM_SLEEP_BETWEEN: float = 3.0

    GITHUB_SOURCES: Dict = field(default_factory=lambda: {
        "atp":          "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
        "wta":          "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
        "atp_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv",
        "wta_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv",
        "football_eu":  "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"})

    FOOTBALL_DATA_UK_LEAGUES: Dict = field(default_factory=lambda: {
        "E0":"Premier League","E1":"Championship","D1":"Bundesliga",
        "SP1":"La Liga","I1":"Serie A","F1":"Ligue 1",
        "N1":"Eredivisie","P1":"Liga Portugal"})

    FOOTBALL_DATA_UK_SEASONS: List[str] = field(default_factory=lambda: ["2223","2324","2425"])


CFG = Config()
CFG.API_FOOTBALL_KEY      = os.getenv("API_FOOTBALL","").strip()
CFG.FOOTBALL_DATA_ORG_KEY = os.getenv("FOOTBALL_DATA_ORG_KEY","").strip()

# =========================================================
# 2. LOGGING
# =========================================================
DEBUG_MODE = os.getenv("DEBUG_MODE","false").lower() == "true"
for _d in [CFG.CACHE_DIR, CFG.LOG_DIR, CFG.HISTORICAL_DIR, CFG.ML_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ZBET90_v91")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
for _h in [logging.StreamHandler(sys.stdout),
           logging.FileHandler(CFG.LOG_FILE, mode="a", encoding="utf-8")]:
    _h.setFormatter(_fmt); logger.addHandler(_h)

# =========================================================
# 3. SHARED CACHE UTILITIES
# =========================================================
_cache_lock = threading.Lock()

class Cache:
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
            tmp = fp.with_name(f"{fp.name}.tmp.{os.getpid()}_{int(time.time()*1000)}")
            content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            with _cache_lock:
                tmp.write_text(content, encoding="utf-8")
                try:    tmp.replace(fp)
                except PermissionError:
                    if fp.exists(): fp.unlink()
                    tmp.rename(fp)
        except Exception as e:
            logger.debug("[CACHE] Save error: %s", e)

    @staticmethod
    def valid(cache: dict, key: str, ttl_h: float) -> bool:
        e = cache.get(key)
        if not isinstance(e, dict) or "ts" not in e: return False
        try:
            t = datetime.fromisoformat(e["ts"])
            if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - t < timedelta(hours=ttl_h)
        except Exception: return False

    @staticmethod
    def set(cache: dict, key: str, val: Any, ts_field="ts") -> dict:
        cache[key] = {"ts": datetime.now(timezone.utc).isoformat(), "data": val}
        return cache

    @staticmethod
    def get(cache: dict, key: str) -> Any:
        return cache.get(key, {}).get("data")


class _BaseAPIClient:
    """Mixin providing throttled GET with file-based cache."""
    _cache: dict; _cache_file: Path; _last_call: float; _min_interval: float

    def _load_cache(self, ttl_h=6.0):
        self._cache = {}
        try:
            if self._cache_file.exists():
                raw = json.loads(self._cache_file.read_text())
                now = datetime.now(timezone.utc)
                for k, v in raw.items():
                    if isinstance(v, dict) and "ts" in v:
                        t = datetime.fromisoformat(v["ts"])
                        if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
                        if (now - t) < timedelta(hours=ttl_h):
                            self._cache[k] = v
        except Exception: pass

    def _save_cache(self):
        try:
            self._cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, default=str))
        except Exception: pass

    def _cache_key(self, endpoint, params):
        return hashlib.md5(
            f"{endpoint}|{json.dumps(params or {}, sort_keys=True)}".encode()
        ).hexdigest()

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _fetch(self, url, params, headers, ttl_h, timeout=15) -> Optional[dict]:
        ck = self._cache_key(url, params)
        if Cache.valid(self._cache, ck, ttl_h):
            return Cache.get(self._cache, ck)
        self._throttle()
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            self._last_call = time.time()
            if r.status_code == 200:
                data = r.json()
                self._cache[ck] = {"ts": datetime.now(timezone.utc).isoformat(), "data": data}
                self._save_cache()
                return data
            elif r.status_code == 429:
                logger.warning("[HTTP] Rate limited: %s", url)
                time.sleep(60)
            else:
                logger.debug("[HTTP] %d: %s", r.status_code, url[:80])
        except Exception as e:
            logger.debug("[HTTP] %s: %s", url[:60], str(e)[:60])
        return None


# =========================================================
# 4. THESPORTSDB CLIENT
# =========================================================
class TheSportsDBClient(_BaseAPIClient):
    BASE = "https://www.thesportsdb.com/api/v1/json/{key}"

    def __init__(self):
        self._key        = os.getenv("TSDB_API_KEY", CFG.TSDB_API_KEY)
        self._cache_file = CFG.CACHE_DIR / "tsdb_cache.json"
        self._last_call  = 0.0
        self._min_interval = 0.5
        self._load_cache(ttl_h=6.0)

    def _get(self, ep, params=None, ttl_h=6.0):
        url = f"{self.BASE.format(key=self._key)}/{ep}"
        return self._fetch(url, params, {"User-Agent":"ZBET90/9.1"}, ttl_h)

    def search_team(self, name) -> Optional[dict]:
        d = self._get("searchteams.php", {"t": name})
        return d["teams"][0] if d and d.get("teams") else None

    def search_player(self, name) -> Optional[dict]:
        d = self._get("searchplayers.php", {"p": name})
        return d["player"][0] if d and d.get("player") else None

    def get_team_last_events(self, tid, n=10) -> List[dict]:
        d = self._get("eventslast.php", {"id": tid})
        return (d.get("results") or [])[:n] if d else []

    def get_team_next_events(self, tid, n=5) -> List[dict]:
        d = self._get("eventsnext.php", {"id": tid})
        return (d.get("events") or [])[:n] if d else []

    def get_team_stats(self, name: str) -> dict:
        team = self.search_team(name)
        if not team: return {}
        tid = team.get("idTeam")
        if not tid: return {}
        result = {
            "team_id":   tid,
            "team_name": team.get("strTeam", name),
            "country":   team.get("strCountry",""),
            "league":    team.get("strLeague",""),
        }
        events = self.get_team_last_events(tid, 15)
        if not events: return result
        wins = draws = losses = gs = gc = 0
        form = []
        for ev in events:
            try:
                hs = int(float(ev.get("intHomeScore",0) or 0))
                as_ = int(float(ev.get("intAwayScore",0) or 0))
            except (ValueError, TypeError): continue
            is_home = _fuzzy_match(name, ev.get("strHomeTeam",""))
            scored, conceded = (hs, as_) if is_home else (as_, hs)
            gs += scored; gc += conceded
            if scored > conceded:   wins += 1;  form.append("W")
            elif scored == conceded: draws += 1; form.append("D")
            else:                   losses += 1; form.append("L")
        total = wins + draws + losses
        if total:
            result.update({
                "form":              "".join(form[:10]),
                "win_rate":          round(wins/total, 3),
                "draw_rate":         round(draws/total, 3),
                "avg_scored":        round(gs/total, 2),
                "avg_conceded":      round(gc/total, 2),
                "matches_analyzed":  total,
                "data_quality":      "good" if total>=10 else "limited" if total>=5 else "poor"
            })
        return result

    def get_league_table(self, league_id: int, season: str = None) -> List[dict]:
        """Free endpoint — league standings table by league ID."""
        if not season:
            y = datetime.now().year
            season = f"{y}-{y+1}"
        d = self._get("lookuptable.php", {"l": str(league_id), "s": season}, ttl_h=4.0)
        if d and d.get("table"):
            return d["table"]
        return []

    def get_events_on_date(self, date_str: str, sport: str = None) -> List[dict]:
        """Get all events on a given date (YYYY-MM-DD), optionally filtered by sport."""
        params = {"d": date_str}
        if sport:
            params["s"] = sport
        d = self._get("eventsday.php", params, ttl_h=1.0)
        return d.get("events") or [] if d else []

    def get_league_events_on_date(self, league_id: int, date_str: str) -> List[dict]:
        """Events for a specific league on a date."""
        d = self._get("eventsday.php", {"d": date_str, "l": str(league_id)}, ttl_h=1.0)
        return d.get("events") or [] if d else []

    def get_team_standing(self, team_name: str, league_id: int) -> dict:
        """Lookup team rank in league table (free)."""
        table = self.get_league_table(league_id)
        for row in table:
            if _fuzzy_match(team_name, str(row.get("strTeam",""))):
                return {
                    "rank":          int(row.get("intRank", 0) or 0),
                    "played":        int(row.get("intPlayed", 0) or 0),
                    "wins":          int(row.get("intWin", 0) or 0),
                    "draws":         int(row.get("intDraw", 0) or 0),
                    "losses":        int(row.get("intLoss", 0) or 0),
                    "points":        int(row.get("intPoints", 0) or 0),
                    "goals_for":     int(row.get("intGoalsFor", 0) or 0),
                    "goals_against": int(row.get("intGoalsAgainst", 0) or 0),
                    "total_teams":   len(table),
                    "source":        "tsdb_table",
                }
        return {}


# =========================================================
# 4b. OPENLIGADB CLIENT — Free, No Auth, 1000 req/h
# Covers: Bundesliga, 2.Bundesliga, Champions League,
#         DFB Pokal, World Cup, many European leagues
# Docs: https://api.openligadb.de
# =========================================================
class OpenLigaDBClient(_BaseAPIClient):
    BASE = "https://api.openligadb.de"

    # league shortcut → human name (all supported by OpenLigaDB)
    LEAGUES = {
        "bl1":  "Bundesliga",
        "bl2":  "2. Bundesliga",
        "bl3":  "3. Liga",
        "ucl":  "Champions League",
        "uel":  "Europa League",
        "uecl": "Conference League",
        "dfb":  "DFB Pokal",
        "gb1":  "Premier League",
        "es1":  "La Liga",
        "it1":  "Serie A",
        "fr1":  "Ligue 1",
        "nl1":  "Eredivisie",
        "pt1":  "Primeira Liga",
        "wm26": "World Cup 2026",
    }

    def __init__(self):
        self._cache_file   = CFG.CACHE_DIR / "openligadb_cache.json"
        self._last_call    = 0.0
        self._min_interval = 0.3   # polite — 1000 req/h limit
        self._load_cache(ttl_h=4.0)
        logger.info("✅ [OPENLIGADB] Client ready (no auth, 1000 req/h)")

    def _get(self, path, ttl_h=4.0) -> Any:
        url = f"{self.BASE}/{path.lstrip('/')}"
        return self._fetch(url, None, {"Accept": "application/json"}, ttl_h)

    def get_table(self, league: str, season: int = None) -> List[dict]:
        """League standings. season=2024 means 2024/25."""
        s = season or datetime.now().year - (1 if datetime.now().month < 7 else 0)
        data = self._get(f"getbltable/{league}/{s}", ttl_h=2.0)
        if not isinstance(data, list): return []
        result = []
        for row in data:
            gp = max((row.get("wins",0) or 0) + (row.get("draws",0) or 0) + (row.get("losses",0) or 0), 1)
            gf = row.get("goals",0) or 0; ga = row.get("opponentGoals",0) or 0
            result.append({
                "rank":          row.get("rank", 0),
                "team":          row.get("teamName",""),
                "short_name":    row.get("shortName",""),
                "points":        row.get("points",0),
                "played":        gp,
                "wins":          row.get("wins",0) or 0,
                "draws":         row.get("draws",0) or 0,
                "losses":        row.get("losses",0) or 0,
                "goals_for":     gf,
                "goals_against": ga,
                "goal_diff":     gf - ga,
                "win_pct":       round((row.get("wins",0) or 0) / gp, 3),
                "avg_scored":    round(gf / gp, 2),
                "avg_conceded":  round(ga / gp, 2),
            })
        return result

    def get_matches(self, league: str, season: int = None) -> List[dict]:
        """All matches for a league/season."""
        s = season or datetime.now().year - (1 if datetime.now().month < 7 else 0)
        data = self._get(f"getmatchdata/{league}/{s}", ttl_h=1.0)
        return data if isinstance(data, list) else []

    def get_team_stats(self, team_name: str, league: str = None, season: int = None) -> dict:
        """
        Derive team stats from OpenLigaDB table + recent matches.
        Tries all known leagues if none specified.
        """
        leagues_to_try = [league] if league else list(self.LEAGUES.keys())
        s = season or datetime.now().year - (1 if datetime.now().month < 7 else 0)

        for lg in leagues_to_try[:6]:   # cap at 6 attempts
            table = self.get_table(lg, s)
            if not table: continue
            for row in table:
                if _fuzzy_match(team_name, row["team"]) or _fuzzy_match(team_name, row.get("short_name","")):
                    # Found — now enrich with recent match form
                    result = {**row, "league": self.LEAGUES.get(lg, lg),
                              "league_shortcut": lg, "source": "openligadb",
                              "data_quality": "good" if row["played"] >= 10 else "limited"}
                    form = self._get_recent_form(team_name, lg, s)
                    if form: result.update(form)
                    return result
        return {}

    def _get_recent_form(self, team_name: str, league: str, season: int) -> dict:
        """Last 10 match form string from finished matches."""
        matches = self.get_matches(league, season)
        if not matches: return {}
        finished = [m for m in matches
                    if m.get("matchIsFinished") and
                    m.get("matchResults")]
        # Sort by date descending
        finished.sort(key=lambda x: x.get("matchDateTime",""), reverse=True)
        form = []; w=d=l=gs=gc=0
        for m in finished[:15]:
            t1 = m.get("team1",{}).get("teamName","")
            t2 = m.get("team2",{}).get("teamName","")
            res = m.get("matchResults",[])
            # Final result is resultOrderID == 2 (fulltime), fallback to last
            ft = next((r for r in res if r.get("resultTypeID")==2),
                      res[-1] if res else None)
            if not ft: continue
            g1 = ft.get("pointsTeam1",0) or 0; g2 = ft.get("pointsTeam2",0) or 0
            is_home = _fuzzy_match(team_name, t1)
            is_away = _fuzzy_match(team_name, t2)
            if not is_home and not is_away: continue
            scored   = g1 if is_home else g2
            conceded = g2 if is_home else g1
            gs += scored; gc += conceded
            if scored > conceded:   w+=1; form.append("W")
            elif scored == conceded: d+=1; form.append("D")
            else:                   l+=1; form.append("L")
        n = len(form)
        if n == 0: return {}
        return {
            "form":         "".join(form),
            "win_rate":     round(w/n, 3),
            "draw_rate":    round(d/n, 3),
            "avg_scored":   round(gs/n, 2),
            "avg_conceded": round(gc/n, 2),
            "over25_rate":  round(sum(1 for i in range(min(n,10)) if
                                      (form[i]!="") and False) / max(n,1), 3),  # placeholder
        }

    def get_h2h(self, team1: str, team2: str, league: str = None, season: int = None) -> dict:
        """H2H stats between two teams from OpenLigaDB matches."""
        leagues_to_try = [league] if league else ["bl1","bl2","ucl","gb1","es1","it1","fr1"]
        s = season or datetime.now().year - (1 if datetime.now().month < 7 else 0)
        t1w = t2w = draws = 0; total_goals = []

        for lg in leagues_to_try[:4]:
            matches = self.get_matches(lg, s)
            for m in matches:
                if not m.get("matchIsFinished"): continue
                ht = m.get("team1",{}).get("teamName","")
                at = m.get("team2",{}).get("teamName","")
                if not ((_fuzzy_match(team1,ht) and _fuzzy_match(team2,at)) or
                        (_fuzzy_match(team2,ht) and _fuzzy_match(team1,at))):
                    continue
                res = m.get("matchResults",[])
                ft  = next((r for r in res if r.get("resultTypeID")==2), res[-1] if res else None)
                if not ft: continue
                g1 = ft.get("pointsTeam1",0) or 0; g2 = ft.get("pointsTeam2",0) or 0
                total_goals.append(g1+g2)
                t1_is_home = _fuzzy_match(team1, ht)
                ts = g1 if t1_is_home else g2; os_ = g2 if t1_is_home else g1
                if ts > os_: t1w += 1
                elif ts < os_: t2w += 1
                else: draws += 1

        n = t1w + t2w + draws
        if n == 0: return {}
        return {
            "total":          n,
            f"{team1}_wins":  t1w,
            f"{team2}_wins":  t2w,
            "draws":          draws,
            "avg_goals":      round(sum(total_goals)/n, 2),
            "over25_rate":    round(sum(1 for g in total_goals if g > 2.5)/n, 3),
            "btts_rate":      round(sum(1 for m in total_goals if m > 0)/n, 3),  # approx
            "dominance":      f"{team1}_dominant" if t1w > t2w*1.5 else
                              f"{team2}_dominant" if t2w > t1w*1.5 else "balanced",
            "source":         "openligadb",
        }


# =========================================================
# 5. API-FOOTBALL CLIENT
# =========================================================
class APIFootballClient(_BaseAPIClient):
    BASE  = "https://v3.football.api-sports.io"
    H_KEY = "x-apisports-key"

    LEAGUE_MAP = {
        "Premier League":39,"Championship":40,"La Liga":140,"Bundesliga":78,
        "Serie A":135,"Ligue 1":61,"Eredivisie":88,"Liga Portugal":94,
        "Champions League":2,"Europa League":3,"MLS":253}

    def __init__(self):
        self._key           = CFG.API_FOOTBALL_KEY
        self._cache_file    = CFG.CACHE_DIR / "apifootball_cache.json"
        self._usage_file    = CFG.CACHE_DIR / "apifootball_usage.json"
        self._last_call     = 0.0
        self._min_interval  = 1.0
        self._calls_today   = 0
        self._load_cache(ttl_h=CFG.API_FOOTBALL_TTL)
        self._load_usage()
        if self._key:
            logger.info("✅ [API-FOOTBALL] Key loaded (calls today: %d/%d)",
                        self._calls_today, CFG.API_FOOTBALL_MAX_CALLS)
        else:
            logger.warning("⚠️  [API-FOOTBALL] No key — disabled")

    def _load_usage(self):
        try:
            u = json.loads(self._usage_file.read_text()) if self._usage_file.exists() else {}
            if u.get("date") == datetime.now(timezone.utc).strftime("%Y-%m-%d"):
                self._calls_today = u.get("calls", 0); return
        except Exception: pass
        self._calls_today = 0

    def _save_usage(self):
        try:
            self._usage_file.write_text(json.dumps({
                "date":  datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "calls": self._calls_today}))
        except Exception: pass

    def _get(self, ep, params=None, ttl_h=None) -> Optional[dict]:
        if not self._key: return None
        if self._calls_today >= CFG.API_FOOTBALL_MAX_CALLS:
            logger.warning("[API-FOOTBALL] Daily limit reached (%d)", self._calls_today)
            return None
        ttl = ttl_h or CFG.API_FOOTBALL_TTL
        ck  = self._cache_key(ep, params)
        if Cache.valid(self._cache, ck, ttl):
            return Cache.get(self._cache, ck)
        self._throttle()
        try:
            r = requests.get(f"{self.BASE}/{ep}", params=params,
                             headers={self.H_KEY: self._key, "Accept":"application/json"},
                             timeout=15)
            self._calls_today += 1; self._save_usage()
            rem = r.headers.get("x-ratelimit-requests-remaining","?")
            logger.debug("[API-FOOTBALL] %s → HTTP %d (remaining: %s)", ep, r.status_code, rem)
            if r.status_code == 200:
                data = r.json()
                if data.get("errors") and data["errors"] != []:
                    logger.warning("[API-FOOTBALL] Error: %s", data["errors"]); return None
                self._cache[ck] = {"ts": datetime.now(timezone.utc).isoformat(), "data": data}
                self._save_cache()
                return data
        except Exception as e:
            logger.debug("[API-FOOTBALL] %s: %s", ep, str(e)[:60])
        return None

    def get_team_id(self, name, league_id=None) -> Optional[int]:
        params = {"search": name}
        if league_id: params["league"] = league_id
        d = self._get("teams", params, ttl_h=24.0)
        if d and d.get("response"): return d["response"][0]["team"]["id"]
        return None

    def get_team_stats(self, name: str, league_id=None, season=None) -> dict:
        if not self._key: return {}
        season = season or datetime.now().year
        tid = self.get_team_id(name, league_id)
        if not tid: return {}
        if not league_id:
            ld = self._get("teams/leagues", {"team": tid}, ttl_h=24.0)
            if ld and ld.get("response"):
                for lg in ld["response"]:
                    if lg.get("league",{}).get("type") == "League":
                        league_id = lg["league"]["id"]
                        season = (lg.get("seasons",[{}])[-1]).get("year", season)
                        break
        if not league_id: return {}
        d = self._get("teams/statistics", {"team":tid,"league":league_id,"season":season})
        if not d or not d.get("response"): return {}
        resp = d["response"]
        fix = resp.get("fixtures",{}); goals = resp.get("goals",{})
        pl = fix.get("played",{}).get("total",0) or 0
        w  = fix.get("wins",{}).get("total",0) or 0
        dr = fix.get("draws",{}).get("total",0) or 0
        gf = goals.get("for",{}).get("total",{}).get("total",0) or 0
        ga = goals.get("against",{}).get("total",{}).get("total",0) or 0
        sp = max(pl,1)
        form_str = resp.get("form","") or ""
        cs  = resp.get("clean_sheet",{}).get("total",0) or 0
        return {
            "team_id":            tid, "league_id": league_id, "season": season,
            "played":             pl, "wins": w, "draws": dr, "losses": pl-w-dr,
            "win_rate":           round(w/sp,3), "draw_rate": round(dr/sp,3),
            "avg_scored":         round(gf/sp,2), "avg_conceded": round(ga/sp,2),
            "clean_sheet_rate":   round(cs/sp,3),
            "form":               form_str[-10:] or "N/A",
            "recent_form_5":      form_str[-5:]  or "N/A",
            "data_quality":       "good" if pl>=15 else "limited" if pl>=8 else "poor",
            "source":             "api_football"}

    def get_h2h(self, t1: str, t2: str, last_n=10) -> dict:
        if not self._key: return {}
        id1 = self.get_team_id(t1); id2 = self.get_team_id(t2)
        if not id1 or not id2: return {}
        d = self._get("fixtures/headtohead", {"h2h":f"{id1}-{id2}","last":last_n}, ttl_h=12.0)
        if not d or not d.get("response"): return {}
        matches = d["response"]
        w1=w2=draws=0; total_goals=[]
        for m in matches:
            teams = m.get("teams",{}); g = m.get("goals",{})
            hg = g.get("home",0) or 0; ag = g.get("away",0) or 0
            total_goals.append(hg+ag)
            if   teams.get("home",{}).get("winner"): (w1 if teams["home"]["id"]==id1 else w2).__class__  # trick
            w1 += 1 if teams.get("home",{}).get("winner") and teams["home"]["id"]==id1 else 0
            w2 += 1 if teams.get("away",{}).get("winner") and teams["away"]["id"]==id2 else 0
            draws += 1 if not teams.get("home",{}).get("winner") and not teams.get("away",{}).get("winner") else 0
        n = len(matches)
        return {
            "total": n, f"{t1}_wins": w1, f"{t2}_wins": w2, "draws": draws,
            "avg_goals":    round(sum(total_goals)/max(n,1),2),
            "over25_rate":  round(sum(1 for g in total_goals if g>2.5)/max(n,1),3),
            "dominance":    f"{t1}_dominant" if w1>w2*1.5 else f"{t2}_dominant" if w2>w1*1.5 else "balanced",
            "source":       "api_football"}

    def get_team_standing(self, name: str, league_id: int) -> dict:
        standings = self._get("standings", {"league":league_id,"season":datetime.now().year}, ttl_h=4.0)
        if not standings or not standings.get("response"): return {}
        try:
            table = standings["response"][0]["league"]["standings"][0]
            for t in table:
                if _fuzzy_match(name, t.get("team",{}).get("name","")):
                    return {
                        "rank":   t.get("rank",0),
                        "points": t.get("points",0),
                        "played": t.get("all",{}).get("played",0),
                        "form":   t.get("form",""),
                        "total_teams": len(table)}
        except (KeyError, IndexError): pass
        return {}


# =========================================================
# 6. FOOTBALL-DATA.ORG CLIENT
# =========================================================
class FootballDataOrgClient(_BaseAPIClient):
    BASE = "https://api.football-data.org/v4"
    COMP = {"Premier League":"PL","Championship":"ELC","La Liga":"PD",
            "Bundesliga":"BL1","Serie A":"SA","Ligue 1":"FL1",
            "Eredivisie":"DED","Champions League":"CL"}

    def __init__(self):
        self._key          = CFG.FOOTBALL_DATA_ORG_KEY
        self._cache_file   = CFG.CACHE_DIR / "football_data_org_cache.json"
        self._last_call    = 0.0
        self._min_interval = 6.0
        self._load_cache(ttl_h=CFG.FOOTBALL_DATA_ORG_TTL)
        if self._key: logger.info("✅ [FOOTBALL-DATA.ORG] Key loaded")
        else:         logger.info("ℹ️  [FOOTBALL-DATA.ORG] No key — limited")

    def _get(self, ep, params=None, ttl_h=6.0):
        hdrs = {"Accept":"application/json"}
        if self._key: hdrs["X-Auth-Token"] = self._key
        return self._fetch(f"{self.BASE}/{ep}", params, hdrs, ttl_h)

    def get_team_matches(self, name: str, last_n=10) -> dict:
        """
        Get recent finished matches for a team.
        Loops through supported competitions to find the team,
        then fetches its last N finished matches directly.
        """
        # Find team ID by scanning competitions
        team_id = self._find_team_id(name)
        if not team_id: return {}

        md = self._get(f"teams/{team_id}/matches",
                       {"status":"FINISHED","limit":last_n}, ttl_h=4.0)
        if not md or not md.get("matches"): return {}
        w=dr=l=gf=ga=0; form=[]
        for m in md["matches"]:
            sc  = m.get("score",{}).get("fullTime",{})
            hs  = sc.get("home",0) or 0; as_ = sc.get("away",0) or 0
            is_home = (m.get("homeTeam",{}).get("id") == team_id)
            ts,os_ = (hs,as_) if is_home else (as_,hs)
            gf+=ts; ga+=os_
            if ts>os_:  w+=1; form.append("W")
            elif ts==os_: dr+=1; form.append("D")
            else:         l+=1; form.append("L")
        n=len(md["matches"]); sp=max(n,1)
        return {"played":n,"win_rate":round(w/sp,3),"draw_rate":round(dr/sp,3),
                "avg_scored":round(gf/sp,2),"avg_conceded":round(ga/sp,2),
                "form":"".join(reversed(form)),
                "data_quality":"good" if n>=8 else "limited" if n>=4 else "poor",
                "source":"football_data_org"}

    def _find_team_id(self, name: str) -> Optional[int]:
        """Find team ID by scanning all supported competitions."""
        for code in self.COMP.values():
            d = self._get(f"competitions/{code}/teams", {}, ttl_h=24.0)
            if not d or not d.get("teams"): continue
            for t in d["teams"]:
                if _fuzzy_match(name, t.get("name","")) or _fuzzy_match(name, t.get("shortName","")):
                    return t.get("id")
        return None

    def get_standings(self, competition_code: str, season: int = None) -> List[dict]:
        """League standings from football-data.org."""
        params = {}
        if season: params["season"] = season
        d = self._get(f"competitions/{competition_code}/standings", params, ttl_h=4.0)
        if not d: return []
        try:
            table = d["standings"][0]["table"]
            return [{"rank":t["position"],"team":t["team"]["name"],
                     "points":t["points"],"played":t["playedGames"],
                     "wins":t["won"],"draws":t["draw"],"losses":t["lost"],
                     "gf":t["goalsFor"],"ga":t["goalsAgainst"],
                     "gd":t["goalDifference"],"form":t.get("form","")} for t in table]
        except (KeyError,IndexError): return []


    def get_player_stats(self, name: str) -> dict:
        p = self.search_player(name)
        if not p: return {}
        return {
            "player_id":   p.get("idPlayer"),
            "player_name": p.get("strPlayer", name),
            "nationality": p.get("strNationality",""),
            "sport":       p.get("strSport",""),
            "position":    p.get("strPosition",""),
        }

# Singletons
tsdb         = TheSportsDBClient()
api_football = APIFootballClient()
fdo          = FootballDataOrgClient()
openligadb   = OpenLigaDBClient()


# =========================================================
# 8. NAME MATCHING HELPERS
# =========================================================
def _fuzzy_match(a: str, b: str, thresh=0.45) -> bool:
    if not a or not b: return False
    na, nb = _norm(a), _norm(b)
    if na==nb or na in nb or nb in na: return True
    t1,t2 = set(na.split()), set(nb.split())
    if t1 and t2 and len(t1&t2)/max(len(t1),len(t2)) >= thresh: return True
    return False

def _norm(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", str(name).lower().strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    for w in ["fc","cf","sc","ac","bk","fk","if","the","united","city","real","atletico"]:
        s = re.sub(rf"\b{w}\b","",s)
    return re.sub(r"\s+"," ",s).strip()


# =========================================================
# 9. AI MANAGER
# =========================================================
import google.genai as genai
from google.genai import types
try:   from groq import Groq; HAS_GROQ=True
except ImportError: HAS_GROQ=False

AI_SYSTEM = """You are an elite sports betting quantitative analyst. Find GENUINE EDGE only.

DECISION FRAMEWORK:
- BET: EV>2% AND (sharp_line=True OR ML_confidence>0.62) AND Kelly>1%
- BET: EV>3.5% even with limited data if no major red flags
- SKIP: EV<1.5% | Kelly<0.8% | Models disagree>20%

CONFIDENCE CALIBRATION: 85-100: All signals agree + EV>5% | 75-84: Sharp+EV>3% | 65-74: Clear edge+EV>2% | <55: Hard skip

Output ONLY valid JSON:
{"decision":"BET" or "SKIP","confidence":<0-100>,"sport_emoji":"<emoji>","risk_level":"Low" or "Medium" or "High","key_factors":["f1","f2","f3"],"logic":"2-3 sentences","red_flags":["r1"]}"""

def _extract_json(raw: str) -> Optional[dict]:
    if not raw: return None
    clean = re.sub(r"<think>[\s\S]*?</think>","",raw,flags=re.IGNORECASE)
    clean = re.sub(r"```(?:json)?","",clean).strip().rstrip("`").strip()
    try: return json.loads(clean)
    except Exception: pass
    for m in reversed(list(re.finditer(r"\{[^{}]*\}",clean))):
        try:
            r = json.loads(m.group(0))
            if isinstance(r,dict) and r: return r
        except Exception: continue
    return None

class AIManager:
    _instance=None; _lock=threading.Lock()
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_done = False
            return cls._instance

    def __init__(self):
        if self._init_done: return
        keys = [k.strip() for k in [os.getenv(e,"") for e in ["GEMINI","GEMINI1","GEMINI2","GEMINI3"]] if k.strip()]
        self.gem_clients = [genai.Client(api_key=k) for k in keys]
        self._safety = [types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
                        for c in [types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                  types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                  types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                  types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT]]
        groq_keys = [k.strip() for k in [os.getenv(e,"") for e in ["GROQ_API_KEY","GROQ1","GROQ2"]] if k.strip()]
        self.groq_clients = [Groq(api_key=k) for k in groq_keys] if groq_keys and HAS_GROQ else []
        self._gem_failed: Dict[int,float] = {}
        self._rate_lock = threading.Lock()
        self._last_call = 0.0
        self.last_provider = "none"
        self._init_done = True
        logger.info("✅ [AI] Gemini:%d | Groq:%d", len(self.gem_clients), len(self.groq_clients))

    def generate(self, prompt: str) -> Optional[dict]:
        with self._rate_lock:
            elapsed = time.time() - self._last_call
            if elapsed < 2.0: time.sleep(2.0 - elapsed)
            self._last_call = time.time()
        # Gemini
        if self.gem_clients:
            cfg = types.GenerateContentConfig(
                temperature=CFG.AI_TEMPERATURE, max_output_tokens=CFG.AI_MAX_TOKENS,
                response_mime_type="application/json", safety_settings=self._safety,
                system_instruction=AI_SYSTEM)
            avail = [i for i in range(len(self.gem_clients))
                     if not (self._gem_failed.get(i,0) and time.time()-self._gem_failed[i] < 900)]
            if not avail: self._gem_failed.clear(); avail=list(range(len(self.gem_clients)))
            for _ in range(3):
                if not avail: break
                idx = random.choice(avail)
                try:
                    resp = self.gem_clients[idx].models.generate_content(
                        model=CFG.AI_MODEL, contents=prompt, config=cfg)
                    if resp.text:
                        self.last_provider = "gemini"
                        return json.loads(resp.text) if resp.text.strip().startswith("{") else _extract_json(resp.text)
                except Exception as e:
                    es = str(e)
                    if "429" in es or "quota" in es.lower():
                        self._gem_failed[idx]=time.time(); avail.remove(idx)
                    else:
                        logger.warning("[GEMINI] Key%d: %s", idx, es[:80]); break
        # Groq
        if self.groq_clients:
            for attempt in range(2):
                try:
                    cc = random.choice(self.groq_clients).chat.completions.create(
                        messages=[{"role":"system","content":AI_SYSTEM},{"role":"user","content":prompt}],
                        model="qwen/qwen3-32b", temperature=CFG.AI_TEMPERATURE,
                        max_completion_tokens=CFG.AI_MAX_TOKENS, response_format={"type":"json_object"})
                    raw = cc.choices[0].message.content
                    if raw:
                        self.last_provider = "groq"
                        return json.loads(raw) if raw.strip().startswith("{") else _extract_json(raw)
                except Exception as e:
                    logger.warning("[GROQ] Attempt%d: %s", attempt+1, str(e)[:80])
                    time.sleep(2)
        return None

ai_manager = AIManager()


# =========================================================
# 10. ODDS API KEY MANAGER
# =========================================================
class OddsKeyManager:
    def __init__(self):
        self.keys = []
        for env,label in [("ODDS_API_KEY","primary"),("ODDS_API_KEY2","backup_1"),("ODDS_API_KEY3","backup_2")]:
            k = os.getenv(env,"").strip()
            if k:
                self.keys.append({"key":k,"label":label,"failed":False,
                                   "fail_time":None,"calls":0,"remaining":-1})
                logger.info("🔑 [KEY] %s loaded", label)
        if not self.keys:
            logger.critical("FATAL: No ODDS_API_KEY"); sys.exit(1)

    def mark_failed(self, idx, reason):
        if 0<=idx<len(self.keys):
            self.keys[idx].update({"failed":True,"fail_time":datetime.now(timezone.utc).isoformat()})
            logger.warning("🔑❌ %s FAILED: %s", self.keys[idx]["label"], reason)

    def update_remaining(self, label, remaining: int):
        for k in self.keys:
            if k["label"] == label:
                k["remaining"] = remaining; break

    def get_active(self) -> List[dict]:
        now = datetime.now(timezone.utc)
        active = []
        for k in self.keys:
            if not k["failed"]: active.append(k)
            elif k.get("fail_time"):
                try:
                    ft = datetime.fromisoformat(k["fail_time"])
                    if ft.tzinfo is None: ft = ft.replace(tzinfo=timezone.utc)
                    if now-ft > timedelta(minutes=30):
                        k["failed"]=False; active.append(k)
                except Exception: pass
        if not active:
            for k in self.keys: k["failed"]=False
            active = list(self.keys)
        return active

    def summary(self) -> str:
        parts = []
        for k in self.keys:
            rem = f" rem:{k['remaining']}" if k["remaining"]>=0 else ""
            parts.append(f"{'❌' if k['failed'] else '✅'} {k['label']}({k['calls']}calls{rem})")
        return " | ".join(parts)

GEMINI_API_KEY     = os.getenv("GEMINI","").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID","").strip()
if not all([GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing env vars"); sys.exit(1)

odds_key_mgr = OddsKeyManager()


# =========================================================
# 11. SENT HISTORY
# =========================================================
class SentHistory:
    def __init__(self):
        self.data = Cache.load(CFG.HISTORY_FILE)
        self._cleanup()

    def _cleanup(self):
        now = datetime.now(timezone.utc)
        for k in [k for k,v in self.data.items()
                  if now - datetime.fromisoformat(v.get("sent_at","2000-01-01T00:00:00+00:00")
                  ).replace(tzinfo=timezone.utc) > timedelta(hours=CFG.TTL_SENT_HISTORY)]:
            del self.data[k]

    @staticmethod
    def _key(h,a,m): return hashlib.md5(f"{h.lower()}|{a.lower()}|{m.lower()}".encode()).hexdigest()
    def was_sent(self,h,a,m): return self._key(h,a,m) in self.data
    def mark_sent(self,h,a,pick,m):
        self.data[self._key(h,a,m)] = {
            "match":f"{h} vs {a}","pick":pick,"market":m,
            "sent_at":datetime.now(timezone.utc).isoformat()}
        Cache.save(CFG.HISTORY_FILE, self.data)


# =========================================================
# 12. FREE DATA ENGINE
# =========================================================
class FreeDataEngine:
    def __init__(self):
        self.atp_matches = self.wta_matches = None
        self.atp_rankings = self.wta_rankings = None
        self.football_data: Dict[str,pd.DataFrame] = {}
        self.nba_data = self.nhl_data = self.mlb_data = None
        self.years = [2022,2023,2024,2025,2026]

    def _dl_csv(self, url, path, timeout=30) -> bool:
        if path.exists() and (time.time()-path.stat().st_mtime)/3600 < CFG.TTL_GITHUB_DATA:
            return True
        logger.info("[DATA] Downloading: %s", path.name)
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=timeout+attempt*10, headers={"User-Agent":"Mozilla/5.0"})
                if r.status_code==200 and len(r.content)>100:
                    path.write_bytes(r.content); return True
                break
            except requests.exceptions.Timeout: time.sleep(2*(attempt+1))
            except Exception as e: logger.warning("[DATA] %s: %s", path.name, str(e)[:60]); break
        return False

    def load_tennis_data(self):
        COLS = ["tourney_date","tourney_name","surface","draw_size","tourney_level","round",
                "winner_id","winner_name","winner_rank","winner_rank_points",
                "loser_id","loser_name","loser_rank","loser_rank_points",
                "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_bpSaved","w_bpFaced",
                "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_bpSaved","l_bpFaced",
                "score","best_of","minutes"]
        for tour, attr, key in [("ATP","atp_matches","atp"),("WTA","wta_matches","wta")]:
            dfs=[]
            for yr in self.years:
                p = CFG.HISTORICAL_DIR/f"{key}_{yr}.csv"
                if self._dl_csv(CFG.GITHUB_SOURCES[key].format(year=yr), p):
                    try:
                        df = pd.read_csv(p,low_memory=False,encoding="utf-8",encoding_errors="replace")
                        sub = df[[c for c in COLS if c in df.columns]].copy()
                        if "tourney_date" in sub.columns:
                            sub["tourney_date"] = pd.to_numeric(sub["tourney_date"],errors="coerce")
                        dfs.append(sub)
                    except Exception as e: logger.error("[TENNIS] %s %s: %s",tour,yr,e)
            if dfs:
                combined = pd.concat(dfs,ignore_index=True)
                if "tourney_date" in combined.columns:
                    combined = combined.sort_values("tourney_date").reset_index(drop=True)
                setattr(self,attr,combined)
                logger.info("✅ [TENNIS] %s: %d matches",tour,len(combined))
        for tour,key,attr in [("ATP","atp_rankings","atp_rankings"),("WTA","wta_rankings","wta_rankings")]:
            p = CFG.HISTORICAL_DIR/f"{key}.csv"
            if self._dl_csv(CFG.GITHUB_SOURCES[key],p):
                try: setattr(self,attr,pd.read_csv(p,low_memory=False)); logger.info("✅ [RANKINGS] %s",tour)
                except Exception as e: logger.error("[RANKINGS] %s: %s",tour,e)

    def get_player_ranking(self, name, is_wta=False) -> Optional[int]:
        df = self.wta_rankings if is_wta else self.atp_rankings
        if df is None or df.empty: return None
        nc = next((c for c in ["player","name","player_name"] if c in df.columns), None)
        if not nc: return None
        nl = df[nc].astype(str).str.lower()
        parts = name.split(); last = parts[-1].lower() if parts else name.lower()
        m = df[nl == name.lower()]
        if m.empty: m = df[nl.str.contains(re.escape(last),na=False)]
        if not m.empty:
            rc = next((c for c in ["rank","ranking","player_rank"] if c in m.columns),None)
            if rc:
                v = m.iloc[0][rc]
                return int(v) if pd.notna(v) else None
        return None

    def _player_rolling(self, df, clean, n=20) -> dict:
        wins  = df[df["winner_name"].str.lower().str.contains(re.escape(clean),na=False)]
        losses= df[df["loser_name"].str.lower().str.contains(re.escape(clean),na=False)]
        total = len(wins)+len(losses)
        if total==0: return {}
        all_r = ([(r.get("tourney_date",0),"W",r) for _,r in wins.iterrows()] +
                 [(r.get("tourney_date",0),"L",r) for _,r in losses.iterrows()])
        all_r.sort(key=lambda x: x[0] if pd.notna(x[0]) else 0, reverse=True)
        recent = all_r[:n]; rw = sum(1 for x in recent if x[1]=="W")
        result = {
            "total_matches":    total,
            "win_rate_overall": round(len(wins)/total,3),
            "recent_form":      "".join(x[1] for x in recent[:10]),
            "recent_win_rate":  round(rw/len(recent),3) if recent else 0}
        rw_df = wins.tail(n//2)
        if "w_ace" in rw_df.columns:
            v = rw_df["w_ace"].dropna()
            if len(v): result["aces_per_match"]=round(float(v.mean()),2)
        if all(c in rw_df.columns for c in ["w_1stIn","w_svpt"]):
            sv=rw_df["w_svpt"].dropna().mean()
            if sv: result["first_serve_in_pct"]=round(float(rw_df["w_1stIn"].dropna().mean()/sv),3)
        if all(c in rw_df.columns for c in ["w_bpSaved","w_bpFaced"]):
            bpf=rw_df["w_bpFaced"].dropna().mean()
            if bpf: result["bp_saved_pct"]=round(float(rw_df["w_bpSaved"].dropna().mean()/bpf),3)
        ss={}
        for surf in ["Hard","Clay","Grass"]:
            if "surface" in wins.columns:
                sw=wins[wins["surface"].str.lower()==surf.lower()]
                sl=(losses[losses["surface"].str.lower()==surf.lower()] if "surface" in losses.columns else pd.DataFrame())
                st=len(sw)+len(sl)
                if st>=5: ss[surf]={"win_rate":round(len(sw)/st,3),"matches":st}
        if ss: result["surface_stats"]=ss
        return result

    def get_tennis_stats(self, pa, pb, is_wta=False) -> dict:
        df = self.wta_matches if is_wta else self.atp_matches
        if df is None or df.empty: return {}
        def clean_name(n):
            n=n.strip(); parts=n.split()
            if len(parts)>=2:
                cand=" ".join(parts[-2:]).lower()
                wn=df["winner_name"].astype(str).str.lower()
                ln=df["loser_name"].astype(str).str.lower()
                if any(wn.str.contains(re.escape(cand),na=False)) or any(ln.str.contains(re.escape(cand),na=False)):
                    return cand
            return parts[-1].lower() if parts else n.lower()
        ca,cb=clean_name(pa),clean_name(pb)
        stats={"player_a":{"name":pa},"player_b":{"name":pb},"h2h":{}}
        for p_c,key,p_f,is_w in [(ca,"player_a",pa,is_wta),(cb,"player_b",pb,is_wta)]:
            s=self._player_rolling(df,p_c)
            if s:
                stats[key].update(s)
                r=self.get_player_ranking(p_f,is_w)
                if r: stats[key]["current_ranking"]=r
                stats[key]["data_quality"]="good" if s.get("total_matches",0)>=20 else "limited" if s.get("total_matches",0)>=5 else "poor"
        h2h_a=df[df["winner_name"].str.lower().str.contains(ca,na=False)&df["loser_name"].str.lower().str.contains(cb,na=False)]
        h2h_b=df[df["winner_name"].str.lower().str.contains(cb,na=False)&df["loser_name"].str.lower().str.contains(ca,na=False)]
        t=len(h2h_a)+len(h2h_b)
        if t:
            stats["h2h"]={"total":t,f"{pa}_wins":len(h2h_a),f"{pb}_wins":len(h2h_b),
                          "dominance":f"{pa}_dominant" if len(h2h_a)>len(h2h_b)*2 else f"{pb}_dominant" if len(h2h_b)>len(h2h_a)*2 else "balanced"}
            if "surface" in h2h_a.columns:
                by_surf={}
                for surf in ["Hard","Clay","Grass"]:
                    sa=h2h_a[h2h_a["surface"].str.lower()==surf.lower()]
                    sb=h2h_b[h2h_b["surface"].str.lower()==surf.lower()]
                    if len(sa)+len(sb)>0: by_surf[surf]={f"{pa}_wins":len(sa),f"{pb}_wins":len(sb)}
                if by_surf: stats["h2h"]["by_surface"]=by_surf
        qa=stats["player_a"].get("data_quality","poor"); qb=stats["player_b"].get("data_quality","poor")
        stats["data_quality_summary"]={
            "player_a":qa,"player_b":qb,"h2h_matches":t,
            "overall":"good" if qa=="good" and qb=="good" and t>=3 else "limited" if qa!="poor" or qb!="poor" else "poor"}
        return stats

    def load_football_data(self):
        COLS=["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
              "HS","AS","HST","AST","HC","AC",
              "B365H","B365D","B365A","BbMxH","BbMxD","BbMxA",
              "BbAvH","BbAvD","BbAvA","BbMx>2.5","BbAv>2.5"]
        all_dfs=[]
        for season in CFG.FOOTBALL_DATA_UK_SEASONS:
            for code,name in CFG.FOOTBALL_DATA_UK_LEAGUES.items():
                url = CFG.GITHUB_SOURCES["football_eu"].format(season=season,league=code)
                p = CFG.HISTORICAL_DIR/f"football_{code}_{season}.csv"
                if self._dl_csv(url,p):
                    try:
                        df=pd.read_csv(p,low_memory=False,encoding="latin-1")
                        avail=[c for c in COLS if c in df.columns]
                        if len(avail)<5: continue
                        sub=df[avail].copy(); sub["League"]=name; sub["Season"]=season
                        if "Date" in sub.columns:
                            sub["Date"]=pd.to_datetime(sub["Date"],format="mixed",dayfirst=True,errors="coerce")
                        if "HomeTeam" in sub.columns: sub=sub.dropna(subset=["HomeTeam","AwayTeam"])
                        all_dfs.append(sub)
                    except Exception as e: logger.warning("[FOOTBALL] %s: %s",p.name,e)
        if all_dfs:
            comb=pd.concat(all_dfs,ignore_index=True)
            if "Date" in comb.columns: comb=comb.sort_values("Date").reset_index(drop=True)
            self.football_data["all"]=comb
            logger.info("✅ [FOOTBALL] %d matches loaded",len(comb))

    def _fz_df(self, team, col):
        cl=team.lower().strip()
        m=col.str.lower().str.strip()==cl
        if m.any(): return m
        for p in cl.split():
            if len(p)>3:
                m2=col.str.lower().str.contains(re.escape(p),na=False)
                if 0<m2.sum()<=20: return m2
        return pd.Series([False]*len(col),index=col.index)

    def get_football_stats(self, home, away) -> dict:
        stats={"home":{},"away":{},"h2h":{}}
        df=self.football_data.get("all")
        if df is None or df.empty: return stats
        for team,key in [(home,"home"),(away,"away")]:
            hm=self._fz_df(team,df["HomeTeam"]); am=self._fz_df(team,df["AwayTeam"])
            all_r=[]
            for _,row in df[hm].iterrows():
                hg=int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                ag=int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                ftr=row.get("FTR","")
                all_r.append({"result":"W" if ftr=="H" else("D" if ftr=="D" else "L"),"scored":hg,"conceded":ag,"total":hg+ag})
            for _,row in df[am].iterrows():
                hg=int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                ag=int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                ftr=row.get("FTR","")
                all_r.append({"result":"W" if ftr=="A" else("D" if ftr=="D" else "L"),"scored":ag,"conceded":hg,"total":hg+ag})
            if len(all_r)>=5:
                recent=all_r[-15:]; n=len(recent)
                sc=[r["scored"] for r in recent]; cn=[r["conceded"] for r in recent]
                totals=[r["total"] for r in recent]
                stats[key]={
                    "form":             "".join(r["result"] for r in recent[-10:]),
                    "win_rate":         round(sum(1 for r in recent if r["result"]=="W")/n,3),
                    "draw_rate":        round(sum(1 for r in recent if r["result"]=="D")/n,3),
                    "avg_scored":       round(float(np.mean(sc)),2),
                    "avg_conceded":     round(float(np.mean(cn)),2),
                    "avg_total":        round(float(np.mean(totals)),2),
                    "over25_rate":      round(sum(1 for r in recent if r["total"]>2.5)/n,3),
                    "over35_rate":      round(sum(1 for r in recent if r["total"]>3.5)/n,3),
                    "btts_rate":        round(sum(1 for r in recent if r["scored"]>0 and r["conceded"]>0)/n,3),
                    "matches_analyzed": len(all_r),
                    "data_quality":     "good" if len(all_r)>=20 else "limited" if len(all_r)>=8 else "poor",
                    "source":           "github"}
        # H2H
        hm2=self._fz_df(home,df["HomeTeam"]); am2=self._fz_df(away,df["AwayTeam"])
        hm3=self._fz_df(away,df["HomeTeam"]); am3=self._fz_df(home,df["AwayTeam"])
        h2h_df=df[(hm2&am2)|(hm3&am3)]
        if len(h2h_df)>=3:
            h2h_r=[{"total":int(row["FTHG"] or 0)+int(row["FTAG"] or 0),
                    "btts":(row.get("FTHG",0) or 0)>0 and (row.get("FTAG",0) or 0)>0}
                   for _,row in h2h_df.iterrows() if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG"))]
            hn=len(h2h_r); gl=[r["total"] for r in h2h_r]
            stats["h2h"]={
                "total_matches":hn,"avg_goals":round(float(np.mean(gl)),2),
                "over25_rate":round(sum(1 for r in h2h_r if r["total"]>2.5)/hn,3),
                "btts_rate":round(sum(1 for r in h2h_r if r["btts"])/hn,3),"source":"github"}
        return stats

    def load_nhl_data(self):
        try:
            r=requests.get("https://api-web.nhle.com/v1/standings/now",timeout=15,headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code==200:
                standings=r.json().get("standings",[])
                rows=[]
                for t in standings:
                    w=t.get("wins",0); l=t.get("losses",0); otl=t.get("otLosses",0)
                    gp=max(w+l+otl,1); gf=t.get("goalFor",0); ga=t.get("goalAgainst",0)
                    rows.append({"team":t.get("teamName",{}).get("default",""),"wins":w,"losses":l,
                                 "points":t.get("points",0),"win_pct":round(w/gp,3),
                                 "avg_gf":round(gf/gp,2),"avg_ga":round(ga/gp,2),
                                 "streak":f"{t.get('streakCode','')}{t.get('streakCount','')}",
                                 "l10_wins":t.get("l10Wins",0)})
                self.nhl_data=pd.DataFrame(rows)
                logger.info("✅ [NHL] %d teams",len(rows)); return
        except Exception as e: logger.warning("[NHL] %s",str(e)[:60])
        self.nhl_data=None

    def load_mlb_data(self):
        season=datetime.now().year
        url=f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason"
        try:
            r=requests.get(url,timeout=15,headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code==200:
                rows=[]
                for record in r.json().get("records",[]):
                    for tr in record.get("teamRecords",[]):
                        ti=tr.get("team",{}); w=tr.get("wins",0); l=tr.get("losses",0)
                        gp=max(w+l,1); rs=tr.get("runsScored",0) or 0; ra=tr.get("runsAllowed",0) or 0
                        rows.append({"team":ti.get("name",""),"wins":w,"losses":l,
                                     "win_pct":float(tr.get("winningPercentage",0) or 0),
                                     "avg_runs_scored":round(rs/gp,2),"avg_runs_allowed":round(ra/gp,2),
                                     "run_diff":round((rs-ra)/gp,2),
                                     "streak":tr.get("streak",{}).get("streakCode","")})
                if rows:
                    self.mlb_data=pd.DataFrame(rows)
                    logger.info("✅ [MLB] %d teams (Season %s)",len(rows),season); return
        except Exception as e: logger.warning("[MLB] %s",str(e)[:60])
        self.mlb_data=None

    def load_nba_data(self):
        try:
            from nba_api.stats.endpoints import leaguestandings
            now=datetime.now()
            s_year=now.year if now.month>8 else now.year-1
            sstr=f"{s_year}-{str(s_year+1)[-2:]}"
            standings=leaguestandings.LeagueStandings(season=sstr,season_type="Regular Season",league_id="00",
                headers={"User-Agent":"Mozilla/5.0","Accept":"application/json",
                         "Referer":"https://www.nba.com/","Origin":"https://www.nba.com"},timeout=10)
            df=standings.get_data_frames()[0]
            if df is not None and not df.empty:
                self.nba_data=df; logger.info("✅ [NBA] %d teams (%s)",len(df),sstr); return
        except Exception as e: logger.warning("[NBA] %s",str(e)[:80])
        self.nba_data=None

    def get_us_sports_stats(self, sport, team) -> dict:
        """Unified lookup for NBA/WNBA/MLB/NHL. Falls back to TSDB for any sport."""
        sport_l = sport.lower(); cl = team.lower().strip()
        is_nba     = "basketball" in sport_l or "nba" in sport_l
        is_wnba    = "wnba" in sport_l or "women" in sport_l
        is_baseball= "baseball" in sport_l or "mlb" in sport_l or "npb" in sport_l
        is_hockey  = "hockey" in sport_l or "nhl" in sport_l

        # MLB
        if is_baseball and self.mlb_data is not None and not self.mlb_data.empty:
            for _,row in self.mlb_data.iterrows():
                if _fuzzy_match(cl, str(row.get("team",""))):
                    d=row.to_dict(); d.update({"data_quality":"good","source":"mlb_api"}); return d

        # NHL
        if is_hockey and self.nhl_data is not None and not self.nhl_data.empty:
            for _,row in self.nhl_data.iterrows():
                if _fuzzy_match(cl, str(row.get("team",""))):
                    d=row.to_dict(); d.update({"data_quality":"good","source":"nhl_api"}); return d

        # NBA standings (non-WNBA only — WNBA teams not in NBA data)
        if is_nba and not is_wnba and self.nba_data is not None and not self.nba_data.empty:
            for col in ["TeamName","TEAM_NAME","TeamCity"]:
                if col in self.nba_data.columns:
                    for _,row in self.nba_data.iterrows():
                        if _fuzzy_match(cl, str(row[col])):
                            return {"win_pct": float(row.get("WinPCT", row.get("WIN_PCT", 0.5)) or 0.5),
                                    "source":"nba_api","data_quality":"limited"}

        # TSDB — works for WNBA, Japanese baseball, any sport TSDB covers
        try:
            ts = tsdb.get_team_stats(team)
            if ts and ts.get("matches_analyzed",0) >= 3:
                return ts
        except Exception as e:
            logger.warning("[TSDB get_us_sports_stats] %s: %s", team, e)

        return {}


# =========================================================
# 13. ML ENGINE
# =========================================================
class MLEngine:
    def __init__(self, de: FreeDataEngine):
        self.de = de
        self.football_pipeline = None
        self.tennis_pipelines: Dict[str, Optional[dict]] = {"atp":None,"wta":None}
        self.is_football_trained = False
        self._team_deques: Dict[str,deque] = defaultdict(lambda: deque(maxlen=10))

    @property
    def is_tennis_trained(self):
        return any(p is not None for p in self.tennis_pipelines.values())

    def load_or_train_football(self):
        p = CFG.ML_DIR/"football_model_v91.pkl"
        if p.exists() and (time.time()-p.stat().st_mtime)/3600 < 24:
            try:
                d=pickle.loads(p.read_bytes())
                self.football_pipeline=d["pipeline"]
                for k,v in d.get("deques",{}).items():
                    self._team_deques[k]=deque(v,maxlen=10)
                self.is_football_trained=True
                logger.info("⚡ [ML FOOTBALL] Loaded from cache"); return
            except Exception: pass
        self._train_football()
        if self.is_football_trained:
            try: p.write_bytes(pickle.dumps({"pipeline":self.football_pipeline,"deques":{k:list(v) for k,v in self._team_deques.items()}}))
            except Exception: pass

    def _train_football(self):
        df=self.de.football_data.get("all")
        if df is None or len(df)<300: return
        if "Date" in df.columns: df=df.sort_values("Date").reset_index(drop=True)
        self._team_deques=defaultdict(lambda: deque(maxlen=10))
        feats,labels=[],[]
        for _,row in df.iterrows():
            ht=str(row.get("HomeTeam","") or ""); at=str(row.get("AwayTeam","") or "")
            ftr=str(row.get("FTR","") or "")
            if not ht or not at or ftr not in ["H","D","A"]: continue
            try: hg=float(row.get("FTHG",0) or 0); ag=float(row.get("FTAG",0) or 0)
            except Exception: continue
            def gs(t):
                h=list(self._team_deques[t])
                if len(h)<3: return None
                w=np.array([1/(i+1) for i in range(len(h))][::-1]); w/=w.sum()
                return {"avg_gs":float(np.dot(w,[x["gs"] for x in h])),
                        "avg_gc":float(np.dot(w,[x["gc"] for x in h])),
                        "form_pts":float(np.dot(w,[x["pts"] for x in h])),
                        "win_rate":sum(1 for x in h if x["pts"]==3)/len(h)}
            hs=gs(ht); aws=gs(at)
            if hs and aws:
                feats.append([hs["avg_gs"],hs["avg_gc"],hs["form_pts"],hs["win_rate"],
                              aws["avg_gs"],aws["avg_gc"],aws["form_pts"],aws["win_rate"],
                              hs["avg_gs"]-aws["avg_gc"],aws["avg_gs"]-hs["avg_gc"]])
                labels.append({"H":0,"D":1,"A":2}[ftr])
            self._team_deques[ht].appendleft({"gs":hg,"gc":ag,"pts":3 if ftr=="H" else(1 if ftr=="D" else 0)})
            self._team_deques[at].appendleft({"gs":ag,"gc":hg,"pts":3 if ftr=="A" else(1 if ftr=="D" else 0)})
        if not feats or len(np.unique(labels))<2: return
        X=np.nan_to_num(np.array(feats,dtype=np.float64)); y=np.array(labels,dtype=np.int32)
        scaler=RobustScaler(); Xs=scaler.fit_transform(X)
        try:
            model=CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42),cv=3,method="isotonic")
            model.fit(Xs,y)
            self.football_pipeline={"model":model,"scaler":scaler}
            self.is_football_trained=True
            logger.info("✅ [ML FOOTBALL] Trained on %d samples",len(X))
        except Exception as e: logger.error("[ML FOOTBALL] %s",e)

    def predict_football(self, home, away) -> Optional[dict]:
        if not self.is_football_trained: return None
        def gf(team):
            cl=team.lower().strip()
            bm=next((k for k in self._team_deques if cl in k.lower() or k.lower() in cl),None)
            if not bm: return None
            h=list(self._team_deques[bm])
            if len(h)<3: return None
            w=np.array([1/(i+1) for i in range(len(h))][::-1]); w/=w.sum()
            return {"avg_gs":float(np.dot(w,[x["gs"] for x in h])),"avg_gc":float(np.dot(w,[x["gc"] for x in h])),
                    "form_pts":float(np.dot(w,[x["pts"] for x in h])),"win_rate":sum(1 for x in h if x["pts"]==3)/len(h)}
        hs=gf(home); aws=gf(away)
        if not hs or not aws: return None
        fv=[hs["avg_gs"],hs["avg_gc"],hs["form_pts"],hs["win_rate"],
            aws["avg_gs"],aws["avg_gc"],aws["form_pts"],aws["win_rate"],
            hs["avg_gs"]-aws["avg_gc"],aws["avg_gs"]-hs["avg_gc"]]
        try:
            X=np.nan_to_num(np.array([fv],dtype=np.float64))
            Xs=self.football_pipeline["scaler"].transform(X)
            probs=self.football_pipeline["model"].predict_proba(Xs)[0]
            lm={0:"home_win",1:"draw",2:"away_win"}
            return {lm.get(int(c),f"c{c}"):round(float(p),4)
                    for c,p in zip(self.football_pipeline["model"].classes_,probs)}
        except Exception as e: logger.warning("[ML FOOTBALL] %s",e); return None

    def load_or_train_tennis(self, is_wta=False):
        tour="wta" if is_wta else "atp"
        p=CFG.ML_DIR/f"tennis_model_{tour}_v91.pkl"
        if p.exists() and (time.time()-p.stat().st_mtime)/3600 < 24:
            try:
                self.tennis_pipelines[tour]=pickle.loads(p.read_bytes())["pipeline"]
                logger.info("⚡ [ML TENNIS %s] Loaded",tour.upper()); return
            except Exception: pass
        self._train_tennis(is_wta)
        if self.tennis_pipelines[tour]:
            try: p.write_bytes(pickle.dumps({"pipeline":self.tennis_pipelines[tour]}))
            except Exception: pass

    def _train_tennis(self, is_wta=False):
        df=self.de.wta_matches if is_wta else self.de.atp_matches
        tour="wta" if is_wta else "atp"
        if df is None or len(df)<500: logger.warning("[ML TENNIS %s] Insufficient data",tour.upper()); return
        df=df.sort_values("tourney_date").reset_index(drop=True)
        ph: Dict[Any,List[dict]]=defaultdict(list); feats,labels,weights=[],[],[]
        for _,row in df.iterrows():
            wid=row.get("winner_id"); lid=row.get("loser_id")
            wr=float(row.get("winner_rank",0) or 0); lr=float(row.get("loser_rank",0) or 0)
            if wr<=0 or lr<=0: continue
            surf=str(row.get("surface","Hard") or "Hard").lower()
            td=float(row.get("tourney_date",20200101) or 20200101)
            def agg(hist):
                recent=hist[-20:] if len(hist)>=20 else hist
                if not recent: return {}
                total=len(recent); wins=sum(1 for h in recent if h["won"])
                svpt=max(sum(h.get("svpt",50) for h in recent),1)
                return {"win_rate":wins/total,"ace_rate":sum(h.get("ace",0) for h in recent)/svpt,"n":total}
            wa=agg(ph.get(wid,[])); la=agg(ph.get(lid,[]))
            if wa.get("n",0)>=3 and la.get("n",0)>=3:
                is_wp1=wr<lr; p1a,p2a=(wa,la) if is_wp1 else (la,wa)
                p1r,p2r=(wr,lr) if is_wp1 else (lr,wr)
                fv=[p1r,p2r,p2r-p1r,1. if surf=="hard" else 0.,1. if surf=="clay" else 0.,1. if surf=="grass" else 0.,
                    p1a.get("win_rate",0.5),p2a.get("win_rate",0.5),
                    p1a.get("win_rate",0.5)-p2a.get("win_rate",0.5),float(p1a.get("n",0)),float(p2a.get("n",0))]
                feats.append(fv); labels.append(1 if is_wp1 else 0)
                weights.append(float(np.clip(0.5+0.5*(td-20200101)/max(20260101-20200101,1),0.5,1.0)))
            if wid is not None: ph[wid].append({"won":True,"ace":float(row.get("w_ace",0) or 0),"svpt":max(float(row.get("w_svpt",50) or 50),1.)})
            if lid is not None: ph[lid].append({"won":False,"ace":float(row.get("l_ace",0) or 0),"svpt":max(float(row.get("l_svpt",50) or 50),1.)})
        if not feats or len(np.unique(labels))<2: return
        X=np.nan_to_num(np.array(feats,dtype=np.float64)); y=np.array(labels,dtype=np.int32); sw=np.array(weights,dtype=np.float64)
        scaler=RobustScaler(); Xs=scaler.fit_transform(X)
        try:
            cal=CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42),cv=3,method="isotonic")
            cal.fit(Xs,y,sample_weight=sw)
            self.tennis_pipelines[tour]={"model":cal,"scaler":scaler}
            logger.info("✅ [ML TENNIS %s] Trained on %d samples",tour.upper(),len(X))
        except Exception as e: logger.error("[ML TENNIS %s] %s",tour.upper(),e)

    def predict_tennis(self, pa, pb, stats, surface="hard") -> Optional[dict]:
        tour="wta" if stats.get("tour","").lower()=="wta" else "atp"
        pipeline=self.tennis_pipelines.get(tour) or next((p for p in self.tennis_pipelines.values() if p),None)
        if not pipeline: return None
        pas=stats.get("player_a",{}); pbs=stats.get("player_b",{})
        ra=float(pas.get("current_ranking",100) or 100); rb=float(pbs.get("current_ranking",100) or 100)
        is_pa_p1=ra<=rb; p1r,p2r=(ra,rb) if is_pa_p1 else (rb,ra)
        p1wr=float(pas.get("recent_win_rate",0.5) or 0.5); p2wr=float(pbs.get("recent_win_rate",0.5) or 0.5)
        if not is_pa_p1: p1wr,p2wr=p2wr,p1wr
        fv=[p1r,p2r,p2r-p1r,1. if surface=="hard" else 0.,1. if surface=="clay" else 0.,1. if surface=="grass" else 0.,
            p1wr,p2wr,p1wr-p2wr,float(pas.get("total_matches",10) or 10),float(pbs.get("total_matches",10) or 10)]
        try:
            X=np.nan_to_num(np.array([fv],dtype=np.float64)); Xs=pipeline["scaler"].transform(X)
            probs=pipeline["model"].predict_proba(Xs)[0]
            pm={int(c):float(p) for c,p in zip(pipeline["model"].classes_,probs)}
            p1w=pm.get(1,0.5); pa_p=p1w if is_pa_p1 else(1-p1w)
            return {f"{pa}_win_prob":round(pa_p,4),f"{pb}_win_prob":round(1-pa_p,4)}
        except Exception as e: logger.warning("[ML TENNIS] %s",e); return None


# =========================================================
# 14. POISSON ENGINE
# =========================================================
class PoissonEngine:
    @staticmethod
    def calculate(home, away, df) -> dict:
        if df is None or df.empty: return {}
        req={"HomeTeam","AwayTeam","FTHG","FTAG"}
        if not req.issubset(df.columns): return {}
        rec=df.dropna(subset=["FTHG","FTAG"]).tail(1500).copy()
        if len(rec)<50: return {}
        la_h=rec["FTHG"].astype(float).mean(); la_a=rec["FTAG"].astype(float).mean()
        if pd.isna(la_h) or la_h==0: return {}
        def fz(t,col):
            cl=t.lower().strip()
            m=col.str.lower().str.strip()==cl
            if m.any(): return m
            for p in cl.split():
                if len(p)>3:
                    m2=col.str.lower().str.contains(re.escape(p),na=False)
                    if m2.any(): return m2
            return pd.Series([False]*len(col),index=col.index)
        hm=rec[fz(home,rec["HomeTeam"])]; am=rec[fz(away,rec["AwayTeam"])]
        if len(hm)<5 or len(am)<5: return {}
        ha=hm["FTHG"].astype(float).mean()/la_h; hd=hm["FTAG"].astype(float).mean()/la_a
        aa=am["FTAG"].astype(float).mean()/la_a; ad=am["FTHG"].astype(float).mean()/la_h
        if any(pd.isna(v) or v==0 for v in [ha,hd,aa,ad]): return {}
        hxg=float(np.clip(ha*ad*la_h,0.1,8.)); axg=float(np.clip(aa*hd*la_a,0.1,8.))
        mg=6; pm=np.zeros((mg+1,mg+1))
        for x in range(mg+1):
            for y in range(mg+1):
                pm[x,y]=stats_scipy.poisson.pmf(x,hxg)*stats_scipy.poisson.pmf(y,axg)
        t=pm.sum()
        if t==0: return {}
        pm/=t
        return {"home_xg":round(hxg,2),"away_xg":round(axg,2),
                "home_win_prob_poisson":round(float(np.sum(np.tril(pm,-1))),3),
                "draw_prob_poisson":    round(float(np.sum(np.diag(pm))),3),
                "away_win_prob_poisson":round(float(np.sum(np.triu(pm,1))),3)}


# =========================================================
# 15. EV ENGINE
# =========================================================
class EVEngine:
    @staticmethod
    def remove_vig(odds_list: List[float]) -> List[float]:
        implied=[1/o for o in odds_list if o>1.]; total=sum(implied)
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
    def kelly(prob, odds) -> float:
        b=odds-1.
        if b<=0 or prob<=0 or prob>=1: return 0.
        return round(min(max(0.,(prob*b-(1-prob))/b)*CFG.KELLY_FRACTION, CFG.MAX_KELLY_PCT/100),4)


def calculate_ev(markets_data: dict) -> list:
    best_per_market: dict = {}
    for mk, ml in markets_data.items():
        if not isinstance(ml, list): continue
        sharp_prices: Dict[Tuple,List[float]] = defaultdict(list)
        soft_prices:  Dict[Tuple,List[float]] = defaultdict(list)
        best_mkt:     Dict[Tuple,Tuple] = {}
        for entry in ml:
            if not isinstance(entry, dict): continue
            bk=entry.get("bookmaker_key",""); bk_name=entry.get("bookmaker",bk)
            is_sharp=bk in CFG.SHARP_BOOKMAKERS
            for o in entry.get("outcomes",[]):
                if not isinstance(o, dict): continue
                raw=o.get("name","")
                if not raw: continue
                pt=o.get("point"); ck=(raw,pt)
                disp=f"{raw} {pt}" if pt is not None else raw
                try: price=float(o["price"])
                except Exception: continue
                if price<=1.: continue
                (sharp_prices if is_sharp else soft_prices)[ck].append(price)
                if ck not in best_mkt or price>best_mkt[ck][0]:
                    best_mkt[ck]=(price,bk_name,disp)
        if not best_mkt: continue
        ref_prices={k:max(p) for k,p in sharp_prices.items() if p}
        has_sharp=bool(ref_prices)
        if not ref_prices: ref_prices={k:max(p) for k,p in soft_prices.items() if p}
        if not ref_prices: continue
        comp_keys=list(ref_prices.keys()); odds_list=[ref_prices[k] for k in comp_keys]
        impl_sum=sum(1/o for o in odds_list if o>0)
        if not (CFG.MIN_VALID_IMPLIED_SUM<=impl_sum<=CFG.MAX_VALID_IMPLIED_SUM): continue
        if len(comp_keys)<CFG.MARKET_EXPECTED_OUTCOMES.get(mk,{}).get("min",2): continue
        try:
            tp_list=EVEngine.remove_vig(odds_list)
            if len(tp_list)!=len(comp_keys): raise ValueError()
            tp=dict(zip(comp_keys,tp_list))
        except Exception:
            tp={comp_keys[i]:(1/odds_list[i])/max(impl_sum,1e-10) for i in range(len(comp_keys))}
        min_odds=CFG.H2H_MIN_ODDS if mk=="h2h" else CFG.TOTALS_MIN_ODDS
        sharp_mult=0.75 if has_sharp else 1.0
        min_ev=(CFG.H2H_MIN_EV if mk=="h2h" else CFG.TOTALS_MIN_EV)*sharp_mult
        best_opp=None
        for ck in comp_keys:
            true_p=tp.get(ck,0)
            if true_p<=0 or true_p>=1: continue
            bp,bbm,disp_name=best_mkt.get(ck,(0,"?","?"))
            if bp<=1.: continue
            ev=true_p*bp-1.
            if ev>0.20:
                logger.warning("[EV] Suspicious EV=%.1f%% for %s — skip",ev*100,disp_name); continue
            if ev<min_ev or bp<min_odds: continue
            bk_count=len(soft_prices.get(ck,[])+sharp_prices.get(ck,[]))
            if bk_count<3 and not has_sharp:
                logger.debug("[EV] Thin market %s (%d books) — skip",disp_name,bk_count); continue
            kelly_p=EVEngine.kelly(true_p,bp)
            sp=ref_prices.get(ck,bp); clv=(bp/sp-1)*100 if sp>0 else 0.
            opp={"pick":disp_name,"market":mk,"market_label":_market_label(mk),
                 "prob":round(true_p,4),"odds":round(bp,3),"bookmaker":bbm,
                 "ev":round(ev,4),"edge_pct":round(ev*100,2),"kelly_pct":round(kelly_p*100,2),
                 "clv_pct":round(clv,2),"has_sharp_line":has_sharp,"steam_pct":None,
                 "bookmaker_count":bk_count}
            if best_opp is None or opp["ev"]>best_opp["ev"]: best_opp=opp
        if best_opp: best_per_market[mk]=best_opp
    return sorted(best_per_market.values(),key=lambda x:x["ev"],reverse=True)


# =========================================================
# 16. CONFIDENCE ENGINE
# =========================================================
class ConfidenceEngine:
    @classmethod
    def score(cls, opp: dict, stats: dict, ml_pred=None, poisson_pred=None) -> int:
        s=42; ev=opp.get("ev",0)*100
        if ev>20: return 20
        s += 16 if ev>8 else 13 if ev>5 else 9 if ev>3 else 6 if ev>2 else 3 if ev>1 else -8
        if opp.get("has_sharp_line"):  s+=14
        clv=opp.get("clv_pct",0);     s += 8 if clv>3 else 4 if clv>1.5 else 0
        kelly=opp.get("kelly_pct",0);  s += 6 if kelly>2 else 3 if kelly>1 else(-4 if kelly<0.5 else 0)
        bk=opp.get("bookmaker_count",1); s += 7 if bk>=8 else 4 if bk>=5 else 1 if bk>=3 else -6
        # Data quality
        dq=_get_dq(stats)
        s += 10 if dq=="good" else 5 if dq=="limited" else -3 if dq=="poor" else -9
        # TSDB
        h_ts=stats.get("tsdb_stats",{}).get("home",{}); a_ts=stats.get("tsdb_stats",{}).get("away",{})
        hq=h_ts.get("data_quality","none"); aq=a_ts.get("data_quality","none")
        if hq=="good" and aq=="good":                              s+=9
        elif hq in("good","limited") and aq in("good","limited"): s+=5
        elif hq in("good","limited") or aq in("good","limited"):   s+=2
        # Form alignment
        h_wr=h_ts.get("win_rate",0); a_wr=a_ts.get("win_rate",0); h_name=h_ts.get("team_name","")
        if h_wr and a_wr and abs(h_wr-a_wr)>0.20 and h_name:
            pick_lower=opp.get("pick","").lower()
            favors_home=any(w in pick_lower for w in h_name.lower().split() if len(w)>3)
            s += 5 if (favors_home==(h_wr>a_wr)) else -4
        # ML
        if ml_pred:
            mx=max((v for v in ml_pred.values() if isinstance(v,(float,int)) and 0<v<=1),default=0)
            s += 11 if mx>0.68 else 7 if mx>0.62 else 3 if mx>0.55 else 0
        if poisson_pred: s+=5; s+=3 if ml_pred else 0
        # Steam
        steam=opp.get("steam_pct")
        if steam is not None: s += 9 if steam>=3.0 else 5 if steam>=1.5 else(-8 if steam<=-5 else 0)
        return int(np.clip(s,0,100))


def _get_dq(stats: dict) -> str:
    """Determine overall data quality from whatever sources are populated."""
    if stats.get("historical_data"):
        return stats["historical_data"].get("data_quality_summary",{}).get("overall","poor")
    if stats.get("football_stats"):
        hq=stats["football_stats"].get("home",{}).get("data_quality","poor")
        aq=stats["football_stats"].get("away",{}).get("data_quality","poor")
        if hq=="good" and aq=="good": return "good"
        if hq!="poor" or aq!="poor": return "limited"
        return "poor"
    if stats.get("us_sports"):
        hq=stats["us_sports"].get("home",{}).get("data_quality","poor")
        aq=stats["us_sports"].get("away",{}).get("data_quality","poor")
        if hq=="good" and aq=="good": return "good"
        if hq!="poor" or aq!="poor": return "limited"
        return "poor"
    if stats.get("tsdb_stats"):
        hq=stats["tsdb_stats"].get("home",{}).get("data_quality","poor")
        aq=stats["tsdb_stats"].get("away",{}).get("data_quality","poor")
        if hq=="good" and aq=="good": return "limited"   # TSDB alone = limited max
        if hq!="poor" or aq!="poor": return "poor"
        return "poor"
    return "none"


# =========================================================
# 17. LINE MOVEMENT TRACKER
# =========================================================
class LineTracker:
    def __init__(self):
        self._path=CFG.CACHE_DIR/"line_movement.json"; self._lock=threading.Lock()
        self.data=Cache.load(self._path)

    def record(self, home, away, market, outcome, odds) -> Optional[float]:
        if odds<=1.: return None
        mk=hashlib.md5(f"{home}|{away}|{market}|{outcome}".encode()).hexdigest()
        with self._lock:
            now=datetime.now(timezone.utc).isoformat()
            if mk not in self.data:
                self.data[mk]={"initial_odds":odds,"current_odds":odds,"timestamp":now}
                Cache.save(self._path,self.data); return None
            init=self.data[mk].get("initial_odds",odds)
            self.data[mk].update({"current_odds":odds,"timestamp":now})
            Cache.save(self._path,self.data)
        return round((init/odds-1)*100,2) if init>0 else 0.

line_tracker=LineTracker()


# =========================================================
# 18. PERFORMANCE TRACKER
# =========================================================
class PerfTracker:
    def __init__(self):
        self._lock=threading.Lock(); self.data=Cache.load(CFG.PERFORMANCE_FILE)
        self.data.setdefault("signals",[])
    def record(self,home,away,pick,market,odds,ev,confidence,prob,sport="other",sport_key=""):
        sig={"id":hashlib.md5(f"{home}|{away}|{market}|{datetime.now(timezone.utc).date()}".encode()).hexdigest()[:8],
             "timestamp":datetime.now(timezone.utc).isoformat(),"sport":sport,"api_sport_key":sport_key,
             "home":home,"away":away,"pick":pick,"market":market,"odds":odds,"ev":ev,
             "confidence":confidence,"implied_prob":prob,"outcome":None,"profit_loss":None}
        with self._lock:
            self.data["signals"].append(sig)
            if len(self.data["signals"])>500: self.data["signals"]=self.data["signals"][-500:]
        Cache.save(CFG.PERFORMANCE_FILE,self.data)

perf_tracker=PerfTracker()


# =========================================================
# 19. HELPERS
# =========================================================
def _market_label(mk): return {"h2h":"Match Winner","totals":"Over/Under","spreads":"Handicap"}.get(mk,mk.replace("_"," ").title())
def _sport_emoji(sk): return {"tennis":"🎾","football":"⚽","basketball":"🏀","baseball":"⚾","hockey":"🏒","cricket":"🏏"}.get(sk,"🏆")

def normalize_sport(title: str) -> str:
    l=(title or "").lower()
    if any(k in l for k in ["tennis","atp","wta"]):                      return "tennis"
    if any(k in l for k in ["soccer","football","premier","liga","bundesliga","serie","ligue","champions"]): return "football"
    if any(k in l for k in ["basketball","nba","wnba","euroleague"]):     return "basketball"
    if any(k in l for k in ["baseball","mlb","npb","softball"]):         return "baseball"
    if any(k in l for k in ["hockey","nhl"]):                            return "hockey"
    if any(k in l for k in ["cricket","ipl","t20"]):                     return "cricket"
    return "other"

def clean_name(n): return re.sub(r"\s*\([^)]*\)","",str(n or "")).strip()

def countdown(ct, now) -> str:
    try:
        mt=datetime.fromisoformat(ct.replace("Z","+00:00"))
        if mt.tzinfo is None: mt=mt.replace(tzinfo=timezone.utc)
        mins=int((mt-now).total_seconds()/60)
        if mins>60: return f"{mins//60}h {mins%60}m"
        return f"{mins}m" if mins>0 else "LIVE"
    except Exception: return "N/A"

def translate_pick(pick, market, home, away) -> str:
    pl=pick.lower().strip()
    if market.lower()=="h2h":
        hs=difflib.SequenceMatcher(None,home.lower(),pl).ratio()
        as_=difflib.SequenceMatcher(None,away.lower(),pl).ratio()
        if hs>as_ and hs>0.3: return f"{home} Win"
        if as_>hs and as_>0.3: return f"{away} Win"
        if "draw" in pl or "tie" in pl: return "Draw"
    elif "total" in market.lower():
        m=re.search(r"\b(over|under)\b\s*([\d.]+)",pl)
        if m: return f"{m.group(1).capitalize()} {m.group(2)}"
    return pick.title()

def conf_label(fc: int) -> str:
    if fc>=78: return "Very Strong 🔥🔥"
    if fc>=70: return "Strong 🔥"
    if fc>=63: return "Good ✅"
    return "Standard ⚡"


# =========================================================
# 20. AI DECISION ENGINE
# =========================================================
def make_ai_decision(home, away, sport, sport_key, opp, stats, math_score, ml_pred=None, poisson_pred=None) -> dict:
    default={"sport_emoji":_sport_emoji(sport_key),"decision":"SKIP",
             "ai_confidence":math_score,"math_confidence":math_score,"final_confidence":math_score,
             "risk_level":"High","logic":"Math score below threshold.","key_factors":[],"red_flags":[]}
    if math_score<CFG.MIN_MATH_SCORE_TO_CALL_AI:
        return {**default,"logic":f"Math score {math_score} < threshold {CFG.MIN_MATH_SCORE_TO_CALL_AI}"}

    lines=[
        f"MATCH: {home} vs {away}",
        f"SPORT: {sport} | MARKET: {opp['market_label']}",
        f"PICK: {opp['pick']} @ {opp['odds']}",
        f"",
        f"MARKET DATA:",
        f"  True Prob: {opp['prob']*100:.1f}%  EV: {opp['edge_pct']:+.2f}%",
        f"  Kelly: {opp.get('kelly_pct',0):.1f}%  Sharp: {opp.get('has_sharp_line',False)}",
        f"  CLV: {opp.get('clv_pct',0):+.1f}%  Books: {opp.get('bookmaker_count',1)}",
        f"  Steam: {opp.get('steam_pct','first_obs')}  Math Score: {math_score}/100",
    ]
    if stats.get("historical_data"):
        pa=stats["historical_data"].get("player_a",{}); pb=stats["historical_data"].get("player_b",{})
        h2h=stats["historical_data"].get("h2h",{}); dq=stats["historical_data"].get("data_quality_summary",{})
        lines+=[f"","f=TENNIS DATA (quality={dq.get('overall','?')}):",
                f"  {home}: Rank #{pa.get('current_ranking','?')}  WR {pa.get('recent_win_rate',0)*100:.1f}%  Form {pa.get('recent_form','N/A')}",
                f"  {away}: Rank #{pb.get('current_ranking','?')}  WR {pb.get('recent_win_rate',0)*100:.1f}%  Form {pb.get('recent_form','N/A')}",
                f"  H2H: {h2h.get('total',0)} matches | {h2h.get('dominance','balanced')}"]
    if stats.get("football_stats"):
        hm=stats["football_stats"].get("home",{}); aw=stats["football_stats"].get("away",{}); h2h=stats["football_stats"].get("h2h",{})
        lines+=[f"","FOOTBALL DATA:",
                f"  {home}: Form {hm.get('form','N/A')}  GS/GA {hm.get('avg_scored',0):.2f}/{hm.get('avg_conceded',0):.2f}  WR {hm.get('win_rate',0)*100:.1f}%  O2.5 {hm.get('over25_rate',0)*100:.1f}%  Qual:{hm.get('data_quality','?')}",
                f"  {away}: Form {aw.get('form','N/A')}  GS/GA {aw.get('avg_scored',0):.2f}/{aw.get('avg_conceded',0):.2f}  WR {aw.get('win_rate',0)*100:.1f}%  O2.5 {aw.get('over25_rate',0)*100:.1f}%  Qual:{aw.get('data_quality','?')}"]
        if h2h: lines.append(f"  H2H {h2h.get('total_matches',0)} matches: Avg Goals {h2h.get('avg_goals',0):.2f}  O2.5 {h2h.get('over25_rate',0)*100:.1f}%  BTTS {h2h.get('btts_rate',0)*100:.1f}%")
    if poisson_pred:
        lines+=[f"","POISSON: xG {poisson_pred.get('home_xg','?')}-{poisson_pred.get('away_xg','?')}",
                f"  H {poisson_pred.get('home_win_prob_poisson',0)*100:.1f}%  D {poisson_pred.get('draw_prob_poisson',0)*100:.1f}%  A {poisson_pred.get('away_win_prob_poisson',0)*100:.1f}%"]
    if ml_pred:
        lines.append(f"ML MODEL: {' | '.join(f'{k}:{v*100:.1f}%' for k,v in ml_pred.items() if isinstance(v,float))}")
    if stats.get("us_sports"):
        us=stats["us_sports"]
        lines+=[f"","US SPORTS:",
                f"  {home}: {json.dumps(us.get('home',{}))}",
                f"  {away}: {json.dumps(us.get('away',{}))}"]

    ai_data=ai_manager.generate("\n".join(lines))
    if not ai_data or not isinstance(ai_data,dict):
        return {**default,"decision":"BET" if math_score>=58 else "SKIP","logic":"AI unavailable — math fallback."}

    decision=str(ai_data.get("decision","SKIP")).upper().strip()
    if decision not in ["BET","SKIP"]: decision="SKIP"
    try: ai_conf=int(np.clip(float(ai_data.get("confidence",math_score)),0,100))
    except Exception: ai_conf=math_score

    # Data quality caps — _get_dq covers all stat types: historical/football/us_sports/tsdb
    dq = _get_dq(stats)
    if   dq == "none"    and ai_conf > 60: ai_conf = 60; logger.warning("[AI] No data → cap 60")
    elif dq == "poor"    and ai_conf > 65: ai_conf = 65; logger.warning("[AI] Poor data → cap 65")
    elif dq == "limited" and ai_conf > 74: ai_conf = 74; logger.warning("[AI] Limited data → cap 74")
    if ai_manager.last_provider=="groq" and ai_conf>=70 and opp.get("ev",0)*100<2.5:
        ai_conf=min(ai_conf-7,68)

    hybrid=ai_conf*CFG.AI_WEIGHT+math_score*CFG.MATH_WEIGHT
    delta=hybrid-math_score
    if delta>CFG.MAX_AI_BOOST: hybrid=math_score+CFG.MAX_AI_BOOST
    elif delta<-CFG.MAX_AI_PENALTY: hybrid=math_score-CFG.MAX_AI_PENALTY
    final=int(np.clip(hybrid,0,100))

    if decision=="BET" and ai_conf<50: decision="SKIP"; logger.warning("[AI] BET conf=%d too low → SKIP",ai_conf)
    if decision=="SKIP" and opp.get("has_sharp_line") and opp.get("ev",0)*100>4.0 and math_score>52:
        decision="BET"; final=max(final,63); logger.info("[AI] Sharp override → BET EV=%.1f%%",opp["ev"]*100)

    logger.info("[AI] %s vs %s | %s | AI:%d Math:%d Final:%d | provider:%s",
                home,away,decision,ai_conf,math_score,final,ai_manager.last_provider)
    return {
        "sport_emoji":    str(ai_data.get("sport_emoji","")).strip() or _sport_emoji(sport_key),
        "decision":       decision,
        "ai_confidence":  ai_conf,
        "math_confidence":math_score,
        "final_confidence":final,
        "risk_level":     str(ai_data.get("risk_level","Medium")),
        "logic":          str(ai_data.get("logic",default["logic"]))[:500],
        "key_factors":    [str(f)[:120] for f in (ai_data.get("key_factors") or [])[:5]],
        "red_flags":      [str(f)[:120] for f in (ai_data.get("red_flags") or [])[:3]]}


# =========================================================
# 21. TELEGRAM
# =========================================================
def send_telegram(msg: str) -> bool:
    MAX=4000
    chunks=[msg] if len(msg)<=MAX else []
    if not chunks:
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


def build_message(home, away, sport, sport_key, opp, ai, stats, math_score, ml_pred, poisson_pred, now_utc, ct) -> str:
    fc=ai["final_confidence"]
    he=html_lib.escape(home); ae=html_lib.escape(away)
    sport_labels={"tennis":"Tennis 🎾","football":"Football ⚽","basketball":"Basketball 🏀",
                  "baseball":"Baseball ⚾","hockey":"Ice Hockey 🏒","cricket":"Cricket 🏏"}
    sport_en=sport_labels.get(sport_key,f"{sport} 🏆")
    pick_esc=html_lib.escape(translate_pick(opp["pick"],opp["market"],home,away))
    sharp="🔪 " if opp.get("has_sharp_line") else ""
    risk_labels={"Low":"Low 🟢","Medium":"Medium 🟠","High":"High 🔴"}
    risk_en=risk_labels.get(ai["risk_level"],"Medium 🟠")
    logic=html_lib.escape(str(ai.get("logic",""))[:350])

    stat_line=""
    if stats.get("football_stats"):
        hm=stats["football_stats"].get("home",{}); aw=stats["football_stats"].get("away",{})
        h_form=hm.get("form",hm.get("recent_form_5","?")); a_form=aw.get("form",aw.get("recent_form_5","?"))
        h_gs=hm.get("avg_scored",0); h_gc=hm.get("avg_conceded",0)
        a_gs=aw.get("avg_scored",0); a_gc=aw.get("avg_conceded",0)
        h_rank=hm.get("standing",{}).get("rank",""); a_rank=aw.get("standing",{}).get("rank","")
        stat_line=(f"\n{he}{f' #{h_rank}' if h_rank else ''}: {h_form}  {h_gs:.1f}/{h_gc:.1f}\n"
                   f"{ae}{f' #{a_rank}' if a_rank else ''}: {a_form}  {a_gs:.1f}/{a_gc:.1f}")
        if poisson_pred:
            stat_line+=(f"\nxG {poisson_pred.get('home_xg','?')}-{poisson_pred.get('away_xg','?')}"
                        f"  H{poisson_pred.get('home_win_prob_poisson',0)*100:.0f}%"
                        f" D{poisson_pred.get('draw_prob_poisson',0)*100:.0f}%"
                        f" A{poisson_pred.get('away_win_prob_poisson',0)*100:.0f}%")
    elif stats.get("historical_data"):
        pa=stats["historical_data"].get("player_a",{}); pb=stats["historical_data"].get("player_b",{}); h2h=stats["historical_data"].get("h2h",{})
        stat_line=(f"\n{he}: #{pa.get('current_ranking','?')}  WR {pa.get('recent_win_rate',0)*100:.0f}%  {pa.get('recent_form','?')[:6]}\n"
                   f"{ae}: #{pb.get('current_ranking','?')}  WR {pb.get('recent_win_rate',0)*100:.0f}%  {pb.get('recent_form','?')[:6]}")
        if h2h.get("total",0)>0: stat_line+=f"\nH2H {h2h['total']} matches — {h2h.get('dominance','balanced')}"
    elif stats.get("us_sports"):
        us=stats["us_sports"]; hs=us.get("home",{}); aw=us.get("away",{})
        stat_line=(f"\n{he}: WR {hs.get('win_rate',hs.get('win_pct',0))*100:.0f}%  {hs.get('form',hs.get('recent_record','N/A'))}\n"
                   f"{ae}: WR {aw.get('win_rate',aw.get('win_pct',0))*100:.0f}%  {aw.get('form',aw.get('recent_record','N/A'))}")
    if ml_pred:
        ml_parts=[f"{k.replace('_win_prob','').replace('_',' ').title()} {v*100:.0f}%" for k,v in ml_pred.items() if isinstance(v,float) and "prob" in k]
        if ml_parts: stat_line+=f"\nML: {' | '.join(ml_parts)}"

    return (f"{ai.get('sport_emoji','🏆')} <b>{sport_en}</b>\n\n"
            f"<b>{he}</b> vs <b>{ae}</b>   {countdown(ct,now_utc)}\n\n"
            f"Pick:\n<code>{sharp}{pick_esc} @ <b>{opp['odds']:.2f}</b></code>\n"
            f"   Stake: <b>{opp.get('kelly_pct',0):.1f}%</b>\n\n"
            f"Signal: <b>{conf_label(fc)}</b> ({fc}%)   Risk: <b>{risk_en}</b>\n\n"
            f"<blockquote>{logic}</blockquote>"
            f"{html_lib.escape(stat_line) if stat_line else ''}\n\n"
            f"<i>{html_lib.escape(CFG.TELEGRAM_ID)}</i>")


# =========================================================
# 22. ODDS FETCHER
# =========================================================
class OddsCache:
    def __init__(self):
        self._fp=CFG.ODDS_CACHE_FILE; self.cache=Cache.load(self._fp)
    def _key(self,wh):
        raw=f"{','.join(sorted(CFG.ODDS_API_MARKETS))}|{wh}|{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H')}"
        return hashlib.md5(raw.encode()).hexdigest()
    def get(self,wh):
        k=self._key(wh)
        if Cache.valid(self.cache,k,CFG.TTL_ODDS_CACHE_MINUTES/60):
            d=Cache.get(self.cache,k)
            if d: logger.info("💾 [ODDS CACHE] HIT %d events",len(d)); return d
        return None
    def save(self,wh,events):
        k=self._key(wh); self.cache=Cache.set(self.cache,k,events); Cache.save(self._fp,self.cache)
        logger.info("💾 [ODDS CACHE] Saved %d events",len(events))
    def get_stale(self,wh,max_ttl=2.):
        k=self._key(wh)
        if Cache.valid(self.cache,k,max_ttl): return Cache.get(self.cache,k)
        return None

odds_cache=OddsCache()


async def fetch_market_async(session, market, now_utc, api_key, label):
    end=now_utc+timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    params={"apiKey":api_key,"regions":CFG.ODDS_API_REGIONS,"markets":market,
            "oddsFormat":"decimal","dateFormat":"iso"}
    try:
        async with session.get("https://api.the-odds-api.com/v4/sports/upcoming/odds",
                               params=params,timeout=aiohttp.ClientTimeout(total=25)) as r:
            remaining=int(r.headers.get("x-requests-remaining",-1))
            used=int(r.headers.get("x-requests-used",-1))
            if r.status==200:
                events=await r.json(content_type=None)
                valid=[e for e in events if isinstance(e,dict) and
                       now_utc<=datetime.fromisoformat(e.get("commence_time","").replace("Z","+00:00")).replace(tzinfo=timezone.utc)<=end]
                logger.info("🔑 [%s] %s → %d events | used:%d remaining:%d",
                            label,market,len(valid),used,remaining)
                return valid,200,None,remaining,used
            err=await r.text()
            reasons={401:"Invalid key",402:"Quota exhausted",429:"Rate limited"}
            return [],[],r.status,reasons.get(r.status,f"HTTP {r.status}"),-1,-1
    except asyncio.TimeoutError: return [],0,"Timeout",-1,-1
    except Exception as e:       return [],0,str(e)[:60],-1,-1


async def fetch_all_odds() -> list:
    now=datetime.now(timezone.utc)
    cached=odds_cache.get(CFG.MATCH_WINDOW_HOURS)
    if cached: return cached

    logger.info("📡 [ODDS] Fetching from API...")
    all_events: Dict[str,dict]={}; pending=list(CFG.ODDS_API_MARKETS)
    total_used=0; min_remaining=-1

    for ki in odds_key_mgr.get_active():
        if not pending: break
        ak=ki["key"]; label=ki["label"]; ki["calls"]+=1
        conn=aiohttp.TCPConnector(limit=10,ssl=False)
        async with aiohttp.ClientSession(connector=conn) as sess:
            tasks=[fetch_market_async(sess,m,now,ak,label) for m in pending]
            results=await asyncio.gather(*tasks,return_exceptions=True)

        failed=[]; hard_fail=None
        for i,res in enumerate(results):
            m=pending[i]
            if isinstance(res,Exception): failed.append(m); continue
            if len(res)==5: events,status,err,remaining,used = res[0],res[1],res[2],res[3],res[4]
            else: events,status,err = res[0],res[1],res[2]; remaining=-1; used=-1
            if status==200:
                if remaining>=0: ki["remaining"]=remaining; odds_key_mgr.update_remaining(label,remaining)
                if used>=0: total_used+=used
                if min_remaining<0 or (remaining>=0 and remaining<min_remaining): min_remaining=remaining
                for e in events:
                    eid=e.get("id")
                    if not eid: continue
                    if eid not in all_events:
                        all_events[eid]={k:e.get(k,"") for k in ["id","sport_key","sport_title","commence_time","home_team","away_team"]}
                        all_events[eid]["_markets_data"]={}
                    for bm in e.get("bookmakers",[]):
                        bk=bm.get("key",""); bt=bm.get("title",bk)
                        for md in bm.get("markets",[]):
                            mk2=md.get("key","")
                            if mk2: all_events[eid]["_markets_data"].setdefault(mk2,[]).append({"bookmaker":bt,"bookmaker_key":bk,"outcomes":md.get("outcomes",[])})
            else:
                failed.append(m)
                if status in [401,402,429]:
                    idx=next((i for i,k in enumerate(odds_key_mgr.keys) if k["label"]==label),-1)
                    if idx>=0: odds_key_mgr.mark_failed(idx,f"HTTP {status}")
                    hard_fail=status
        pending=failed

    final=list(all_events.values())
    if final: odds_cache.save(CFG.MATCH_WINDOW_HOURS,final)
    else:
        stale=odds_cache.get_stale(CFG.MATCH_WINDOW_HOURS,2.)
        if stale: logger.warning("💾 [STALE] Using stale cache: %d events",len(stale)); return stale

    used_str=f"used:{total_used}" if total_used>0 else "used:N/A"
    rem_str=f"remaining:{min_remaining}" if min_remaining>=0 else "remaining:N/A"
    logger.info("📊 Odds API usage → %s | %s", used_str, rem_str)
    logger.info("📊 Keys: %s", odds_key_mgr.summary())
    return final


# =========================================================
# 23. MAIN PIPELINE
# =========================================================
async def async_main():
    logger.info("="*65)
    logger.info("  ZBET90 ENGINE v9.1 | Production | Multi-Source")
    logger.info("="*65)
    sent=SentHistory(); now=datetime.now(timezone.utc)

    # ── Phase 1: Load data ────────────────────────────────
    logger.info("📥 [PHASE 1] Loading data sources...")
    de=FreeDataEngine()
    de.load_tennis_data(); de.load_football_data()
    de.load_nba_data(); de.load_nhl_data(); de.load_mlb_data()

    # Data source status report
    logger.info("─"*40)
    logger.info("📦 DATA SOURCE STATUS:")
    logger.info("  ├─ Tennis ATP:  %s", f"{len(de.atp_matches)} matches" if de.atp_matches is not None else "❌ unavailable")
    logger.info("  ├─ Tennis WTA:  %s", f"{len(de.wta_matches)} matches" if de.wta_matches is not None else "❌ unavailable")
    logger.info("  ├─ Football:    %s", f"{len(de.football_data.get('all',pd.DataFrame()))} matches" if de.football_data.get("all") is not None else "❌ unavailable")
    logger.info("  ├─ NBA:         %s", f"{len(de.nba_data)} teams" if de.nba_data is not None else "❌ unavailable")
    logger.info("  ├─ NHL:         %s", f"{len(de.nhl_data)} teams" if de.nhl_data is not None else "❌ unavailable")
    logger.info("  ├─ MLB:         %s", f"{len(de.mlb_data)} teams" if de.mlb_data is not None else "❌ unavailable")
    logger.info("  ├─ API-Football: %s", f"✅ active ({api_football._calls_today}/{CFG.API_FOOTBALL_MAX_CALLS} calls)" if CFG.API_FOOTBALL_KEY else "⚠️  no key")
    logger.info("  └─ FDOrg:       %s", "✅ active" if CFG.FOOTBALL_DATA_ORG_KEY else "⚠️  no key")
    logger.info("─"*40)

    # ── Phase 2: ML models ────────────────────────────────
    logger.info("🧠 [PHASE 2] Training/loading ML models...")
    ml=MLEngine(de)
    ml.load_or_train_football()
    ml.load_or_train_tennis(is_wta=False)
    ml.load_or_train_tennis(is_wta=True)
    logger.info("  ├─ Football ML:  %s", "✅ ready" if ml.is_football_trained else "❌ not trained")
    logger.info("  ├─ ATP ML:       %s", "✅ ready" if ml.tennis_pipelines.get("atp") else "❌ not trained")
    logger.info("  └─ WTA ML:       %s", "✅ ready" if ml.tennis_pipelines.get("wta") else "❌ not trained")

    # ── Phase 3: Odds ─────────────────────────────────────
    logger.info("📡 [PHASE 3] Fetching odds (%.1fh window)...", CFG.MATCH_WINDOW_HOURS)
    events=await fetch_all_odds()
    if not events: logger.info("❌ No events in window."); return

    logger.info("🔍 [PHASE 4] Analyzing %d events...", len(events))
    events.sort(key=lambda x: x.get("commence_time",""))

    total_sent=total_analyzed=0
    skip_counts={"no_opp":0,"ev":0,"sent":0,"math":0,"ai":0,"conf":0}

    for event in events:
        home=clean_name(event.get("home_team",""))
        away=clean_name(event.get("away_team",""))
        sport=event.get("sport_title","Unknown")
        sport_key=normalize_sport(sport)
        if not home or not away: continue

        opps=calculate_ev(event.get("_markets_data",{}))
        if not opps: skip_counts["no_opp"]+=1; continue
        opp=opps[0]; total_analyzed+=1

        if opp["ev"]<CFG.MATH_MIN_EV_TO_ANALYZE: skip_counts["ev"]+=1; continue
        if sent.was_sent(home,away,opp["market"]): skip_counts["sent"]+=1; logger.info("⏭️  SENT: %s vs %s",home,away); continue

        opp["steam_pct"]=line_tracker.record(home,away,opp["market"],opp["pick"],opp["odds"])

        # ── Data gathering ─────────────────────────────────────
        stats: dict={}; ml_pred=poisson_pred=None
        src_log=[]  # track which sources returned data

        if sport_key=="tennis":
            is_wta="wta" in sport.lower()
            try:
                ts=de.get_tennis_stats(home,away,is_wta)
                if ts:
                    ts["tour"]="wta" if is_wta else "atp"
                    stats["historical_data"]=ts
                    dq=ts.get("data_quality_summary",{}).get("overall","?")
                    src_log.append(f"GitHub-Tennis(Q={dq})")
            except Exception as e: logger.warning("  [Tennis GitHub] %s",e)
            for p_name,p_key in [(home,"player_a"),(away,"player_b")]:
                try:
                    tp=tsdb.get_player_stats(p_name)
                    if tp and tp.get("player_id"):
                        if "historical_data" in stats:
                            stats["historical_data"][p_key]["tsdb_nationality"]=tp.get("nationality","")
                        src_log.append(f"TSDB-{p_name[:10]}")
                except Exception: pass
            if ml.is_tennis_trained and ts:
                sl=sport.lower()
                surf="grass" if any(k in sl for k in ["wimbledon","grass","queens"]) else "clay" if any(k in sl for k in ["clay","roland","monte"]) else "hard"
                ml_pred=ml.predict_tennis(home,away,ts,surf)
                if ml_pred: stats["ml_prediction"]=ml_pred; src_log.append("ML-Tennis")

        elif sport_key=="football":
            # 1. API-Football (premium stats if key available)
            if CFG.API_FOOTBALL_KEY:
                try:
                    h_af=api_football.get_team_stats(home); a_af=api_football.get_team_stats(away)
                    if h_af or a_af:
                        h2h_af=api_football.get_h2h(home,away) if h_af and a_af else {}
                        stats["football_stats"]={"home":h_af or {},"away":a_af or {},"h2h":h2h_af}
                        src_log.append(f"API-Football(H={h_af.get('data_quality','?') if h_af else 'none'},A={a_af.get('data_quality','?') if a_af else 'none'},H2H={h2h_af.get('total',0)})")
                        if h_af and h_af.get("league_id"):
                            h_st=api_football.get_team_standing(home,h_af["league_id"])
                            if h_st: stats["football_stats"]["home"]["standing"]=h_st
                        if a_af and a_af.get("league_id"):
                            a_st=api_football.get_team_standing(away,a_af["league_id"])
                            if a_st: stats["football_stats"]["away"]["standing"]=a_st
                except Exception as e: logger.warning("  [API-Football] %s",str(e)[:80])

            # 2. OpenLigaDB — free, no auth, real-time standings + H2H
            try:
                h_ol = openligadb.get_team_stats(home)
                a_ol = openligadb.get_team_stats(away)
                if h_ol or a_ol:
                    if "football_stats" not in stats:
                        stats["football_stats"] = {"home": h_ol or {}, "away": a_ol or {}, "h2h": {}}
                    else:
                        # Enrich: fill in form/standing if API-Football didn't supply
                        if h_ol and not stats["football_stats"]["home"].get("form"):
                            stats["football_stats"]["home"].update({
                                "form": h_ol.get("form",""), "win_rate": h_ol.get("win_rate",0),
                                "avg_scored": h_ol.get("avg_scored",0), "avg_conceded": h_ol.get("avg_conceded",0),
                                "standing": {"rank": h_ol.get("rank",0), "points": h_ol.get("points",0),
                                             "played": h_ol.get("played",0)}})
                        if a_ol and not stats["football_stats"]["away"].get("form"):
                            stats["football_stats"]["away"].update({
                                "form": a_ol.get("form",""), "win_rate": a_ol.get("win_rate",0),
                                "avg_scored": a_ol.get("avg_scored",0), "avg_conceded": a_ol.get("avg_conceded",0),
                                "standing": {"rank": a_ol.get("rank",0), "points": a_ol.get("points",0),
                                             "played": a_ol.get("played",0)}})
                    h2h_ol = openligadb.get_h2h(home, away)
                    if h2h_ol and not stats.get("football_stats",{}).get("h2h"):
                        stats["football_stats"]["h2h"] = h2h_ol
                    src_log.append(f"OpenLigaDB(H={h_ol.get('data_quality','?') if h_ol else 'none'},A={a_ol.get('data_quality','?') if a_ol else 'none'})")
            except Exception as e: logger.warning("  [OpenLigaDB] %s", str(e)[:80])

            # 3. Football-Data.org (fallback — 10 competitions, unlimited calls)
            if not stats.get("football_stats") and CFG.FOOTBALL_DATA_ORG_KEY:
                try:
                    h_fd=fdo.get_team_matches(home); a_fd=fdo.get_team_matches(away)
                    if h_fd or a_fd:
                        stats["football_stats"]={"home":h_fd or {},"away":a_fd or {},"h2h":{}}
                        src_log.append(f"FDOrg(H={h_fd.get('data_quality','?') if h_fd else 'none'},A={a_fd.get('data_quality','?') if a_fd else 'none'})")
                except Exception as e: logger.warning("  [FDOrg] %s",str(e)[:60])

            # 4. GitHub CSV (historical baseline + H2H for any league)
            try:
                fs=de.get_football_stats(home,away)
                if fs:
                    if "football_stats" not in stats:
                        stats["football_stats"]=fs
                        src_log.append(f"GitHub-Football(H={fs.get('home',{}).get('data_quality','?')},A={fs.get('away',{}).get('data_quality','?')})")
                    else:
                        stats["football_stats"]["github"]=fs
                        if not stats["football_stats"].get("h2h"): stats["football_stats"]["h2h"]=fs.get("h2h",{})
                        src_log.append("GitHub-Football(H2H+history)")
            except Exception as e: logger.warning("  [GitHub Football] %s",str(e)[:60])

            # 5. TSDB form enrich (last resort for any league not covered above)
            try:
                h_ts=tsdb.get_team_stats(home); a_ts=tsdb.get_team_stats(away)
                if h_ts or a_ts:
                    if "football_stats" not in stats:
                        stats["football_stats"]={"home":h_ts or {},"away":a_ts or {},"h2h":{}}
                    else:
                        if h_ts and not stats["football_stats"]["home"].get("form"):
                            stats["football_stats"]["home"].update({"tsdb_form":h_ts.get("form",""),"tsdb_win_rate":h_ts.get("win_rate",0)})
                        if a_ts and not stats["football_stats"]["away"].get("form"):
                            stats["football_stats"]["away"].update({"tsdb_form":a_ts.get("form",""),"tsdb_win_rate":a_ts.get("win_rate",0)})
                    src_log.append(f"TSDB(H={h_ts.get('data_quality','?') if h_ts else 'none'},A={a_ts.get('data_quality','?') if a_ts else 'none'})")
            except Exception as e: logger.warning("  [TSDB Football] %s",str(e)[:60])

            # 6. ML + Poisson
            if ml.is_football_trained:
                try:
                    ml_pred=ml.predict_football(home,away)
                    if ml_pred: stats["ml_prediction"]=ml_pred; src_log.append(f"ML-Football({','.join(f'{k}:{v*100:.0f}%' for k,v in ml_pred.items() if isinstance(v,float))})")
                except Exception as e: logger.warning("  [ML Football] %s",e)
            try:
                poisson_pred=PoissonEngine.calculate(home,away,de.football_data.get("all"))
                if poisson_pred: stats["poisson_prediction"]=poisson_pred; src_log.append(f"Poisson(H={poisson_pred.get('home_win_prob_poisson',0)*100:.0f}%,D={poisson_pred.get('draw_prob_poisson',0)*100:.0f}%,A={poisson_pred.get('away_win_prob_poisson',0)*100:.0f}%)")
            except Exception as e: logger.warning("  [Poisson] %s",e)

        elif sport_key in ("basketball","baseball","hockey"):
            is_wnba = "wnba" in sport.lower() or ("women" in sport.lower() and "basketball" in sport.lower())
            try:
                hs  = de.get_us_sports_stats(sport, home)
                aws = de.get_us_sports_stats(sport, away)
                if hs or aws:
                    stats["us_sports"] = {"home": hs or {}, "away": aws or {}}
                    h_src = hs.get("source","?") if hs else "none"
                    a_src = aws.get("source","?") if aws else "none"
                    h_dq  = hs.get("data_quality","?") if hs else "none"
                    a_dq  = aws.get("data_quality","?") if aws else "none"
                    src_log.append(f"US-Sports(H={h_src}/{h_dq},A={a_src}/{a_dq})")
                else:
                    logger.info("  ⚠️  No US sports data found for %s vs %s [%s]", home, away, sport)
            except Exception as e:
                logger.warning("  [US Sports pipeline] %s", e)
            # Derive win-rate ML
            us  = stats.get("us_sports", {})
            hs  = us.get("home", {}); aws = us.get("away", {})
            h_wr = hs.get("win_rate", hs.get("win_pct", 0))
            a_wr = aws.get("win_rate", aws.get("win_pct", 0))
            if h_wr and a_wr:
                cap   = 0.80 if sport_key == "basketball" else 0.75
                floor = 0.20 if sport_key == "basketball" else 0.25
                hp    = min(cap, max(floor, (h_wr / (h_wr + a_wr)) * 0.9 + 0.05))
                ml_pred = {f"{home}_win_prob": round(hp, 4), f"{away}_win_prob": round(1-hp, 4)}
                stats["ml_prediction"] = ml_pred
                src_log.append(f"ML-WinRate(H={hp*100:.0f}%,A={(1-hp)*100:.0f}%)")

        else:  # cricket / NPB / other
            # Try US sports lookup first (covers NPB etc. via TSDB)
            try:
                hs  = de.get_us_sports_stats(sport, home)
                aws = de.get_us_sports_stats(sport, away)
                if hs or aws:
                    stats["us_sports"] = {"home": hs or {}, "away": aws or {}}
                    src_log.append(f"TSDB-Other(H={hs.get('data_quality','?') if hs else 'none'},A={aws.get('data_quality','?') if aws else 'none'})")
                else:
                    # Direct TSDB fallback
                    h_ts = tsdb.get_team_stats(home); a_ts = tsdb.get_team_stats(away)
                    if h_ts or a_ts:
                        stats["tsdb_stats"] = {"home": h_ts or {}, "away": a_ts or {}}
                        src_log.append(f"TSDB(H={h_ts.get('data_quality','?') if h_ts else 'none'},A={a_ts.get('data_quality','?') if a_ts else 'none'})")
            except Exception as e:
                logger.warning("  [Other sports pipeline] %s", e)

        logger.info("  📦 [%s vs %s] Sources: %s",
                    home[:20], away[:20], " | ".join(src_log) if src_log else "NONE ⚠️")

        # ── Math score ──────────────────────────────────────
        math_score=ConfidenceEngine.score(opp,stats,ml_pred,poisson_pred)
        min_math=50 if sport_key=="other" else CFG.MIN_MATH_SCORE_TO_CALL_AI
        if math_score<min_math:
            skip_counts["math"]+=1
            logger.info("⏭️  SKIP(math:%d<%d) %s vs %s EV=%.2f%%",math_score,min_math,home,away,opp["edge_pct"])
            continue

        # ── AI decision ──────────────────────────────────────
        ai=make_ai_decision(home,away,sport,sport_key,opp,stats,math_score,ml_pred,poisson_pred)
        fc=ai["final_confidence"]
        if ai.get("decision")=="SKIP":
            skip_counts["ai"]+=1
            logger.info("⏭️  AI_SKIP: %s vs %s Math:%d AI:%d Final:%d",home,away,math_score,ai["ai_confidence"],fc); continue
        if fc<CFG.MIN_CONFIDENCE_TO_SEND:
            skip_counts["conf"]+=1
            logger.info("⏭️  SKIP(conf:%d<%d) %s vs %s",fc,CFG.MIN_CONFIDENCE_TO_SEND,home,away); continue

        logger.info("✅ SIGNAL: %s vs %s | Math:%d AI:%d Final:%d | EV=%.2f%% | Kelly=%.1f%% | Sharp=%s",
                    home,away,math_score,ai["ai_confidence"],fc,opp["edge_pct"],opp.get("kelly_pct",0),opp.get("has_sharp_line",False))

        msg=build_message(home,away,sport,sport_key,opp,ai,stats,math_score,ml_pred,poisson_pred,now,event.get("commence_time",""))
        if send_telegram(msg):
            sent.mark_sent(home,away,opp["pick"],opp["market"])
            perf_tracker.record(home,away,opp["pick"],opp["market"],opp["odds"],opp["ev"],fc,opp["prob"],sport_key,event.get("sport_key",""))
            total_sent+=1
            logger.info("📤 SENT: %s vs %s | EV=%.2f%% | Conf=%d%%",home,away,opp["edge_pct"],fc)
        else:
            logger.error("❌ Telegram failed: %s vs %s",home,away)
        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    # ── Final report ─────────────────────────────────────
    logger.info("="*65)
    logger.info("📊 FINAL REPORT:")
    logger.info("  Events total:   %d", len(events))
    logger.info("  Analyzed:       %d", total_analyzed)
    logger.info("  Signals sent:   %d", total_sent)
    logger.info("  Skip breakdown: no_opp=%d ev=%d sent=%d math=%d ai=%d conf=%d",
                skip_counts["no_opp"],skip_counts["ev"],skip_counts["sent"],
                skip_counts["math"],skip_counts["ai"],skip_counts["conf"])
    logger.info("  API-Football:   %d/%d calls used today", api_football._calls_today, CFG.API_FOOTBALL_MAX_CALLS)
    logger.info("  Keys:           %s", odds_key_mgr.summary())
    logger.info("="*65)


if __name__=="__main__":
    try: asyncio.run(async_main())
    except KeyboardInterrupt: logger.info("Stopped.")
    except Exception as e: logger.critical("SYSTEM FAILURE: %s",e,exc_info=True); sys.exit(1)
