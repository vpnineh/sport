# =========================================================
# ZBET90 ENGINE v9.0 | Complete Rewrite | Production Grade
# =========================================================
# File: main.py
# =========================================================
import os, sys, time, json, re, random, logging, html as html_lib
import hashlib, asyncio, aiohttp, requests, numpy as np, pandas as pd
import pickle, warnings, threading, difflib
from io import StringIO
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict, deque
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import scipy.stats as stats_scipy
from scipy.optimize import brentq

# =========================================================
# 1. CONFIG
# =========================================================
@dataclass
class Config:
    # Directories
    CACHE_DIR: Path = Path("api_cache")
    LOG_DIR: Path = Path("log")
    HISTORICAL_DIR: Path = Path("api_cache/historical")
    ML_DIR: Path = Path("api_cache/ml_models")
    HISTORY_FILE: Path = Path("api_cache/sent_history.json")
    ODDS_CACHE_FILE: Path = Path("api_cache/odds_cache.json")
    API_USAGE_FILE: Path = Path("api_cache/api_usage_tracker.json")
    PERFORMANCE_FILE: Path = Path("api_cache/performance_tracker.json")
    LOG_FILE: Path = Path("api_cache/execution_logs.log")

    # Match window
    MATCH_WINDOW_HOURS: float = 10.0

    # Odds API
    ODDS_API_MARKETS: List[str] = field(default_factory=lambda: ["h2h", "totals"])
    ODDS_API_REGIONS: str = "eu,us,uk,au"
    TTL_ODDS_CACHE_MINUTES: float = 10.0
    TTL_SENT_HISTORY: float = 48.0
    TTL_GITHUB_DATA: float = 12.0

    # EV thresholds
    H2H_MIN_ODDS: float = 1.30
    H2H_MIN_EV: float = 0.010
    TOTALS_MIN_ODDS: float = 1.40
    TOTALS_MIN_EV: float = 0.012
    MAX_REALISTIC_EV: float = 0.18
    MATH_MIN_EV_TO_ANALYZE: float = 0.008
    MAX_VALID_IMPLIED_SUM: float = 1.20
    MIN_VALID_IMPLIED_SUM: float = 0.65

    # Kelly
    KELLY_FRACTION: float = 0.25
    MAX_KELLY_PCT: float = 5.0

    # Pipeline thresholds
    MIN_MATH_SCORE_TO_CALL_AI: int = 32
    MIN_CONFIDENCE_TO_SEND: int = 60
    HIGH_CONFIDENCE: int = 75
    MEDIUM_CONFIDENCE: int = 60

    # AI settings
    AI_WEIGHT: float = 0.50
    MATH_WEIGHT: float = 0.50
    MAX_AI_BOOST: int = 18
    MAX_AI_PENALTY: int = 12
    AI_MODEL: str = "gemini-2.0-flash"
    AI_MAX_TOKENS: int = 2000
    AI_TEMPERATURE: float = 0.05

    # Sharp bookmakers
    SHARP_BOOKMAKERS: List[str] = field(default_factory=lambda: [
        "pinnacle", "betfair_ex_eu", "matchbook", "betfair_ex_uk",
        "sport888", "betsson", "nordicbet", "unibet_eu"
    ])

    # API-Football
    API_FOOTBALL_KEY: str = ""
    API_FOOTBALL_TTL: float = 6.0       # hours
    API_FOOTBALL_MAX_CALLS: int = 95    # keep 5 buffer from 100/day

    # football-data.org
    FOOTBALL_DATA_ORG_KEY: str = ""
    FOOTBALL_DATA_ORG_TTL: float = 6.0

    # TheSportsDB
    TSDB_API_KEY: str = "123"

    # BallDontLie (NBA)
    BALLDONTLIE_API_KEY: str = ""

    # Market validation
    MARKET_EXPECTED_OUTCOMES: Dict[str, Any] = field(default_factory=lambda: {
        "h2h": {"min": 2, "max": 3},
        "totals": {"min": 2, "max": 2}
    })

    TELEGRAM_ID: str = "@zBET90"
    TELEGRAM_SLEEP_BETWEEN: float = 3.0

    # GitHub data sources
    GITHUB_SOURCES: Dict[str, Any] = field(default_factory=lambda: {
        "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
        "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
        "atp_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv",
        "wta_rankings": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv",
        "football_eu": "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv",
    })

    FOOTBALL_DATA_UK_LEAGUES: Dict[str, str] = field(default_factory=lambda: {
        "E0": "Premier League", "E1": "Championship",
        "D1": "Bundesliga", "SP1": "La Liga",
        "I1": "Serie A", "F1": "Ligue 1",
        "N1": "Eredivisie", "P1": "Liga Portugal",
    })

    FOOTBALL_DATA_UK_SEASONS: List[str] = field(default_factory=lambda: [
        "2223", "2324", "2425"
    ])


CFG = Config()
# Load API keys from environment
CFG.API_FOOTBALL_KEY = os.getenv("API_FOOTBALL", "").strip()
CFG.FOOTBALL_DATA_ORG_KEY = os.getenv("FOOTBALL_DATA_ORG_KEY", "").strip()
CFG.BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "").strip()

# =========================================================
# 2. LOGGING
# =========================================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
for d in [CFG.CACHE_DIR, CFG.LOG_DIR, CFG.HISTORICAL_DIR, CFG.ML_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ZBET90_v9")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)
_fh = logging.FileHandler(CFG.LOG_FILE, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

# =========================================================
# 3. THESPORTSDB CLIENT (Free API - Primary Data Source)
# =========================================================
class TheSportsDBClient:
    """
    Free TheSportsDB API client.
    Key "3" = free tier with rate limits.
    Docs: https://www.thesportsdb.com/documentation
    """
    BASE = "https://www.thesportsdb.com/api/v1/json/{key}"
    BASE2 = "https://www.thesportsdb.com/api/v2/json/{key}"

    # Sport name → TheSportsDB sport name
    SPORT_MAP = {
        "football": "Soccer",
        "soccer": "Soccer",
        "tennis": "Tennis",
        "basketball": "Basketball",
        "baseball": "Baseball",
        "hockey": "Ice Hockey",
        "cricket": "Cricket",
        "american football": "American Football",
        "rugby": "Rugby",
    }

    # Popular league IDs for direct lookup
    LEAGUE_IDS = {
        "Premier League": 4328,
        "La Liga": 4335,
        "Bundesliga": 4331,
        "Serie A": 4332,
        "Ligue 1": 4334,
        "Champions League": 4480,
        "Europa League": 4481,
        "MLS": 4346,
        "NBA": 4387,
        "MLB": 4424,
        "NHL": 4380,
        "NFL": 4391,
        "ATP": 4659,
        "WTA": 4658,
        "IPL": 4425,
    }

    def __init__(self):
        self._key = os.getenv("TSDB_API_KEY", CFG.TSDB_API_KEY)
        self._cache: Dict[str, Any] = {}
        self._cache_file = CFG.CACHE_DIR / "tsdb_cache.json"
        self._load_cache()
        self._last_call = 0.0
        self._min_interval = 0.5  # 500ms between calls (free tier limit)

    def _load_cache(self):
        try:
            if self._cache_file.exists():
                raw = json.loads(self._cache_file.read_text())
                now = datetime.now(timezone.utc)
                for k, v in raw.items():
                    if isinstance(v, dict) and "ts" in v:
                        ts = datetime.fromisoformat(v["ts"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if (now - ts) < timedelta(hours=6):
                            self._cache[k] = v
        except Exception:
            pass

    def _save_cache(self):
        try:
            self._cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, default=str)
            )
        except Exception:
            pass

    def _get(self, endpoint: str, params: dict = None,
             cache_ttl_hours: float = 6.0) -> Optional[dict]:
        """Make a throttled GET request with caching."""
        ck = hashlib.md5(f"{endpoint}|{json.dumps(params or {}, sort_keys=True)}".encode()).hexdigest()

        # Check cache
        if ck in self._cache:
            try:
                ts = datetime.fromisoformat(self._cache[ck]["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - ts) < timedelta(hours=cache_ttl_hours):
                    return self._cache[ck].get("data")
            except Exception:
                pass

        # Rate limiting
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        base = self.BASE.format(key=self._key)
        url = f"{base}/{endpoint}"

        try:
            r = requests.get(url, params=params,
                             timeout=15,
                             headers={"User-Agent": "ZBET90/9.0"})
            self._last_call = time.time()

            if r.status_code == 200:
                data = r.json()
                self._cache[ck] = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "data": data
                }
                self._save_cache()
                return data
            else:
                logger.debug("[TSDB] %s → HTTP %d", endpoint, r.status_code)
        except Exception as e:
            logger.debug("[TSDB] %s → %s", endpoint, str(e)[:60])

        return None

    def search_team(self, team_name: str) -> Optional[dict]:
        """Search for a team by name."""
        data = self._get(f"searchteams.php", {"t": team_name})
        if data and data.get("teams"):
            return data["teams"][0]
        return None

    def search_player(self, player_name: str) -> Optional[dict]:
        """Search for a player."""
        data = self._get(f"searchplayers.php", {"p": player_name})
        if data and data.get("player"):
            return data["player"][0]
        return None

    def get_team_last_events(self, team_id: str, count: int = 10) -> List[dict]:
        """Get last N events for a team."""
        data = self._get(f"eventslast.php", {"id": team_id})
        if data and data.get("results"):
            return data["results"][:count]
        return []

    def get_team_next_events(self, team_id: str, count: int = 5) -> List[dict]:
        """Get next N events for a team."""
        data = self._get(f"eventsnext.php", {"id": team_id})
        if data and data.get("events"):
            return data["events"][:count]
        return []

    def get_league_events_on_date(self, league_id: int, date_str: str) -> List[dict]:
        """Get events for a league on a specific date (YYYY-MM-DD)."""
        data = self._get(f"eventsday.php",
                         {"d": date_str, "l": str(league_id)},
                         cache_ttl_hours=1.0)
        if data and data.get("events"):
            return data["events"]
        return []

    def get_events_on_date(self, date_str: str, sport: str = None) -> List[dict]:
        """Get all events on a date, optionally filtered by sport."""
        params = {"d": date_str}
        if sport:
            params["s"] = sport
        data = self._get(f"eventsday.php", params, cache_ttl_hours=1.0)
        if data and data.get("events"):
            return data["events"]
        return []

    def get_team_stats(self, team_name: str) -> dict:
        """
        Get comprehensive team stats from TheSportsDB.
        Returns form, league position, recent results.
        """
        result = {}

        # 1. Find team
        team = self.search_team(team_name)
        if not team:
            logger.debug("[TSDB] Team not found: %s", team_name)
            return result

        team_id = team.get("idTeam")
        if not team_id:
            return result

        result["team_id"] = team_id
        result["team_name"] = team.get("strTeam", team_name)
        result["country"] = team.get("strCountry", "")
        result["league"] = team.get("strLeague", "")
        result["stadium"] = team.get("strStadium", "")

        # 2. Get last events
        last_events = self.get_team_last_events(team_id, 15)
        if last_events:
            form_str = ""
            wins = draws = losses = 0
            goals_scored = goals_conceded = 0

            for ev in last_events:
                h_score_str = ev.get("intHomeScore", "")
                a_score_str = ev.get("intAwayScore", "")
                if h_score_str is None or a_score_str is None:
                    continue
                try:
                    hs = int(float(h_score_str))
                    as_ = int(float(a_score_str))
                except (ValueError, TypeError):
                    continue

                # Determine if team is home or away
                h_team = ev.get("strHomeTeam", "")
                a_team = ev.get("strAwayTeam", "")
                is_home = _fuzzy_match_name(team_name, h_team)

                if is_home:
                    scored, conceded = hs, as_
                else:
                    scored, conceded = as_, hs

                goals_scored += scored
                goals_conceded += conceded

                if scored > conceded:
                    form_str += "W"
                    wins += 1
                elif scored == conceded:
                    form_str += "D"
                    draws += 1
                else:
                    form_str += "L"
                    losses += 1

            total = wins + draws + losses
            if total > 0:
                result["form"] = form_str[:10]
                result["win_rate"] = round(wins / total, 3)
                result["draw_rate"] = round(draws / total, 3)
                result["loss_rate"] = round(losses / total, 3)
                result["avg_scored"] = round(goals_scored / total, 2)
                result["avg_conceded"] = round(goals_conceded / total, 2)
                result["matches_analyzed"] = total
                result["data_quality"] = (
                    "good" if total >= 10
                    else "limited" if total >= 5
                    else "poor"
                )

        return result

    def get_player_stats(self, player_name: str) -> dict:
        """Get player stats from TheSportsDB."""
        result = {}
        player = self.search_player(player_name)
        if not player:
            return result

        result["player_id"] = player.get("idPlayer")
        result["player_name"] = player.get("strPlayer", player_name)
        result["nationality"] = player.get("strNationality", "")
        result["sport"] = player.get("strSport", "")
        result["position"] = player.get("strPosition", "")
        result["description"] = (player.get("strDescriptionEN") or "")[:200]

        return result

    def get_upcoming_events_for_match(self, home: str, away: str,
                                      date_str: str = None) -> Optional[dict]:
        """
        Try to find a specific upcoming match between two teams.
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Search by home team
        home_team = self.search_team(home)
        if not home_team:
            return None

        team_id = home_team.get("idTeam")
        if not team_id:
            return None

        next_events = self.get_team_next_events(team_id, 10)
        for ev in next_events:
            a_team = ev.get("strAwayTeam", "")
            if _fuzzy_match_name(away, a_team):
                return ev

        return None

    def get_league_table(self, league_id: int, season: str = None) -> List[dict]:
        """Get league standings table."""
        if season is None:
            season = f"{datetime.now().year}-{datetime.now().year + 1}"

        data = self._get(f"lookuptable.php",
                         {"l": str(league_id), "s": season},
                         cache_ttl_hours=4.0)
        if data and data.get("table"):
            return data["table"]
        return []

    def get_h2h(self, team1_id: str, team2_id: str) -> List[dict]:
        """Get head-to-head results."""
        # Note: Only available in premium tier
        # Use free alternative: search events
        data = self._get(f"searchevents.php",
                         {"e": "", "s": ""},
                         cache_ttl_hours=12.0)
        return []

    def get_events_results_by_date(self, date_str: str) -> List[dict]:
        """Get all completed event results for a date."""
        data = self._get(f"eventsday.php",
                         {"d": date_str},
                         cache_ttl_hours=2.0)
        if data and data.get("events"):
            return [e for e in data["events"]
                    if e.get("strStatus") in
                    ("Match Finished", "FT", "AET", "After ET",
                     "Finished", "Complete", "ft", "aet")]
        return []


# Singleton
tsdb = TheSportsDBClient()


def _fuzzy_match_name(name1: str, name2: str, threshold: float = 0.45) -> bool:
    """Fuzzy name matching."""
    if not name1 or not name2:
        return False
    n1 = _normalize_name(name1)
    n2 = _normalize_name(name2)
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    # Token overlap
    t1 = set(n1.split())
    t2 = set(n2.split())
    if t1 and t2:
        overlap = len(t1 & t2) / max(len(t1), len(t2))
        if overlap >= threshold:
            return True
    return False


def _normalize_name(name: str) -> str:
    """Normalize team/player name for comparison."""
    import unicodedata
    if not name:
        return ""
    s = str(name).lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Remove common noise words
    for noise in ["fc", "cf", "sc", "ac", "bk", "fk", "if", "the",
                  "united", "city", "real", "atletico"]:
        s = re.sub(rf"\b{noise}\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

# =========================================================
# 3b. API-FOOTBALL CLIENT (api-football.com)
# Free: 100 calls/day — cache aggressively
# =========================================================
class APIFootballClient:
    BASE = "https://v3.football.api-sports.io"
    HEADERS_KEY = "x-apisports-key"

    # Map our sport_key/league names to API-Football league IDs
    LEAGUE_MAP = {
        "Premier League": 39,
        "Championship": 40,
        "La Liga": 140,
        "Bundesliga": 78,
        "Serie A": 135,
        "Ligue 1": 61,
        "Eredivisie": 88,
        "Liga Portugal": 94,
        "Champions League": 2,
        "Europa League": 3,
        "MLS": 253,
        "Brasileirao": 71,
        "Argentine Liga": 128,
        "Super Lig": 203,
        "Jupiler League": 144,
    }

    def __init__(self):
        self._key = CFG.API_FOOTBALL_KEY
        self._cache_file = CFG.CACHE_DIR / "api_football_cache.json"
        self._usage_file = CFG.CACHE_DIR / "api_football_usage.json"
        self._cache: dict = {}
        self._calls_today = 0
        self._load_cache()
        self._load_usage()
        if self._key:
            logger.info("✅ [API-FOOTBALL] Key loaded (calls today: %d/%d)",
                        self._calls_today, CFG.API_FOOTBALL_MAX_CALLS)
        else:
            logger.warning("⚠️ [API-FOOTBALL] No key — skipping")

    def _load_cache(self):
        try:
            if self._cache_file.exists():
                raw = json.loads(self._cache_file.read_text())
                now = datetime.now(timezone.utc)
                for k, v in raw.items():
                    if isinstance(v, dict) and "ts" in v:
                        ts = datetime.fromisoformat(v["ts"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if (now - ts) < timedelta(hours=CFG.API_FOOTBALL_TTL):
                            self._cache[k] = v
        except Exception:
            pass

    def _save_cache(self):
        try:
            self._cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, default=str)
            )
        except Exception:
            pass

    def _load_usage(self):
        try:
            if self._usage_file.exists():
                u = json.loads(self._usage_file.read_text())
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if u.get("date") == today:
                    self._calls_today = u.get("calls", 0)
                    return
        except Exception:
            pass
        self._calls_today = 0

    def _save_usage(self):
        try:
            self._usage_file.write_text(json.dumps({
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "calls": self._calls_today
            }))
        except Exception:
            pass

    def _get(self, endpoint: str, params: dict = None,
             ttl_hours: float = None) -> Optional[dict]:
        if not self._key:
            return None
        if self._calls_today >= CFG.API_FOOTBALL_MAX_CALLS:
            logger.warning("[API-FOOTBALL] Daily limit reached (%d)", self._calls_today)
            return None

        ttl = ttl_hours or CFG.API_FOOTBALL_TTL
        ck = hashlib.md5(
            f"{endpoint}|{json.dumps(params or {}, sort_keys=True)}".encode()
        ).hexdigest()

        # Check cache
        if ck in self._cache:
            try:
                ts = datetime.fromisoformat(self._cache[ck]["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - ts) < timedelta(hours=ttl):
                    return self._cache[ck].get("data")
            except Exception:
                pass

        try:
            r = requests.get(
                f"{self.BASE}/{endpoint}",
                params=params,
                headers={
                    self.HEADERS_KEY: self._key,
                    "Accept": "application/json"
                },
                timeout=15
            )
            self._calls_today += 1
            self._save_usage()

            remaining = r.headers.get("x-ratelimit-requests-remaining", "?")
            logger.debug("[API-FOOTBALL] %s → HTTP %d (rem: %s)",
                         endpoint, r.status_code, remaining)

            if r.status_code == 200:
                data = r.json()
                if data.get("errors") and data["errors"] != []:
                    logger.warning("[API-FOOTBALL] API error: %s", data["errors"])
                    return None
                self._cache[ck] = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "data": data
                }
                self._save_cache()
                return data
            elif r.status_code == 429:
                logger.warning("[API-FOOTBALL] Rate limited")
            else:
                logger.debug("[API-FOOTBALL] HTTP %d for %s", r.status_code, endpoint)
        except Exception as e:
            logger.debug("[API-FOOTBALL] %s: %s", endpoint, str(e)[:60])
        return None

    def get_team_id(self, team_name: str, league_id: int = None) -> Optional[int]:
        params = {"search": team_name}
        if league_id:
            params["league"] = league_id
        data = self._get("teams", params, ttl_hours=24.0)
        if data and data.get("response"):
            return data["response"][0]["team"]["id"]
        return None

    def get_team_stats(self, team_name: str,
                       league_id: int = None,
                       season: int = None) -> dict:
        """Get comprehensive team stats from API-Football."""
        if not self._key:
            return {}

        season = season or datetime.now().year
        # Try to find team ID
        team_id = self.get_team_id(team_name, league_id)
        if not team_id:
            return {}

        # If no league_id, try to find it
        if not league_id:
            league_data = self._get("teams/leagues",
                                     {"team": team_id}, ttl_hours=24.0)
            if league_data and league_data.get("response"):
                # Get most recent active league
                for lg in league_data["response"]:
                    lgs = lg.get("league", {})
                    if lgs.get("type") == "League":
                        league_id = lgs.get("id")
                        season = lg.get("seasons", [{}])[-1].get("year", season)
                        break

        if not league_id:
            return {}

        data = self._get("teams/statistics",
                          {"team": team_id, "league": league_id, "season": season},
                          ttl_hours=CFG.API_FOOTBALL_TTL)
        if not data or not data.get("response"):
            return {}

        resp = data["response"]
        fixtures = resp.get("fixtures", {})
        goals = resp.get("goals", {})
        form_str = resp.get("form", "") or ""

        played = fixtures.get("played", {}).get("total", 0) or 0
        wins   = fixtures.get("wins", {}).get("total", 0) or 0
        draws  = fixtures.get("draws", {}).get("total", 0) or 0
        losses = fixtures.get("loses", {}).get("total", 0) or 0

        gf_total = goals.get("for", {}).get("total", {}).get("total", 0) or 0
        ga_total = goals.get("against", {}).get("total", {}).get("total", 0) or 0

        gf_home = goals.get("for", {}).get("total", {}).get("home", 0) or 0
        gf_away = goals.get("for", {}).get("total", {}).get("away", 0) or 0
        ga_home = goals.get("against", {}).get("total", {}).get("home", 0) or 0
        ga_away = goals.get("against", {}).get("total", {}).get("away", 0) or 0

        # Biggest wins/losses
        biggest = resp.get("biggest", {})

        # Clean sheets
        cs = resp.get("clean_sheet", {})
        cs_total = cs.get("total", 0) or 0

        # Failed to score
        fts = resp.get("failed_to_score", {})
        fts_total = fts.get("total", 0) or 0

        safe_played = max(played, 1)

        result = {
            "team_id": team_id,
            "league_id": league_id,
            "season": season,
            "played": played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": round(wins / safe_played, 3),
            "draw_rate": round(draws / safe_played, 3),
            "avg_scored": round(gf_total / safe_played, 2),
            "avg_conceded": round(ga_total / safe_played, 2),
            "avg_scored_home": round(gf_home / max(played // 2, 1), 2),
            "avg_scored_away": round(gf_away / max(played // 2, 1), 2),
            "avg_conceded_home": round(ga_home / max(played // 2, 1), 2),
            "avg_conceded_away": round(ga_away / max(played // 2, 1), 2),
            "clean_sheet_rate": round(cs_total / safe_played, 3),
            "failed_to_score_rate": round(fts_total / safe_played, 3),
            "form": form_str[-10:] if form_str else "N/A",
            "recent_form_5": form_str[-5:] if form_str else "N/A",
            "data_quality": (
                "good" if played >= 15
                else "limited" if played >= 8
                else "poor"
            ),
            "source": "api_football"
        }

        # Add biggest results
        if biggest:
            result["biggest_win"] = biggest.get("wins", {}).get("total", "?")
            result["biggest_loss"] = biggest.get("loses", {}).get("total", "?")

        return result

    def get_h2h(self, team1_name: str, team2_name: str,
                last_n: int = 10) -> dict:
        """Get head-to-head stats between two teams."""
        if not self._key:
            return {}

        t1_id = self.get_team_id(team1_name)
        t2_id = self.get_team_id(team2_name)
        if not t1_id or not t2_id:
            return {}

        data = self._get("fixtures/headtohead",
                          {"h2h": f"{t1_id}-{t2_id}", "last": last_n},
                          ttl_hours=12.0)
        if not data or not data.get("response"):
            return {}

        matches = data["response"]
        if not matches:
            return {}

        t1_wins = t2_wins = draws = 0
        total_goals = []

        for m in matches:
            teams = m.get("teams", {})
            goals = m.get("goals", {})
            hg = goals.get("home", 0) or 0
            ag = goals.get("away", 0) or 0
            total_goals.append(hg + ag)

            home_id = teams.get("home", {}).get("id")
            winner_id = None
            if teams.get("home", {}).get("winner"):
                winner_id = home_id
            elif teams.get("away", {}).get("winner"):
                winner_id = teams.get("away", {}).get("id")
            else:
                draws += 1
                continue

            if winner_id == t1_id:
                t1_wins += 1
            else:
                t2_wins += 1

        total = len(matches)
        avg_goals = round(sum(total_goals) / max(total, 1), 2)

        return {
            "total": total,
            f"{team1_name}_wins": t1_wins,
            f"{team2_name}_wins": t2_wins,
            "draws": draws,
            "avg_goals": avg_goals,
            "over25_rate": round(
                sum(1 for g in total_goals if g > 2.5) / max(total, 1), 3
            ),
            "btts_rate": round(
                sum(1 for m in matches
                    if (m.get("goals", {}).get("home", 0) or 0) > 0
                    and (m.get("goals", {}).get("away", 0) or 0) > 0
                    ) / max(total, 1), 3
            ),
            "dominance": (
                f"{team1_name}_dominant" if t1_wins > t2_wins * 1.5
                else f"{team2_name}_dominant" if t2_wins > t1_wins * 1.5
                else "balanced"
            ),
            "source": "api_football"
        }

    def get_fixture_stats(self, home: str, away: str,
                          date_str: str = None) -> dict:
        """Get pre-match stats for a specific fixture."""
        if not self._key:
            return {}

        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Search for the fixture
        home_id = self.get_team_id(home)
        away_id = self.get_team_id(away)
        if not home_id or not away_id:
            return {}

        data = self._get("fixtures",
                          {"team": home_id, "date": date_str},
                          ttl_hours=2.0)
        if not data or not data.get("response"):
            return {}

        # Find matching fixture
        for fix in data["response"]:
            teams = fix.get("teams", {})
            away_match = teams.get("away", {}).get("id") == away_id
            if away_match:
                fixture_id = fix.get("fixture", {}).get("id")
                venue = fix.get("fixture", {}).get("venue", {})
                return {
                    "fixture_id": fixture_id,
                    "venue": venue.get("name", ""),
                    "city": venue.get("city", ""),
                    "referee": fix.get("fixture", {}).get("referee", ""),
                    "status": fix.get("fixture", {}).get("status", {}).get("long", ""),
                }
        return {}

    def get_standings(self, league_id: int,
                      season: int = None) -> List[dict]:
        """Get current league standings."""
        season = season or datetime.now().year
        data = self._get("standings",
                          {"league": league_id, "season": season},
                          ttl_hours=4.0)
        if not data or not data.get("response"):
            return []

        try:
            standings = data["response"][0]["league"]["standings"][0]
            result = []
            for team in standings:
                result.append({
                    "rank": team.get("rank", 0),
                    "team": team.get("team", {}).get("name", ""),
                    "team_id": team.get("team", {}).get("id"),
                    "points": team.get("points", 0),
                    "played": team.get("all", {}).get("played", 0),
                    "wins": team.get("all", {}).get("win", 0),
                    "draws": team.get("all", {}).get("draw", 0),
                    "losses": team.get("all", {}).get("lose", 0),
                    "gf": team.get("all", {}).get("goals", {}).get("for", 0),
                    "ga": team.get("all", {}).get("goals", {}).get("against", 0),
                    "form": team.get("form", ""),
                    "description": team.get("description", ""),
                })
            return result
        except (KeyError, IndexError):
            return []

    def get_team_standing(self, team_name: str,
                          league_id: int = None) -> dict:
        """Get a specific team's standing in their league."""
        if not league_id:
            # Try to detect league
            team_id = self.get_team_id(team_name)
            if not team_id:
                return {}
            league_data = self._get("teams/leagues",
                                     {"team": team_id}, ttl_hours=24.0)
            if league_data and league_data.get("response"):
                for lg in league_data["response"]:
                    if lg.get("league", {}).get("type") == "League":
                        league_id = lg["league"]["id"]
                        break

        if not league_id:
            return {}

        standings = self.get_standings(league_id)
        name_lower = team_name.lower()
        for s in standings:
            if _fuzzy_match_name(team_name, s["team"]):
                return {**s, "total_teams": len(standings)}
        return {}

    def get_injuries(self, team_name: str,
                     fixture_id: int = None) -> List[dict]:
        """Get injury/suspension list for a team."""
        if not self._key or not fixture_id:
            return []
        team_id = self.get_team_id(team_name)
        if not team_id:
            return []
        data = self._get("injuries",
                          {"fixture": fixture_id, "team": team_id},
                          ttl_hours=2.0)
        if not data or not data.get("response"):
            return []
        injuries = []
        for p in data["response"]:
            player = p.get("player", {})
            injuries.append({
                "name": player.get("name", ""),
                "type": p.get("type", ""),
                "reason": p.get("reason", ""),
            })
        return injuries


# =========================================================
# 3c. FOOTBALL-DATA.ORG CLIENT (Free backup)
# No strict call limits for free tier
# =========================================================
class FootballDataOrgClient:
    BASE = "https://api.football-data.org/v4"

    COMPETITION_MAP = {
        "Premier League": "PL",
        "Championship": "ELC",
        "La Liga": "PD",
        "Bundesliga": "BL1",
        "Serie A": "SA",
        "Ligue 1": "FL1",
        "Eredivisie": "DED",
        "Champions League": "CL",
        "Europa League": "EL",
        "MLS": "MLS",
    }

    def __init__(self):
        self._key = CFG.FOOTBALL_DATA_ORG_KEY
        self._cache_file = CFG.CACHE_DIR / "football_data_org_cache.json"
        self._cache: dict = {}
        self._last_call = 0.0
        self._min_interval = 6.0  # free tier: 10 calls/min
        self._load_cache()
        if self._key:
            logger.info("✅ [FOOTBALL-DATA.ORG] Key loaded")
        else:
            logger.info("ℹ️ [FOOTBALL-DATA.ORG] No key — limited access")

    def _load_cache(self):
        try:
            if self._cache_file.exists():
                raw = json.loads(self._cache_file.read_text())
                now = datetime.now(timezone.utc)
                for k, v in raw.items():
                    if isinstance(v, dict) and "ts" in v:
                        ts = datetime.fromisoformat(v["ts"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if (now - ts) < timedelta(hours=CFG.FOOTBALL_DATA_ORG_TTL):
                            self._cache[k] = v
        except Exception:
            pass

    def _save_cache(self):
        try:
            self._cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, default=str)
            )
        except Exception:
            pass

    def _get(self, endpoint: str, params: dict = None,
             ttl_hours: float = 6.0) -> Optional[dict]:
        ck = hashlib.md5(
            f"{endpoint}|{json.dumps(params or {}, sort_keys=True)}".encode()
        ).hexdigest()

        if ck in self._cache:
            try:
                ts = datetime.fromisoformat(self._cache[ck]["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - ts) < timedelta(hours=ttl_hours):
                    return self._cache[ck].get("data")
            except Exception:
                pass

        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        headers = {"Accept": "application/json"}
        if self._key:
            headers["X-Auth-Token"] = self._key

        try:
            r = requests.get(
                f"{self.BASE}/{endpoint}",
                params=params,
                headers=headers,
                timeout=15
            )
            self._last_call = time.time()

            if r.status_code == 200:
                data = r.json()
                self._cache[ck] = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "data": data
                }
                self._save_cache()
                return data
            elif r.status_code == 429:
                logger.warning("[FOOTBALL-DATA.ORG] Rate limited — sleeping 60s")
                time.sleep(60)
            else:
                logger.debug("[FOOTBALL-DATA.ORG] HTTP %d: %s",
                             r.status_code, endpoint)
        except Exception as e:
            logger.debug("[FOOTBALL-DATA.ORG] %s: %s", endpoint, str(e)[:60])
        return None

    def get_team_matches(self, team_name: str,
                         last_n: int = 10) -> dict:
        """Get recent matches for a team."""
        # Search for team
        data = self._get(f"teams", {"name": team_name}, ttl_hours=24.0)
        if not data or not data.get("teams"):
            return {}

        team = data["teams"][0]
        team_id = team.get("id")
        if not team_id:
            return {}

        matches_data = self._get(
            f"teams/{team_id}/matches",
            {"status": "FINISHED", "limit": last_n},
            ttl_hours=4.0
        )
        if not matches_data or not matches_data.get("matches"):
            return {}

        matches = matches_data["matches"]
        wins = draws = losses = 0
        gf = ga = 0
        form = []

        for m in matches:
            home_team = m.get("homeTeam", {}).get("name", "")
            away_team = m.get("awayTeam", {}).get("name", "")
            score = m.get("score", {}).get("fullTime", {})
            h_score = score.get("home", 0) or 0
            a_score = score.get("away", 0) or 0

            is_home = _fuzzy_match_name(team_name, home_team)
            t_score = h_score if is_home else a_score
            o_score = a_score if is_home else h_score

            gf += t_score
            ga += o_score

            if t_score > o_score:
                wins += 1
                form.append("W")
            elif t_score == o_score:
                draws += 1
                form.append("D")
            else:
                losses += 1
                form.append("L")

        total = len(matches)
        safe_total = max(total, 1)

        return {
            "team_id": team_id,
            "team_name": team.get("name", team_name),
            "competition": team.get("runningCompetitions", [{}])[0].get("name", "?") if team.get("runningCompetitions") else "?",
            "played": total,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": round(wins / safe_total, 3),
            "avg_scored": round(gf / safe_total, 2),
            "avg_conceded": round(ga / safe_total, 2),
            "form": "".join(reversed(form)),
            "data_quality": (
                "good" if total >= 8
                else "limited" if total >= 4
                else "poor"
            ),
            "source": "football_data_org"
        }

    def get_standings(self, competition_code: str,
                      season: int = None) -> List[dict]:
        """Get standings for a competition."""
        params = {}
        if season:
            params["season"] = season
        data = self._get(
            f"competitions/{competition_code}/standings",
            params, ttl_hours=4.0
        )
        if not data:
            return []
        try:
            table = data["standings"][0]["table"]
            return [{
                "rank": t["position"],
                "team": t["team"]["name"],
                "points": t["points"],
                "played": t["playedGames"],
                "wins": t["won"],
                "draws": t["draw"],
                "losses": t["lost"],
                "gf": t["goalsFor"],
                "ga": t["goalsAgainst"],
                "gd": t["goalDifference"],
                "form": t.get("form", ""),
            } for t in table]
        except (KeyError, IndexError):
            return []


# =========================================================
# 3d. BALLDONTLIE CLIENT (NBA — Free, no strict limits)
# =========================================================
class BallDontLieClient:
    BASE = "https://api.balldontlie.io/v1"

    def __init__(self):
        self._key = CFG.BALLDONTLIE_API_KEY
        self._cache_file = CFG.CACHE_DIR / "balldontlie_cache.json"
        self._cache: dict = {}
        self._last_call = 0.0
        self._min_interval = 1.0
        self._load_cache()
        logger.info("✅ [BALLDONTLIE] NBA client ready")

    def _load_cache(self):
        try:
            if self._cache_file.exists():
                raw = json.loads(self._cache_file.read_text())
                now = datetime.now(timezone.utc)
                for k, v in raw.items():
                    if isinstance(v, dict) and "ts" in v:
                        ts = datetime.fromisoformat(v["ts"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if (now - ts) < timedelta(hours=6):
                            self._cache[k] = v
        except Exception:
            pass

    def _save_cache(self):
        try:
            self._cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, default=str)
            )
        except Exception:
            pass

    def _get(self, endpoint: str, params: dict = None,
             ttl_hours: float = 6.0) -> Optional[dict]:
        ck = hashlib.md5(
            f"{endpoint}|{json.dumps(params or {}, sort_keys=True)}".encode()
        ).hexdigest()

        if ck in self._cache:
            try:
                ts = datetime.fromisoformat(self._cache[ck]["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - ts) < timedelta(hours=ttl_hours):
                    return self._cache[ck].get("data")
            except Exception:
                pass

        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        headers = {"Accept": "application/json"}
        if self._key:
            headers["Authorization"] = self._key

        try:
            r = requests.get(
                f"{self.BASE}/{endpoint}",
                params=params,
                headers=headers,
                timeout=15
            )
            self._last_call = time.time()
            if r.status_code == 200:
                data = r.json()
                self._cache[ck] = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "data": data
                }
                self._save_cache()
                return data
            else:
                logger.debug("[BALLDONTLIE] HTTP %d: %s", r.status_code, endpoint)
        except Exception as e:
            logger.debug("[BALLDONTLIE] %s: %s", endpoint, str(e)[:60])
        return None

    def get_team_stats(self, team_name: str) -> dict:
        """Get NBA team season averages."""
        # Find team
        data = self._get("teams", {"search": team_name}, ttl_hours=24.0)
        if not data or not data.get("data"):
            return {}

        team = data["data"][0]
        team_id = team.get("id")
        if not team_id:
            return {}

        # Get season averages
        season = datetime.now().year - (1 if datetime.now().month < 9 else 0)
        stats_data = self._get(
            "season_averages",
            {"season": season, "team_ids[]": team_id},
            ttl_hours=4.0
        )

        result = {
            "team_id": team_id,
            "team_name": team.get("full_name", team_name),
            "abbreviation": team.get("abbreviation", ""),
            "conference": team.get("conference", ""),
            "division": team.get("division", ""),
            "source": "balldontlie"
        }

        if stats_data and stats_data.get("data"):
            s = stats_data["data"][0]
            result.update({
                "avg_pts": round(float(s.get("pts", 0) or 0), 1),
                "avg_reb": round(float(s.get("reb", 0) or 0), 1),
                "avg_ast": round(float(s.get("ast", 0) or 0), 1),
                "fg_pct": round(float(s.get("fg_pct", 0) or 0), 3),
                "fg3_pct": round(float(s.get("fg3_pct", 0) or 0), 3),
                "ft_pct": round(float(s.get("ft_pct", 0) or 0), 3),
            })
            result["data_quality"] = "limited"
        else:
            result["data_quality"] = "poor"

        return result

    def get_recent_games(self, team_name: str, last_n: int = 10) -> dict:
        """Get recent game results for an NBA team."""
        team_data = self._get("teams", {"search": team_name}, ttl_hours=24.0)
        if not team_data or not team_data.get("data"):
            return {}

        team_id = team_data["data"][0].get("id")
        if not team_id:
            return {}

        season = datetime.now().year - (1 if datetime.now().month < 9 else 0)
        games_data = self._get(
            "games",
            {
                "seasons[]": season,
                "team_ids[]": team_id,
                "per_page": last_n,
                "postseason": False
            },
            ttl_hours=4.0
        )

        if not games_data or not games_data.get("data"):
            return {}

        games = sorted(
            [g for g in games_data["data"] if g.get("status") == "Final"],
            key=lambda x: x.get("date", ""),
            reverse=True
        )[:last_n]

        wins = losses = pts_for = pts_against = 0
        form = []

        for g in games:
            home_id = g.get("home_team", {}).get("id")
            is_home = home_id == team_id
            t_pts = g.get("home_team_score", 0) if is_home else g.get("visitor_team_score", 0)
            o_pts = g.get("visitor_team_score", 0) if is_home else g.get("home_team_score", 0)
            t_pts = t_pts or 0
            o_pts = o_pts or 0
            pts_for += t_pts
            pts_against += o_pts
            if t_pts > o_pts:
                wins += 1
                form.append("W")
            else:
                losses += 1
                form.append("L")

        total = len(games)
        safe_total = max(total, 1)

        return {
            "recent_record": f"{wins}W-{losses}L",
            "win_rate": round(wins / safe_total, 3),
            "avg_pts_scored": round(pts_for / safe_total, 1),
            "avg_pts_allowed": round(pts_against / safe_total, 1),
            "pt_diff": round((pts_for - pts_against) / safe_total, 1),
            "form": "".join(reversed(form)),
            "games_analyzed": total,
            "data_quality": "good" if total >= 8 else "limited",
            "source": "balldontlie"
        }


# Singletons
api_football = APIFootballClient()
football_data_org = FootballDataOrgClient()
balldontlie = BallDontLieClient()

# =========================================================
# 4. AI MANAGER
# =========================================================
import google.genai as genai
from google.genai import types

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

AI_SYSTEM_PROMPT = """You are an elite sports betting quantitative analyst. Find GENUINE EDGE only.

DECISION FRAMEWORK:
- BET: EV > 2% AND (sharp_line=True OR ML_confidence > 0.62) AND Kelly > 1%
- BET: EV > 3.5% even with limited data if no major red flags
- SKIP: EV < 1.5% for any reason
- SKIP: Kelly < 0.8%
- SKIP: Models disagree > 20%
- Missing data = lower confidence, NOT automatic skip

CONFIDENCE CALIBRATION:
85-100: Multiple signals all agree + EV > 5%
75-84:  Sharp line + EV > 3% + stats support
65-74:  Clear primary signal + EV > 2%
55-64:  Marginal - skip unless sharp line present
< 55:   Hard skip

Output ONLY valid JSON:
{"decision":"BET" or "SKIP","confidence":<0-100>,"sport_emoji":"<emoji>","risk_level":"Low" or "Medium" or "High","key_factors":["f1","f2","f3"],"logic":"2-3 sentences","red_flags":["r1"]}"""


class AIManager:
    _instance = None
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
        gem_keys = [k.strip() for k in [
            os.getenv("GEMINI", ""), os.getenv("GEMINI1", ""),
            os.getenv("GEMINI2", ""), os.getenv("GEMINI3", "")
        ] if k.strip()]
        self.gemini_clients = [genai.Client(api_key=k) for k in gem_keys] if gem_keys else []
        self._safety = [
            types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            for c in [types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                      types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                      types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                      types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT]
        ]
        groq_keys = [k.strip() for k in [
            os.getenv("GROQ_API_KEY", ""), os.getenv("GROQ1", ""), os.getenv("GROQ2", "")
        ] if k.strip()]
        self.groq_clients = [Groq(api_key=k) for k in groq_keys] if (groq_keys and HAS_GROQ) else []
        self._gemini_failed: Dict[int, float] = {}
        self._last_call_time = 0.0
        self._rate_lock = threading.Lock()
        self._last_provider = "none"
        self._initialized = True
        logger.info("✅ [AI] Gemini:%d | Groq:%d", len(self.gemini_clients), len(self.groq_clients))

    def _is_key_failed(self, idx: int) -> bool:
        ft = self._gemini_failed.get(idx)
        if ft is None:
            return False
        if time.time() - ft > 900:
            del self._gemini_failed[idx]
            return False
        return True

    def generate(self, prompt: str) -> Optional[dict]:
        with self._rate_lock:
            elapsed = time.time() - self._last_call_time
            if elapsed < 2.0:
                time.sleep(2.0 - elapsed)
            self._last_call_time = time.time()

        # Try Gemini
        if self.gemini_clients:
            gen_cfg = types.GenerateContentConfig(
                temperature=CFG.AI_TEMPERATURE,
                max_output_tokens=CFG.AI_MAX_TOKENS,
                response_mime_type="application/json",
                safety_settings=self._safety,
                system_instruction=AI_SYSTEM_PROMPT
            )
            available = [i for i in range(len(self.gemini_clients))
                         if not self._is_key_failed(i)]
            if not available:
                self._gemini_failed.clear()
                available = list(range(len(self.gemini_clients)))

            for _ in range(3):
                if not available:
                    break
                idx = random.choice(available)
                try:
                    resp = self.gemini_clients[idx].models.generate_content(
                        model=CFG.AI_MODEL, contents=prompt, config=gen_cfg
                    )
                    raw = resp.text
                    if raw:
                        self._last_provider = "gemini"
                        try:
                            return json.loads(raw)
                        except Exception:
                            return _robust_json_extract(raw)
                except Exception as e:
                    es = str(e)
                    if "429" in es or "quota" in es.lower():
                        self._gemini_failed[idx] = time.time()
                        available.remove(idx)
                        continue
                    logger.warning("[GEMINI] Key%d: %s", idx, es[:80])
                    break

        # Try Groq
        if self.groq_clients:
            for attempt in range(2):
                try:
                    client = random.choice(self.groq_clients)
                    cc = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": AI_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        model="qwen/qwen3-32b",
                        temperature=CFG.AI_TEMPERATURE,
                        max_completion_tokens=CFG.AI_MAX_TOKENS,
                        response_format={"type": "json_object"}
                    )
                    raw = cc.choices[0].message.content
                    if raw:
                        self._last_provider = "groq"
                        try:
                            return json.loads(raw)
                        except Exception:
                            return _robust_json_extract(raw)
                except Exception as e:
                    logger.warning("[GROQ] Attempt%d: %s", attempt + 1, str(e)[:80])
                    time.sleep(2)

        return None


ai_manager = AIManager()


def _robust_json_extract(raw: str) -> Optional[dict]:
    if not raw:
        return None
    clean = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE)
    clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", clean))):
        try:
            r = json.loads(m.group(0))
            if isinstance(r, dict) and r:
                return r
        except Exception:
            continue
    return None


# =========================================================
# 5. API KEY MANAGER
# =========================================================
class OddsAPIKeyManager:
    def __init__(self):
        self.keys = []
        for env, label in [("ODDS_API_KEY", "primary"),
                            ("ODDS_API_KEY2", "backup_1"),
                            ("ODDS_API_KEY3", "backup_2")]:
            k = os.getenv(env, "").strip()
            if k:
                self.keys.append({
                    "key": k, "label": label,
                    "failed": False, "fail_time": None, "calls": 0
                })
                logger.info("🔑 [KEY] %s loaded", label)
        if not self.keys:
            logger.critical("FATAL: No ODDS_API_KEY!")
            sys.exit(1)

    def mark_failed(self, idx: int, reason: str):
        if 0 <= idx < len(self.keys):
            self.keys[idx]["failed"] = True
            self.keys[idx]["fail_time"] = datetime.now(timezone.utc).isoformat()
            logger.warning("🔑❌ %s FAILED: %s", self.keys[idx]["label"], reason)

    def get_active_keys(self) -> List[dict]:
        now = datetime.now(timezone.utc)
        active = []
        for k in self.keys:
            if not k["failed"]:
                active.append(k)
            elif k.get("fail_time"):
                try:
                    ft = datetime.fromisoformat(k["fail_time"])
                    if ft.tzinfo is None:
                        ft = ft.replace(tzinfo=timezone.utc)
                    if now - ft > timedelta(minutes=30):
                        k["failed"] = False
                        active.append(k)
                except Exception:
                    pass
        if not active:
            for k in self.keys:
                k["failed"] = False
            active = list(self.keys)
        return active

    def get_summary(self) -> str:
        return " | ".join(
            f"{'❌' if k['failed'] else '✅'} {k['label']}({k['calls']}calls)"
            for k in self.keys
        )


GEMINI_API_KEY = os.getenv("GEMINI", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
if not all([GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing env vars")
    sys.exit(1)
odds_key_manager = OddsAPIKeyManager()

# =========================================================
# 6. CACHE MANAGER
# =========================================================
_cache_lock = threading.Lock()


class CacheManager:
    @staticmethod
    def load(fp: Path) -> dict:
        try:
            if fp.exists():
                return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    @staticmethod
    def save(fp: Path, data: dict):
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp_name = f"{fp.name}.tmp.{os.getpid()}_{int(time.time()*1000)}"
            tmp = fp.with_name(tmp_name)
            content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            with _cache_lock:
                tmp.write_text(content, encoding="utf-8")
                try:
                    tmp.replace(fp)
                except PermissionError:
                    if fp.exists():
                        fp.unlink()
                    tmp.rename(fp)
        except Exception as e:
            logger.debug("[CACHE] Save error: %s", e)

    @staticmethod
    def is_valid(cache: dict, key: str, ttl_hours: float) -> bool:
        e = cache.get(key)
        if not isinstance(e, dict) or "timestamp" not in e:
            return False
        try:
            t = datetime.fromisoformat(e["timestamp"])
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - t < timedelta(hours=ttl_hours)
        except Exception:
            return False

    @staticmethod
    def set(cache: dict, key: str, value: Any) -> dict:
        cache[key] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": value
        }
        return cache

    @staticmethod
    def get(cache: dict, key: str) -> Any:
        return cache.get(key, {}).get("data")


# =========================================================
# 7. SENT HISTORY
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
                t = datetime.fromisoformat(v.get("sent_at", "2000-01-01T00:00:00+00:00"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if now - t > timedelta(hours=CFG.TTL_SENT_HISTORY):
                    to_del.append(k)
            except Exception:
                to_del.append(k)
        for k in to_del:
            del self.history[k]

    @staticmethod
    def _key(home, away, market) -> str:
        return hashlib.md5(
            f"{home.lower()}|{away.lower()}|{market.lower()}".encode()
        ).hexdigest()

    def was_sent(self, home, away, market) -> bool:
        return self._key(home, away, market) in self.history

    def mark_sent(self, home, away, pick, market):
        self.history[self._key(home, away, market)] = {
            "match": f"{home} vs {away}",
            "pick": pick,
            "market": market,
            "sent_at": datetime.now(timezone.utc).isoformat()
        }
        CacheManager.save(CFG.HISTORY_FILE, self.history)


# =========================================================
# 8. FREE DATA ENGINE
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
        self.years = [2022, 2023, 2024, 2025]

    def _download_csv(self, url: str, path: Path, timeout: int = 30) -> bool:
        if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < CFG.TTL_GITHUB_DATA:
            return True
        logger.info("[DATA] Downloading: %s", path.name)
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=timeout + attempt * 10,
                                 headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200 and len(r.content) > 100:
                    path.write_bytes(r.content)
                    return True
                break
            except requests.exceptions.Timeout:
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                logger.warning("[DATA] %s: %s", path.name, str(e)[:60])
                break
        return False

    def load_tennis_data(self):
        COLS = ["tourney_date", "tourney_name", "surface", "draw_size",
                "tourney_level", "round",
                "winner_id", "winner_name", "winner_rank", "winner_rank_points",
                "loser_id", "loser_name", "loser_rank", "loser_rank_points",
                "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
                "w_bpSaved", "w_bpFaced",
                "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
                "l_bpSaved", "l_bpFaced",
                "score", "best_of", "minutes"]
        for tour, attr, key in [("ATP", "atp_matches", "atp"),
                                  ("WTA", "wta_matches", "wta")]:
            dfs = []
            for year in self.years:
                url = CFG.GITHUB_SOURCES[key].format(year=year)
                path = CFG.HISTORICAL_DIR / f"{key}_{year}.csv"
                if self._download_csv(url, path):
                    try:
                        df = pd.read_csv(path, low_memory=False,
                                         encoding="utf-8", encoding_errors="replace")
                        sub = df[[c for c in COLS if c in df.columns]].copy()
                        if "tourney_date" in sub.columns:
                            sub["tourney_date"] = pd.to_numeric(
                                sub["tourney_date"], errors="coerce")
                        dfs.append(sub)
                    except Exception as e:
                        logger.error("[TENNIS] %s %s: %s", tour, year, e)
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                if "tourney_date" in combined.columns:
                    combined = combined.sort_values("tourney_date").reset_index(drop=True)
                setattr(self, attr, combined)
                logger.info("✅ [TENNIS] %s: %d matches", tour, len(combined))

        for tour, key, attr in [("ATP", "atp_rankings", "atp_rankings"),
                                  ("WTA", "wta_rankings", "wta_rankings")]:
            path = CFG.HISTORICAL_DIR / f"{key}.csv"
            if self._download_csv(CFG.GITHUB_SOURCES[key], path):
                try:
                    setattr(self, attr, pd.read_csv(path, low_memory=False))
                    logger.info("✅ [RANKINGS] %s", tour)
                except Exception as e:
                    logger.error("[RANKINGS] %s: %s", tour, e)

    def get_player_ranking(self, name: str, is_wta: bool = False) -> Optional[int]:
        df = self.wta_rankings if is_wta else self.atp_rankings
        if df is None or df.empty:
            return None
        nc = next((c for c in ["player", "name", "player_name"] if c in df.columns), None)
        if not nc:
            return None
        name_lower = name.lower().strip()
        col_lower = df[nc].astype(str).str.lower()
        exact = df[col_lower == name_lower]
        if not exact.empty:
            m = exact
        else:
            parts = name.split()
            last = parts[-1].lower() if parts else name_lower
            m = df[col_lower.str.contains(re.escape(last), na=False)]
        if not m.empty:
            rc = next((c for c in ["rank", "ranking", "player_rank"]
                       if c in m.columns), None)
            if rc:
                v = m.iloc[0][rc]
                return int(v) if pd.notna(v) else None
        return None

    def _player_rolling(self, df: pd.DataFrame, clean: str, n: int = 20) -> dict:
        wins = df[df["winner_name"].str.lower().str.contains(
            re.escape(clean), na=False)]
        losses = df[df["loser_name"].str.lower().str.contains(
            re.escape(clean), na=False)]
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
            "recent_win_rate": round(rw / len(recent), 3) if recent else 0
        }
        rw_df = wins.tail(n // 2)
        if "w_ace" in rw_df.columns:
            v = rw_df["w_ace"].dropna()
            if len(v):
                result["aces_per_match"] = round(float(v.mean()), 2)
        if all(c in rw_df.columns for c in ["w_1stIn", "w_svpt"]):
            sv = rw_df["w_svpt"].dropna().mean()
            if sv:
                result["first_serve_in_pct"] = round(
                    float(rw_df["w_1stIn"].dropna().mean() / sv), 3)
        if all(c in rw_df.columns for c in ["w_bpSaved", "w_bpFaced"]):
            bpf = rw_df["w_bpFaced"].dropna().mean()
            if bpf:
                result["bp_saved_pct"] = round(
                    float(rw_df["w_bpSaved"].dropna().mean() / bpf), 3)
        ss = {}
        for surf in ["Hard", "Clay", "Grass"]:
            if "surface" in wins.columns:
                sw = wins[wins["surface"].str.lower() == surf.lower()]
                sl = (losses[losses["surface"].str.lower() == surf.lower()]
                      if "surface" in losses.columns else pd.DataFrame())
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

        def clean_name(n):
            n = n.strip()
            parts = n.split()
            if len(parts) >= 2:
                candidate = " ".join(parts[-2:]).lower()
                if df is not None:
                    wn = df["winner_name"].astype(str).str.lower()
                    ln = df["loser_name"].astype(str).str.lower()
                    if (any(wn.str.contains(re.escape(candidate), na=False)) or
                            any(ln.str.contains(re.escape(candidate), na=False))):
                        return candidate
            return parts[-1].lower() if parts else n.lower()

        ca, cb = clean_name(pa), clean_name(pb)
        stats = {"player_a": {"name": pa}, "player_b": {"name": pb}, "h2h": {}}

        for p_c, key, p_f, is_w in [(ca, "player_a", pa, is_wta),
                                      (cb, "player_b", pb, is_wta)]:
            s = self._player_rolling(df, p_c)
            if s:
                stats[key].update(s)
                r = self.get_player_ranking(p_f, is_w)
                if r:
                    stats[key]["current_ranking"] = r
                stats[key]["data_quality"] = (
                    "good" if s.get("total_matches", 0) >= 20
                    else "limited" if s.get("total_matches", 0) >= 5
                    else "poor"
                )

        # TSDB player enrichment
        for p_name, key in [(pa, "player_a"), (pb, "player_b")]:
            try:
                tp = tsdb.get_player_stats(p_name)
                if tp:
                    stats[key]["nationality"] = tp.get("nationality", "")
            except Exception:
                pass

        # H2H from GitHub
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
                )
            }
            if "surface" in h2h_a.columns:
                by_surf = {}
                for surf in ["Hard", "Clay", "Grass"]:
                    sa = h2h_a[h2h_a["surface"].str.lower() == surf.lower()]
                    sb = h2h_b[h2h_b["surface"].str.lower() == surf.lower()]
                    if len(sa) + len(sb) > 0:
                        by_surf[surf] = {
                            f"{pa}_wins": len(sa),
                            f"{pb}_wins": len(sb)
                        }
                if by_surf:
                    stats["h2h"]["by_surface"] = by_surf

        qa = stats["player_a"].get("data_quality", "poor")
        qb = stats["player_b"].get("data_quality", "poor")
        stats["data_quality_summary"] = {
            "player_a": qa, "player_b": qb, "h2h_matches": t,
            "overall": (
                "good" if qa == "good" and qb == "good" and t >= 3
                else "limited" if qa != "poor" or qb != "poor"
                else "poor"
            )
        }
        return stats

    def load_football_data(self):
        COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                "HS", "AS", "HST", "AST", "HC", "AC",
                "B365H", "B365D", "B365A",
                "BbMxH", "BbMxD", "BbMxA",
                "BbAvH", "BbAvD", "BbAvA",
                "BbMx>2.5", "BbAv>2.5"]
        all_dfs = []
        for season in CFG.FOOTBALL_DATA_UK_SEASONS:
            for code, name in CFG.FOOTBALL_DATA_UK_LEAGUES.items():
                url = CFG.GITHUB_SOURCES["football_eu"].format(
                    season=season, league=code)
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
                                sub["Date"], format="mixed",
                                dayfirst=True, errors="coerce")
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
            logger.info("✅ [FOOTBALL] %d matches loaded", len(comb))

    def _fuzzy_df(self, team: str, col: pd.Series) -> pd.Series:
        clean = team.lower().strip()
        m = col.str.lower().str.strip() == clean
        if m.any():
            return m
        for p in clean.split():
            if len(p) > 3:
                m2 = col.str.lower().str.contains(re.escape(p), na=False)
                if m2.sum() <= 20:
                    if m2.any():
                        return m2
        return pd.Series([False] * len(col), index=col.index)

    def get_football_stats(self, home: str, away: str) -> dict:
        """
        Multi-source football stats:
        1. API-Football (primary — best data)
        2. Football-Data.org (backup)
        3. TheSportsDB (fallback)
        4. GitHub CSV (historical baseline)
        """
        stats: dict = {"home": {}, "away": {}, "h2h": {}}

        # ── 1. API-Football ───────────────────────────────────
        if CFG.API_FOOTBALL_KEY:
            try:
                h_apif = api_football.get_team_stats(home)
                a_apif = api_football.get_team_stats(away)
                if h_apif:
                    stats["home"].update(h_apif)
                    logger.debug("[API-FOOTBALL] %s: Q=%s",
                                 home, h_apif.get("data_quality"))
                if a_apif:
                    stats["away"].update(a_apif)
                    logger.debug("[API-FOOTBALL] %s: Q=%s",
                                 away, a_apif.get("data_quality"))

                # H2H from API-Football
                h2h_apif = api_football.get_h2h(home, away)
                if h2h_apif:
                    stats["h2h"].update(h2h_apif)

                # Standings
                if h_apif.get("league_id"):
                    h_standing = api_football.get_team_standing(
                        home, h_apif["league_id"])
                    if h_standing:
                        stats["home"]["standing"] = h_standing
                if a_apif.get("league_id"):
                    a_standing = api_football.get_team_standing(
                        away, a_apif["league_id"])
                    if a_standing:
                        stats["away"]["standing"] = a_standing

            except Exception as e:
                logger.warning("[API-FOOTBALL] get_football_stats: %s", e)

        # ── 2. Football-Data.org (if API-Football missing) ────
        if not stats["home"].get("win_rate") and CFG.FOOTBALL_DATA_ORG_KEY:
            try:
                h_fdo = football_data_org.get_team_matches(home)
                a_fdo = football_data_org.get_team_matches(away)
                if h_fdo:
                    stats["home"].update(h_fdo)
                if a_fdo:
                    stats["away"].update(a_fdo)
            except Exception as e:
                logger.debug("[FOOTBALL-DATA.ORG] %s", e)

        # ── 3. TheSportsDB (always enrich form) ───────────────
        try:
            h_tsdb = tsdb.get_team_stats(home)
            a_tsdb = tsdb.get_team_stats(away)
            if h_tsdb and not stats["home"].get("form"):
                stats["home"]["tsdb_form"] = h_tsdb.get("form", "")
                stats["home"]["tsdb_win_rate"] = h_tsdb.get("win_rate", 0)
            if a_tsdb and not stats["away"].get("form"):
                stats["away"]["tsdb_form"] = a_tsdb.get("form", "")
                stats["away"]["tsdb_win_rate"] = a_tsdb.get("win_rate", 0)
        except Exception as e:
            logger.debug("[TSDB] football enrich: %s", e)

        # ── 4. GitHub CSV (historical baseline always) ─────────
        df = self.football_data.get("all")
        if df is not None and not df.empty:
            for team, key, is_home in [(home, "home", True), (away, "away", False)]:
                hm = self._fuzzy_df(team, df["HomeTeam"])
                am = self._fuzzy_df(team, df["AwayTeam"])
                all_r = []
                for _, row in df[hm].iterrows():
                    hg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                    ag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                    ftr = row.get("FTR", "")
                    all_r.append({
                        "result": "W" if ftr == "H" else ("D" if ftr == "D" else "L"),
                        "scored": hg, "conceded": ag,
                        "total": hg + ag
                    })
                for _, row in df[am].iterrows():
                    hg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                    ag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                    ftr = row.get("FTR", "")
                    all_r.append({
                        "result": "W" if ftr == "A" else ("D" if ftr == "D" else "L"),
                        "scored": ag, "conceded": hg,
                        "total": hg + ag
                    })
                if len(all_r) >= 5:
                    recent = all_r[-15:]
                    n = len(recent)
                    sc = [r["scored"] for r in recent]
                    cn = [r["conceded"] for r in recent]
                    totals = [r["total"] for r in recent]
                    github_stats = {
                        "github_form": "".join(r["result"] for r in recent[-10:]),
                        "github_win_rate": round(
                            sum(1 for r in recent if r["result"] == "W") / n, 3),
                        "github_draw_rate": round(
                            sum(1 for r in recent if r["result"] == "D") / n, 3),
                        "github_avg_scored": round(float(np.mean(sc)), 2),
                        "github_avg_conceded": round(float(np.mean(cn)), 2),
                        "github_avg_total": round(float(np.mean(totals)), 2),
                        "github_over25_rate": round(
                            sum(1 for r in recent if r["total"] > 2.5) / n, 3),
                        "github_over35_rate": round(
                            sum(1 for r in recent if r["total"] > 3.5) / n, 3),
                        "github_btts_rate": round(
                            sum(1 for r in recent
                                if r["scored"] > 0 and r["conceded"] > 0) / n, 3),
                        "github_matches": len(all_r),
                        "data_quality": (
                            "good" if len(all_r) >= 20
                            else "limited" if len(all_r) >= 8
                            else "poor"
                        )
                    }
                    # Only override if API-Football didn't provide
                    if not stats[key].get("win_rate"):
                        stats[key].update(github_stats)
                    else:
                        # Add as supplement
                        stats[key]["github"] = github_stats

            # GitHub H2H
            if not stats["h2h"]:
                hm2 = self._fuzzy_df(home, df["HomeTeam"])
                am2 = self._fuzzy_df(away, df["AwayTeam"])
                hm3 = self._fuzzy_df(away, df["HomeTeam"])
                am3 = self._fuzzy_df(home, df["AwayTeam"])
                h2h_df = df[(hm2 & am2) | (hm3 & am3)]
                if len(h2h_df) >= 3:
                    h2h_r = []
                    for _, row in h2h_df.iterrows():
                        hg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                        ag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                        h2h_r.append({"total": hg + ag, "btts": hg > 0 and ag > 0})
                    hn = len(h2h_r)
                    gl = [r["total"] for r in h2h_r]
                    stats["h2h"] = {
                        "total_matches": hn,
                        "avg_goals": round(float(np.mean(gl)), 2),
                        "over25_rate": round(
                            sum(1 for r in h2h_r if r["total"] > 2.5) / hn, 3),
                        "btts_rate": round(
                            sum(1 for r in h2h_r if r["btts"]) / hn, 3),
                        "source": "github"
                    }

        return stats

    def load_nhl_data(self):
        url = "https://api-web.nhle.com/v1/standings/now"
        cache_path = CFG.HISTORICAL_DIR / "nhl_standings.json"
        try:
            r = requests.get(url, timeout=15,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                data = r.json()
                standings = data.get("standings", [])
                if standings:
                    rows = []
                    for team in standings:
                        w = team.get("wins", 0)
                        l = team.get("losses", 0)
                        otl = team.get("otLosses", 0)
                        gp = max(w + l + otl, 1)
                        gf = team.get("goalFor", 0)
                        ga = team.get("goalAgainst", 0)
                        rows.append({
                            "team": team.get("teamName", {}).get("default", ""),
                            "wins": w, "losses": l, "otLosses": otl,
                            "points": team.get("points", 0),
                            "goalsFor": gf, "goalsAgainst": ga,
                            "win_pct": round(w / gp, 3),
                            "avg_gf": round(gf / gp, 2),
                            "avg_ga": round(ga / gp, 2),
                            "l10Wins": team.get("l10Wins", 0),
                            "l10Losses": team.get("l10Losses", 0),
                            "streakCode": team.get("streakCode", ""),
                            "streakCount": team.get("streakCount", 0),
                        })
                    self.nhl_data = pd.DataFrame(rows)
                    logger.info("✅ [NHL] %d teams", len(rows))
                    return
        except Exception as e:
            logger.warning("[NHL] %s", str(e)[:60])
        self.nhl_data = None

    def load_mlb_data(self):
        url = ("https://statsapi.mlb.com/api/v1/standings"
               "?leagueId=103,104&season=2025&standingsTypes=regularSeason")
        try:
            r = requests.get(url, timeout=15,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                data = r.json()
                rows = []
                for record in data.get("records", []):
                    for tr in record.get("teamRecords", []):
                        ti = tr.get("team", {})
                        w = tr.get("wins", 0)
                        l = tr.get("losses", 0)
                        gp = max(w + l, 1)
                        rs = tr.get("runsScored", 0) or 0
                        ra = tr.get("runsAllowed", 0) or 0
                        rows.append({
                            "team": ti.get("name", ""),
                            "wins": w, "losses": l,
                            "win_pct": float(tr.get("winningPercentage", 0) or 0),
                            "avg_runs_scored": round(rs / gp, 2),
                            "avg_runs_allowed": round(ra / gp, 2),
                            "run_diff": round((rs - ra) / gp, 2),
                            "streak": tr.get("streak", {}).get("streakCode", ""),
                        })
                if rows:
                    self.mlb_data = pd.DataFrame(rows)
                    logger.info("✅ [MLB] %d teams", len(rows))
                    return
        except Exception as e:
            logger.warning("[MLB] %s", str(e)[:60])
        self.mlb_data = None

    def load_nba_data(self):
        try:
            from nba_api.stats.endpoints import leaguestandings
            standings = leaguestandings.LeagueStandings(
                season="2024-25", season_type="Regular Season", league_id="00",
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json",
                    "Referer": "https://www.nba.com/",
                    "Origin": "https://www.nba.com"
                }, timeout=10)
            df = standings.get_data_frames()[0]
            if df is not None and not df.empty:
                self.nba_data = df
                logger.info("✅ [NBA] %d teams", len(df))
                return
        except Exception as e:
            logger.warning("[NBA] %s", str(e)[:60])
        self.nba_data = None

    def get_us_sports_stats(self, sport: str, team: str) -> dict:
        sport_lower = sport.lower()

        # NBA
        if "basketball" in sport_lower or "nba" in sport_lower:
            # Try BallDontLie first
            try:
                recent = balldontlie.get_recent_games(team)
                if recent and recent.get("data_quality") != "poor":
                    avg_stats = balldontlie.get_team_stats(team)
                    return {**recent, **avg_stats}
            except Exception:
                pass
            # Fallback to nba_api standings
            if self.nba_data is not None and not self.nba_data.empty:
                clean = team.lower().strip()
                for col in ["TeamName", "TEAM_NAME"]:
                    if col in self.nba_data.columns:
                        m = self.nba_data[
                            self.nba_data[col].astype(str).str.lower()
                            .str.contains(re.escape(clean), na=False)
                        ]
                        if not m.empty:
                            row = m.iloc[0]
                            return {
                                "win_pct": float(
                                    row.get("WinPCT", row.get("WIN_PCT", 0.5)) or 0.5),
                                "source": "nba_api"
                            }

        # MLB
        elif "baseball" in sport_lower or "mlb" in sport_lower:
            if self.mlb_data is not None:
                clean = team.lower().strip()
                m = self.mlb_data[
                    self.mlb_data["team"].str.lower()
                    .str.contains(re.escape(clean), na=False)
                ]
                if not m.empty:
                    return m.iloc[0].to_dict()

        # NHL
        elif "hockey" in sport_lower or "nhl" in sport_lower:
            if self.nhl_data is not None:
                clean = team.lower().strip()
                m = self.nhl_data[
                    self.nhl_data["team"].str.lower()
                    .str.contains(re.escape(clean), na=False)
                ]
                if not m.empty:
                    return m.iloc[0].to_dict()

        # TSDB fallback for any sport
        try:
            ts = tsdb.get_team_stats(team)
            if ts and ts.get("data_quality") in ("good", "limited"):
                return ts
        except Exception:
            pass

        return {}


# =========================================================
# 9. ML ENGINE
# =========================================================
class MLPredictionEngine:
    def __init__(self, de: FreeDataEngine):
        self.de = de
        self.football_pipeline: Optional[dict] = None
        self.tennis_pipelines: Dict[str, Optional[dict]] = {"atp": None, "wta": None}
        self.is_football_trained = False
        self._football_team_deques: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self._rng = np.random.RandomState(42)

    @property
    def is_tennis_trained(self) -> bool:
        return any(p is not None for p in self.tennis_pipelines.values())

    def load_or_train_football_model(self):
        path = CFG.ML_DIR / "football_model_v90.pkl"
        if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < 24:
            try:
                d = pickle.loads(path.read_bytes())
                self.football_pipeline = d["pipeline"]
                deques = d.get("deques", {})
                for k, v in deques.items():
                    self._football_team_deques[k] = deque(v, maxlen=10)
                self.is_football_trained = True
                logger.info("⚡ [ML FOOTBALL] Loaded from cache")
                return
            except Exception:
                pass
        self._train_football()
        if self.is_football_trained:
            try:
                path.write_bytes(pickle.dumps({
                    "pipeline": self.football_pipeline,
                    "deques": {k: list(v) for k, v in self._football_team_deques.items()}
                }))
            except Exception:
                pass

    def _train_football(self):
        df = self.de.football_data.get("all")
        if df is None or len(df) < 300:
            return
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        X, y = self._build_features(df)
        if len(X) < 200 or len(np.unique(y)) < 2:
            return

        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
        model = CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, random_state=42
            ),
            cv=3, method="isotonic"
        )
        try:
            model.fit(Xs, y)
            self.football_pipeline = {"model": model, "scaler": scaler}
            self.is_football_trained = True
            logger.info("✅ [ML FOOTBALL] Trained on %d samples", len(X))
        except Exception as e:
            logger.error("[ML FOOTBALL] %s", e)

    def _build_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self._football_team_deques = defaultdict(lambda: deque(maxlen=10))
        feats, labels = [], []

        for _, row in df.iterrows():
            ht = str(row.get("HomeTeam", "") or "")
            at = str(row.get("AwayTeam", "") or "")
            ftr = str(row.get("FTR", "") or "")
            if not ht or not at or ftr not in ["H", "D", "A"]:
                continue
            try:
                hg = float(row.get("FTHG", 0) or 0)
                ag = float(row.get("FTAG", 0) or 0)
            except Exception:
                continue

            def get_stats(team):
                h = list(self._football_team_deques[team])
                if len(h) < 3:
                    return None
                w = np.array([1 / (i + 1) for i in range(len(h))][::-1])
                w /= w.sum()
                return {
                    "avg_gs": float(np.dot(w, [x["gs"] for x in h])),
                    "avg_gc": float(np.dot(w, [x["gc"] for x in h])),
                    "form_pts": float(np.dot(w, [x["pts"] for x in h])),
                    "win_rate": sum(1 for x in h if x["pts"] == 3) / len(h)
                }

            hs = get_stats(ht)
            aws = get_stats(at)
            if hs and aws:
                feats.append([
                    hs["avg_gs"], hs["avg_gc"], hs["form_pts"], hs["win_rate"],
                    aws["avg_gs"], aws["avg_gc"], aws["form_pts"], aws["win_rate"],
                    hs["avg_gs"] - aws["avg_gc"],
                    aws["avg_gs"] - hs["avg_gc"]
                ])
                labels.append({"H": 0, "D": 1, "A": 2}[ftr])

            self._football_team_deques[ht].appendleft({
                "gs": hg, "gc": ag,
                "pts": 3 if ftr == "H" else (1 if ftr == "D" else 0)
            })
            self._football_team_deques[at].appendleft({
                "gs": ag, "gc": hg,
                "pts": 3 if ftr == "A" else (1 if ftr == "D" else 0)
            })

        if not feats:
            return np.array([]), np.array([])
        return (
            np.nan_to_num(np.array(feats, dtype=np.float64)),
            np.array(labels, dtype=np.int32)
        )

    def predict_football(self, home: str, away: str) -> Optional[dict]:
        if not self.is_football_trained:
            return None

        def get_team_features(team):
            cl = team.lower().strip()
            bm = next(
                (k for k in self._football_team_deques
                 if cl in k.lower() or k.lower() in cl),
                None
            )
            if not bm:
                return None
            h = list(self._football_team_deques[bm])
            if len(h) < 3:
                return None
            w = np.array([1 / (i + 1) for i in range(len(h))][::-1])
            w /= w.sum()
            return {
                "avg_gs": float(np.dot(w, [x["gs"] for x in h])),
                "avg_gc": float(np.dot(w, [x["gc"] for x in h])),
                "form_pts": float(np.dot(w, [x["pts"] for x in h])),
                "win_rate": sum(1 for x in h if x["pts"] == 3) / len(h)
            }

        hs = get_team_features(home)
        aws = get_team_features(away)
        if not hs or not aws:
            return None

        fv = [
            hs["avg_gs"], hs["avg_gc"], hs["form_pts"], hs["win_rate"],
            aws["avg_gs"], aws["avg_gc"], aws["form_pts"], aws["win_rate"],
            hs["avg_gs"] - aws["avg_gc"],
            aws["avg_gs"] - hs["avg_gc"]
        ]
        X = np.nan_to_num(np.array([fv], dtype=np.float64))
        Xs = self.football_pipeline["scaler"].transform(X)
        try:
            probs = self.football_pipeline["model"].predict_proba(Xs)[0]
            classes = self.football_pipeline["model"].classes_
            lm = {0: "home_win", 1: "draw", 2: "away_win"}
            return {lm.get(int(c), f"c{c}"): round(float(p), 4)
                    for c, p in zip(classes, probs)}
        except Exception as e:
            logger.warning("[ML FOOTBALL] %s", e)
            return None

    def load_or_train_tennis_model(self, is_wta: bool = False):
        tour = "wta" if is_wta else "atp"
        path = CFG.ML_DIR / f"tennis_model_{tour}_v90.pkl"
        if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < 24:
            try:
                d = pickle.loads(path.read_bytes())
                self.tennis_pipelines[tour] = d["pipeline"]
                logger.info("⚡ [ML TENNIS %s] Loaded", tour.upper())
                return
            except Exception:
                pass
        self._train_tennis(is_wta)
        if self.tennis_pipelines[tour]:
            try:
                path.write_bytes(pickle.dumps(
                    {"pipeline": self.tennis_pipelines[tour]}
                ))
            except Exception:
                pass

    def _train_tennis(self, is_wta: bool = False):
        df = self.de.wta_matches if is_wta else self.de.atp_matches
        tour = "wta" if is_wta else "atp"
        if df is None or len(df) < 500:
            logger.warning("[ML TENNIS %s] Insufficient data", tour.upper())
            return

        df = df.sort_values("tourney_date").reset_index(drop=True)
        player_history: Dict[Any, List[dict]] = defaultdict(list)
        feats, labels, weights = [], [], []

        for _, row in df.iterrows():
            wid = row.get("winner_id")
            lid = row.get("loser_id")
            wr = float(row.get("winner_rank", 0) or 0)
            lr = float(row.get("loser_rank", 0) or 0)
            if wr <= 0 or lr <= 0:
                continue

            surf = str(row.get("surface", "Hard") or "Hard").lower()
            td = float(row.get("tourney_date", 20200101) or 20200101)

            def agg(history):
                recent = history[-20:] if len(history) >= 20 else history
                if not recent:
                    return {}
                total = len(recent)
                wins = sum(1 for h in recent if h["won"])
                svpt = max(sum(h.get("svpt", 50) for h in recent), 1)
                return {
                    "win_rate": wins / total,
                    "ace_rate": sum(h.get("ace", 0) for h in recent) / svpt,
                    "n": total
                }

            w_hist = player_history.get(wid, [])
            l_hist = player_history.get(lid, [])
            w_agg = agg(w_hist)
            l_agg = agg(l_hist)

            if w_agg.get("n", 0) >= 3 and l_agg.get("n", 0) >= 3:
                is_w_p1 = wr < lr
                p1r, p2r = (wr, lr) if is_w_p1 else (lr, wr)
                p1a, p2a = (w_agg, l_agg) if is_w_p1 else (l_agg, w_agg)
                label = 1 if is_w_p1 else 0
                fv = [
                    p1r, p2r, p2r - p1r,
                    1. if surf == "hard" else 0.,
                    1. if surf == "clay" else 0.,
                    1. if surf == "grass" else 0.,
                    p1a.get("win_rate", 0.5),
                    p2a.get("win_rate", 0.5),
                    p1a.get("win_rate", 0.5) - p2a.get("win_rate", 0.5),
                    float(p1a.get("n", 0)),
                    float(p2a.get("n", 0))
                ]
                feats.append(fv)
                labels.append(label)
                weights.append(float(np.clip(
                    0.5 + 0.5 * (td - 20200101) / max(20260101 - 20200101, 1),
                    0.5, 1.0
                )))

            if wid is not None:
                player_history[wid].append({
                    "won": True,
                    "ace": float(row.get("w_ace", 0) or 0),
                    "svpt": max(float(row.get("w_svpt", 50) or 50), 1.)
                })
            if lid is not None:
                player_history[lid].append({
                    "won": False,
                    "ace": float(row.get("l_ace", 0) or 0),
                    "svpt": max(float(row.get("l_svpt", 50) or 50), 1.)
                })

        if not feats or len(np.unique(labels)) < 2:
            return

        X = np.nan_to_num(np.array(feats, dtype=np.float64))
        y = np.array(labels, dtype=np.int32)
        sw = np.array(weights, dtype=np.float64)

        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
        try:
            gb = GradientBoostingClassifier(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, random_state=42
            )
            cal = CalibratedClassifierCV(gb, cv=3, method="isotonic")
            cal.fit(Xs, y, sample_weight=sw)
            self.tennis_pipelines[tour] = {"model": cal, "scaler": scaler}
            logger.info("✅ [ML TENNIS %s] Trained on %d samples", tour.upper(), len(X))
        except Exception as e:
            logger.error("[ML TENNIS %s] %s", tour.upper(), e)

    def predict_tennis(self, pa: str, pb: str, stats: dict,
                       surface: str = "hard") -> Optional[dict]:
        tour = "wta" if stats.get("tour", "").lower() == "wta" else "atp"
        pipeline = self.tennis_pipelines.get(tour)
        if not pipeline:
            for t in ["atp", "wta"]:
                if self.tennis_pipelines.get(t):
                    pipeline = self.tennis_pipelines[t]
                    break
        if not pipeline:
            return None

        pas = stats.get("player_a", {})
        pbs = stats.get("player_b", {})
        ra = float(pas.get("current_ranking", 100) or 100)
        rb = float(pbs.get("current_ranking", 100) or 100)

        is_pa_p1 = (ra <= rb)
        p1r, p2r = (ra, rb) if is_pa_p1 else (rb, ra)
        p1wr = float(pas.get("recent_win_rate", 0.5) or 0.5)
        p2wr = float(pbs.get("recent_win_rate", 0.5) or 0.5)
        if not is_pa_p1:
            p1wr, p2wr = p2wr, p1wr

        fv = [
            p1r, p2r, p2r - p1r,
            1. if surface == "hard" else 0.,
            1. if surface == "clay" else 0.,
            1. if surface == "grass" else 0.,
            p1wr, p2wr, p1wr - p2wr,
            float(pas.get("total_matches", 10) or 10),
            float(pbs.get("total_matches", 10) or 10)
        ]

        try:
            X = np.nan_to_num(np.array([fv], dtype=np.float64))
            Xs = pipeline["scaler"].transform(X)
            probs = pipeline["model"].predict_proba(Xs)[0]
            pm = {int(c): float(p) for c, p in zip(pipeline["model"].classes_, probs)}
            p1_win = pm.get(1, 0.5)
            pa_p = p1_win if is_pa_p1 else (1 - p1_win)
            return {
                f"{pa}_win_prob": round(pa_p, 4),
                f"{pb}_win_prob": round(1 - pa_p, 4)
            }
        except Exception as e:
            logger.warning("[ML TENNIS] %s", e)
            return None


# =========================================================
# 10. POISSON ENGINE
# =========================================================
class PoissonEngine:
    @staticmethod
    def calculate(home: str, away: str,
                  df: Optional[pd.DataFrame]) -> dict:
        if df is None or df.empty:
            return {}
        req = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not req.issubset(df.columns):
            return {}
        rec = df.dropna(subset=["FTHG", "FTAG"]).tail(1500).copy()
        if len(rec) < 50:
            return {}
        la_home = rec["FTHG"].astype(float).mean()
        la_away = rec["FTAG"].astype(float).mean()
        if pd.isna(la_home) or la_home == 0:
            return {}

        def fz(t, col):
            cl = t.lower().strip()
            m = col.str.lower().str.strip() == cl
            if m.any():
                return m
            for p in cl.split():
                if len(p) > 3:
                    m2 = col.str.lower().str.contains(re.escape(p), na=False)
                    if m2.any():
                        return m2
            return pd.Series([False] * len(col), index=col.index)

        hm = rec[fz(home, rec["HomeTeam"])]
        am = rec[fz(away, rec["AwayTeam"])]
        if len(hm) < 5 or len(am) < 5:
            return {}

        ha = hm["FTHG"].astype(float).mean() / la_home
        hd = hm["FTAG"].astype(float).mean() / la_away
        aa = am["FTAG"].astype(float).mean() / la_away
        ad = am["FTHG"].astype(float).mean() / la_home

        if any(pd.isna(v) or v == 0 for v in [ha, hd, aa, ad]):
            return {}

        hxg = float(np.clip(ha * ad * la_home, 0.1, 8.))
        axg = float(np.clip(aa * hd * la_away, 0.1, 8.))

        mg = 6
        pm = np.zeros((mg + 1, mg + 1))
        for x in range(mg + 1):
            for y in range(mg + 1):
                pm[x, y] = (
                    stats_scipy.poisson.pmf(x, hxg) *
                    stats_scipy.poisson.pmf(y, axg)
                )
        t = pm.sum()
        if t == 0:
            return {}
        pm /= t

        return {
            "home_xg": round(hxg, 2),
            "away_xg": round(axg, 2),
            "home_win_prob_poisson": round(float(np.sum(np.tril(pm, -1))), 3),
            "draw_prob_poisson": round(float(np.sum(np.diag(pm))), 3),
            "away_win_prob_poisson": round(float(np.sum(np.triu(pm, 1))), 3),
        }


# =========================================================
# 11. EV ENGINE
# =========================================================
class EVEngine:
    @staticmethod
    def remove_vig(odds_list: List[float]) -> List[float]:
        implied = [1 / o for o in odds_list if o > 1.]
        if not implied:
            return []
        total = sum(implied)
        if abs(total - 1.) < 0.001:
            return implied

        def f(k):
            return sum(p ** k for p in implied) - 1.

        try:
            fa, fb = f(0.5), f(3.0)
            if fa * fb >= 0:
                return [p / total for p in implied]
            k = brentq(f, 0.5, 3.0, xtol=1e-6)
            tp = [p ** k for p in implied]
            s = sum(tp)
            return [p / s for p in tp] if s > 0 else [p / total for p in implied]
        except Exception:
            return [p / total for p in implied]

    @staticmethod
    def kelly(prob: float, odds: float) -> float:
        b = odds - 1.
        if b <= 0 or prob <= 0 or prob >= 1:
            return 0.
        k = max(0., (prob * b - (1 - prob)) / b)
        return round(min(k * CFG.KELLY_FRACTION, CFG.MAX_KELLY_PCT / 100), 4)


def calculate_ev(markets_data: dict) -> list:
    best_per_market: dict = {}

    for mk, ml in markets_data.items():
        if not isinstance(ml, list):
            continue

        sharp_prices: Dict[Tuple, List[float]] = defaultdict(list)
        soft_prices: Dict[Tuple, List[float]] = defaultdict(list)
        best_mkt: Dict[Tuple, Tuple[float, str, str]] = {}

        for entry in ml:
            if not isinstance(entry, dict):
                continue
            bk = entry.get("bookmaker_key", "")
            bk_name = entry.get("bookmaker", bk)
            is_sharp = bk in CFG.SHARP_BOOKMAKERS

            for o in entry.get("outcomes", []):
                if not isinstance(o, dict):
                    continue
                raw_name = o.get("name", "")
                if not raw_name:
                    continue
                point = o.get("point")
                comp_key = (raw_name, point)
                display_name = f"{raw_name} {point}" if point is not None else raw_name
                try:
                    price = float(o["price"])
                except Exception:
                    continue
                if price <= 1.:
                    continue

                (sharp_prices if is_sharp else soft_prices)[comp_key].append(price)
                if comp_key not in best_mkt or price > best_mkt[comp_key][0]:
                    best_mkt[comp_key] = (price, bk_name, display_name)

        if not best_mkt:
            continue

        ref_prices = {k: max(p) for k, p in sharp_prices.items() if p}
        has_sharp = bool(ref_prices)
        if not ref_prices:
            ref_prices = {k: max(p) for k, p in soft_prices.items() if p}
        if not ref_prices:
            continue

        comp_keys = list(ref_prices.keys())
        odds_list = [ref_prices[k] for k in comp_keys]
        impl_sum = sum(1 / o for o in odds_list if o > 0)

        if not (CFG.MIN_VALID_IMPLIED_SUM <= impl_sum <= CFG.MAX_VALID_IMPLIED_SUM):
            continue
        min_outcomes = CFG.MARKET_EXPECTED_OUTCOMES.get(mk, {}).get("min", 2)
        if len(comp_keys) < min_outcomes:
            continue

        try:
            tp_list = EVEngine.remove_vig(odds_list)
            if len(tp_list) != len(comp_keys):
                raise ValueError()
            tp = dict(zip(comp_keys, tp_list))
        except Exception:
            tp = {
                comp_keys[i]: (1 / odds_list[i]) / max(impl_sum, 1e-10)
                for i in range(len(comp_keys))
            }

        min_odds = CFG.H2H_MIN_ODDS if mk == "h2h" else CFG.TOTALS_MIN_ODDS
        sharp_mult = 0.75 if has_sharp else 1.0
        min_ev = (CFG.H2H_MIN_EV if mk == "h2h" else CFG.TOTALS_MIN_EV) * sharp_mult

        best_opp = None
        for ck in comp_keys:
            true_p = tp.get(ck, 0)
            if true_p <= 0 or true_p >= 1:
                continue
            bp, bbm, disp_name = best_mkt.get(ck, (0, "?", "?"))
            if bp <= 1.:
                continue

            ev = true_p * bp - 1.

            # Sanity check — EV > 20% almost always means bad/thin market data
            if ev > 0.20:
                logger.warning(
                    "[EV] Suspicious EV=%.1f%% for %s (odds=%.2f prob=%.3f bk_count=%d) — skip",
                    ev * 100, disp_name, bp, true_p,
                    len(soft_prices.get(ck, []) + sharp_prices.get(ck, []))
                )
                continue

            if ev < min_ev or bp < min_odds:
                continue

            kelly_p = EVEngine.kelly(true_p, bp)
            sp = ref_prices.get(ck, bp)
            clv = (bp / sp - 1) * 100 if sp > 0 else 0.
            bk_count = len(soft_prices.get(ck, []) + sharp_prices.get(ck, []))

            # Thin market penalty — fewer than 3 books = unreliable
            if bk_count < 3 and not has_sharp:
                logger.debug(
                    "[EV] Thin market %s: only %d books — skip",
                    disp_name, bk_count
                )
                continue

            opp = {
                "pick": disp_name,
                "market": mk,
                "market_label": _get_market_label(mk),
                "prob": round(true_p, 4),
                "odds": round(bp, 3),
                "bookmaker": bbm,
                "ev": round(ev, 4),
                "edge_pct": round(ev * 100, 2),
                "kelly_pct": round(kelly_p * 100, 2),
                "clv_pct": round(clv, 2),
                "has_sharp_line": has_sharp,
                "steam_pct": None,
                "bookmaker_count": bk_count
            }
            if best_opp is None or opp["ev"] > best_opp["ev"]:
                best_opp = opp

        if best_opp:
            best_per_market[mk] = best_opp

    return sorted(best_per_market.values(), key=lambda x: x["ev"], reverse=True)


# =========================================================
# 12. CONFIDENCE ENGINE
# =========================================================
class ConfidenceEngine:
    @classmethod
    def score(cls, opp: dict, stats: dict,
               ml_pred: Optional[dict] = None,
               poisson_pred: Optional[dict] = None) -> int:
        s = 42  # base

        ev = opp.get("ev", 0) * 100

        # Hard cap on suspicious EV
        if ev > 20:
            return 20

        # EV tiers
        if ev > 8:
            s += 16
        elif ev > 5:
            s += 13
        elif ev > 3:
            s += 9
        elif ev > 2:
            s += 6
        elif ev > 1:
            s += 3
        else:
            s -= 8

        # Sharp line — strongest signal
        if opp.get("has_sharp_line"):
            s += 14

        # CLV
        clv = opp.get("clv_pct", 0)
        if clv > 3:
            s += 8
        elif clv > 1.5:
            s += 4

        # Kelly
        kelly = opp.get("kelly_pct", 0)
        if kelly > 2:
            s += 6
        elif kelly > 1:
            s += 3
        elif kelly < 0.5:
            s -= 4

        # Book count
        bk_count = opp.get("bookmaker_count", 1)
        if bk_count >= 8:
            s += 7
        elif bk_count >= 5:
            s += 4
        elif bk_count >= 3:
            s += 1
        else:
            s -= 6  # very thin market

        # ── Historical data quality (GitHub) ─────────────────
        dq = "none"
        if stats.get("historical_data"):
            dq = stats["historical_data"].get(
                "data_quality_summary", {}
            ).get("overall", "poor")
        elif stats.get("football_stats"):
            hq = stats["football_stats"].get("home", {}).get("data_quality", "poor")
            aq = stats["football_stats"].get("away", {}).get("data_quality", "poor")
            if hq == "good" and aq == "good":
                dq = "good"
            elif hq != "poor" or aq != "poor":
                dq = "limited"
            else:
                dq = "poor"

        if dq == "good":
            s += 10
        elif dq == "limited":
            s += 5
        elif dq == "poor":
            s -= 3
        elif dq == "none":
            s -= 9

        # ── TSDB live data ────────────────────────────────────
        tsdb_stats = stats.get("tsdb_stats", {})
        h_tsdb = tsdb_stats.get("home", {})
        a_tsdb = tsdb_stats.get("away", {})
        h_tsdb_q = h_tsdb.get("data_quality", "none")
        a_tsdb_q = a_tsdb.get("data_quality", "none")

        if h_tsdb_q == "good" and a_tsdb_q == "good":
            s += 9
        elif h_tsdb_q in ("good", "limited") and a_tsdb_q in ("good", "limited"):
            s += 5
        elif h_tsdb_q in ("good", "limited") or a_tsdb_q in ("good", "limited"):
            s += 2

        # Form alignment check
        h_wr = h_tsdb.get("win_rate", 0)
        a_wr = a_tsdb.get("win_rate", 0)
        h_name = h_tsdb.get("team_name", "")
        if h_wr and a_wr and abs(h_wr - a_wr) > 0.20 and h_name:
            pick_lower = opp.get("pick", "").lower()
            pick_favors_home = any(
                w in pick_lower
                for w in h_name.lower().split()
                if len(w) > 3
            )
            stronger_is_home = h_wr > a_wr
            if pick_favors_home == stronger_is_home:
                s += 5   # pick aligned with better form team
            else:
                s -= 4   # pick against form

        # ── ML model ─────────────────────────────────────────
        if ml_pred:
            mx = max(
                (v for v in ml_pred.values()
                 if isinstance(v, (float, int)) and 0 < v <= 1),
                default=0
            )
            if mx > 0.68:
                s += 11
            elif mx > 0.62:
                s += 7
            elif mx > 0.55:
                s += 3

        # ── Poisson ───────────────────────────────────────────
        if poisson_pred:
            s += 5
            if ml_pred:
                s += 3  # consensus bonus

        # ── US sports: standings-only penalty ─────────────────
        us = stats.get("us_sports", {})
        h_src = us.get("home", {}).get("source", "")
        if h_src in ["mlb_api", "nhl_api"] and dq == "none":
            s -= 7

        # ── Steam / line movement ─────────────────────────────
        steam = opp.get("steam_pct")
        if steam is not None:
            if steam >= 3.0:
                s += 9
            elif steam >= 1.5:
                s += 5
            elif steam <= -5:
                s -= 8

        return int(np.clip(s, 0, 100))


# =========================================================
# 13. UTILITIES
# =========================================================
def _get_market_label(mk: str) -> str:
    return {
        "h2h": "Match Winner",
        "totals": "Over/Under",
        "spreads": "Handicap"
    }.get(mk, mk.replace("_", " ").title())


def _get_sport_emoji(sk: str) -> str:
    return {
        "tennis": "🎾", "football": "⚽", "basketball": "🏀",
        "baseball": "⚾", "hockey": "🏒", "cricket": "🏏"
    }.get(sk, "🏆")


def normalize_sport_key(sport_title: str) -> str:
    lower = (sport_title or "").lower()
    if any(k in lower for k in ["tennis", "atp", "wta"]):
        return "tennis"
    if any(k in lower for k in ["soccer", "football", "premier", "liga",
                                  "bundesliga", "serie", "ligue", "champions"]):
        return "football"
    if any(k in lower for k in ["basketball", "nba", "euroleague"]):
        return "basketball"
    if any(k in lower for k in ["baseball", "mlb"]):
        return "baseball"
    if any(k in lower for k in ["hockey", "nhl"]):
        return "hockey"
    if any(k in lower for k in ["cricket", "ipl", "t20"]):
        return "cricket"
    return "other"


def clean_team_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(name or "")).strip()


def get_countdown_str(ct: str, now: datetime) -> str:
    try:
        mt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        if mt.tzinfo is None:
            mt = mt.replace(tzinfo=timezone.utc)
        mins = int((mt - now).total_seconds() / 60)
        if mins > 60:
            return f"{mins // 60}h {mins % 60}m"
        if mins > 0:
            return f"{mins}m"
        return "LIVE"
    except Exception:
        return "N/A"


def translate_pick(pick: str, market: str, home: str, away: str) -> str:
    pick_lower = pick.lower().strip()
    if market.lower() == "h2h":
        home_sim = difflib.SequenceMatcher(None, home.lower(), pick_lower).ratio()
        away_sim = difflib.SequenceMatcher(None, away.lower(), pick_lower).ratio()
        if home_sim > away_sim and home_sim > 0.3:
            return f"{home} to Win"
        elif away_sim > home_sim and away_sim > 0.3:
            return f"{away} to Win"
        elif "draw" in pick_lower or "tie" in pick_lower:
            return "Draw"
    elif "total" in market.lower():
        m = re.search(r"\b(over|under)\b\s*([\d.]+)", pick_lower)
        if m:
            direction = m.group(1).capitalize()
            line = m.group(2)
            return f"{direction} {line}"
    return pick.title()


def get_confidence_label(fc: int) -> str:
    if fc >= 78:
        return "خیلی قوی 🔥🔥"
    elif fc >= 70:
        return "قوی 🔥"
    elif fc >= 63:
        return "متوسط ✅"
    return "استاندارد ⚡"


# =========================================================
# 14. LINE MOVEMENT TRACKER
# =========================================================
class LineMovementTracker:
    def __init__(self):
        self._path = CFG.CACHE_DIR / "line_movement.json"
        self._lock = threading.Lock()
        self.data = CacheManager.load(self._path)
        self._cleanup()

    def _cleanup(self):
        now = datetime.now(timezone.utc)
        to_del = [
            k for k, v in self.data.items()
            if isinstance(v, dict) and "timestamp" in v
            and (now - datetime.fromisoformat(v["timestamp"]).replace(
                tzinfo=timezone.utc)) > timedelta(hours=48)
        ]
        for k in to_del:
            self.data.pop(k, None)

    def record(self, home: str, away: str, market: str,
               outcome: str, odds: float) -> Optional[float]:
        if odds <= 1.:
            return None
        mk = hashlib.md5(
            f"{home}|{away}|{market}|{outcome}".encode()
        ).hexdigest()
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            if mk not in self.data:
                self.data[mk] = {
                    "initial_odds": odds,
                    "current_odds": odds,
                    "timestamp": now
                }
                CacheManager.save(self._path, self.data)
                return None
            init = self.data[mk].get("initial_odds", odds)
            self.data[mk].update({"current_odds": odds, "timestamp": now})
            CacheManager.save(self._path, self.data)
        return round((init / odds - 1) * 100, 2) if init > 0 else 0.


line_tracker = LineMovementTracker()


# =========================================================
# 15. PERFORMANCE TRACKER
# =========================================================
class PerformanceTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.data = CacheManager.load(CFG.PERFORMANCE_FILE)
        self.data.setdefault("signals", [])
        self.data.setdefault("summary", {})

    def record(self, home, away, pick, market, odds, ev,
               confidence, prob, sport="other", api_sport_key=""):
        sig = {
            "id": hashlib.md5(
                f"{home}|{away}|{market}|{datetime.now(timezone.utc).date()}".encode()
            ).hexdigest()[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sport": sport,
            "api_sport_key": api_sport_key,
            "home": home, "away": away,
            "pick": pick, "market": market,
            "odds": odds, "ev": ev,
            "confidence": confidence,
            "implied_prob": prob,
            "outcome": None,
            "profit_loss": None
        }
        with self._lock:
            self.data["signals"].append(sig)
            if len(self.data["signals"]) > 500:
                self.data["signals"] = self.data["signals"][-500:]
        CacheManager.save(CFG.PERFORMANCE_FILE, self.data)


perf_tracker = PerformanceTracker()


# =========================================================
# 16. AI DECISION ENGINE
# =========================================================
def make_ai_decision(home: str, away: str, sport: str, sport_key: str,
                     opp: dict, stats: dict, math_score: int,
                     ml_pred: Optional[dict] = None,
                     poisson_pred: Optional[dict] = None) -> dict:
    default = {
        "sport_emoji": _get_sport_emoji(sport_key),
        "decision": "SKIP",
        "ai_confidence": math_score,
        "math_confidence": math_score,
        "final_confidence": math_score,
        "risk_level": "High",
        "logic": "Math score below threshold.",
        "key_factors": [],
        "red_flags": []
    }

    if math_score < CFG.MIN_MATH_SCORE_TO_CALL_AI:
        return {
            **default,
            "logic": f"Math score {math_score} < threshold {CFG.MIN_MATH_SCORE_TO_CALL_AI}"
        }

    # ── Build structured prompt ────────────────────────────
    lines = [
        f"MATCH: {home} vs {away}",
        f"SPORT: {sport} | MARKET: {opp['market_label']}",
        f"PICK: {opp['pick']} @ {opp['odds']}",
        f"",
        f"MARKET DATA:",
        f"  True Prob: {opp['prob']*100:.1f}%",
        f"  EV: {opp['edge_pct']:+.2f}%",
        f"  Kelly: {opp.get('kelly_pct',0):.1f}%",
        f"  Sharp Line: {opp.get('has_sharp_line', False)}",
        f"  CLV: {opp.get('clv_pct',0):+.1f}%",
        f"  Books: {opp.get('bookmaker_count',1)}",
        f"  Steam: {opp.get('steam_pct','first_obs')}",
        f"  Math Score: {math_score}/100",
    ]

    # Tennis stats
    if stats.get("historical_data"):
        pa = stats["historical_data"].get("player_a", {})
        pb = stats["historical_data"].get("player_b", {})
        h2h = stats["historical_data"].get("h2h", {})
        dq = stats["historical_data"].get("data_quality_summary", {})
        lines += [
            f"",
            f"TENNIS DATA (quality={dq.get('overall','?')}):",
            f"  {home}:",
            f"    Ranking: #{pa.get('current_ranking','?')}",
            f"    Win Rate (recent): {pa.get('recent_win_rate',0)*100:.1f}%",
            f"    Form (last 10): {pa.get('recent_form','N/A')}",
            f"    Total Matches: {pa.get('total_matches',0)}",
            f"    Aces/Match: {pa.get('aces_per_match','?')}",
            f"    1st Serve In: {pa.get('first_serve_in_pct',0)*100:.1f}%",
            f"    BP Saved: {pa.get('bp_saved_pct',0)*100:.1f}%",
            f"  {away}:",
            f"    Ranking: #{pb.get('current_ranking','?')}",
            f"    Win Rate (recent): {pb.get('recent_win_rate',0)*100:.1f}%",
            f"    Form (last 10): {pb.get('recent_form','N/A')}",
            f"    Total Matches: {pb.get('total_matches',0)}",
            f"    Aces/Match: {pb.get('aces_per_match','?')}",
            f"    1st Serve In: {pb.get('first_serve_in_pct',0)*100:.1f}%",
            f"    BP Saved: {pb.get('bp_saved_pct',0)*100:.1f}%",
            f"  H2H: {h2h.get('total',0)} matches | {h2h.get('dominance','balanced')}",
            f"    {home} wins: {h2h.get(home+'_wins', h2h.get('home_wins',0))}",
            f"    {away} wins: {h2h.get(away+'_wins', h2h.get('away_wins',0))}",
        ]
        if h2h.get("by_surface"):
            lines.append(f"  H2H by surface: {json.dumps(h2h['by_surface'])}")

    # Football stats
    if stats.get("football_stats"):
        hm = stats["football_stats"].get("home", {})
        aw = stats["football_stats"].get("away", {})
        h2h = stats["football_stats"].get("h2h", {})
        lines += [
            f"",
            f"FOOTBALL DATA:",
            f"  {home} (Home):",
            f"    Form: {hm.get('github_form', hm.get('form','N/A'))}",
            f"    Goals Scored/Game: {hm.get('github_avg_scored', hm.get('avg_scored',0)):.2f}",
            f"    Goals Conceded/Game: {hm.get('github_avg_conceded', hm.get('avg_conceded',0)):.2f}",
            f"    Win Rate: {hm.get('github_win_rate', hm.get('win_rate',0))*100:.1f}%",
            f"    Over 2.5 Rate: {hm.get('github_over25_rate', hm.get('over25_rate',0))*100:.1f}%",
            f"    Data Quality: {hm.get('data_quality','?')}",
            f"    Matches Analyzed: {hm.get('github_matches', hm.get('matches_analyzed',0))}",
            f"  {away} (Away):",
            f"    Form: {aw.get('github_form', aw.get('form','N/A'))}",
            f"    Goals Scored/Game: {aw.get('github_avg_scored', aw.get('avg_scored',0)):.2f}",
            f"    Goals Conceded/Game: {aw.get('github_avg_conceded', aw.get('avg_conceded',0)):.2f}",
            f"    Win Rate: {aw.get('github_win_rate', aw.get('win_rate',0))*100:.1f}%",
            f"    Over 2.5 Rate: {aw.get('github_over25_rate', aw.get('over25_rate',0))*100:.1f}%",
            f"    Data Quality: {aw.get('data_quality','?')}",
            f"    Matches Analyzed: {aw.get('github_matches', aw.get('matches_analyzed',0))}",
        ]
        if h2h:
            lines += [
                f"  H2H ({h2h.get('total_matches',0)} matches):",
                f"    Avg Goals: {h2h.get('avg_goals',0):.2f}",
                f"    Over 2.5 Rate: {h2h.get('over25_rate',0)*100:.1f}%",
                f"    BTTS Rate: {h2h.get('btts_rate',0)*100:.1f}%",
            ]
    # Poisson
    if poisson_pred:
        lines += [
            f"",
            f"POISSON MODEL:",
            f"  Home xG: {poisson_pred.get('home_xg','?')}",
            f"  Away xG: {poisson_pred.get('away_xg','?')}",
            f"  Home Win: {poisson_pred.get('home_win_prob_poisson',0)*100:.1f}%",
            f"  Draw: {poisson_pred.get('draw_prob_poisson',0)*100:.1f}%",
            f"  Away Win: {poisson_pred.get('away_win_prob_poisson',0)*100:.1f}%",
        ]

    # ML model
    if ml_pred:
        lines += [f"", f"ML MODEL:"]
        for k, v in ml_pred.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v*100:.1f}%")

    # TSDB live stats
    if stats.get("tsdb_stats"):
        h_ts = stats["tsdb_stats"].get("home", {})
        a_ts = stats["tsdb_stats"].get("away", {})
        if h_ts or a_ts:
            lines += [f"", f"LIVE TEAM DATA (TheSportsDB):"]
            if h_ts:
                lines += [
                    f"  {home}:",
                    f"    League: {h_ts.get('league','?')}",
                    f"    Form: {h_ts.get('form','?')}",
                    f"    Win Rate: {h_ts.get('win_rate',0)*100:.1f}%",
                    f"    Avg Scored: {h_ts.get('avg_scored',0):.2f}",
                    f"    Avg Conceded: {h_ts.get('avg_conceded',0):.2f}",
                    f"    Matches: {h_ts.get('matches_analyzed',0)}",
                ]
            if a_ts:
                lines += [
                    f"  {away}:",
                    f"    League: {a_ts.get('league','?')}",
                    f"    Form: {a_ts.get('form','?')}",
                    f"    Win Rate: {a_ts.get('win_rate',0)*100:.1f}%",
                    f"    Avg Scored: {a_ts.get('avg_scored',0):.2f}",
                    f"    Avg Conceded: {a_ts.get('avg_conceded',0):.2f}",
                    f"    Matches: {a_ts.get('matches_analyzed',0)}",
                ]

    # US Sports
    if stats.get("us_sports"):
        us = stats["us_sports"]
        lines += [
            f"",
            f"US SPORTS DATA:",
            f"  {home}: {json.dumps(us.get('home',{}))}",
            f"  {away}: {json.dumps(us.get('away',{}))}",
        ]

    prompt = "\n".join(lines)
    ai_data = ai_manager.generate(prompt)

    if not ai_data or not isinstance(ai_data, dict):
        fb_decision = "BET" if math_score >= 58 else "SKIP"
        return {
            **default,
            "decision": fb_decision,
            "logic": "AI unavailable — math fallback."
        }

    decision = str(ai_data.get("decision", "SKIP")).upper().strip()
    if decision not in ["BET", "SKIP"]:
        decision = "SKIP"

    try:
        ai_conf = int(np.clip(float(ai_data.get("confidence", math_score)), 0, 100))
    except Exception:
        ai_conf = math_score

    # ── Data quality caps ──────────────────────────────────
    dq_overall = "none"
    if stats.get("historical_data"):
        dq_overall = stats["historical_data"].get(
            "data_quality_summary", {}
        ).get("overall", "poor")
    elif stats.get("football_stats"):
        hq = stats["football_stats"].get("home", {}).get("data_quality", "poor")
        aq = stats["football_stats"].get("away", {}).get("data_quality", "poor")
        if hq == "good" and aq == "good":
            dq_overall = "good"
        elif hq != "poor" or aq != "poor":
            dq_overall = "limited"
        else:
            dq_overall = "poor"

    # TSDB can upgrade dq if GitHub has none
    if dq_overall == "none":
        h_tsdb_q = stats.get("tsdb_stats", {}).get("home", {}).get("data_quality", "none")
        a_tsdb_q = stats.get("tsdb_stats", {}).get("away", {}).get("data_quality", "none")
        if h_tsdb_q == "good" and a_tsdb_q == "good":
            dq_overall = "limited"  # TSDB alone = limited
        elif h_tsdb_q in ("good", "limited") or a_tsdb_q in ("good", "limited"):
            dq_overall = "poor"

    if dq_overall == "none" and ai_conf > 60:
        ai_conf = 60
        logger.warning("[AI] No data at all → cap 60")
    elif dq_overall == "poor" and ai_conf > 65:
        ai_conf = 65
        logger.warning("[AI] Poor data → cap 65")
    elif dq_overall == "limited" and ai_conf > 74:
        ai_conf = 74
        logger.warning("[AI] Limited data → cap 74")

    # Groq calibration
    if (ai_manager._last_provider == "groq" and
            ai_conf >= 70 and opp.get("ev", 0) * 100 < 2.5):
        ai_conf = min(ai_conf - 7, 68)

    # ── Blend AI + Math 50/50 ──────────────────────────────
    hybrid = ai_conf * CFG.AI_WEIGHT + math_score * CFG.MATH_WEIGHT
    ai_delta = hybrid - math_score
    if ai_delta > CFG.MAX_AI_BOOST:
        hybrid = math_score + CFG.MAX_AI_BOOST
    elif ai_delta < -CFG.MAX_AI_PENALTY:
        hybrid = math_score - CFG.MAX_AI_PENALTY
    final = int(np.clip(hybrid, 0, 100))

    # Consistency check
    if decision == "BET" and ai_conf < 50:
        decision = "SKIP"
        logger.warning("[AI] BET conf=%d too low → SKIP", ai_conf)

    # Sharp line override
    if (decision == "SKIP" and
            opp.get("has_sharp_line") and
            opp.get("ev", 0) * 100 > 4.0 and
            math_score > 52):
        decision = "BET"
        final = max(final, 63)
        logger.info("[AI] Sharp override → BET EV=%.1f%%", opp["ev"] * 100)

    kf = [str(f)[:120] for f in (ai_data.get("key_factors") or [])[:5]]
    rf = [str(f)[:120] for f in (ai_data.get("red_flags") or [])[:3]]
    rl = str(ai_data.get("risk_level", "Medium"))
    if rl not in ["Low", "Medium", "High"]:
        rl = "Medium"
    se = str(ai_data.get("sport_emoji", "")).strip() or _get_sport_emoji(sport_key)
    logic = str(ai_data.get("logic", default["logic"]))[:500]

    logger.info(
        "[AI] %s vs %s | %s | AI:%d Math:%d Final:%d | %s",
        home, away, decision, ai_conf, math_score, final,
        ai_manager._last_provider
    )

    return {
        "sport_emoji": se,
        "decision": decision,
        "ai_confidence": ai_conf,
        "math_confidence": math_score,
        "final_confidence": final,
        "risk_level": rl,
        "logic": logic,
        "key_factors": kf,
        "red_flags": rf
    }


# =========================================================
# 17. TELEGRAM
# =========================================================
def send_telegram(msg: str) -> bool:
    MAX = 4000
    chunks = []
    if len(msg) <= MAX:
        chunks = [msg]
    else:
        cur = ""
        for line in msg.split("\n"):
            if len(cur) + len(line) + 1 > MAX:
                if cur:
                    chunks.append(cur.strip())
                cur = line + "\n"
            else:
                cur += line + "\n"
        if cur.strip():
            chunks.append(cur.strip())
    ok = True
    for chunk in chunks:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                      "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=15)
            if not r.ok:
                logger.error("Telegram [%d]: %s", r.status_code, r.text[:200])
                ok = False
        except requests.RequestException as e:
            logger.error("Telegram: %s", e)
            ok = False
    return ok


def build_message(home, away, sport, sport_key, opp, ai_data, stats,
                   math_score, ml_pred, poisson_pred, now_utc,
                   commence_time) -> str:
    fc   = ai_data["final_confidence"]
    he   = html_lib.escape(home)
    ae   = html_lib.escape(away)

    sport_labels = {
        "tennis": "Tennis 🎾", "football": "Football ⚽",
        "basketball": "Basketball 🏀", "baseball": "Baseball ⚾",
        "hockey": "Ice Hockey 🏒", "cricket": "Cricket 🏏"
    }
    sport_en = sport_labels.get(sport_key, f"{sport} 🏆")

    public_pick  = translate_pick(opp["pick"], opp["market"], home, away)
    pick_escaped = html_lib.escape(public_pick)
    countdown    = get_countdown_str(commence_time, now_utc)
    sharp_badge  = "🔪 " if opp.get("has_sharp_line") else ""

    if fc >= 78:
        conf_label = "Very Strong 🔥🔥"
    elif fc >= 70:
        conf_label = "Strong 🔥"
    elif fc >= 63:
        conf_label = "Good ✅"
    else:
        conf_label = "Standard ⚡"

    risk_labels = {"Low": "Low 🟢", "Medium": "Medium 🟠", "High": "High 🔴"}
    risk_en = risk_labels.get(ai_data["risk_level"], "Medium 🟠")

    logic = html_lib.escape(str(ai_data.get("logic", ""))[:350])

    # Compact stat line
    stat_line = ""

    if stats.get("football_stats"):
        hm = stats["football_stats"].get("home", {})
        aw = stats["football_stats"].get("away", {})
        h_form = hm.get("form", hm.get("github_form", hm.get("recent_form_5", "?")))
        a_form = aw.get("form", aw.get("github_form", aw.get("recent_form_5", "?")))
        h_gs   = hm.get("avg_scored", hm.get("github_avg_scored", 0))
        h_gc   = hm.get("avg_conceded", hm.get("github_avg_conceded", 0))
        a_gs   = aw.get("avg_scored", aw.get("github_avg_scored", 0))
        a_gc   = aw.get("avg_conceded", aw.get("github_avg_conceded", 0))
        h_rank = hm.get("standing", {}).get("rank", "")
        a_rank = aw.get("standing", {}).get("rank", "")
        h_rank_str = f" #{h_rank}" if h_rank else ""
        a_rank_str = f" #{a_rank}" if a_rank else ""
        stat_line += (
            f"\n{he}{h_rank_str}: {h_form}  {h_gs:.1f}/{h_gc:.1f}\n"
            f"{ae}{a_rank_str}: {a_form}  {a_gs:.1f}/{a_gc:.1f}"
        )
        if poisson_pred:
            stat_line += (
                f"\nxG {poisson_pred.get('home_xg','?')}-{poisson_pred.get('away_xg','?')}"
                f"  H{poisson_pred.get('home_win_prob_poisson',0)*100:.0f}%"
                f" D{poisson_pred.get('draw_prob_poisson',0)*100:.0f}%"
                f" A{poisson_pred.get('away_win_prob_poisson',0)*100:.0f}%"
            )

    elif stats.get("historical_data"):
        pa  = stats["historical_data"].get("player_a", {})
        pb  = stats["historical_data"].get("player_b", {})
        h2h = stats["historical_data"].get("h2h", {})
        stat_line += (
            f"\n{he}: #{pa.get('current_ranking','?')}  "
            f"WR {pa.get('recent_win_rate',0)*100:.0f}%  "
            f"{pa.get('recent_form','?')[:6]}\n"
            f"{ae}: #{pb.get('current_ranking','?')}  "
            f"WR {pb.get('recent_win_rate',0)*100:.0f}%  "
            f"{pb.get('recent_form','?')[:6]}"
        )
        if h2h.get("total", 0) > 0:
            stat_line += (
                f"\nH2H {h2h.get('total',0)}  {h2h.get('dominance','balanced')}"
            )

    elif stats.get("us_sports"):
        us = stats["us_sports"]
        hs = us.get("home", {})
        aw = us.get("away", {})
        if hs or aw:
            stat_line += (
                f"\n{he}: WR {hs.get('win_rate', hs.get('win_pct',0))*100:.0f}%"
                f"  {hs.get('form', hs.get('recent_record','?'))}\n"
                f"{ae}: WR {aw.get('win_rate', aw.get('win_pct',0))*100:.0f}%"
                f"  {aw.get('form', aw.get('recent_record','?'))}"
            )

    if ml_pred:
        ml_parts = [
            f"{k.replace('_win_prob','').replace('_',' ').title()} {v*100:.0f}%"
            for k, v in ml_pred.items()
            if isinstance(v, float) and "prob" in k
        ]
        if ml_parts:
            stat_line += f"\nML: {' | '.join(ml_parts)}"

    return (
        f"{ai_data.get('sport_emoji','🏆')} <b>{sport_en}</b>\n\n"
        f"<b>{he}</b> vs <b>{ae}</b>   {countdown}\n\n"
        f"Pick: \n<code>{sharp_badge}{pick_escaped} @ <b>{opp['odds']:.2f}</b></code>\n"
        #f"Odds: <b>{opp['odds']:.2f}</b>   Edge: <b>{opp['edge_pct']:.1f}%</b>"
        f"   Stake: <b>{opp.get('kelly_pct',0):.1f}%</b>\n\n"
        f"Signal: <b>{conf_label}</b> ({fc}%)   Risk: <b>{risk_en}</b>\n\n"
        f"<blockquote>{logic}</blockquote>"
        f"{html_lib.escape(stat_line) if stat_line else ''}\n\n"
        f"<i>{html_lib.escape(CFG.TELEGRAM_ID)}</i>"
    )
                       
# =========================================================
# 18. ODDS FETCHER
# =========================================================
class OddsCache:
    def __init__(self):
        self.cache = CacheManager.load(CFG.ODDS_CACHE_FILE)

    def _key(self, markets, wh):
        raw = f"{','.join(sorted(markets))}|{wh}|{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H')}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, markets, wh):
        k = self._key(markets, wh)
        if CacheManager.is_valid(self.cache, k, CFG.TTL_ODDS_CACHE_MINUTES / 60):
            d = CacheManager.get(self.cache, k)
            if d:
                logger.info("💾 [ODDS CACHE] HIT %d events", len(d))
                return d
        return None

    def save(self, markets, wh, events):
        k = self._key(markets, wh)
        self.cache = CacheManager.set(self.cache, k, events)
        CacheManager.save(CFG.ODDS_CACHE_FILE, self.cache)
        logger.info("💾 [ODDS CACHE] Saved %d events", len(events))

    def get_stale(self, markets, wh, max_ttl=2.):
        k = self._key(markets, wh)
        if CacheManager.is_valid(self.cache, k, max_ttl):
            return CacheManager.get(self.cache, k)
        return None


odds_cache = OddsCache()


async def fetch_market_async(session, market, now_utc, api_key, label):
    end = now_utc + timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    params = {
        "apiKey": api_key,
        "regions": CFG.ODDS_API_REGIONS,
        "markets": market,
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    try:
        async with session.get(
            "https://api.the-odds-api.com/v4/sports/upcoming/odds",
            params=params,
            timeout=aiohttp.ClientTimeout(total=25)
        ) as r:
            rem = int(r.headers.get("x-requests-remaining", -1))
            if r.status == 200:
                events = await r.json(content_type=None)
                valid = []
                for e in events:
                    if not isinstance(e, dict):
                        continue
                    try:
                        mt = datetime.fromisoformat(
                            e.get("commence_time", "").replace("Z", "+00:00")
                        )
                        if mt.tzinfo is None:
                            mt = mt.replace(tzinfo=timezone.utc)
                        if now_utc <= mt <= end:
                            valid.append(e)
                    except Exception:
                        continue
                logger.info("🔑 [%s] %s → %d events (rem:%d)",
                            label, market, len(valid), rem)
                return valid, 200, None
            err = await r.text()
            reasons = {401: "Invalid key", 402: "Quota exhausted", 429: "Rate limited"}
            return [], r.status, reasons.get(r.status, f"HTTP {r.status}")
    except asyncio.TimeoutError:
        return [], 0, "Timeout"
    except Exception as e:
        return [], 0, str(e)[:60]


async def fetch_all_odds() -> list:
    now = datetime.now(timezone.utc)
    cached = odds_cache.get(CFG.ODDS_API_MARKETS, CFG.MATCH_WINDOW_HOURS)
    if cached:
        return cached

    logger.info("📡 [ODDS] Fetching from API...")
    all_events: Dict[str, dict] = {}
    pending_markets = list(CFG.ODDS_API_MARKETS)

    for ki in odds_key_manager.get_active_keys():
        if not pending_markets:
            break
        ak = ki["key"]
        label = ki["label"]
        ki["calls"] += 1

        conn = aiohttp.TCPConnector(limit=10, ssl=False)
        async with aiohttp.ClientSession(connector=conn) as sess:
            tasks = [fetch_market_async(sess, m, now, ak, label)
                     for m in pending_markets]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        failed_markets = []
        hard_fail = None

        for i, res in enumerate(results):
            m = pending_markets[i]
            if isinstance(res, Exception):
                failed_markets.append(m)
                continue
            events, status, err = res
            if status == 200:
                for e in events:
                    eid = e.get("id")
                    if not eid:
                        continue
                    if eid not in all_events:
                        all_events[eid] = {
                            "id": eid,
                            "sport_key": e.get("sport_key", ""),
                            "sport_title": e.get("sport_title", ""),
                            "commence_time": e.get("commence_time", ""),
                            "home_team": e.get("home_team", ""),
                            "away_team": e.get("away_team", ""),
                            "_markets_data": {}
                        }
                    for bm in e.get("bookmakers", []):
                        bk = bm.get("key", "")
                        bt = bm.get("title", bk)
                        for md in bm.get("markets", []):
                            mk = md.get("key", "")
                            if not mk:
                                continue
                            all_events[eid]["_markets_data"].setdefault(mk, []).append({
                                "bookmaker": bt,
                                "bookmaker_key": bk,
                                "outcomes": md.get("outcomes", [])
                            })
            else:
                failed_markets.append(m)
                if status in [401, 402, 429]:
                    hard_fail = status

        if hard_fail:
            idx = next((i for i, k in enumerate(odds_key_manager.keys)
                        if k["label"] == label), -1)
            if idx >= 0:
                odds_key_manager.mark_failed(idx, f"HTTP {hard_fail}")

        pending_markets = failed_markets

    final = list(all_events.values())
    if final:
        odds_cache.save(CFG.ODDS_API_MARKETS, CFG.MATCH_WINDOW_HOURS, final)
    elif not final:
        stale = odds_cache.get_stale(CFG.ODDS_API_MARKETS, CFG.MATCH_WINDOW_HOURS, 2.)
        if stale:
            logger.warning("💾 [STALE] Using stale cache: %d events", len(stale))
            return stale

    logger.info("📊 %s", odds_key_manager.get_summary())
    return final


# =========================================================
# 19. MAIN PIPELINE
# =========================================================
async def async_main():
    logger.info("=" * 65)
    logger.info("  ZBET90 ENGINE v9.0 | Multi-Source | API-Football+TSDB+GitHub")
    logger.info("=" * 65)

    sent = SentHistory()
    now  = datetime.now(timezone.utc)

    logger.info("📥 [PHASE 1] Loading data sources...")
    de = FreeDataEngine()
    de.load_tennis_data()
    de.load_football_data()
    de.load_nba_data()
    de.load_nhl_data()
    de.load_mlb_data()

    logger.info("🧠 [PHASE 2] Training ML models...")
    ml = MLPredictionEngine(de)
    ml.load_or_train_football_model()
    ml.load_or_train_tennis_model(is_wta=False)
    ml.load_or_train_tennis_model(is_wta=True)

    logger.info("📡 [PHASE 3] Fetching odds (%.1fh window)...",
                CFG.MATCH_WINDOW_HOURS)
    events = await fetch_all_odds()
    if not events:
        logger.info("❌ No events in window.")
        return

    logger.info("🔍 [PHASE 4] Analyzing %d events...", len(events))
    events.sort(key=lambda x: x.get("commence_time", ""))

    total_sent = total_analyzed = 0
    skip_counts = {
        "no_opp": 0, "ev": 0, "sent": 0,
        "math": 0, "ai": 0, "conf": 0
    }

    for event in events:
        home      = clean_team_name(event.get("home_team", ""))
        away      = clean_team_name(event.get("away_team", ""))
        sport     = event.get("sport_title", "Unknown")
        sport_key = normalize_sport_key(sport)
        if not home or not away:
            continue

        markets_data = event.get("_markets_data", {})
        opps = calculate_ev(markets_data)
        if not opps:
            skip_counts["no_opp"] += 1
            continue

        opp = opps[0]
        total_analyzed += 1

        if opp["ev"] < CFG.MATH_MIN_EV_TO_ANALYZE:
            skip_counts["ev"] += 1
            continue

        if sent.was_sent(home, away, opp["market"]):
            skip_counts["sent"] += 1
            logger.info("⏭️ SENT: %s vs %s", home, away)
            continue

        opp["steam_pct"] = line_tracker.record(
            home, away, opp["market"], opp["pick"], opp["odds"]
        )

        # ── Gather stats ───────────────────────────────────────
        stats: dict  = {}
        ml_pred      = None
        poisson_pred = None

        logger.info("📊 [DATA] Gathering stats: %s vs %s [%s]",
                    home, away, sport_key)

        # ── TENNIS ────────────────────────────────────────────
        if sport_key == "tennis":
            is_wta = "wta" in sport.lower()

            # 1. GitHub CSV stats (primary for tennis)
            ts = de.get_tennis_stats(home, away, is_wta)
            if ts:
                ts["tour"] = "wta" if is_wta else "atp"
                stats["historical_data"] = ts
                dq = ts.get("data_quality_summary", {}).get("overall", "?")
                logger.info("  ✅ GitHub tennis: Q=%s", dq)
            else:
                logger.info("  ⚠️ GitHub tennis: no data")

            # 2. TSDB player lookup (ranking, nationality)
            for p_name, p_key in [(home, "player_a"), (away, "player_b")]:
                try:
                    tp = tsdb.get_player_stats(p_name)
                    if tp and tp.get("player_id"):
                        if "historical_data" in stats:
                            stats["historical_data"][p_key]["tsdb_nationality"] = \
                                tp.get("nationality", "")
                            stats["historical_data"][p_key]["tsdb_sport"] = \
                                tp.get("sport", "")
                        logger.debug("  ✅ TSDB player: %s", p_name)
                except Exception as e:
                    logger.debug("  [TSDB player] %s: %s", p_name, e)

            # 3. ML prediction
            if ml.is_tennis_trained and ts:
                sport_lower = sport.lower()
                if any(k in sport_lower for k in ["wimbledon", "grass", "queens"]):
                    surf = "grass"
                elif any(k in sport_lower
                          for k in ["clay", "roland garros", "monte carlo"]):
                    surf = "clay"
                else:
                    surf = "hard"
                ml_pred = ml.predict_tennis(home, away, ts, surf)
                if ml_pred:
                    stats["ml_prediction"] = ml_pred
                    logger.info("  ✅ ML tennis: %s",
                                {k: f"{v*100:.1f}%" for k, v in ml_pred.items()
                                 if isinstance(v, float)})

        # ── FOOTBALL ──────────────────────────────────────────
        elif sport_key == "football":

            # 1. API-Football (primary — best stats)
            if CFG.API_FOOTBALL_KEY:
                try:
                    h_apif = api_football.get_team_stats(home)
                    a_apif = api_football.get_team_stats(away)
                    h2h_apif = {}
                    if h_apif and a_apif:
                        h2h_apif = api_football.get_h2h(home, away)
                    if h_apif or a_apif:
                        stats["football_stats"] = {
                            "home": h_apif or {},
                            "away": a_apif or {},
                            "h2h":  h2h_apif or {}
                        }
                        logger.info("  ✅ API-Football: H=%s A=%s H2H=%d",
                                    h_apif.get("data_quality", "?") if h_apif else "none",
                                    a_apif.get("data_quality", "?") if a_apif else "none",
                                    h2h_apif.get("total", 0) if h2h_apif else 0)
                        # Standings
                        if h_apif and h_apif.get("league_id"):
                            h_st = api_football.get_team_standing(
                                home, h_apif["league_id"])
                            if h_st:
                                stats["football_stats"]["home"]["standing"] = h_st
                        if a_apif and a_apif.get("league_id"):
                            a_st = api_football.get_team_standing(
                                away, a_apif["league_id"])
                            if a_st:
                                stats["football_stats"]["away"]["standing"] = a_st
                except Exception as e:
                    logger.warning("  [API-Football] %s", str(e)[:80])

            # 2. Football-Data.org (if API-Football missing)
            if not stats.get("football_stats") and CFG.FOOTBALL_DATA_ORG_KEY:
                try:
                    h_fdo = football_data_org.get_team_matches(home)
                    a_fdo = football_data_org.get_team_matches(away)
                    if h_fdo or a_fdo:
                        stats["football_stats"] = {
                            "home": h_fdo or {},
                            "away": a_fdo or {},
                            "h2h":  {}
                        }
                        logger.info("  ✅ Football-Data.org: H=%s A=%s",
                                    h_fdo.get("data_quality","?") if h_fdo else "none",
                                    a_fdo.get("data_quality","?") if a_fdo else "none")
                except Exception as e:
                    logger.debug("  [Football-Data.org] %s", str(e)[:60])

            # 3. GitHub CSV (always — historical baseline + H2H)
            try:
                fs_github = de.get_football_stats(home, away)
                if fs_github:
                    if "football_stats" not in stats:
                        stats["football_stats"] = fs_github
                        logger.info("  ✅ GitHub football: H=%s A=%s",
                                    fs_github.get("home",{}).get("data_quality","?"),
                                    fs_github.get("away",{}).get("data_quality","?"))
                    else:
                        # Supplement: add GitHub data as extra field
                        stats["football_stats"]["github"] = fs_github
                        # Fill H2H if missing
                        if not stats["football_stats"].get("h2h"):
                            stats["football_stats"]["h2h"] = \
                                fs_github.get("h2h", {})
                        logger.info("  ✅ GitHub football: supplemented")
            except Exception as e:
                logger.debug("  [GitHub football] %s", str(e)[:60])

            # 4. TSDB team form (quick enrich)
            try:
                h_tsdb = tsdb.get_team_stats(home)
                a_tsdb = tsdb.get_team_stats(away)
                if h_tsdb or a_tsdb:
                    if "football_stats" not in stats:
                        stats["football_stats"] = {
                            "home": h_tsdb or {},
                            "away": a_tsdb or {},
                            "h2h":  {}
                        }
                    else:
                        # Add TSDB form if not already present
                        if h_tsdb and not stats["football_stats"]["home"].get("form"):
                            stats["football_stats"]["home"]["tsdb_form"] = \
                                h_tsdb.get("form", "")
                            stats["football_stats"]["home"]["tsdb_win_rate"] = \
                                h_tsdb.get("win_rate", 0)
                        if a_tsdb and not stats["football_stats"]["away"].get("form"):
                            stats["football_stats"]["away"]["tsdb_form"] = \
                                a_tsdb.get("form", "")
                            stats["football_stats"]["away"]["tsdb_win_rate"] = \
                                a_tsdb.get("win_rate", 0)
                    logger.info("  ✅ TSDB football: H=%s A=%s",
                                h_tsdb.get("data_quality","?") if h_tsdb else "none",
                                a_tsdb.get("data_quality","?") if a_tsdb else "none")
            except Exception as e:
                logger.debug("  [TSDB football] %s", str(e)[:60])

            # 5. ML model
            if ml.is_football_trained:
                try:
                    ml_pred = ml.predict_football(home, away)
                    if ml_pred:
                        stats["ml_prediction"] = ml_pred
                        logger.info("  ✅ ML football: %s",
                                    {k: f"{v*100:.1f}%" for k, v in ml_pred.items()
                                     if isinstance(v, float)})
                except Exception as e:
                    logger.debug("  [ML football] %s", e)

            # 6. Poisson
            try:
                poisson_pred = PoissonEngine.calculate(
                    home, away, de.football_data.get("all"))
                if poisson_pred:
                    stats["poisson_prediction"] = poisson_pred
                    logger.info("  ✅ Poisson: H=%.0f%% D=%.0f%% A=%.0f%%",
                                poisson_pred.get("home_win_prob_poisson",0)*100,
                                poisson_pred.get("draw_prob_poisson",0)*100,
                                poisson_pred.get("away_win_prob_poisson",0)*100)
            except Exception as e:
                logger.debug("  [Poisson] %s", e)

        # ── BASKETBALL ────────────────────────────────────────
        elif sport_key == "basketball":

            # 1. BallDontLie (NBA recent games)
            try:
                h_bdl = balldontlie.get_recent_games(home)
                a_bdl = balldontlie.get_recent_games(away)
                if h_bdl or a_bdl:
                    h_avg = balldontlie.get_team_stats(home)
                    a_avg = balldontlie.get_team_stats(away)
                    stats["us_sports"] = {
                        "home": {**(h_bdl or {}), **(h_avg or {})},
                        "away": {**(a_bdl or {}), **(a_avg or {})}
                    }
                    logger.info("  ✅ BallDontLie: H=%s A=%s",
                                h_bdl.get("data_quality","?") if h_bdl else "none",
                                a_bdl.get("data_quality","?") if a_bdl else "none")
            except Exception as e:
                logger.debug("  [BallDontLie] %s", e)

            # 2. Fallback: nba_api standings
            if not stats.get("us_sports"):
                try:
                    hs  = de.get_us_sports_stats(sport, home)
                    aws = de.get_us_sports_stats(sport, away)
                    if hs or aws:
                        stats["us_sports"] = {"home": hs, "away": aws}
                        logger.info("  ✅ NBA standings fallback")
                except Exception as e:
                    logger.debug("  [NBA fallback] %s", e)

            # 3. ML from win rates
            us = stats.get("us_sports", {})
            hs  = us.get("home", {})
            aws = us.get("away", {})
            h_wr = hs.get("win_rate", hs.get("win_pct", 0))
            a_wr = aws.get("win_rate", aws.get("win_pct", 0))
            if h_wr and a_wr:
                total = h_wr + a_wr
                home_prob = min(0.80, max(0.20, (h_wr / total) * 0.95 + 0.025))
                ml_pred = {
                    f"{home}_win_prob": round(home_prob, 4),
                    f"{away}_win_prob": round(1 - home_prob, 4)
                }
                stats["ml_prediction"] = ml_pred
                logger.info("  ✅ NBA ML: H=%.0f%% A=%.0f%%",
                            home_prob*100, (1-home_prob)*100)

        # ── BASEBALL ──────────────────────────────────────────
        elif sport_key == "baseball":

            # MLB official API
            try:
                hs  = de.get_us_sports_stats(sport, home)
                aws = de.get_us_sports_stats(sport, away)
                if hs or aws:
                    stats["us_sports"] = {"home": hs, "away": aws}
                    logger.info("  ✅ MLB: H=%s A=%s",
                                hs.get("source","?") if hs else "none",
                                aws.get("source","?") if aws else "none")
            except Exception as e:
                logger.debug("  [MLB] %s", e)

            # TSDB as backup
            if not stats.get("us_sports"):
                try:
                    h_ts = tsdb.get_team_stats(home)
                    a_ts = tsdb.get_team_stats(away)
                    if h_ts or a_ts:
                        stats["us_sports"] = {
                            "home": h_ts or {},
                            "away": a_ts or {}
                        }
                        logger.info("  ✅ TSDB baseball fallback")
                except Exception as e:
                    logger.debug("  [TSDB baseball] %s", e)

            # Simple ML from win rates
            us = stats.get("us_sports", {})
            h_wr = us.get("home", {}).get("win_pct",
                   us.get("home", {}).get("win_rate", 0))
            a_wr = us.get("away", {}).get("win_pct",
                   us.get("away", {}).get("win_rate", 0))
            if h_wr and a_wr:
                total = h_wr + a_wr
                home_prob = min(0.75, max(0.25, (h_wr / total) * 0.90 + 0.05))
                ml_pred = {
                    f"{home}_win_prob": round(home_prob, 4),
                    f"{away}_win_prob": round(1 - home_prob, 4)
                }
                stats["ml_prediction"] = ml_pred
                logger.info("  ✅ MLB ML: H=%.0f%% A=%.0f%%",
                            home_prob*100, (1-home_prob)*100)

        # ── HOCKEY ────────────────────────────────────────────
        elif sport_key == "hockey":

            try:
                hs  = de.get_us_sports_stats(sport, home)
                aws = de.get_us_sports_stats(sport, away)
                if hs or aws:
                    stats["us_sports"] = {"home": hs, "away": aws}
                    logger.info("  ✅ NHL: H=%s A=%s",
                                hs.get("source","?") if hs else "none",
                                aws.get("source","?") if aws else "none")
            except Exception as e:
                logger.debug("  [NHL] %s", e)

            # TSDB backup
            if not stats.get("us_sports"):
                try:
                    h_ts = tsdb.get_team_stats(home)
                    a_ts = tsdb.get_team_stats(away)
                    if h_ts or a_ts:
                        stats["us_sports"] = {
                            "home": h_ts or {},
                            "away": a_ts or {}
                        }
                        logger.info("  ✅ TSDB hockey fallback")
                except Exception as e:
                    logger.debug("  [TSDB hockey] %s", e)

            # ML from win rates
            us = stats.get("us_sports", {})
            h_wr = us.get("home", {}).get("win_pct",
                   us.get("home", {}).get("win_rate", 0))
            a_wr = us.get("away", {}).get("win_pct",
                   us.get("away", {}).get("win_rate", 0))
            if h_wr and a_wr:
                total = h_wr + a_wr
                home_prob = min(0.75, max(0.25, (h_wr / total) * 0.90 + 0.05))
                ml_pred = {
                    f"{home}_win_prob": round(home_prob, 4),
                    f"{away}_win_prob": round(1 - home_prob, 4)
                }
                stats["ml_prediction"] = ml_pred
                logger.info("  ✅ NHL ML: H=%.0f%% A=%.0f%%",
                            home_prob*100, (1-home_prob)*100)

        # ── CRICKET / OTHER ───────────────────────────────────
        else:
            try:
                h_ts = tsdb.get_team_stats(home)
                a_ts = tsdb.get_team_stats(away)
                if h_ts or a_ts:
                    stats["tsdb_stats"] = {
                        "home": h_ts or {},
                        "away": a_ts or {}
                    }
                    logger.info("  ✅ TSDB other sport")
            except Exception as e:
                logger.debug("  [TSDB other] %s", e)

        # Log data summary
        data_sources = []
        if stats.get("historical_data"):   data_sources.append("GitHub-Tennis")
        if stats.get("football_stats"):    data_sources.append("Football-Stats")
        if stats.get("us_sports"):         data_sources.append("US-Sports")
        if stats.get("tsdb_stats"):        data_sources.append("TSDB")
        if ml_pred:                        data_sources.append("ML")
        if poisson_pred:                   data_sources.append("Poisson")
        logger.info("  📦 Data: [%s]", ", ".join(data_sources) if data_sources else "NONE")

        # ── Math score ─────────────────────────────────────────
        math_score = ConfidenceEngine.score(opp, stats, ml_pred, poisson_pred)

        min_math = 50 if sport_key == "other" else CFG.MIN_MATH_SCORE_TO_CALL_AI
        if math_score < min_math:
            skip_counts["math"] += 1
            logger.info("⏭️ SKIP(math:%d<%d) %s vs %s EV=%.2f%%",
                        math_score, min_math, home, away, opp["edge_pct"])
            continue

        # ── AI decision ────────────────────────────────────────
        ai = make_ai_decision(
            home, away, sport, sport_key,
            opp, stats, math_score, ml_pred, poisson_pred
        )
        fc = ai["final_confidence"]

        if ai.get("decision") == "SKIP":
            skip_counts["ai"] += 1
            logger.info("⏭️ AI_SKIP: %s vs %s Math:%d AI:%d Final:%d",
                        home, away, math_score, ai["ai_confidence"], fc)
            continue

        if fc < CFG.MIN_CONFIDENCE_TO_SEND:
            skip_counts["conf"] += 1
            logger.info("⏭️ SKIP(conf:%d<%d) %s vs %s",
                        fc, CFG.MIN_CONFIDENCE_TO_SEND, home, away)
            continue

        logger.info("✅ SIGNAL: %s vs %s | Math:%d AI:%d Final:%d EV=%.2f%%",
                    home, away, math_score, ai["ai_confidence"],
                    fc, opp["edge_pct"])

        msg = build_message(
            home, away, sport, sport_key,
            opp, ai, stats, math_score,
            ml_pred, poisson_pred,
            now, event.get("commence_time", "")
        )

        if send_telegram(msg):
            sent.mark_sent(home, away, opp["pick"], opp["market"])
            perf_tracker.record(
                home, away, opp["pick"], opp["market"],
                opp["odds"], opp["ev"], fc, opp["prob"],
                sport_key, event.get("sport_key", "")
            )
            total_sent += 1
            logger.info("📤 SENT: %s vs %s EV=%.2f%% Conf=%d%%",
                        home, away, opp["edge_pct"], fc)
        else:
            logger.error("❌ Telegram failed: %s vs %s", home, away)

        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)

    logger.info("=" * 65)
    logger.info("📊 Events:%d  Analyzed:%d  Sent:%d",
                len(events), total_analyzed, total_sent)
    logger.info("   Skip → %s", skip_counts)
    logger.info("📊 %s", odds_key_manager.get_summary())
    logger.info("=" * 65)


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
    except Exception as e:
        logger.critical("SYSTEM FAILURE: %s", e, exc_info=True)
        sys.exit(1)
