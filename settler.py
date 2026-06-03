# =========================================================
# ZBET90 SETTLER ENGINE v5.0 | Production Grade
# =========================================================
import os
import sys
import json
import logging
import asyncio
import re
import unicodedata
import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import requests

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
PERFORMANCE_FILE = Path("api_cache/performance_tracker.json")
PENDING_FILE     = Path("api_cache/pending_settlement.json")
LOG_FILE         = Path("api_cache/settler_logs.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

ODDS_KEYS = [k for k in [
    os.getenv("ODDS_API_KEY",  "").strip(),
    os.getenv("ODDS_API_KEY2", "").strip(),
    os.getenv("ODDS_API_KEY3", "").strip(),
] if k]

# ── Logging ───────────────────────────────────────────────
logger = logging.getLogger("SETTLER")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
for _h in [logging.StreamHandler(sys.stdout),
           logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")]:
    _h.setFormatter(_fmt)
    logger.addHandler(_h)

if not ODDS_KEYS:
    logger.critical("FATAL: No ODDS_API_KEY found!")
    sys.exit(1)


# =========================================================
# UTILITIES
# =========================================================
def normalize_str(s: str) -> str:
    """Lowercase, strip accents, remove non-alpha chars."""
    if not s:
        return ""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # remove common suffixes / prefixes that differ between sources
    for noise in ["fc", "cf", "sc", "ac", "bk", "fk", "if", "rsc", "afc", "rfc"]:
        s = re.sub(rf"\b{noise}\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str) -> set:
    return {t for t in normalize_str(s).split() if len(t) > 2}


def team_similarity(a: str, b: str) -> float:
    """
    0-1 similarity between two team/player name strings.
    Uses token overlap + substring bonus.
    """
    na, nb = normalize_str(a), normalize_str(b)
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.9
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    score = overlap / max(len(ta), len(tb))
    return round(score, 3)


def save_json_safe(filepath: Path, data: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix(f".tmp_{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(filepath)


def is_tennis_sport(sport: str) -> bool:
    return any(k in sport.lower() for k in ("tennis", "atp", "wta"))


def is_soccer_sport(sport: str) -> bool:
    return any(k in sport.lower() for k in ("football", "soccer"))


# =========================================================
# BET RESOLVER  —  sport-aware, market-aware
# =========================================================
def resolve_bet(bet: dict,
                h_score: int,
                a_score: int,
                api_h: str,
                api_a: str) -> str:
    """
    Determine win/loss/void for a settled match.

    For TENNIS:
      - h_score / a_score = sets won by home / away player
        (some APIs give games; we treat >0 as indicator)
      - h2h market: pick matches winner name → win
      - totals: total games in match (use games field if available)

    For all other sports:
      - h_score / a_score = goals / points / runs
    """
    pick   = normalize_str(bet.get("pick", ""))
    market = bet.get("market", "h2h").lower().strip()
    sport  = bet.get("sport", "")

    api_h_n = normalize_str(api_h)
    api_a_n = normalize_str(api_a)

    # ── determine actual winner ──────────────────────────
    if h_score > a_score:
        winner_n = api_h_n
    elif a_score > h_score:
        winner_n = api_a_n
    else:
        winner_n = "draw"   # only valid for non-tennis

    def pick_matches(team_n: str) -> bool:
        """Does our pick string refer to this team/player?"""
        if not team_n or team_n == "draw":
            return "draw" in pick or "tie" in pick
        sim = team_similarity(pick, team_n)
        if sim >= 0.5:
            return True
        # token overlap with raw pick
        pt = tokenize(pick)
        tt = tokenize(team_n)
        if pt and tt and len(pt & tt) / max(len(pt), len(tt)) >= 0.5:
            return True
        return False

    # ── h2h / match winner ───────────────────────────────
    if market == "h2h":
        if winner_n == "draw":
            return "win" if pick_matches("draw") else "loss"
        if pick_matches(winner_n):
            return "win"
        return "loss"

    # ── h2h lay ──────────────────────────────────────────
    if market == "h2h_lay":
        if winner_n == "draw":
            return "loss" if pick_matches("draw") else "win"
        return "loss" if pick_matches(winner_n) else "win"

    # ── totals / over-under ──────────────────────────────
    if market in ("totals", "over/under", "totals"):
        nums = re.findall(r"\d+\.?\d*", pick)
        if not nums:
            return "void"
        line  = float(nums[0])
        total = h_score + a_score

        # For tennis totals the line is usually in games (e.g. 22.5)
        # APIs often give sets; if total looks like sets (≤6) and line>10,
        # we can't reliably resolve → void
        if is_tennis_sport(sport) and total <= 6 and line > 10:
            logger.warning("[RESOLVE] Tennis totals: set-score (%d) vs game-line (%.1f) → void", total, line)
            return "void"

        is_over  = "over"  in pick or pick.startswith("o ")
        is_under = "under" in pick or pick.startswith("u ")

        if total == line:
            return "void"          # push / no action
        if is_over:
            return "win" if total > line else "loss"
        if is_under:
            return "win" if total < line else "loss"
        return "void"

    # ── spreads / handicap ───────────────────────────────
    if market in ("spreads", "handicap"):
        nums = re.findall(r"[+-]?\d+\.?\d*", pick)
        if not nums:
            return "void"
        hcap = float(nums[0])
        if pick_matches(api_h_n):
            adjusted = h_score + hcap
            if adjusted > a_score:  return "win"
            if adjusted < a_score:  return "loss"
            return "void"
        elif pick_matches(api_a_n):
            adjusted = a_score + hcap
            if adjusted > h_score:  return "win"
            if adjusted < h_score:  return "loss"
            return "void"
        return "void"

    return "void"


def calculate_profit(odds: float, outcome: str) -> float:
    if outcome == "win":
        return round(float(odds) - 1.0, 2)
    if outcome == "loss":
        return -1.0
    return 0.0


# =========================================================
# PENDING MANAGER
# =========================================================
class PendingManager:
    MAX_RETRIES = 6          # give up after 6 retry cycles (~6 runs × 4h = 24h)
    MAX_AGE_DAYS = 5

    def __init__(self):
        self.data = self._load()

    def _load(self) -> dict:
        try:
            if PENDING_FILE.exists():
                with open(PENDING_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                now = datetime.now(timezone.utc)
                d["pending"] = [
                    p for p in d.get("pending", [])
                    if self._is_recent(p.get("timestamp", ""), self.MAX_AGE_DAYS, now)
                    and p.get("_retry_count", 0) < self.MAX_RETRIES
                ]
                return d
        except Exception:
            pass
        return {"pending": [], "last_updated": ""}

    @staticmethod
    def _is_recent(ts: str, days: int, now: datetime) -> bool:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (now - dt) < timedelta(days=days)
        except Exception:
            return False

    def _make_id(self, b: dict) -> str:
        raw = f"{b.get('home','')}|{b.get('away','')}|{b.get('market','')}|{b.get('timestamp','')}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def save(self):
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        save_json_safe(PENDING_FILE, self.data)

    def add(self, bet: dict):
        bid = self._make_id(bet)
        existing = {self._make_id(p) for p in self.data["pending"]}
        if bid not in existing:
            self.data["pending"].append({**bet,
                                         "_pending_id": bid,
                                         "_retry_count": 0,
                                         "_pending_since": datetime.now(timezone.utc).isoformat()})

    def remove(self, bet: dict):
        bid = self._make_id(bet)
        self.data["pending"] = [p for p in self.data["pending"]
                                 if self._make_id(p) != bid]

    def increment_retry(self, bet: dict):
        bid = self._make_id(bet)
        for p in self.data["pending"]:
            if self._make_id(p) == bid:
                p["_retry_count"] = p.get("_retry_count", 0) + 1
                p["_last_retry"]  = datetime.now(timezone.utc).isoformat()
                break

    def get_all(self) -> list:
        return list(self.data["pending"])


# =========================================================
# SCRAPING ENGINES  —  async, multi-source, robust
# =========================================================
class ResultScraper:
    """
    Fetches finished match results from 3 free sources.
    Results are stored in two pools:
      self.soccer_pool  — football / soccer
      self.other_pool   — tennis, basketball, baseball, hockey, cricket
    """

    def __init__(self):
        self.soccer_pool: List[dict] = []
        self.other_pool:  List[dict] = []
        self._lock = asyncio.Lock()

    # ── internal adder (dedup by home+away) ─────────────
    async def _add(self, pool_name: str, result: dict):
        async with self._lock:
            pool = self.soccer_pool if pool_name == "soccer" else self.other_pool
            h, a = result["home_n"], result["away_n"]
            if not any(e["home_n"] == h and e["away_n"] == a for e in pool):
                pool.append(result)

    def _make_entry(self, home: str, away: str,
                    h_score: int, a_score: int,
                    source: str,
                    raw_home: str = "", raw_away: str = "") -> dict:
        return {
            "home_n":   normalize_str(home),
            "away_n":   normalize_str(away),
            "raw_home": raw_home or home,
            "raw_away": raw_away or away,
            "h_score":  h_score,
            "a_score":  a_score,
            "source":   source,
        }

    # ── FotMob ──────────────────────────────────────────
    async def _fetch_fotmob(self, date_str: str):
        url = f"https://www.fotmob.com/api/matches?date={date_str}"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15,
                                         headers={"User-Agent": "Mozilla/5.0"}) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return
                for league in r.json().get("leagues", []):
                    for m in league.get("matches", []):
                        if not m.get("status", {}).get("finished"):
                            continue
                        home    = m.get("home", {}).get("name", "")
                        away    = m.get("away", {}).get("name", "")
                        h_score = int(m.get("home", {}).get("score", 0) or 0)
                        a_score = int(m.get("away", {}).get("score", 0) or 0)
                        if home and away:
                            await self._add("soccer",
                                            self._make_entry(home, away, h_score, a_score,
                                                             "fotmob", home, away))
        except Exception as e:
            logger.debug("[FotMob] %s: %s", date_str, e)

    # ── ESPN ────────────────────────────────────────────
    async def _fetch_espn(self, date_str: str):
        """
        Fetches multiple ESPN endpoints for both soccer and non-soccer.
        date_str format: YYYYMMDD
        """
        endpoints = [
            # soccer
            (f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard?dates={date_str}", "soccer"),
            # US sports
            (f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}", "other"),
            (f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date_str}",    "other"),
            (f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={date_str}",      "other"),
            # tennis
            (f"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard?dates={date_str}", "other"),
            (f"https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard?dates={date_str}", "other"),
        ]
        try:
            import httpx
            async with httpx.AsyncClient(timeout=12,
                                         headers={"User-Agent": "Mozilla/5.0"}) as client:
                for url, pool in endpoints:
                    try:
                        r = await client.get(url)
                        if r.status_code != 200:
                            continue
                        for event in r.json().get("events", []):
                            state = event.get("status", {}).get("type", {}).get("state")
                            if state != "post":
                                continue
                            comps_list = event.get("competitions", [])
                            if not comps_list:
                                continue
                            competitors = comps_list[0].get("competitors", [])
                            home_name = away_name = ""
                            h_score = a_score = 0
                            for comp in competitors:
                                name = (comp.get("team", {}).get("displayName")
                                        or comp.get("athlete", {}).get("displayName", ""))
                                try:
                                    score_val = int(float(comp.get("score", "0") or "0"))
                                except (ValueError, TypeError):
                                    score_val = 0
                                if comp.get("homeAway") == "home":
                                    home_name = name
                                    h_score   = score_val
                                else:
                                    away_name = name
                                    a_score   = score_val
                            if home_name and away_name:
                                await self._add(pool,
                                                self._make_entry(home_name, away_name,
                                                                 h_score, a_score,
                                                                 "espn",
                                                                 home_name, away_name))
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("[ESPN] %s: %s", date_str, e)

    # ── SofaScore ────────────────────────────────────────
    async def _fetch_sofascore(self, date_str: str):
        """
        date_str format: YYYY-MM-DD
        """
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
            "Referer": "https://www.sofascore.com/",
            "Accept":  "application/json",
        }
        sports_pools = [
            ("football",    "soccer"),
            ("tennis",      "other"),
            ("basketball",  "other"),
            ("baseball",    "other"),
            ("ice-hockey",  "other"),
            ("cricket",     "other"),
        ]
        try:
            import httpx
            async with httpx.AsyncClient(timeout=12, headers=headers) as client:
                for sport, pool in sports_pools:
                    try:
                        url = (f"https://api.sofascore.com/api/v1/sport/"
                               f"{sport}/scheduled-events/{date_str}")
                        r = await client.get(url)
                        if r.status_code != 200:
                            continue
                        for ev in r.json().get("events", []):
                            if ev.get("status", {}).get("type") != "finished":
                                continue
                            home = ev.get("homeTeam", {}).get("name", "")
                            away = ev.get("awayTeam", {}).get("name", "")
                            # SofaScore: for tennis homeScore.current = sets won
                            h_score = ev.get("homeScore", {}).get("current", 0) or 0
                            a_score = ev.get("awayScore", {}).get("current", 0) or 0
                            if home and away:
                                await self._add(pool,
                                                self._make_entry(home, away,
                                                                 h_score, a_score,
                                                                 "sofascore",
                                                                 home, away))
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("[SofaScore] %s: %s", date_str, e)

    # ── FlashScore (simple requests fallback) ────────────
    async def _fetch_thesportsdb(self, date_str_dash: str):
        """
        TheSportsDB free tier — broad sport coverage.
        date_str_dash: YYYY-MM-DD
        """
        # Free API, no key needed for basic lookups
        url = f"https://www.thesportsdb.com/api/v1/json/3/eventsday.php?d={date_str_dash}"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=12,
                                         headers={"User-Agent": "Mozilla/5.0"}) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return
                events = r.json().get("events") or []
                for ev in events:
                    status = (ev.get("strStatus") or "").lower()
                    if status not in ("ft", "aet", "finished", "complete",
                                     "match finished", "ht", "fulltime"):
                        if "final" not in status and "finish" not in status:
                            continue
                    home = ev.get("strHomeTeam", "")
                    away = ev.get("strAwayTeam", "")
                    try:
                        h_score = int(float(ev.get("intHomeScore") or 0))
                        a_score = int(float(ev.get("intAwayScore") or 0))
                    except (ValueError, TypeError):
                        continue
                    sport = (ev.get("strSport") or "").lower()
                    pool  = "soccer" if "soccer" in sport or "football" in sport else "other"
                    if home and away:
                        await self._add(pool,
                                        self._make_entry(home, away, h_score, a_score,
                                                         "thesportsdb", home, away))
        except Exception as e:
            logger.debug("[TheSportsDB] %s: %s", date_str_dash, e)

    # ── master loader ────────────────────────────────────
    async def load_recent_results(self, days_back: int = 3):
        logger.info("🌍 [SCRAPER] Loading results (last %d days)...", days_back)
        now  = datetime.now(timezone.utc)
        tasks = []
        for i in range(days_back):
            target    = now - timedelta(days=i)
            date_fotm = target.strftime("%Y%m%d")
            date_soft = target.strftime("%Y-%m-%d")
            tasks += [
                self._fetch_fotmob(date_fotm),
                self._fetch_espn(date_fotm),
                self._fetch_sofascore(date_soft),
                self._fetch_thesportsdb(date_soft),
            ]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("✅ [SCRAPER] Soccer pool: %d | Other pool: %d",
                    len(self.soccer_pool), len(self.other_pool))

    # ── fuzzy match ──────────────────────────────────────
    def find_result(self, home: str, away: str, sport: str) -> Optional[dict]:
        """
        Returns the best matching result dict or None.
        Threshold: both home AND away must score >= 0.45 similarity.
        """
        pool = self.soccer_pool if is_soccer_sport(sport) else self.other_pool
        if not pool:
            return None

        best: Optional[dict] = None
        best_score            = 0.0

        for entry in pool:
            sh = team_similarity(home, entry["home_n"])
            sa = team_similarity(away, entry["away_n"])
            combined = sh * 0.5 + sa * 0.5
            # also try swapped (some APIs list differently)
            sh2 = team_similarity(home, entry["away_n"])
            sa2 = team_similarity(away, entry["home_n"])
            combined2 = sh2 * 0.5 + sa2 * 0.5

            if combined2 > combined and combined2 >= 0.45:
                # home/away swapped in the API response — swap scores
                swapped = {**entry,
                           "home_n":  entry["away_n"],
                           "away_n":  entry["home_n"],
                           "h_score": entry["a_score"],
                           "a_score": entry["h_score"],
                           "raw_home": entry["raw_away"],
                           "raw_away": entry["raw_home"],
                           "_swapped": True}
                if combined2 > best_score:
                    best_score = combined2
                    best       = swapped
            elif combined >= 0.45 and combined > best_score:
                best_score = combined
                best       = entry

        if best:
            logger.debug("[FUZZY] %.2f | %s vs %s → %s vs %s (%s)",
                         best_score, home, away,
                         best["raw_home"], best["raw_away"], best["source"])
        return best


# =========================================================
# ODDS-API RESULTS ENGINE  —  minimal calls, cached
# =========================================================
class OddsAPIResultsEngine:
    """
    Fetches completed scores from the-odds-api.
    Caches per sport_key to avoid hammering the quota.
    """

    SUPPORTED_SPORTS = {
        "soccer_epl", "soccer_spain_la_liga", "soccer_germany_bundesliga",
        "soccer_italy_serie_a", "soccer_france_ligue_one",
        "soccer_uefa_champs_league", "soccer_uefa_europa_league",
        "basketball_nba", "baseball_mlb", "icehockey_nhl",
        "tennis_atp", "tennis_wta",
        "cricket_icc_world_cup", "cricket_odi",
        # broad fallbacks
        "americanfootball_nfl", "basketball_euroleague",
    }

    def __init__(self):
        self._cache: Dict[str, list] = {}
        self._fetched: set = set()
        self._cache_file = Path("api_cache/odds_api_scores_cache.json")
        self._load_cache()

    def _load_cache(self):
        try:
            if self._cache_file.exists():
                raw = json.loads(self._cache_file.read_text())
                now = datetime.now(timezone.utc)
                for k, v in raw.items():
                    ts_str = v.get("_cached_at", "")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if (now - ts) < timedelta(hours=4):
                            self._cache[k] = v.get("data", [])
                            self._fetched.add(k)
        except Exception:
            pass

    def _save_cache(self):
        try:
            out = {}
            for k, data in self._cache.items():
                out[k] = {"_cached_at": datetime.now(timezone.utc).isoformat(),
                           "data": data}
            save_json_safe(self._cache_file, out)
        except Exception:
            pass

    def fetch(self, sport_key: str, days_from: int = 3) -> list:
        if not sport_key:
            return []
        # normalise key
        sk = sport_key.lower().strip()
        if sk in self._fetched:
            return self._cache.get(sk, [])

        for key in ODDS_KEYS:
            url = (f"https://api.the-odds-api.com/v4/sports/{sk}/scores/"
                   f"?daysFrom={days_from}&apiKey={key}")
            try:
                r = requests.get(url, timeout=15)
                remaining = int(r.headers.get("x-requests-remaining", -1))
                if r.status_code == 200:
                    data = r.json()
                    logger.info("📡 [ODDS-API] %s → %d results (rem:%d)", sk, len(data), remaining)
                    self._cache[sk] = data
                    self._fetched.add(sk)
                    self._save_cache()
                    return data
                elif r.status_code == 422:
                    logger.debug("[ODDS-API] Sport key '%s' unsupported", sk)
                    self._fetched.add(sk)  # don't retry
                    return []
                elif r.status_code in (401, 402):
                    logger.warning("[ODDS-API] Key exhausted/invalid: %s", key[:8])
                    continue
            except Exception as e:
                logger.debug("[ODDS-API] %s: %s", sk, e)
                continue
        return []

    def find_result(self, bet: dict) -> Optional[dict]:
        """
        Given a bet dict (with api_sport_key, home, away),
        return a resolved result or None.
        """
        sk = bet.get("api_sport_key", "")
        results = self.fetch(sk)
        if not results:
            return None

        home_n = normalize_str(bet.get("home", ""))
        away_n = normalize_str(bet.get("away", ""))

        best: Optional[dict] = None
        best_score            = 0.0

        for match in results:
            if not match.get("completed"):
                continue
            api_h = normalize_str(match.get("home_team", ""))
            api_a = normalize_str(match.get("away_team", ""))
            sh    = team_similarity(bet.get("home", ""), api_h)
            sa    = team_similarity(bet.get("away", ""), api_a)
            # also try swapped
            sh2   = team_similarity(bet.get("home", ""), api_a)
            sa2   = team_similarity(bet.get("away", ""), api_h)

            fwd = (sh + sa) / 2
            rev = (sh2 + sa2) / 2

            if fwd >= 0.45 and fwd > best_score:
                best_score = fwd
                best       = {"match": match, "swapped": False,
                               "api_h": api_h, "api_a": api_a}
            if rev >= 0.45 and rev > best_score:
                best_score = rev
                best       = {"match": match, "swapped": True,
                               "api_h": api_a, "api_a": api_h}

        if not best:
            return None

        match  = best["match"]
        api_h  = best["api_h"]
        api_a  = best["api_a"]
        scores = match.get("scores") or []

        # extract scores  —  try name-matching within scores list
        h_score = a_score = 0
        if scores:
            for s in scores:
                sn = normalize_str(s.get("name", ""))
                try:
                    val = int(float(s.get("score", 0) or 0))
                except (ValueError, TypeError):
                    val = 0
                sim_h = team_similarity(s.get("name", ""), best["match"].get("home_team", ""))
                sim_a = team_similarity(s.get("name", ""), best["match"].get("away_team", ""))
                if sim_h >= sim_a:
                    h_score = val
                else:
                    a_score = val
            if best["swapped"]:
                h_score, a_score = a_score, h_score

        return {
            "h_score": h_score,
            "a_score": a_score,
            "api_h":   api_h,
            "api_a":   api_a,
            "source":  "odds_api",
        }


# =========================================================
# TRACKER SYNC
# =========================================================
def sync_bet_to_tracker(bet: dict, tracker: dict):
    """Update the signal in the performance tracker in-place."""
    bid = bet.get("id")
    bts = bet.get("timestamp")
    bhm = bet.get("home")

    for signal in tracker.get("signals", []):
        match_by_id = bid and signal.get("id") == bid
        match_by_ts = (not bid
                       and bts
                       and signal.get("timestamp") == bts
                       and signal.get("home") == bhm)
        if match_by_id or match_by_ts:
            signal["outcome"]      = bet["outcome"]
            signal["profit_loss"]  = bet["profit_loss"]
            signal["_settled_by"]  = bet.get("_settled_by", "system")
            signal["_settled_at"]  = datetime.now(timezone.utc).isoformat()
            return True
    return False


# =========================================================
# TELEGRAM REPORT
# =========================================================
def send_telegram_report(settled: list, summary: dict):
    if not settled or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    import html as html_lib

    lines = ["🧾 <b>ZBET90 SETTLEMENT REPORT v5.0</b>\n"]
    daily_pl = 0.0

    SOURCE_ICON = {
        "fotmob":     "⚽",
        "espn":       "📺",
        "sofascore":  "📱",
        "thesportsdb":"🗄️",
        "odds_api":   "📡",
        "void":       "⚪️",
    }

    for bet in settled:
        outcome = bet.get("outcome", "unknown")
        source  = bet.get("_settled_by", "unknown")
        icon    = SOURCE_ICON.get(source, "📡")

        if outcome == "void":
            result_icon, pl_str = "⚪️", "0.0u"
        else:
            pl = bet.get("profit_loss", 0.0) or 0.0
            daily_pl   += pl
            result_icon = "🟢" if outcome == "win" else "🔴"
            pl_str      = f"+{pl:.2f}u" if pl > 0 else f"{pl:.2f}u"

        lines.append(
            f"⚔️ <b>{html_lib.escape(str(bet.get('home','?')))} vs "
            f"{html_lib.escape(str(bet.get('away','?')))}</b>\n"
            f"🎯 {html_lib.escape(str(bet.get('pick','?')))} @ {bet.get('odds','?')}\n"
            f"🏁 <b>{outcome.upper()}</b> {result_icon} | {pl_str} {icon}\n"
        )

    total_icon = "📈" if daily_pl > 0 else "📉"
    lines += [
        "══════════════════",
        f"{total_icon} <b>Session PnL:</b> {daily_pl:+.2f} units",
        f"🏆 <b>Win Rate:</b> {summary.get('win_rate', 0) * 100:.1f}%",
        f"💰 <b>ROI:</b> {summary.get('roi_pct', 0):.1f}%",
        f"📊 <b>Resolved:</b> {summary.get('resolved', 0)} / {summary.get('total_signals', 0)}",
        f"⏳ <b>Pending:</b> {summary.get('pending_count', 0)}",
    ]

    msg = "\n".join(lines)
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
        logger.info("📤 Telegram report sent.")
    except Exception as e:
        logger.error("Telegram error: %s", e)


# =========================================================
# MAIN SETTLER
# =========================================================
async def async_settle():
    logger.info("=" * 60)
    logger.info("⚡ ZBET90 SETTLER v5.0 | Scraper + OddsAPI Pipeline")
    logger.info("=" * 60)

    if not PERFORMANCE_FILE.exists():
        logger.info("❌ No performance_tracker.json found. Exiting.")
        return

    with open(PERFORMANCE_FILE, "r", encoding="utf-8") as f:
        tracker = json.load(f)

    now         = datetime.now(timezone.utc)
    pending_mgr = PendingManager()

    # ── 1. collect unsettled signals older than 3h ──────
    existing_ids = {pending_mgr._make_id(p) for p in pending_mgr.get_all()}
    new_added    = 0
    for sig in tracker.get("signals", []):
        if sig.get("outcome") is not None:
            continue
        try:
            t = datetime.fromisoformat(sig["timestamp"])
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if (now - t) < timedelta(hours=3):
                continue          # too fresh — match may still be live
        except Exception:
            continue
        bid = pending_mgr._make_id(sig)
        if bid not in existing_ids:
            pending_mgr.add(sig)
            existing_ids.add(bid)
            new_added += 1

    to_check = pending_mgr.get_all()
    logger.info("📋 Pending: %d total | %d newly added", len(to_check), new_added)
    if not to_check:
        logger.info("ℹ️ Nothing to settle.")
        return

    # ── 2. Scraper phase ─────────────────────────────────
    scraper = ResultScraper()
    await scraper.load_recent_results(days_back=3)

    settled_session: list = []
    need_api: list        = []

    for bet in to_check:
        home  = bet.get("home", "")
        away  = bet.get("away", "")
        sport = bet.get("sport", "")

        result = scraper.find_result(home, away, sport)
        if result:
            outcome = resolve_bet(bet,
                                  result["h_score"], result["a_score"],
                                  result["raw_home"], result["raw_away"])
            bet["outcome"]      = outcome
            bet["profit_loss"]  = calculate_profit(bet.get("odds", 2.0), outcome)
            bet["_settled_by"]  = result["source"]
            settled_session.append(bet)
            pending_mgr.remove(bet)
            logger.info("✅ [%s] %s vs %s → %s (%.2fu)",
                        result["source"].upper(), home, away,
                        outcome.upper(), bet["profit_loss"])
        else:
            need_api.append(bet)

    # ── 3. Odds-API fallback ──────────────────────────────
    if need_api:
        logger.info("📡 %d not found by scraper → Odds-API...", len(need_api))
        api_engine = OddsAPIResultsEngine()

        for bet in need_api:
            result = api_engine.find_result(bet)
            if result:
                outcome = resolve_bet(bet,
                                      result["h_score"], result["a_score"],
                                      result["api_h"],   result["api_a"])
                bet["outcome"]      = outcome
                bet["profit_loss"]  = calculate_profit(bet.get("odds", 2.0), outcome)
                bet["_settled_by"]  = "odds_api"
                settled_session.append(bet)
                pending_mgr.remove(bet)
                logger.info("✅ [ODDS-API] %s vs %s → %s (%.2fu)",
                            bet["home"], bet["away"],
                            outcome.upper(), bet["profit_loss"])
            else:
                # check age → void or keep pending
                try:
                    t = datetime.fromisoformat(bet["timestamp"])
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                    age_h = (now - t).total_seconds() / 3600
                except Exception:
                    age_h = 0

                if age_h > 72:                      # 3 days → give up
                    bet["outcome"]     = "void"
                    bet["profit_loss"] = 0.0
                    bet["_settled_by"] = "void"
                    settled_session.append(bet)
                    pending_mgr.remove(bet)
                    logger.warning("⚪️ [VOID] %s vs %s (%.0fh old)",
                                   bet["home"], bet["away"], age_h)
                else:
                    pending_mgr.increment_retry(bet)
                    logger.info("⏳ [RETRY %d] %s vs %s",
                                bet.get("_retry_count", 0),
                                bet["home"], bet["away"])

    # ── 4. persist & report ──────────────────────────────
    pending_mgr.save()

    if not settled_session:
        logger.info("⏳ No new settlements this session.")
        return

    synced = 0
    for bet in settled_session:
        if sync_bet_to_tracker(bet, tracker):
            synced += 1

    resolved = [s for s in tracker["signals"]
                if s.get("outcome") and s["outcome"] != "void"]
    wins     = [s for s in resolved if s["outcome"] == "win"]
    total_pl = sum(s.get("profit_loss", 0) or 0 for s in resolved)

    tracker["summary"] = {
        "total_signals":         len(tracker["signals"]),
        "resolved":              len(resolved),
        "win_rate":              round(len(wins) / max(len(resolved), 1), 3),
        "total_profit_loss_units": round(total_pl, 2),
        "roi_pct":               round(total_pl / max(len(resolved), 1) * 100, 2),
        "last_updated":          now.isoformat(),
        "pending_count":         len(pending_mgr.get_all()),
    }

    save_json_safe(PERFORMANCE_FILE, tracker)
    logger.info("💾 Synced %d/%d bets to tracker.", synced, len(settled_session))

    send_telegram_report(settled_session, tracker["summary"])

    logger.info("=" * 60)
    logger.info("📊 Settled:%d | Pending:%d | WR:%.1f%% | ROI:%.1f%%",
                len(settled_session),
                len(pending_mgr.get_all()),
                tracker["summary"]["win_rate"] * 100,
                tracker["summary"]["roi_pct"])
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(async_settle())
