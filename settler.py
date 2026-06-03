# =========================================================
# ZBET90 SETTLER ENGINE v6.0 | TheSportsDB Primary
# =========================================================
# File: settler.py
# =========================================================
import os, sys, json, logging, asyncio, re, unicodedata
import hashlib, time, requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict

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

# ── Logging ──────────────────────────────────────────────
logger = logging.getLogger("SETTLER_v6")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                         "%Y-%m-%d %H:%M:%S")
for _h in [logging.StreamHandler(sys.stdout),
           logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")]:
    _h.setFormatter(_fmt)
    logger.addHandler(_h)

if not ODDS_KEYS:
    logger.critical("FATAL: No ODDS_API_KEY!")
    sys.exit(1)


# =========================================================
# UTILITIES
# =========================================================
def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    for noise in ["fc", "cf", "sc", "ac", "bk", "fk", "afc", "rfc",
                  "united", "city", "real", "atletico"]:
        s = re.sub(rf"\b{noise}\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str) -> set:
    return {t for t in normalize_str(s).split() if len(t) > 2}


def team_sim(a: str, b: str) -> float:
    na, nb = normalize_str(a), normalize_str(b)
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.92
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(len(ta), len(tb))
    return round(overlap, 3)


def save_json(filepath: Path, data: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix(f".tmp_{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(filepath)


def is_soccer(sport: str) -> bool:
    return any(k in sport.lower() for k in ("football", "soccer"))


def is_tennis(sport: str) -> bool:
    return any(k in sport.lower() for k in ("tennis", "atp", "wta"))


# =========================================================
# BET RESOLVER
# =========================================================
def resolve_bet(bet: dict, h_score: int, a_score: int,
                api_h: str, api_a: str) -> str:
    pick   = normalize_str(bet.get("pick", ""))
    market = bet.get("market", "h2h").lower().strip()
    sport  = bet.get("sport", "")

    api_h_n = normalize_str(api_h)
    api_a_n = normalize_str(api_a)

    if h_score > a_score:
        winner_n = api_h_n
    elif a_score > h_score:
        winner_n = api_a_n
    else:
        winner_n = "draw"

    def pick_matches_team(team_n: str) -> bool:
        if not team_n or team_n == "draw":
            return "draw" in pick or "tie" in pick
        if team_sim(pick, team_n) >= 0.45:
            return True
        pt = tokenize(pick)
        tt = tokenize(team_n)
        if pt and tt and len(pt & tt) / max(len(pt), len(tt)) >= 0.45:
            return True
        return False

    # h2h
    if market == "h2h":
        if winner_n == "draw":
            return "win" if pick_matches_team("draw") else "loss"
        return "win" if pick_matches_team(winner_n) else "loss"

    # totals
    if market in ("totals", "over/under"):
        nums = re.findall(r"\d+\.?\d*", pick)
        if not nums:
            return "void"
        line  = float(nums[0])
        total = h_score + a_score

        # Tennis: if scores look like sets and line is game-based → void
        if is_tennis(sport) and total <= 6 and line > 10:
            return "void"

        is_over  = "over" in pick
        is_under = "under" in pick

        if total == line:
            return "void"
        if is_over:
            return "win" if total > line else "loss"
        if is_under:
            return "win" if total < line else "loss"
        return "void"

    # spreads
    if market in ("spreads", "handicap"):
        nums = re.findall(r"[+-]?\d+\.?\d*", pick)
        if not nums:
            return "void"
        hcap = float(nums[0])
        if pick_matches_team(api_h_n):
            adj = h_score + hcap
            if adj > a_score:
                return "win"
            elif adj < a_score:
                return "loss"
            return "void"
        elif pick_matches_team(api_a_n):
            adj = a_score + hcap
            if adj > h_score:
                return "win"
            elif adj < h_score:
                return "loss"
            return "void"
        return "void"

    return "void"


def calc_profit(odds: float, outcome: str) -> float:
    if outcome == "win":
        return round(float(odds) - 1.0, 2)
    if outcome == "loss":
        return -1.0
    return 0.0


# =========================================================
# THESPORTSDB SETTLER (Primary Source)
# =========================================================
class TSDBSettler:
    """
    Uses TheSportsDB free API to find match results.
    Strategy:
    1. Search by date (eventsday.php)
    2. Search by team name (searchevents.php)
    3. Look at recent events for each team
    """
    BASE = "https://www.thesportsdb.com/api/v1/json/3"
    FINISHED_STATUSES = {
        "match finished", "ft", "aet", "after et", "finished",
        "complete", "fulltime", "full time", "final",
        "end", "3", "5"  # some APIs use numeric codes
    }

    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._cache_path = Path("api_cache/tsdb_settler_cache.json")
        self._last_call = 0.0
        self._min_interval = 0.4
        self._load_cache()

    def _load_cache(self):
        try:
            if self._cache_path.exists():
                raw = json.loads(self._cache_path.read_text())
                now = datetime.now(timezone.utc)
                for k, v in raw.items():
                    if isinstance(v, dict) and "ts" in v:
                        ts = datetime.fromisoformat(v["ts"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        # Keep results for 24h
                        if (now - ts) < timedelta(hours=24):
                            self._cache[k] = v
        except Exception:
            pass

    def _save_cache(self):
        try:
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, default=str)
            )
        except Exception:
            pass

    def _get(self, endpoint: str, params: dict = None,
             ttl_hours: float = 4.0) -> Optional[dict]:
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

        url = f"{self.BASE}/{endpoint}"
        try:
            r = requests.get(
                url, params=params,
                timeout=15,
                headers={"User-Agent": "ZBET90-Settler/6.0"}
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
        except Exception as e:
            logger.debug("[TSDB] %s: %s", endpoint, str(e)[:50])
        return None

    def _is_finished(self, event: dict) -> bool:
        status = (
            event.get("strStatus") or
            event.get("strProgress") or
            ""
        ).lower().strip()
        return (
            status in self.FINISHED_STATUSES or
            "finish" in status or
            "final" in status or
            "complete" in status or
            "ft" == status
        )

    def _parse_score(self, event: dict) -> Optional[Tuple[int, int]]:
        """Parse home/away score from TSDB event."""
        try:
            hs_raw = event.get("intHomeScore")
            as_raw = event.get("intAwayScore")
            if hs_raw is None or as_raw is None:
                return None
            hs = int(float(hs_raw))
            as_ = int(float(as_raw))
            return hs, as_
        except (ValueError, TypeError):
            return None

    def _match_event(self, event: dict,
                     home: str, away: str) -> float:
        """Return similarity score for an event vs our bet."""
        ev_home = event.get("strHomeTeam", "")
        ev_away = event.get("strAwayTeam", "")

        # Forward match
        fwd = (team_sim(home, ev_home) + team_sim(away, ev_away)) / 2
        # Reversed (some sources swap home/away)
        rev = (team_sim(home, ev_away) + team_sim(away, ev_home)) / 2

        return max(fwd, rev), fwd >= rev

    def find_result(self, bet: dict) -> Optional[dict]:
        """
        Multi-strategy result finder:
        1. Events on commence date
        2. Recent events for home team
        3. Search by event name
        """
        home  = bet.get("home", "")
        away  = bet.get("away", "")
        sport = bet.get("sport", "")

        # Determine search dates (commence_time ± 2 days)
        search_dates = []
        try:
            ts = datetime.fromisoformat(bet.get("timestamp", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            for delta in [0, 1, -1, 2]:
                d = (ts + timedelta(days=delta)).strftime("%Y-%m-%d")
                search_dates.append(d)
        except Exception:
            now = datetime.now(timezone.utc)
            for delta in [0, -1, 1]:
                search_dates.append(
                    (now + timedelta(days=delta)).strftime("%Y-%m-%d")
                )

        best_event = None
        best_score  = 0.0
        best_fwd    = True

        # ── Strategy 1: eventsday.php ────────────────────
        for date_str in search_dates:
            data = self._get("eventsday.php", {"d": date_str}, ttl_hours=2.0)
            if not data:
                continue
            events = data.get("events") or []
            for ev in events:
                if not self._is_finished(ev):
                    continue
                score, is_fwd = self._match_event(ev, home, away)
                if score > best_score and score >= 0.42:
                    best_score  = score
                    best_event  = ev
                    best_fwd    = is_fwd

        if best_event and best_score >= 0.42:
            return self._build_result(best_event, best_fwd)

        # ── Strategy 2: searchevents.php ─────────────────
        search_term = f"{home.split()[0]} {away.split()[0]}"
        data = self._get("searchevents.php",
                         {"e": search_term}, ttl_hours=6.0)
        if data:
            events = data.get("event") or []
            for ev in events:
                if not self._is_finished(ev):
                    continue
                score, is_fwd = self._match_event(ev, home, away)
                if score > best_score and score >= 0.42:
                    best_score  = score
                    best_event  = ev
                    best_fwd    = is_fwd

        if best_event and best_score >= 0.42:
            return self._build_result(best_event, best_fwd)

        # ── Strategy 3: team's last events ───────────────
        team_data = self._get("searchteams.php", {"t": home}, ttl_hours=12.0)
        if team_data and team_data.get("teams"):
            team_id = team_data["teams"][0].get("idTeam")
            if team_id:
                last_data = self._get("eventslast.php",
                                      {"id": team_id}, ttl_hours=2.0)
                if last_data:
                    events = last_data.get("results") or []
                    for ev in events:
                        score, is_fwd = self._match_event(ev, home, away)
                        if score > best_score and score >= 0.42:
                            best_score  = score
                            best_event  = ev
                            best_fwd    = is_fwd

        if best_event and best_score >= 0.42:
            logger.debug("[TSDB] Match found (%.2f): %s vs %s",
                         best_score, home, away)
            return self._build_result(best_event, best_fwd)

        logger.debug("[TSDB] No result: %s vs %s", home, away)
        return None

    def _build_result(self, ev: dict, is_fwd: bool) -> Optional[dict]:
        scores = self._parse_score(ev)
        if scores is None:
            return None
        hs, as_ = scores
        if not is_fwd:
            hs, as_ = as_, hs  # swap if teams were reversed

        return {
            "h_score":   hs,
            "a_score":   as_,
            "api_h":     ev.get("strHomeTeam", "") if is_fwd else ev.get("strAwayTeam", ""),
            "api_a":     ev.get("strAwayTeam", "") if is_fwd else ev.get("strHomeTeam", ""),
            "source":    "thesportsdb",
            "event_id":  ev.get("idEvent", ""),
            "status":    ev.get("strStatus", ""),
        }


# =========================================================
# ODDS-API SETTLER (Fallback)
# =========================================================
class OddsAPISettler:
    CACHE_FILE = Path("api_cache/odds_api_scores.json")

    def __init__(self):
        self._cache: Dict[str, list] = {}
        self._fetched: set = set()
        self._load_cache()

    def _load_cache(self):
        try:
            if self.CACHE_FILE.exists():
                raw = json.loads(self.CACHE_FILE.read_text())
                now = datetime.now(timezone.utc)
                for k, v in raw.items():
                    if isinstance(v, dict):
                        ts_str = v.get("ts", "")
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            if (now - ts) < timedelta(hours=6):
                                self._cache[k] = v.get("data", [])
                                self._fetched.add(k)
                        except Exception:
                            pass
        except Exception:
            pass

    def _save_cache(self):
        try:
            out = {
                k: {"ts": datetime.now(timezone.utc).isoformat(), "data": v}
                for k, v in self._cache.items()
            }
            save_json(self.CACHE_FILE, out)
        except Exception:
            pass

    def fetch(self, sport_key: str, days_from: int = 3) -> list:
        sk = (sport_key or "").lower().strip()
        if not sk or sk in self._fetched:
            return self._cache.get(sk, [])

        for key in ODDS_KEYS:
            url = (f"https://api.the-odds-api.com/v4/sports/{sk}/scores/"
                   f"?daysFrom={days_from}&apiKey={key}")
            try:
                r = requests.get(url, timeout=15)
                rem = int(r.headers.get("x-requests-remaining", -1))
                if r.status_code == 200:
                    data = r.json()
                    logger.info("📡 [ODDS-API] %s → %d (rem:%d)", sk, len(data), rem)
                    self._cache[sk] = data
                    self._fetched.add(sk)
                    self._save_cache()
                    return data
                elif r.status_code == 422:
                    logger.debug("[ODDS-API] Unsupported: %s", sk)
                    self._fetched.add(sk)
                    return []
                elif r.status_code in (401, 402):
                    logger.warning("[ODDS-API] Key exhausted")
                    continue
            except Exception as e:
                logger.debug("[ODDS-API] %s: %s", sk, str(e)[:50])
        return []

    def find_result(self, bet: dict) -> Optional[dict]:
        sk = bet.get("api_sport_key", "")
        results = self.fetch(sk)
        if not results:
            return None

        home = bet.get("home", "")
        away = bet.get("away", "")
        best = None
        best_score = 0.0

        for match in results:
            if not match.get("completed"):
                continue
            api_h = match.get("home_team", "")
            api_a = match.get("away_team", "")
            fwd = (team_sim(home, api_h) + team_sim(away, api_a)) / 2
            rev = (team_sim(home, api_a) + team_sim(away, api_h)) / 2

            if fwd >= 0.42 and fwd > best_score:
                best_score = fwd
                best = {"match": match, "swapped": False,
                         "api_h": api_h, "api_a": api_a}
            if rev >= 0.42 and rev > best_score:
                best_score = rev
                best = {"match": match, "swapped": True,
                         "api_h": api_a, "api_a": api_h}

        if not best:
            return None

        match  = best["match"]
        scores = match.get("scores") or []
        h_score = a_score = 0

        for s in scores:
            name_sim_h = team_sim(s.get("name", ""), match.get("home_team", ""))
            name_sim_a = team_sim(s.get("name", ""), match.get("away_team", ""))
            try:
                val = int(float(s.get("score", 0) or 0))
            except (ValueError, TypeError):
                val = 0
            if name_sim_h >= name_sim_a:
                h_score = val
            else:
                a_score = val

        if best["swapped"]:
            h_score, a_score = a_score, h_score

        return {
            "h_score": h_score,
            "a_score": a_score,
            "api_h":   best["api_h"],
            "api_a":   best["api_a"],
            "source":  "odds_api",
        }


# =========================================================
# ESPN SETTLER (Second Fallback)
# =========================================================
class ESPNSettler:
    """
    ESPN public API - no auth required.
    Covers: Soccer, NBA, MLB, NHL, Tennis.
    """
    ENDPOINTS = {
        "football":   "soccer/all",
        "basketball": "basketball/nba",
        "baseball":   "baseball/mlb",
        "hockey":     "ice-hockey/nhl",
        "tennis":     "tennis/atp",
    }

    def __init__(self):
        self._results: List[dict] = []
        self._loaded = False

    def _get_sport_endpoint(self, sport: str) -> str:
        for key, ep in self.ENDPOINTS.items():
            if key in sport.lower():
                return ep
        return "soccer/all"

    def load(self, days_back: int = 3):
        if self._loaded:
            return
        now = datetime.now(timezone.utc)
        for i in range(days_back):
            date = (now - timedelta(days=i)).strftime("%Y%m%d")
            for sport_key, ep in self.ENDPOINTS.items():
                url = (f"https://site.api.espn.com/apis/site/v2/sports/"
                       f"{ep}/scoreboard?dates={date}")
                try:
                    r = requests.get(url, timeout=10,
                                     headers={"User-Agent": "Mozilla/5.0"})
                    if r.status_code != 200:
                        continue
                    for event in r.json().get("events", []):
                        state = event.get("status", {}).get("type", {}).get("state")
                        if state != "post":
                            continue
                        comps = event.get("competitions", [{}])[0]
                        competitors = comps.get("competitors", [])
                        home_name = away_name = ""
                        h_score = a_score = 0
                        for comp in competitors:
                            name = (comp.get("team", {}).get("displayName") or
                                    comp.get("athlete", {}).get("displayName", ""))
                            try:
                                score = int(float(comp.get("score", "0") or "0"))
                            except (ValueError, TypeError):
                                score = 0
                            if comp.get("homeAway") == "home":
                                home_name = name
                                h_score   = score
                            else:
                                away_name = name
                                a_score   = score
                        if home_name and away_name:
                            self._results.append({
                                "home":    home_name,
                                "away":    away_name,
                                "h_score": h_score,
                                "a_score": a_score,
                                "sport":   sport_key,
                                "source":  "espn",
                            })
                except Exception as e:
                    logger.debug("[ESPN] %s %s: %s", sport_key, date, str(e)[:50])

        logger.info("✅ [ESPN] %d results loaded", len(self._results))
        self._loaded = True

    def find_result(self, bet: dict) -> Optional[dict]:
        if not self._loaded:
            self.load()
        home  = bet.get("home", "")
        away  = bet.get("away", "")
        best  = None
        best_score = 0.0

        for res in self._results:
            fwd = (team_sim(home, res["home"]) + team_sim(away, res["away"])) / 2
            rev = (team_sim(home, res["away"]) + team_sim(away, res["home"])) / 2

            if fwd >= 0.42 and fwd > best_score:
                best_score = fwd
                best = {
                    "h_score": res["h_score"],
                    "a_score": res["a_score"],
                    "api_h":   res["home"],
                    "api_a":   res["away"],
                    "source":  "espn",
                }
            if rev >= 0.42 and rev > best_score:
                best_score = rev
                best = {
                    "h_score": res["a_score"],
                    "a_score": res["h_score"],
                    "api_h":   res["away"],
                    "api_a":   res["home"],
                    "source":  "espn",
                }

        return best


# =========================================================
# PENDING MANAGER
# =========================================================
class PendingManager:
    MAX_RETRIES = 8
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
                    if self._is_recent(p.get("timestamp", ""), now)
                    and p.get("_retry_count", 0) < self.MAX_RETRIES
                ]
                return d
        except Exception:
            pass
        return {"pending": [], "last_updated": ""}

    def _is_recent(self, ts: str, now: datetime) -> bool:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (now - dt) < timedelta(days=self.MAX_AGE_DAYS)
        except Exception:
            return False

    def _make_id(self, b: dict) -> str:
        raw = f"{b.get('home','')}|{b.get('away','')}|{b.get('market','')}|{b.get('timestamp','')}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def save(self):
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        save_json(PENDING_FILE, self.data)

    def add(self, bet: dict):
        bid = self._make_id(bet)
        existing = {self._make_id(p) for p in self.data["pending"]}
        if bid not in existing:
            self.data["pending"].append({
                **bet,
                "_pending_id":    bid,
                "_retry_count":   0,
                "_pending_since": datetime.now(timezone.utc).isoformat()
            })

    def remove(self, bet: dict):
        bid = self._make_id(bet)
        self.data["pending"] = [
            p for p in self.data["pending"]
            if self._make_id(p) != bid
        ]

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
# TRACKER SYNC
# =========================================================
def sync_to_tracker(bet: dict, tracker: dict) -> bool:
    bid = bet.get("id")
    bts = bet.get("timestamp")
    bhm = bet.get("home")

    for signal in tracker.get("signals", []):
        match_by_id  = bid and signal.get("id") == bid
        match_by_ts  = (not bid and bts and
                        signal.get("timestamp") == bts and
                        signal.get("home") == bhm)
        if match_by_id or match_by_ts:
            signal["outcome"]     = bet["outcome"]
            signal["profit_loss"] = bet["profit_loss"]
            signal["_settled_by"] = bet.get("_settled_by", "system")
            signal["_settled_at"] = datetime.now(timezone.utc).isoformat()
            return True
    return False


# =========================================================
# TELEGRAM REPORT
# =========================================================
def send_report(settled: list, summary: dict):
    if not settled or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import html as html_lib

    src_icon = {
        "thesportsdb": "🗄️",
        "odds_api":    "📡",
        "espn":        "📺",
        "void":        "⚪️",
    }

    lines = ["🧾 <b>ZBET90 SETTLEMENT v6.0</b>\n"]
    daily_pl = 0.0

    for bet in settled:
        outcome = bet.get("outcome", "unknown")
        source  = bet.get("_settled_by", "?")
        icon    = src_icon.get(source, "📡")
        pl      = bet.get("profit_loss", 0.0) or 0.0
        daily_pl += pl if outcome != "void" else 0.0

        result_icon = {"win": "🟢", "loss": "🔴", "void": "⚪️"}.get(outcome, "❓")
        pl_str = f"+{pl:.2f}u" if pl > 0 else f"{pl:.2f}u"

        lines.append(
            f"⚔️ <b>{html_lib.escape(str(bet.get('home','?')))} vs "
            f"{html_lib.escape(str(bet.get('away','?')))}</b>\n"
            f"🎯 {html_lib.escape(str(bet.get('pick','?')))} @ {bet.get('odds','?')}\n"
            f"🏁 <b>{outcome.upper()}</b> {result_icon} | {pl_str} {icon}\n"
        )

    total_icon = "📈" if daily_pl > 0 else "📉"
    wr  = summary.get("win_rate", 0) * 100
    roi = summary.get("roi_pct", 0)

    lines += [
        "══════════════════",
        f"{total_icon} <b>Session PnL:</b> {daily_pl:+.2f} units",
        f"🏆 <b>Win Rate:</b> {wr:.1f}%",
        f"💰 <b>ROI:</b> {roi:.1f}%",
        f"📊 <b>Resolved:</b> {summary.get('resolved',0)} / {summary.get('total_signals',0)}",
        f"⏳ <b>Pending:</b> {summary.get('pending_count',0)}",
    ]

    msg = "\n".join(lines)
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
        logger.info("📤 Report sent.")
    except Exception as e:
        logger.error("Telegram: %s", e)


# =========================================================
# MAIN SETTLER
# =========================================================
async def async_settle():
    logger.info("=" * 60)
    logger.info("⚡ ZBET90 SETTLER v6.0 | TSDB → Odds-API → ESPN")
    logger.info("=" * 60)

    if not PERFORMANCE_FILE.exists():
        logger.info("❌ No performance_tracker.json found.")
        return

    with open(PERFORMANCE_FILE, "r", encoding="utf-8") as f:
        tracker = json.load(f)

    now         = datetime.now(timezone.utc)
    pending_mgr = PendingManager()

    # ── 1. Collect unsettled signals > 3h old ──────────
    existing_ids = {pending_mgr._make_id(p) for p in pending_mgr.get_all()}
    new_added = 0
    for sig in tracker.get("signals", []):
        if sig.get("outcome") is not None:
            continue
        try:
            t = datetime.fromisoformat(sig["timestamp"])
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if (now - t) < timedelta(hours=3):
                continue  # too fresh
        except Exception:
            continue
        bid = pending_mgr._make_id(sig)
        if bid not in existing_ids:
            pending_mgr.add(sig)
            existing_ids.add(bid)
            new_added += 1

    to_check = pending_mgr.get_all()
    logger.info("📋 Pending: %d | New: %d", len(to_check), new_added)
    if not to_check:
        logger.info("ℹ️ Nothing to settle.")
        return

    # ── 2. Initialize settlers ──────────────────────────
    tsdb_settler  = TSDBSettler()
    odds_settler  = OddsAPISettler()
    espn_settler  = ESPNSettler()
    espn_settler.load(days_back=3)

    settled_session: list = []

    # ── 3. Try each settler in order ───────────────────
    for bet in to_check:
        home  = bet.get("home", "")
        away  = bet.get("away", "")
        result = None

        # TSDB (primary)
        result = tsdb_settler.find_result(bet)
        if result:
            logger.info("✅ [TSDB] %s vs %s → %d-%d",
                        home, away, result["h_score"], result["a_score"])
        else:
            # ESPN (second)
            result = espn_settler.find_result(bet)
            if result:
                logger.info("✅ [ESPN] %s vs %s → %d-%d",
                            home, away, result["h_score"], result["a_score"])
            else:
                # Odds API (third)
                result = odds_settler.find_result(bet)
                if result:
                    logger.info("✅ [ODDS-API] %s vs %s → %d-%d",
                                home, away, result["h_score"], result["a_score"])

        if result:
            outcome = resolve_bet(
                bet,
                result["h_score"], result["a_score"],
                result["api_h"], result["api_a"]
            )
            bet["outcome"]     = outcome
            bet["profit_loss"] = calc_profit(bet.get("odds", 2.0), outcome)
            bet["_settled_by"] = result["source"]
            settled_session.append(bet)
            pending_mgr.remove(bet)
            logger.info("💰 %s vs %s → %s (%.2fu) [%s]",
                        home, away, outcome.upper(),
                        bet["profit_loss"], result["source"])
        else:
            # Not found → check age
            try:
                t = datetime.fromisoformat(bet["timestamp"])
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                age_h = (now - t).total_seconds() / 3600
            except Exception:
                age_h = 0

            if age_h > 72:
                bet["outcome"]     = "void"
                bet["profit_loss"] = 0.0
                bet["_settled_by"] = "void"
                settled_session.append(bet)
                pending_mgr.remove(bet)
                logger.warning("⚪️ VOID: %s vs %s (%.0fh old)", home, away, age_h)
            else:
                pending_mgr.increment_retry(bet)
                logger.info("⏳ RETRY %d: %s vs %s",
                            bet.get("_retry_count", 0), home, away)

    # ── 4. Persist & report ────────────────────────────
    pending_mgr.save()

    if not settled_session:
        logger.info("⏳ No new settlements.")
        return

    synced = 0
    for bet in settled_session:
        if sync_to_tracker(bet, tracker):
            synced += 1

    resolved = [s for s in tracker["signals"]
                if s.get("outcome") and s["outcome"] != "void"]
    wins     = [s for s in resolved if s["outcome"] == "win"]
    total_pl = sum(s.get("profit_loss", 0) or 0 for s in resolved)

    tracker["summary"] = {
        "total_signals":           len(tracker["signals"]),
        "resolved":                len(resolved),
        "win_rate":                round(len(wins) / max(len(resolved), 1), 3),
        "total_profit_loss_units": round(total_pl, 2),
        "roi_pct":                 round(total_pl / max(len(resolved), 1) * 100, 2),
        "last_updated":            now.isoformat(),
        "pending_count":           len(pending_mgr.get_all()),
    }

    save_json(PERFORMANCE_FILE, tracker)
    logger.info("💾 Synced %d/%d bets.", synced, len(settled_session))

    send_report(settled_session, tracker["summary"])

    logger.info("=" * 60)
    logger.info("📊 Settled:%d Pending:%d WR:%.1f%% ROI:%.1f%%",
                len(settled_session), len(pending_mgr.get_all()),
                tracker["summary"]["win_rate"] * 100,
                tracker["summary"]["roi_pct"])
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(async_settle())
