import os
import sys
import json
import logging
import asyncio
import unicodedata
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import requests
from google import genai
from google.genai import types
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from curl_cffi.requests import AsyncSession

# =========================================================
# CONFIGURATION
# =========================================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
PERFORMANCE_FILE = Path("api_cache/performance_tracker.json")
PENDING_FILE = Path("api_cache/pending_settlement.json")
LOG_FILE = Path("api_cache/settler_logs.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI", "").strip()

ODDS_KEYS = [k for k in [
    os.getenv("ODDS_API_KEY", "").strip(),
    os.getenv("ODDS_API_KEY2", "").strip(),
    os.getenv("ODDS_API_KEY3", "").strip(),
] if k]

# ─── Logging ──────────────────────────────────────────────
logger = logging.getLogger("SETTLER")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
for h in [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
]:
    h.setFormatter(formatter)
    logger.addHandler(h)

if not GEMINI_API_KEY:
    logger.critical("FATAL: GEMINI env var not set!")
    sys.exit(1)


# =========================================================
# GEMINI RESULT ENGINE (Async & New SDK)
# =========================================================
class GeminiResultEngine:
    _instance: Optional["GeminiResultEngine"] = None
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

        self.client = genai.Client(api_key=GEMINI_API_KEY)

        self._system = (
            "You are a sports results database. Your ONLY job is to return VERIFIED match results.\n"
            "RULES:\n"
            "1. Only report results you are CERTAIN about from your training data.\n"
            "2. If the match happened after your knowledge cutoff OR you are not sure → set 'known': false.\n"
            "3. For soccer/football: winner can be 'home', 'away', or 'draw'.\n"
            "4. For tennis/basketball/baseball: winner is always 'home' or 'away' (no draw).\n"
            "5. NEVER guess or hallucinate. Uncertainty = unknown.\n\n"
            "Output schema (strict JSON):\n"
            "{\n"
            '  "known": bool,\n'
            '  "winner": "home" | "away" | "draw" | null,\n'
            '  "home_score": int | null,\n'
            '  "away_score": int | null,\n'
            '  "confidence": "high" | "medium" | "low",\n'
            '  "note": "string (optional)"\n'
            "}"
        )

        self._config = types.GenerateContentConfig(
            system_instruction=self._system,
            temperature=0.0,
            max_output_tokens=256,
            response_mime_type="application/json",
            safety_settings=[
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
        )

        self._request_times: list = []
        # استفاده از قفل غیرهم‌زمان
        self._rate_lock = asyncio.Lock()

        self._initialized = True
        logger.info("✅ [GEMINI ENGINE] Initialized (New SDK, Async, gemini-1.5-flash, temp=0.0)")

    async def _rate_limit_wait(self):
        """حداکثر ۱۴ درخواست در دقیقه با پشتیبانی Async."""
        async with self._rate_lock:
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 60]

            if len(self._request_times) >= 14:
                wait = 60 - (now - self._request_times[0]) + 1
                if wait > 0:
                    logger.info("[GEMINI] Rate limit reached, waiting %.1fs...", wait)
                    await asyncio.sleep(wait)  # فریز نشدن کل برنامه

            self._request_times.append(time.time())

    async def query_match_result(
        self, home: str, away: str, sport: str, match_date: str, max_retries: int = 2
    ) -> Optional[dict]:
        prompt = (
            f"Match result query:\n"
            f"Sport: {sport}\n"
            f"Home team: {home}\n"
            f"Away team: {away}\n"
            f"Match date (UTC): {match_date}\n\n"
            f"Did this match finish? Who won?\n"
            f"Return the JSON result."
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                await self._rate_limit_wait()
                
                # فراخوانی به صورت کاملاً غیرهم‌زمان
                response = await self.client.aio.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                    config=self._config
                )

                raw = response.text
                if not raw:
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    import re
                    m = re.search(r"\{[\s\S]*\}", raw)
                    if m:
                        try:
                            data = json.loads(m.group(0))
                        except Exception:
                            continue
                    else:
                        continue

                if not isinstance(data, dict):
                    continue

                if not data.get("known", False) or data.get("confidence") == "low":
                    return None

                winner_raw = data.get("winner")
                if winner_raw not in ["home", "away", "draw"]:
                    return None

                if winner_raw == "home":
                    winner_name = home
                elif winner_raw == "away":
                    winner_name = away
                else:
                    winner_name = "draw"

                result = {
                    "home": normalize_str(home),
                    "away": normalize_str(away),
                    "winner": normalize_str(winner_name),
                    "home_score": data.get("home_score"),
                    "away_score": data.get("away_score"),
                    "source": "gemini",
                    "confidence": data.get("confidence", "medium"),
                }
                
                logger.info(
                    "🤖 [GEMINI] %s vs %s → %s",
                    home, away, result["winner"].upper(),
                )
                return result

            except Exception as e:
                err_str = str(e)
                last_error = err_str

                if "429" in err_str or "quota" in err_str.lower():
                    wait = (attempt + 1) * 15
                    logger.warning("[GEMINI] Rate limited, waiting %ds", wait)
                    await asyncio.sleep(wait) # فریز نشدن کل برنامه
                elif "400" in err_str:
                    return None
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)

        return None


# =========================================================
# PENDING MANAGER  ← مدیریت بت‌های بی‌جواب
# =========================================================
class PendingManager:
    """
    بت‌هایی که هنوز نتیجه نگرفتیم رو نگه می‌داره.
    دور بعدی اجرا دوباره چک می‌کنه.
    """

    def __init__(self):
        self.data = self._load()

    def _load(self) -> dict:
        try:
            if PENDING_FILE.exists():
                with open(PENDING_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    # پاک کردن ورودی‌های خیلی قدیمی (>5 روز)
                    now = datetime.now(timezone.utc)
                    d["pending"] = [
                        p for p in d.get("pending", [])
                        if self._is_recent(p.get("timestamp", ""), days=5, now=now)
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

    def save(self):
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        save_json_safe(PENDING_FILE, self.data)

    def add(self, bet: dict):
        """اضافه کردن bet به لیست pending (بدون duplicate)."""
        bet_id = bet.get("id") or f"{bet.get('home')}|{bet.get('away')}|{bet.get('timestamp')}"
        existing_ids = {
            (p.get("id") or f"{p.get('home')}|{p.get('away')}|{p.get('timestamp')}")
            for p in self.data["pending"]
        }
        if bet_id not in existing_ids:
            self.data["pending"].append({**bet, "_pending_since": datetime.now(timezone.utc).isoformat()})

    def remove(self, bet: dict):
        """حذف bet از pending بعد از settle شدن."""
        bet_id = bet.get("id")
        ts = bet.get("timestamp")
        home = bet.get("home")

        self.data["pending"] = [
            p for p in self.data["pending"]
            if not (
                (bet_id and p.get("id") == bet_id)
                or (ts and p.get("timestamp") == ts and p.get("home") == home)
            )
        ]

    def get_all(self) -> list:
        return list(self.data["pending"])

    def retry_count(self, bet: dict) -> int:
        """تعداد دفعاتی که برای این bet تلاش کردیم."""
        bet_id = bet.get("id")
        for p in self.data["pending"]:
            if bet_id and p.get("id") == bet_id:
                return p.get("_retry_count", 0)
        return 0

    def increment_retry(self, bet: dict):
        """شمارنده retry رو زیاد می‌کنه."""
        bet_id = bet.get("id")
        ts = bet.get("timestamp")
        home = bet.get("home")
        for p in self.data["pending"]:
            match = (
                (bet_id and p.get("id") == bet_id)
                or (ts and p.get("timestamp") == ts and p.get("home") == home)
            )
            if match:
                p["_retry_count"] = p.get("_retry_count", 0) + 1
                p["_last_retry"] = datetime.now(timezone.utc).isoformat()
                break


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def is_pick_winner(pick: str, winner: str) -> bool:
    """تطبیق pick با winner با fuzzy matching."""
    if not pick or not winner:
        return False
    pick_norm = normalize_str(pick)
    winner_norm = normalize_str(winner)

    if winner_norm == "draw":
        return "draw" in pick_norm

    p_tokens = {t for t in pick_norm.split() if len(t) > 2}
    w_tokens = set(winner_norm.split())

    if not p_tokens:
        return pick_norm == winner_norm

    overlap = len(p_tokens & w_tokens) / len(p_tokens)
    return overlap >= 0.5


def match_teams_in_fallback(
    bet_home: str, bet_away: str, api_h: str, api_a: str
) -> bool:
    h_norm = normalize_str(bet_home)
    a_norm = normalize_str(bet_away)
    score = 0
    if h_norm in api_h or api_h in h_norm:
        score += 2
    if a_norm in api_a or api_a in a_norm:
        score += 2
    score += sum(1 for t in h_norm.split() if len(t) > 2 and t in api_h)
    score += sum(1 for t in a_norm.split() if len(t) > 2 and t in api_a)
    return score >= 3


def find_score_in_fallback(scores: list, team_name: str) -> int:
    team_norm = normalize_str(team_name)
    for s in scores:
        s_norm = normalize_str(s.get("name", ""))
        if s_norm in team_norm or team_norm in s_norm:
            try:
                return int(float(s["score"]))
            except (ValueError, KeyError):
                pass
    return 0


def save_json_safe(filepath: Path, data: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    tmp_path.replace(filepath)


def sync_bet_to_tracker(bet: dict, tracker: dict):
    """همگام‌سازی نتیجه با tracker."""
    bet_id = bet.get("id")
    for signal in tracker["signals"]:
        matched = (bet_id and signal.get("id") == bet_id) or (
            not bet_id
            and signal.get("timestamp") == bet.get("timestamp")
            and signal.get("home") == bet.get("home")
        )
        if matched:
            signal["outcome"] = bet["outcome"]
            signal["profit_loss"] = bet["profit_loss"]
            signal["_settled_by"] = bet.get("_settled_by", "unknown")
            signal["_settled_at"] = datetime.now(timezone.utc).isoformat()
            break


def calculate_profit(odds: float, outcome: str) -> float:
    if outcome == "win":
        return round(float(odds) - 1.0, 2)
    if outcome == "loss":
        return -1.0
    return 0.0


# =========================================================
# SCRAPING ENGINES  ← Fallback اول بعد از Gemini
# =========================================================
class ResultScraper:
    def __init__(self):
        self.soccer_results: list = []
        self.other_sports_results: list = []
        self._lock = asyncio.Lock()

    async def _add(self, pool_type: str, result: dict):
        async with self._lock:
            pool = (
                self.soccer_results
                if pool_type == "soccer"
                else self.other_sports_results
            )
            h, a = result["home"], result["away"]
            if not any(e["home"] == h and e["away"] == a for e in pool):
                pool.append(result)

    async def fetch_fotmob_soccer(self, target_date: datetime):
        date_str = target_date.strftime("%Y%m%d")
        url = f"https://www.fotmob.com/api/matches?date={date_str}"
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                res = await session.get(url, timeout=15)
                if res.status_code != 200:
                    return
                for league in res.json().get("leagues", []):
                    for match in league.get("matches", []):
                        if not match.get("status", {}).get("finished", False):
                            continue
                        home = match.get("home", {}).get("name", "")
                        away = match.get("away", {}).get("name", "")
                        h_score = match.get("home", {}).get("score", 0) or 0
                        a_score = match.get("away", {}).get("score", 0) or 0
                        winner = "draw"
                        if h_score > a_score:
                            winner = home
                        elif a_score > h_score:
                            winner = away
                        await self._add("soccer", {
                            "home": normalize_str(home),
                            "away": normalize_str(away),
                            "winner": normalize_str(winner),
                            "source": "fotmob",
                        })
        except Exception as e:
            logger.debug("FotMob error [%s]: %s", date_str, e)

    async def fetch_espn_by_date(self, target_date: datetime):
        date_str = target_date.strftime("%Y%m%d")
        endpoints = [
            (
                f"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard?dates={date_str}",
                "other",
            ),
            (
                f"https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard?dates={date_str}",
                "other",
            ),
            (
                f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}",
                "other",
            ),
            (
                f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date_str}",
                "other",
            ),
            (
                f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard?dates={date_str}",
                "soccer",
            ),
        ]
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                for url, pool in endpoints:
                    try:
                        res = await session.get(url, timeout=10)
                        if res.status_code != 200:
                            continue
                        for event in res.json().get("events", []):
                            try:
                                state = (
                                    event.get("status", {})
                                    .get("type", {})
                                    .get("state")
                                )
                                if state != "post":
                                    continue
                                comps_list = event.get("competitions", [])
                                if not comps_list:
                                    continue
                                competitors = comps_list[0].get("competitors", [])
                                if not competitors:
                                    continue

                                home_team, away_team, winner = "", "", "draw"
                                h_score, a_score = 0, 0

                                for comp in competitors:
                                    name = comp.get("team", {}).get(
                                        "displayName"
                                    ) or comp.get("athlete", {}).get(
                                        "displayName", ""
                                    )
                                    is_home = comp.get("homeAway") == "home"
                                    try:
                                        score_val = int(
                                            float(comp.get("score", "0") or "0")
                                        )
                                    except (ValueError, TypeError):
                                        score_val = 0

                                    if is_home:
                                        home_team = name
                                        h_score = score_val
                                    else:
                                        away_team = name
                                        a_score = score_val

                                    if comp.get("winner"):
                                        winner = name

                                if not home_team or not away_team:
                                    continue

                                if pool == "soccer":
                                    if winner == "draw":
                                        if h_score > a_score:
                                            winner = home_team
                                        elif a_score > h_score:
                                            winner = away_team

                                await self._add(pool, {
                                    "home": normalize_str(home_team),
                                    "away": normalize_str(away_team),
                                    "winner": normalize_str(winner),
                                    "source": "espn",
                                })
                            except Exception:
                                continue
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("ESPN error: %s", e)

    async def fetch_sofascore_by_date(self, target_date: datetime):
        date_str = target_date.strftime("%Y-%m-%d")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/110.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.sofascore.com/",
            "Origin": "https://www.sofascore.com",
        }
        sports = ["tennis", "baseball", "basketball", "ice-hockey"]
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                for sport in sports:
                    try:
                        url = (
                            f"https://api.sofascore.com/api/v1/sport/{sport}"
                            f"/scheduled-events/{date_str}"
                        )
                        res = await session.get(url, headers=headers, timeout=10)
                        if res.status_code != 200:
                            continue
                        for event in res.json().get("events", []):
                            if event.get("status", {}).get("type") != "finished":
                                continue
                            home = event.get("homeTeam", {}).get("name", "")
                            away = event.get("awayTeam", {}).get("name", "")
                            h_score = event.get("homeScore", {}).get("current", 0) or 0
                            a_score = event.get("awayScore", {}).get("current", 0) or 0
                            winner = "draw"
                            if h_score > a_score:
                                winner = home
                            elif a_score > h_score:
                                winner = away
                            await self._add("other", {
                                "home": normalize_str(home),
                                "away": normalize_str(away),
                                "winner": normalize_str(winner),
                                "source": "sofascore",
                            })
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("SofaScore error: %s", e)

    async def load_recent_results(self):
        logger.info("🌍 [SCRAPER] Fetching from FotMob / ESPN / SofaScore...")
        now = datetime.now(timezone.utc)
        tasks = []
        for i in range(3):
            target = now - timedelta(days=i)
            tasks += [
                self.fetch_fotmob_soccer(target),
                self.fetch_espn_by_date(target),
                self.fetch_sofascore_by_date(target),
            ]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            "✅ [SCRAPER] Soccer: %d | Other: %d",
            len(self.soccer_results),
            len(self.other_sports_results),
        )

    def fuzzy_match(self, home: str, away: str, sport: str) -> dict:
        is_soccer = any(k in sport.lower() for k in ("football", "soccer"))
        pool = self.soccer_results if is_soccer else self.other_sports_results
        if not pool:
            return {}

        h_norm = normalize_str(home)
        a_norm = normalize_str(away)
        best_match, best_score = {}, 0

        for match in pool:
            mh, ma = match["home"], match["away"]
            score = 0
            if h_norm in mh or mh in h_norm:
                score += 2
            if a_norm in ma or ma in a_norm:
                score += 2
            score += sum(1 for t in h_norm.split() if len(t) > 2 and t in mh)
            score += sum(1 for t in a_norm.split() if len(t) > 2 and t in ma)
            if score > best_score and score >= 3:
                best_score = score
                best_match = match

        return best_match


# =========================================================
# ODDS-API FALLBACK  ← Fallback آخر
# =========================================================
def fetch_odds_api_results(sport_key: str, days_from: int = 3) -> list:
    if not ODDS_KEYS:
        return []
    for key in ODDS_KEYS:
        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/"
            f"?daysFrom={days_from}&apiKey={key}"
        )
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200:
                logger.info(
                    "🔄 [ODDS-API] Got %d results for %s",
                    len(res.json()), sport_key
                )
                return res.json()
            elif res.status_code == 422:
                logger.debug("[ODDS-API] sport_key '%s' not supported", sport_key)
                return []
        except Exception as e:
            logger.debug("[ODDS-API] Error: %s", e)
            continue
    return []


# =========================================================
# TELEGRAM REPORT
# =========================================================
def send_telegram_report(settled_bets: list, summary: dict):
    if not settled_bets or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    import html as html_lib

    lines = ["🧾 <b>ZBET90 DAILY SETTLEMENT REPORT</b>\n"]
    daily_profit = 0.0

    for bet in settled_bets:
        outcome = bet.get("outcome", "unknown")
        source = bet.get("_settled_by", "unknown")
        source_icon = {
            "gemini": "🤖",
            "scraper": "🌐",
            "odds_api": "📡",
            "void": "⚪️",
        }.get(source, "❓")

        if outcome == "void":
            icon, profit_str = "⚪️", "0.0u"
        else:
            icon = "🟢" if outcome == "win" else "🔴"
            pl = bet.get("profit_loss", 0.0)
            profit_str = f"+{pl:.2f}u" if pl > 0 else f"{pl:.2f}u"
            daily_profit += pl

        lines.append(
            f"⚔️ <b>{html_lib.escape(str(bet.get('home', '?')))} "
            f"vs {html_lib.escape(str(bet.get('away', '?')))}</b>"
        )
        lines.append(
            f"🎯 Pick: {html_lib.escape(str(bet.get('pick', '?')))} "
            f"@ {bet.get('odds', '?')}"
        )
        lines.append(
            f"🏁 Result: <b>{outcome.upper()}</b> {icon} | "
            f"PnL: {profit_str} {source_icon}\n"
        )

    total_icon = "📈" if daily_profit > 0 else "📉"
    lines += [
        "══════════════════",
        f"{total_icon} <b>Session PnL:</b> {daily_profit:+.2f} units",
        f"🏆 <b>Win Rate:</b> {summary.get('win_rate', 0) * 100:.1f}%",
        f"💰 <b>ROI:</b> {summary.get('roi_pct', 0):.1f}%",
        f"📊 <b>Resolved:</b> {summary.get('resolved', 0)} / {summary.get('total_signals', 0)}",
    ]

    message_html = "\n".join(lines)
    MAX_LEN = 4000
    chunks, current = [], ""
    for line in message_html.split("\n"):
        if len(current) + len(line) + 1 > MAX_LEN:
            chunks.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current.strip())

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
                timeout=10,
            )
            if not res.ok:
                logger.error("Telegram error [%d]: %s", res.status_code, res.text[:100])
        except Exception as e:
            logger.error("Telegram send error: %s", e)


# =========================================================
# MAIN SETTLER
# =========================================================
async def async_settle():
    logger.info("=" * 55)
    logger.info("🤖 ZBET90 SETTLER ENGINE v2.0 | Gemini-Powered")
    logger.info("=" * 55)

    # ─── Load Tracker ────────────────────────────────────
    if not PERFORMANCE_FILE.exists():
        logger.info("❌ No performance tracker file found. Exiting.")
        return

    with open(PERFORMANCE_FILE, "r", encoding="utf-8") as f:
        tracker = json.load(f)

    now = datetime.now(timezone.utc)

    # ─── Load Pending ─────────────────────────────────────
    pending_mgr = PendingManager()

    # بت‌های تازه که هنوز settle نشدن
    fresh_unsettled = []
    for b in tracker.get("signals", []):
        if b.get("outcome") is not None:
            continue
        try:
            bet_time = datetime.fromisoformat(b["timestamp"])
            if bet_time.tzinfo is None:
                bet_time = bet_time.replace(tzinfo=timezone.utc)
            # فقط بت‌هایی که بیش از 4 ساعت گذشته
            if (now - bet_time) > timedelta(hours=4):
                fresh_unsettled.append(b)
        except Exception as e:
            logger.error("Time parse error [%s]: %s", b.get("id", "?"), e)

    # ترکیب: pending قبلی + تازه‌ها (بدون تکرار)
    existing_pending_ids = {
        (p.get("id") or f"{p.get('home')}|{p.get('timestamp')}")
        for p in pending_mgr.get_all()
    }
    for b in fresh_unsettled:
        bid = b.get("id") or f"{b.get('home')}|{b.get('timestamp')}"
        if bid not in existing_pending_ids:
            pending_mgr.add(b)

    to_check = pending_mgr.get_all()
    if not to_check:
        logger.info("ℹ️ No bets ready to settle.")
        return

    logger.info("🔍 %d bets to check (including pending from previous runs).", len(to_check))

    # ─── Initialize Engines ──────────────────────────────
    gemini_engine = GeminiResultEngine()
    scraper = ResultScraper()

    # scraper رو parallel اجرا کن
    scraper_task = asyncio.create_task(scraper.load_recent_results())

    # ─── Phase 1: Gemini Query ────────────────────────────
    logger.info("🤖 [PHASE 1] Querying Gemini for %d matches...", len(to_check))
    gemini_results: dict = {}   # bet_key → result

    for bet in to_check:
        bet_key = bet.get("id") or f"{bet.get('home')}|{bet.get('timestamp')}"

        try:
            bet_time = datetime.fromisoformat(bet["timestamp"])
            if bet_time.tzinfo is None:
                bet_time = bet_time.replace(tzinfo=timezone.utc)
            match_date_str = bet_time.strftime("%Y-%m-%d")
        except Exception:
            match_date_str = "unknown date"

        result = await gemini_engine.query_match_result(
            home=bet.get("home", ""),
            away=bet.get("away", ""),
            sport=bet.get("sport", "soccer"),
            match_date=match_date_str,
        )

        if result:
            gemini_results[bet_key] = result

    gemini_found = len(gemini_results)
    gemini_missed = len(to_check) - gemini_found
    logger.info(
        "🤖 [GEMINI] Found: %d | Missed: %d → going to scraper/odds-api",
        gemini_found, gemini_missed
    )

    # ─── Wait for Scraper ─────────────────────────────────
    await scraper_task

    # ─── Phase 2 & 3: Scraper + Odds-API for misses ───────
    odds_api_cache: dict = {}
    scraper_results: dict = {}    # bet_key → result

    for bet in to_check:
        bet_key = bet.get("id") or f"{bet.get('home')}|{bet.get('timestamp')}"
        if bet_key in gemini_results:
            continue    # Gemini قبلاً پیدا کرده

        # Scraper
        match = scraper.fuzzy_match(
            bet.get("home", ""),
            bet.get("away", ""),
            bet.get("sport", "soccer"),
        )
        if match:
            scraper_results[bet_key] = {**match, "source": match.get("source", "scraper")}
            continue

        # Odds-API
        api_sport_key = bet.get("api_sport_key", "")
        if not api_sport_key:
            continue

        if api_sport_key not in odds_api_cache:
            odds_api_cache[api_sport_key] = fetch_odds_api_results(
                api_sport_key, days_from=3
            )

        for api_match in odds_api_cache.get(api_sport_key, []):
            api_h = normalize_str(api_match.get("home_team", ""))
            api_a = normalize_str(api_match.get("away_team", ""))

            if not match_teams_in_fallback(
                bet["home"], bet["away"], api_h, api_a
            ):
                continue

            if not api_match.get("completed"):
                logger.info(
                    "⏳ [ODDS-API] Not finished yet: %s vs %s",
                    bet["home"], bet["away"]
                )
                break

            scores = api_match.get("scores") or []
            if scores:
                h_score = find_score_in_fallback(
                    scores, api_match.get("home_team", "")
                )
                a_score = find_score_in_fallback(
                    scores, api_match.get("away_team", "")
                )
                winner = "draw"
                if h_score > a_score:
                    winner = api_h
                elif a_score > h_score:
                    winner = api_a

                scraper_results[bet_key] = {
                    "home": api_h,
                    "away": api_a,
                    "winner": winner,
                    "source": "odds_api",
                }
                logger.info(
                    "📡 [ODDS-API] Found: %s vs %s → %s",
                    bet["home"], bet["away"], winner.upper()
                )
            break

    # ─── Phase 4: Settle ──────────────────────────────────
    settled_this_session: list = []
    still_pending: list = []

    for bet in to_check:
        bet_key = bet.get("id") or f"{bet.get('home')}|{bet.get('timestamp')}"

        # پیدا کردن نتیجه
        result = gemini_results.get(bet_key) or scraper_results.get(bet_key)

        if result:
            winner = result.get("winner", "")
            outcome = "win" if is_pick_winner(bet["pick"], winner) else "loss"
            source = result.get("source", "unknown")

            bet["outcome"] = outcome
            bet["profit_loss"] = calculate_profit(bet.get("odds", 2.0), outcome)
            bet["_settled_by"] = source

            settled_this_session.append(bet)
            pending_mgr.remove(bet)

            logger.info(
                "✅ [%s] %s vs %s → %s (%.2f units)",
                source.upper(),
                bet["home"], bet["away"],
                outcome.upper(),
                bet["profit_loss"],
            )

        else:
            # هنوز بی‌جواب - بررسی void rule
            try:
                bet_time = datetime.fromisoformat(bet["timestamp"])
                if bet_time.tzinfo is None:
                    bet_time = bet_time.replace(tzinfo=timezone.utc)
                hours_elapsed = (now - bet_time).total_seconds() / 3600
            except Exception:
                hours_elapsed = 0

            if hours_elapsed > 48:
                # VOID
                bet["outcome"] = "void"
                bet["profit_loss"] = 0.0
                bet["_settled_by"] = "void"
                settled_this_session.append(bet)
                pending_mgr.remove(bet)
                logger.warning(
                    "⚪️ [VOID] %s vs %s (%.0fh elapsed)",
                    bet["home"], bet["away"], hours_elapsed
                )
            else:
                # هنوز صبر می‌کنیم
                pending_mgr.increment_retry(bet)
                retry_n = pending_mgr.retry_count(bet)
                still_pending.append(bet)
                logger.info(
                    "⏳ [PENDING] %s vs %s (retry #%d, %.0fh elapsed)",
                    bet["home"], bet["away"], retry_n, hours_elapsed
                )

    # ─── Save Pending ─────────────────────────────────────
    pending_mgr.save()
    logger.info(
        "💾 [PENDING] Saved %d unsettled bets for next run",
        len(still_pending)
    )

    if not settled_this_session:
        logger.info("⏳ No matches settled this session.")
        return

    # ─── Sync to Tracker ──────────────────────────────────
    for bet in settled_this_session:
        sync_bet_to_tracker(bet, tracker)

    # ─── Update Summary ───────────────────────────────────
    resolved = [
        s for s in tracker["signals"]
        if s.get("outcome") not in (None, "void")
    ]
    wins = [s for s in resolved if s["outcome"] == "win"]
    total_pl = sum(s.get("profit_loss", 0) or 0 for s in resolved)

    tracker["summary"] = {
        "total_signals": len(tracker["signals"]),
        "resolved": len(resolved),
        "win_rate": round(len(wins) / len(resolved), 3) if resolved else 0.0,
        "total_profit_loss_units": round(total_pl, 2),
        "roi_pct": round((total_pl / len(resolved)) * 100, 2) if resolved else 0.0,
        "last_updated": now.isoformat(),
        "pending_count": len(still_pending),
    }

    save_json_safe(PERFORMANCE_FILE, tracker)
    logger.info(
        "📊 Settled: %d | Win: %d | Loss: %d | Void: %d",
        len(settled_this_session),
        sum(1 for b in settled_this_session if b["outcome"] == "win"),
        sum(1 for b in settled_this_session if b["outcome"] == "loss"),
        sum(1 for b in settled_this_session if b["outcome"] == "void"),
    )

    # ─── Telegram Report ──────────────────────────────────
    send_telegram_report(settled_this_session, tracker["summary"])
    logger.info("=" * 55)


if __name__ == "__main__":
    asyncio.run(async_settle())
