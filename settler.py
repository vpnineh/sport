import os
import sys
import json
import logging
import asyncio
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
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

if not ODDS_KEYS:
    logger.critical("FATAL: No ODDS_API_KEY found!")
    sys.exit(1)


# =========================================================
# PENDING MANAGER
# =========================================================
class PendingManager:
    """مدیریت بت‌های در انتظار تسویه"""
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
        bet_id = bet.get("id") or f"{bet.get('home')}|{bet.get('away')}|{bet.get('timestamp')}"
        existing_ids = {
            (p.get("id") or f"{p.get('home')}|{p.get('away')}|{p.get('timestamp')}")
            for p in self.data["pending"]
        }
        if bet_id not in existing_ids:
            self.data["pending"].append({**bet, "_pending_since": datetime.now(timezone.utc).isoformat()})

    def remove(self, bet: dict):
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

    def increment_retry(self, bet: dict):
        bet_id = bet.get("id")
        ts = bet.get("timestamp")
        home = bet.get("home")
        for p in self.data["pending"]:
            if (bet_id and p.get("id") == bet_id) or (ts and p.get("timestamp") == ts and p.get("home") == home):
                p["_retry_count"] = p.get("_retry_count", 0) + 1
                p["_last_retry"] = datetime.now(timezone.utc).isoformat()
                break


# =========================================================
# HELPER FUNCTIONS & BET RESOLVER
# =========================================================
def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def resolve_bet(bet: dict, h_score: int, a_score: int, api_h: str, api_a: str) -> str:
    """
    بررسی دقیق نتیجه بر اساس نوع مارکت (h2h, h2h_lay, totals)
    """
    pick = normalize_str(bet.get("pick", ""))
    market = bet.get("market", "h2h").lower()
    
    # پیدا کردن برنده واقعی در دنیای واقعی
    winner = "draw"
    if h_score > a_score:
        winner = normalize_str(api_h)
    elif a_score > h_score:
        winner = normalize_str(api_a)
        
    def is_match(p_str, w_str):
        if w_str == "draw":
            return "draw" in p_str
        p_tokens = {t for t in p_str.split() if len(t) > 2}
        w_tokens = set(w_str.split())
        if not p_tokens: 
            return p_str == w_str
        return (len(p_tokens & w_tokens) / len(p_tokens)) >= 0.5

    # ── ۱. مارکت برد مستقیم (Match Winner) ──
    if market == "h2h":
        return "win" if is_match(pick, winner) else "loss"
        
    # ── ۲. مارکت ضدِ برد (Lay) ──
    elif market == "h2h_lay":
        # در Lay، اگر تیمی که انتخاب کردیم ببرد، شرط را باخته‌ایم!
        return "loss" if is_match(pick, winner) else "win"
        
    # ── ۳. مارکت مجموع گل/امتیاز (Over/Under) ──
    elif market in ["totals", "over/under"]:
        total_points = h_score + a_score
        # استخراج عدد لاین از پیک (مثل Over 2.5 -> 2.5)
        numbers = re.findall(r"\d+\.?\d*", pick)
        
        if not numbers:
            return "void"
        
        line = float(numbers[0])
        is_over = "over" in pick or "ov" in pick
        is_under = "under" in pick or "un" in pick
        
        if is_over:
            if total_points > line: return "win"
            if total_points < line: return "loss"
            return "void" # Push
            
        elif is_under:
            if total_points < line: return "win"
            if total_points > line: return "loss"
            return "void"
            
    # مارکت پشتیبانی نشده
    return "void"

def match_teams_in_fallback(bet_home: str, bet_away: str, api_h: str, api_a: str) -> bool:
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
    bet_id = bet.get("id")
    for signal in tracker["signals"]:
        if (bet_id and signal.get("id") == bet_id) or (not bet_id and signal.get("timestamp") == bet.get("timestamp") and signal.get("home") == bet.get("home")):
            signal["outcome"] = bet["outcome"]
            signal["profit_loss"] = bet["profit_loss"]
            signal["_settled_by"] = bet.get("_settled_by", "system")
            signal["_settled_at"] = datetime.now(timezone.utc).isoformat()
            break

def calculate_profit(odds: float, outcome: str) -> float:
    if outcome == "win":
        return round(float(odds) - 1.0, 2)
    if outcome == "loss":
        return -1.0
    return 0.0


# =========================================================
# SCRAPING ENGINES
# =========================================================
class ResultScraper:
    def __init__(self):
        self.soccer_results: list = []
        self.other_sports_results: list = []
        self._lock = asyncio.Lock()

    async def _add(self, pool_type: str, result: dict):
        async with self._lock:
            pool = self.soccer_results if pool_type == "soccer" else self.other_sports_results
            h, a = result["home"], result["away"]
            if not any(e["home"] == h and e["away"] == a for e in pool):
                pool.append(result)

    async def fetch_fotmob_soccer(self, target_date: datetime):
        date_str = target_date.strftime("%Y%m%d")
        url = f"https://www.fotmob.com/api/matches?date={date_str}"
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                res = await session.get(url, timeout=15)
                if res.status_code != 200: return
                for league in res.json().get("leagues", []):
                    for match in league.get("matches", []):
                        if not match.get("status", {}).get("finished", False): continue
                        home = match.get("home", {}).get("name", "")
                        away = match.get("away", {}).get("name", "")
                        h_score = match.get("home", {}).get("score", 0) or 0
                        a_score = match.get("away", {}).get("score", 0) or 0
                        
                        await self._add("soccer", {
                            "home": normalize_str(home),
                            "away": normalize_str(away),
                            "home_score": h_score,
                            "away_score": a_score,
                            "source": "fotmob",
                        })
        except Exception as e:
            logger.debug("FotMob error [%s]: %s", date_str, e)

    async def fetch_espn_by_date(self, target_date: datetime):
        date_str = target_date.strftime("%Y%m%d")
        endpoints = [
            (f"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard?dates={date_str}", "other"),
            (f"https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard?dates={date_str}", "other"),
            (f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}", "other"),
            (f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date_str}", "other"),
            (f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard?dates={date_str}", "soccer"),
        ]
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                for url, pool in endpoints:
                    try:
                        res = await session.get(url, timeout=10)
                        if res.status_code != 200: continue
                        for event in res.json().get("events", []):
                            try:
                                state = event.get("status", {}).get("type", {}).get("state")
                                if state != "post": continue
                                comps_list = event.get("competitions", [])
                                if not comps_list: continue
                                competitors = comps_list[0].get("competitors", [])
                                if not competitors: continue

                                home_team, away_team = "", ""
                                h_score, a_score = 0, 0

                                for comp in competitors:
                                    name = comp.get("team", {}).get("displayName") or comp.get("athlete", {}).get("displayName", "")
                                    is_home = comp.get("homeAway") == "home"
                                    try: score_val = int(float(comp.get("score", "0") or "0"))
                                    except (ValueError, TypeError): score_val = 0

                                    if is_home:
                                        home_team = name
                                        h_score = score_val
                                    else:
                                        away_team = name
                                        a_score = score_val

                                if not home_team or not away_team: continue

                                await self._add(pool, {
                                    "home": normalize_str(home_team),
                                    "away": normalize_str(away_team),
                                    "home_score": h_score,
                                    "away_score": a_score,
                                    "source": "espn",
                                })
                            except Exception: continue
                    except Exception: pass
        except Exception as e:
            logger.debug("ESPN error: %s", e)

    async def fetch_sofascore_by_date(self, target_date: datetime):
        date_str = target_date.strftime("%Y-%m-%d")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
            "Referer": "https://www.sofascore.com/",
            "Origin": "https://www.sofascore.com",
        }
        sports = ["tennis", "baseball", "basketball", "ice-hockey"]
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                for sport in sports:
                    try:
                        url = f"https://api.sofascore.com/api/v1/sport/{sport}/scheduled-events/{date_str}"
                        res = await session.get(url, headers=headers, timeout=10)
                        if res.status_code != 200: continue
                        for event in res.json().get("events", []):
                            if event.get("status", {}).get("type") != "finished": continue
                            home = event.get("homeTeam", {}).get("name", "")
                            away = event.get("awayTeam", {}).get("name", "")
                            h_score = event.get("homeScore", {}).get("current", 0) or 0
                            a_score = event.get("awayScore", {}).get("current", 0) or 0
                            
                            await self._add("other", {
                                "home": normalize_str(home),
                                "away": normalize_str(away),
                                "home_score": h_score,
                                "away_score": a_score,
                                "source": "sofascore",
                            })
                    except Exception: pass
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
        logger.info("✅ [SCRAPER] Soccer: %d | Other: %d", len(self.soccer_results), len(self.other_sports_results))

    def fuzzy_match(self, home: str, away: str, sport: str) -> dict:
        is_soccer = any(k in sport.lower() for k in ("football", "soccer"))
        pool = self.soccer_results if is_soccer else self.other_sports_results
        if not pool: return {}

        h_norm = normalize_str(home)
        a_norm = normalize_str(away)
        best_match, best_score = {}, 0

        for match in pool:
            mh, ma = match["home"], match["away"]
            score = 0
            if h_norm in mh or mh in h_norm: score += 2
            if a_norm in ma or ma in a_norm: score += 2
            score += sum(1 for t in h_norm.split() if len(t) > 2 and t in mh)
            score += sum(1 for t in a_norm.split() if len(t) > 2 and t in ma)
            if score > best_score and score >= 3:
                best_score = score
                best_match = match

        return best_match


# =========================================================
# ODDS-API ENGINE
# =========================================================
def fetch_odds_api_results(sport_key: str, days_from: int = 3) -> list:
    for key in ODDS_KEYS:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/?daysFrom={days_from}&apiKey={key}"
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200:
                logger.info("🔄 [ODDS-API] Got %d results for %s", len(res.json()), sport_key)
                return res.json()
            elif res.status_code == 422:
                logger.debug("[ODDS-API] sport_key '%s' not supported or no recent events", sport_key)
                return []
        except Exception as e:
            logger.debug("[ODDS-API] Error using key %s: %s", key[:8], e)
            continue
    return []


# =========================================================
# TELEGRAM REPORT
# =========================================================
def send_telegram_report(settled_bets: list, summary: dict):
    if not settled_bets or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    import html as html_lib

    lines = ["🧾 <b>ZBET90 SETTLEMENT REPORT</b>\n"]
    daily_profit = 0.0

    for bet in settled_bets:
        outcome = bet.get("outcome", "unknown")
        source = bet.get("_settled_by", "unknown")
        source_icon = {
            "scraper": "🌐",
            "fotmob": "⚽",
            "espn": "📺",
            "sofascore": "📱",
            "odds_api": "📡",
            "void": "⚪️",
        }.get(source, "📡")
        
        if outcome == "void":
            icon, profit_str = "⚪️", "0.0u"
        else:
            icon = "🟢" if outcome == "win" else "🔴"
            pl = bet.get("profit_loss", 0.0)
            profit_str = f"+{pl:.2f}u" if pl > 0 else f"{pl:.2f}u"
            daily_profit += pl

        lines.append(f"⚔️ <b>{html_lib.escape(str(bet.get('home', '?')))} vs {html_lib.escape(str(bet.get('away', '?')))}</b>")
        lines.append(f"🎯 Pick: {html_lib.escape(str(bet.get('pick', '?')))} @ {bet.get('odds', '?')}")
        lines.append(f"🏁 Result: <b>{outcome.upper()}</b> {icon} | PnL: {profit_str} {source_icon}\n")

    total_icon = "📈" if daily_profit > 0 else "📉"
    lines += [
        "══════════════════",
        f"{total_icon} <b>Session PnL:</b> {daily_profit:+.2f} units",
        f"🏆 <b>Win Rate:</b> {summary.get('win_rate', 0) * 100:.1f}%",
        f"💰 <b>ROI:</b> {summary.get('roi_pct', 0):.1f}%",
        f"📊 <b>Resolved:</b> {summary.get('resolved', 0)} / {summary.get('total_signals', 0)}",
    ]

    message_html = "\n".join(lines)
    
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message_html, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception as e:
        logger.error("Telegram send error: %s", e)


# =========================================================
# MAIN SETTLER
# =========================================================
async def async_settle():
    logger.info("=" * 55)
    logger.info("⚡ ZBET90 SETTLER ENGINE v4.0 | Scraper -> API Pipeline")
    logger.info("=" * 55)

    if not PERFORMANCE_FILE.exists():
        logger.info("❌ No performance tracker file found. Exiting.")
        return

    with open(PERFORMANCE_FILE, "r", encoding="utf-8") as f:
        tracker = json.load(f)

    now = datetime.now(timezone.utc)
    pending_mgr = PendingManager()

    fresh_unsettled = []
    for b in tracker.get("signals", []):
        if b.get("outcome") is not None:
            continue
        try:
            bet_time = datetime.fromisoformat(b["timestamp"])
            if bet_time.tzinfo is None:
                bet_time = bet_time.replace(tzinfo=timezone.utc)
            if (now - bet_time) > timedelta(hours=3): 
                fresh_unsettled.append(b)
        except Exception:
            pass

    existing_pending_ids = {(p.get("id") or f"{p.get('home')}|{p.get('timestamp')}") for p in pending_mgr.get_all()}
    for b in fresh_unsettled:
        bid = b.get("id") or f"{b.get('home')}|{b.get('timestamp')}"
        if bid not in existing_pending_ids:
            pending_mgr.add(b)

    to_check = pending_mgr.get_all()
    if not to_check:
        logger.info("ℹ️ No bets ready to settle.")
        return

    logger.info("🔍 %d bets to check.", len(to_check))

    # ─── 1. اجرای Scraper ───
    scraper = ResultScraper()
    await scraper.load_recent_results()

    settled_this_session = []
    need_api_check = []

    # ─── 2. بررسی نتایج در Scraper ───
    for bet in to_check:
        match = scraper.fuzzy_match(bet.get("home", ""), bet.get("away", ""), bet.get("sport", "soccer"))
        
        if match:
            h_score = match.get("home_score", 0)
            a_score = match.get("away_score", 0)
            
            outcome = resolve_bet(bet, h_score, a_score, match["home"], match["away"])
            
            bet["outcome"] = outcome
            bet["profit_loss"] = calculate_profit(bet.get("odds", 2.0), outcome)
            bet["_settled_by"] = match.get("source", "scraper")
            
            settled_this_session.append(bet)
            pending_mgr.remove(bet)
            logger.info("✅ [%s] %s vs %s → %s (%.2f units)", bet["_settled_by"].upper(), bet["home"], bet["away"], outcome.upper(), bet["profit_loss"])
        else:
            need_api_check.append(bet)

    # ─── 3. اجرای Odds-API برای مسابقات باقی‌مانده ───
    if need_api_check:
        logger.info("📡 %d matches not found by scraper. Passing to Odds-API...", len(need_api_check))
        odds_api_cache: dict = {}
        needed_sports = {bet.get("api_sport_key") for bet in need_api_check if bet.get("api_sport_key")}
        
        for sport in needed_sports:
            odds_api_cache[sport] = fetch_odds_api_results(sport, days_from=3)

        still_pending = []

        for bet in need_api_check:
            api_sport_key = bet.get("api_sport_key", "")
            result_found = False
            
            if api_sport_key in odds_api_cache:
                for api_match in odds_api_cache[api_sport_key]:
                    api_h = normalize_str(api_match.get("home_team", ""))
                    api_a = normalize_str(api_match.get("away_team", ""))

                    if not match_teams_in_fallback(bet["home"], bet["away"], api_h, api_a):
                        continue

                    if not api_match.get("completed"):
                        logger.info("⏳ [ODDS-API] Not finished yet: %s vs %s", bet["home"], bet["away"])
                        break

                    scores = api_match.get("scores") or []
                    if scores:
                        h_score = find_score_in_fallback(scores, api_match.get("home_team", ""))
                        a_score = find_score_in_fallback(scores, api_match.get("away_team", ""))
                        
                        outcome = resolve_bet(bet, h_score, a_score, api_h, api_a)
                        
                        bet["outcome"] = outcome
                        bet["profit_loss"] = calculate_profit(bet.get("odds", 2.0), outcome)
                        bet["_settled_by"] = "odds_api"

                        settled_this_session.append(bet)
                        pending_mgr.remove(bet)
                        result_found = True
                        
                        logger.info("✅ [ODDS-API] %s vs %s → %s (%.2f units)", bet["home"], bet["away"], outcome.upper(), bet["profit_loss"])
                    break

            if not result_found:
                try:
                    bet_time = datetime.fromisoformat(bet["timestamp"])
                    if bet_time.tzinfo is None:
                        bet_time = bet_time.replace(tzinfo=timezone.utc)
                    hours_elapsed = (now - bet_time).total_seconds() / 3600
                except Exception:
                    hours_elapsed = 0

                # اگر 48 ساعت گذشت و پیدا نشد، Void کن
                if hours_elapsed > 48:
                    bet["outcome"] = "void"
                    bet["profit_loss"] = 0.0
                    bet["_settled_by"] = "void"
                    settled_this_session.append(bet)
                    pending_mgr.remove(bet)
                    logger.warning("⚪️ [VOID] %s vs %s (%.0fh elapsed)", bet["home"], bet["away"], hours_elapsed)
                else:
                    pending_mgr.increment_retry(bet)
                    still_pending.append(bet)

    # ─── 4. همگام‌سازی و گزارش ───
    pending_mgr.save()

    if not settled_this_session:
        logger.info("⏳ No matches settled this session.")
        return

    for bet in settled_this_session:
        sync_bet_to_tracker(bet, tracker)

    resolved = [s for s in tracker["signals"] if s.get("outcome") not in (None, "void")]
    wins = [s for s in resolved if s["outcome"] == "win"]
    total_pl = sum(s.get("profit_loss", 0) or 0 for s in resolved)

    tracker["summary"] = {
        "total_signals": len(tracker["signals"]),
        "resolved": len(resolved),
        "win_rate": round(len(wins) / len(resolved), 3) if resolved else 0.0,
        "total_profit_loss_units": round(total_pl, 2),
        "roi_pct": round((total_pl / len(resolved)) * 100, 2) if resolved else 0.0,
        "last_updated": now.isoformat(),
        "pending_count": len(pending_mgr.get_all()),
    }

    save_json_safe(PERFORMANCE_FILE, tracker)
    send_telegram_report(settled_this_session, tracker["summary"])
    logger.info("=" * 55)

if __name__ == "__main__":
    asyncio.run(async_settle())
