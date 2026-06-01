import os
import sys
import json
import logging
import asyncio
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from curl_cffi.requests import AsyncSession

# =========================================================
# CONFIGURATION & DIRECTORY INITIALIZATION
# =========================================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
PERFORMANCE_FILE = Path("api_cache/performance_tracker.json")
LOG_FILE = Path("api_cache/settler_logs.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

ODDS_KEYS = [k for k in [
    os.getenv("ODDS_API_KEY", "").strip(),
    os.getenv("ODDS_API_KEY2", "").strip(),
    os.getenv("ODDS_API_KEY3", "").strip()
] if k]

logger = logging.getLogger("SETTLER")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")]:
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_str(s: str) -> str:
    if not s: return ""
    s = str(s).lower().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def is_pick_winner(pick: str, winner: str) -> bool:
    if not pick or not winner: return False
    pick_norm = normalize_str(pick)
    winner_norm = normalize_str(winner)
    if winner_norm == "draw": return "draw" in pick_norm
    p_tokens = set(t for t in pick_norm.split() if len(t) > 2)
    w_tokens = set(winner_norm.split())
    if not p_tokens: return pick_norm == winner_norm
    return len(p_tokens & w_tokens) / len(p_tokens) >= 0.5

def match_teams_in_fallback(bet_home: str, bet_away: str, api_h: str, api_a: str) -> bool:
    h_norm = normalize_str(bet_home)
    a_norm = normalize_str(bet_away)
    score = 0
    if h_norm in api_h or api_h in h_norm: score += 2
    if a_norm in api_a or api_a in a_norm: score += 2
    score += sum(1 for t in h_norm.split() if len(t) > 2 and t in api_h)
    score += sum(1 for t in a_norm.split() if len(t) > 2 and t in api_a)
    return score >= 3

def find_score_in_fallback(scores: list, team_name: str) -> int:
    team_norm = normalize_str(team_name)
    for s in scores:
        s_norm = normalize_str(s.get("name", ""))
        if s_norm in team_norm or team_norm in s_norm:
            try: return int(float(s["score"]))
            except (ValueError, KeyError): pass
    return 0

def save_json_safe(filepath: Path, data: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix('.tmp')
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(filepath)

def sync_bet_to_tracker(bet: dict, tracker: dict):
    """همگام‌سازی نتیجه bet با tracker - با fallback به timestamp+home"""
    bet_id = bet.get("id")
    for signal in tracker["signals"]:
        matched = (
            (bet_id and signal.get("id") == bet_id) or
            (not bet_id and
             signal.get("timestamp") == bet.get("timestamp") and
             signal.get("home") == bet.get("home"))
        )
        if matched:
            signal["outcome"] = bet["outcome"]
            signal["profit_loss"] = bet["profit_loss"]
            break

# =========================================================
# SCRAPING ENGINES
# =========================================================
class ResultScraper:
    def __init__(self):
        self.soccer_results = []
        self.other_sports_results = []
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
                        h_score = match.get("home", {}).get("score", 0)
                        a_score = match.get("away", {}).get("score", 0)
                        winner = "draw"
                        if h_score > a_score: winner = home
                        elif a_score > h_score: winner = away
                        await self._add("soccer", {
                            "home": normalize_str(home),
                            "away": normalize_str(away),
                            "winner": normalize_str(winner)
                        })
        except Exception as e:
            logger.debug("FotMob error [%s]: %s", date_str, e)

    async def fetch_espn_by_date(self, target_date: datetime):
        date_str = target_date.strftime("%Y%m%d")
        endpoints = [
            f"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard?dates={date_str}",
            f"https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard?dates={date_str}",
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}",
            f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date_str}",
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard?dates={date_str}",
        ]
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                for url in endpoints:
                    try:
                        res = await session.get(url, timeout=10)
                        if res.status_code != 200: continue
                        for event in res.json().get("events", []):
                            try:
                                if event.get("status", {}).get("type", {}).get("state") != "post": continue
                                comps = event.get("competitions", [])
                                if not comps: continue
                                competitors = comps[0].get("competitors", [])
                                if not competitors: continue

                                home_team, away_team, winner = "", "", "draw"
                                h_score, a_score = 0, 0

                                for comp in competitors:
                                    name = (comp.get("team", {}).get("displayName") or
                                            comp.get("athlete", {}).get("displayName", ""))
                                    is_home = comp.get("homeAway") == "home"
                                    score_str = comp.get("score", "0")
                                    try:
                                        score_val = int(float(score_str))
                                    except: score_val = 0

                                    if is_home:
                                        home_team = name
                                        h_score = score_val
                                    else:
                                        away_team = name
                                        a_score = score_val

                                    if comp.get("winner"): winner = name

                                if not home_team or not away_team: continue

                                pool = "soccer" if "soccer" in url else "other"
                                if pool == "soccer":
                                    if winner == "draw":
                                        if h_score > a_score: winner = home_team
                                        elif a_score > h_score: winner = away_team

                                await self._add(pool, {
                                    "home": normalize_str(home_team),
                                    "away": normalize_str(away_team),
                                    "winner": normalize_str(winner)
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
            "x-requested-with": "XMLHttpRequest"
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
                            h_score = event.get("homeScore", {}).get("current", 0)
                            a_score = event.get("awayScore", {}).get("current", 0)
                            winner = "draw"
                            if h_score > a_score: winner = home
                            elif a_score > h_score: winner = away
                            await self._add("other", {
                                "home": normalize_str(home),
                                "away": normalize_str(away),
                                "winner": normalize_str(winner)
                            })
                    except Exception: pass
        except Exception as e:
            logger.debug("SofaScore error: %s", e)

    async def load_recent_results(self):
        logger.info("🌍 [SCRAPER] Fetching results (FotMob / ESPN / SofaScore)...")
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
# ODDS-API FALLBACK
# =========================================================
def fetch_odds_api_results(sport_key: str, days_from: int = 3) -> list:
    if not ODDS_KEYS: return []
    logger.info("🔄 [FALLBACK] Odds-API for %s...", sport_key)
    for key in ODDS_KEYS:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/?daysFrom={days_from}&apiKey={key}"
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200: return res.json()
        except: continue
    return []

# =========================================================
# SETTLEMENT HELPERS
# =========================================================
def calculate_profit(odds: float, outcome: str) -> float:
    if outcome == "win": return round(odds - 1.0, 2)
    if outcome == "loss": return -1.0
    return 0.0

def send_telegram_report(settled_bets: list, summary: dict):
    if not settled_bets: return
    lines = ["🧾 <b>ZBET90 DAILY SETTLEMENT REPORT</b>\n"]
    daily_profit = 0.0

    for bet in settled_bets:
        outcome = bet.get("outcome", "")
        if outcome == "void":
            icon, profit_str = "⚪️", "0.0u"
        else:
            icon = "🟢" if outcome == "win" else "🔴"
            pl = bet.get("profit_loss", 0)
            profit_str = f"+{pl}u" if pl > 0 else f"{pl}u"
            daily_profit += pl
        lines.append(f"⚔️ <b>{bet['home']} vs {bet['away']}</b>")
        lines.append(f"🎯 Pick: {bet['pick']} @ {bet['odds']}")
        lines.append(f"🏁 Result: <b>{outcome.upper()}</b> {icon} | PnL: {profit_str}\n")

    total_icon = "📈" if daily_profit > 0 else "📉"
    lines += [
        "══════════════════",
        f"{total_icon} <b>Session PnL:</b> {daily_profit:+.2f} units",
        f"🏆 <b>Overall Win Rate:</b> {summary.get('win_rate', 0)*100:.1f}%",
        f"💰 <b>Overall ROI:</b> {summary.get('roi_pct', 0):.1f}%",
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
    if current: chunks.append(current.strip())

    for chunk in chunks:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"},
                timeout=10
            )
        except Exception as e:
            logger.error("Telegram error: %s", e)

# =========================================================
# MAIN
# =========================================================
async def async_settle():
    logger.info("=" * 50)
    logger.info("🤖 ZBET90 SETTLER ENGINE v1.6 | Final Edition")
    logger.info("=" * 50)

    if not PERFORMANCE_FILE.exists():
        logger.info("❌ No performance tracker file found. Exiting.")
        return

    with open(PERFORMANCE_FILE, "r", encoding="utf-8") as f:
        tracker = json.load(f)

    now = datetime.now(timezone.utc)
    to_check = []
    for b in [s for s in tracker.get("signals", []) if s.get("outcome") is None]:
        try:
            bet_time = datetime.fromisoformat(b["timestamp"])
            if bet_time.tzinfo is None:
                bet_time = bet_time.replace(tzinfo=timezone.utc)
            if (now - bet_time) > timedelta(hours=4):
                to_check.append(b)
        except Exception as e:
            logger.error("Time parse error [%s]: %s", b.get("id", "?"), e)

    if not to_check:
        logger.info("ℹ️ No bets ready to settle.")
        return

    logger.info("🔍 %d bets to check.", len(to_check))
    scraper = ResultScraper()
    await scraper.load_recent_results()

    settled_this_session = []
    odds_api_cache = {}

    for bet in to_check:
        match_result = scraper.fuzzy_match(bet["home"], bet["away"], bet.get("sport", "soccer"))
        api_sport_key = bet.get("api_sport_key")

        # Fallback: Odds-API
        if not match_result and "market" in bet and api_sport_key:
            if api_sport_key not in odds_api_cache:
                odds_api_cache[api_sport_key] = fetch_odds_api_results(api_sport_key, days_from=3)

            for api_match in odds_api_cache.get(api_sport_key, []):
                api_h = normalize_str(api_match.get("home_team", ""))
                api_a = normalize_str(api_match.get("away_team", ""))

                if not match_teams_in_fallback(bet["home"], bet["away"], api_h, api_a):
                    continue

                if not api_match.get("completed"):
                    logger.info("⏳ Match found but not completed: %s vs %s", bet["home"], bet["away"])
                    break

                scores = api_match.get("scores")
                if scores:
                    h_score = find_score_in_fallback(scores, api_match.get("home_team", ""))
                    a_score = find_score_in_fallback(scores, api_match.get("away_team", ""))
                    winner = "draw"
                    if h_score > a_score: winner = api_h
                    elif a_score > h_score: winner = api_a
                    match_result = {"home": api_h, "away": api_a, "winner": winner}
                    logger.info("🔄 [FALLBACK SUCCESS] %s vs %s", bet["home"], bet["away"])
                break

        # Settle
        if match_result:
            outcome = "win" if is_pick_winner(bet["pick"], match_result.get("winner", "")) else "loss"
            bet["outcome"] = outcome
            bet["profit_loss"] = calculate_profit(bet["odds"], outcome)
            settled_this_session.append(bet)
            logger.info("✅ %s vs %s → %s (%.2f)", bet["home"], bet["away"], outcome.upper(), bet["profit_loss"])

        else:
            # VOID RULE: 48 ساعت
            try:
                bet_time = datetime.fromisoformat(bet["timestamp"])
                if bet_time.tzinfo is None:
                    bet_time = bet_time.replace(tzinfo=timezone.utc)
                if (now - bet_time) > timedelta(hours=48):
                    bet["outcome"] = "void"
                    bet["profit_loss"] = 0.0
                    settled_this_session.append(bet)
                    logger.warning("⚠️ VOID: %s vs %s", bet["home"], bet["away"])
            except Exception as e:
                logger.error("Void check error: %s", e)

    if not settled_this_session:
        logger.info("⏳ No matches finished yet.")
        return

    # Sync به tracker
    for bet in settled_this_session:
        sync_bet_to_tracker(bet, tracker)

    resolved = [s for s in tracker["signals"] if s.get("outcome") not in (None, "void")]
    wins = [s for s in resolved if s["outcome"] == "win"]
    total_pl = sum(s.get("profit_loss", 0) for s in resolved)

    tracker["summary"] = {
        "total_signals": len(tracker["signals"]),
        "resolved": len(resolved),
        "win_rate": round(len(wins) / len(resolved), 3) if resolved else 0,
        "total_profit_loss_units": round(total_pl, 2),
        "roi_pct": round((total_pl / len(resolved)) * 100, 2) if resolved else 0,
        "last_updated": now.isoformat(),
    }

    save_json_safe(PERFORMANCE_FILE, tracker)
    send_telegram_report(settled_this_session, tracker["summary"])

if __name__ == "__main__":
    asyncio.run(async_settle())
