import os
import sys
import json
import logging
import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from curl_cffi.requests import AsyncSession

# =========================================================
# CONFIGURATION
# =========================================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
PERFORMANCE_FILE = Path("api_cache/performance_tracker.json")
LOG_FILE = Path("api_cache/settler_logs.log")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# گرفتن کلیدهای Odds-API برای Fallback
ODDS_KEYS = [
    os.getenv("ODDS_API_KEY", "").strip(),
    os.getenv("ODDS_API_KEY2", "").strip(),
    os.getenv("ODDS_API_KEY3", "").strip()
]
ODDS_KEYS = [k for k in ODDS_KEYS if k]

logger = logging.getLogger("SETTLER")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =========================================================
# 1. SCRAPING ENGINES (PRIMARY)
# =========================================================
class ResultScraper:
    def __init__(self):
        self.soccer_results = []
        self.other_sports_results = []

    async def fetch_fotmob_soccer(self, target_date: datetime):
        """استخراج نتایج فوتبال از API مخفی FotMob"""
        date_str = target_date.strftime("%Y%m%d")
        url = f"https://www.fotmob.com/api/matches?date={date_str}"
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                res = await session.get(url, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    for league in data.get("leagues", []):
                        for match in league.get("matches", []):
                            if match.get("status", {}).get("finished", False):
                                home = match.get("home", {}).get("name", "")
                                away = match.get("away", {}).get("name", "")
                                home_score = match.get("home", {}).get("score", 0)
                                away_score = match.get("away", {}).get("score", 0)
                                
                                # تشخیص برنده
                                winner = "Draw"
                                if home_score > away_score: winner = home
                                elif away_score > home_score: winner = away
                                
                                self.soccer_results.append({
                                    "home": home.lower(), "away": away.lower(),
                                    "home_score": home_score, "away_score": away_score,
                                    "winner": winner.lower()
                                })
        except Exception as e:
            logger.warning("FotMob scrape error: %s", e)

    async def fetch_espn_other_sports(self):
        """استخراج نتایج تنیس و بسکتبال/بیسبال از ESPN"""
        endpoints = [
            "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard",
            "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard",
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
        ]
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                for url in endpoints:
                    res = await session.get(url, timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        for event in data.get("events", []):
                            status = event.get("status", {}).get("type", {}).get("state", "")
                            if status == "post":  # تمام شده
                                comps = event.get("competitions", [])[0].get("competitors", [])
                                home_team, away_team, winner = "", "", ""
                                
                                for comp in comps:
                                    name = comp.get("team", {}).get("displayName", "") or comp.get("athlete", {}).get("displayName", "")
                                    is_home = comp.get("homeAway") == "home"
                                    is_winner = comp.get("winner", False)
                                    
                                    if is_home: home_team = name
                                    else: away_team = name
                                    if is_winner: winner = name
                                    
                                self.other_sports_results.append({
                                    "home": home_team.lower(), "away": away_team.lower(),
                                    "winner": winner.lower()
                                })
        except Exception as e:
            logger.warning("ESPN scrape error: %s", e)

    async def load_recent_results(self):
        logger.info("🌍 [SCRAPER] Fetching results from Web (FotMob/ESPN)...")
        now = datetime.now(timezone.utc)
        tasks = [
            self.fetch_fotmob_soccer(now),
            self.fetch_fotmob_soccer(now - timedelta(days=1)),
            self.fetch_fotmob_soccer(now - timedelta(days=2)),
            self.fetch_espn_other_sports()
        ]
        await asyncio.gather(*tasks)
        logger.info("✅ [SCRAPER] Fetched %d Soccer + %d Other matches", len(self.soccer_results), len(self.other_sports_results))

    def fuzzy_match(self, home: str, away: str, sport: str) -> dict:
        target_pool = self.soccer_results if "football" in sport.lower() or "soccer" in sport.lower() else self.other_sports_results
        h_clean, a_clean = home.lower().split()[-1], away.lower().split()[-1]
        
        for match in target_pool:
            mh, ma = match["home"], match["away"]
            if (h_clean in mh and a_clean in ma) or (h_clean in ma and a_clean in mh):
                return match
        return {}

# =========================================================
# 2. ODDS-API FALLBACK
# =========================================================
def fetch_odds_api_results(sport_key: str, days_from: int = 3) -> list:
    if not ODDS_KEYS: return []
    logger.info("🔄 [FALLBACK] Fetching results from The-Odds-API for %s...", sport_key)
    for key in ODDS_KEYS:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/?daysFrom={days_from}&apiKey={key}"
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200:
                return res.json()
        except: continue
    return []

# =========================================================
# 3. SETTLEMENT LOGIC
# =========================================================
def calculate_profit(odds: float, outcome: str) -> float:
    # 1 Unit flat betting assumption
    if outcome == "win":
        return round(odds - 1.0, 2)
    elif outcome == "loss":
        return -1.0
    return 0.0

def send_telegram_report(settled_bets: list, summary: dict):
    if not settled_bets: return
    
    lines = ["🧾 <b>ZBET90 DAILY SETTLEMENT REPORT</b>\n"]
    daily_profit = 0.0
    
    for bet in settled_bets:
        icon = "🟢" if bet["outcome"] == "win" else "🔴"
        profit = f"+{bet['profit_loss']}u" if bet['profit_loss'] > 0 else f"{bet['profit_loss']}u"
        daily_profit += bet['profit_loss']
        
        lines.append(f"⚔️ <b>{bet['home']} vs {bet['away']}</b>")
        lines.append(f"🎯 Pick: {bet['pick']} @ {bet['odds']}")
        lines.append(f"🏁 Result: <b>{bet['outcome'].upper()}</b> {icon} | PnL: {profit}\n")
    
    total_icon = "📈" if daily_profit > 0 else "📉"
    lines.append(f"══════════════════")
    lines.append(f"{total_icon} <b>Session PnL:</b> {daily_profit:+.2f} units")
    lines.append(f"🏆 <b>Overall Win Rate:</b> {summary.get('win_rate', 0)*100:.1f}%")
    lines.append(f"💰 <b>Overall ROI:</b> {summary.get('roi_pct', 0):.1f}%")
    
    msg = "\n".join(lines)
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

# =========================================================
# MAIN EXECUTION
# =========================================================
async def async_settle():
    logger.info("=" * 50)
    logger.info("🤖 ZBET90 SETTLER ENGINE v1.0")
    logger.info("=" * 50)

    if not PERFORMANCE_FILE.exists():
        logger.info("❌ No performance tracker file found. Exiting.")
        return

    with open(PERFORMANCE_FILE, "r", encoding="utf-8") as f:
        tracker = json.load(f)

    pending_bets = [s for s in tracker.get("signals", []) if s.get("outcome") is None]
    
    # Filter bets older than 4 hours
    now = datetime.now(timezone.utc)
    to_check = []
    for b in pending_bets:
        try:
            bet_time = datetime.fromisoformat(b["timestamp"])
            if (now - bet_time) > timedelta(hours=4):
                to_check.append(b)
        except: pass

    if not to_check:
        logger.info("ℹ️ No finished pending bets found to settle.")
        return

    logger.info("🔍 Found %d pending bets to check.", len(to_check))
    
    scraper = ResultScraper()
    await scraper.load_recent_results()

    settled_this_session = []

    for bet in to_check:
        match_result = scraper.fuzzy_match(bet["home"], bet["away"], bet.get("sport", "soccer"))
        
        # اگر در وب پیدا نشد، سراغ Odds-API می‌رود
        if not match_result and "market" in bet:
            # (در نسخه پایه، ما برای سادگی فقط از اسکرپر استفاده میکنیم، اما تابع fallback آماده است)
            pass

        if match_result:
            pick_clean = bet["pick"].lower()
            winner_clean = match_result.get("winner", "")
            
            # تطبیق انتخاب با برنده (برای مارکت Match Winner)
            if winner_clean == "draw" and "draw" in pick_clean:
                outcome = "win"
            elif winner_clean != "draw" and winner_clean in pick_clean or pick_clean in winner_clean:
                outcome = "win"
            else:
                outcome = "loss"

            bet["outcome"] = outcome
            bet["profit_loss"] = calculate_profit(bet["odds"], outcome)
            settled_this_session.append(bet)
            logger.info("✅ Settled: %s vs %s -> %s (Profit: %.2f)", bet['home'], bet['away'], outcome.upper(), bet['profit_loss'])

    if settled_this_session:
        # آپدیت فایل JSON
        resolved = [s for s in tracker["signals"] if s.get("outcome") is not None]
        wins = [s for s in resolved if s["outcome"] == "win"]
        total_pl = sum(s.get("profit_loss", 0) for s in resolved)
        
        tracker["summary"] = {
            "total_signals": len(tracker["signals"]),
            "resolved": len(resolved),
            "win_rate": round(len(wins) / len(resolved), 3) if resolved else 0,
            "total_profit_loss_units": round(total_pl, 2),
            "roi_pct": round((total_pl / len(resolved)) * 100, 2) if resolved else 0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(PERFORMANCE_FILE, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)
            
        send_telegram_report(settled_this_session, tracker["summary"])
    else:
        logger.info("⏳ Matches might not be finished yet. Will check next time.")

if __name__ == "__main__":
    asyncio.run(async_settle())
