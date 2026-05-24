#!/usr/bin/env python3
"""
zBET90 — Positive Expected Value (+EV) Sports Betting Predictor (v3)
================================================================
Fetches upcoming events, enriches them with historical form/stats from SofaScore (RapidAPI),
calculates +EV mathematically in Python, analyzes qualitatively with AI, 
and broadcasts via Telegram.

Author: @zBET90
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: IMPORTS & TYPE HINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import sys
import time
import json
import re
import logging
import html
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: LOGGING CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Config:
    """Centralized configuration loaded from environment variables with defaults."""

    # --- API Keys (required) ---
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "YOUR_ODDS_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
    RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "YOUR_RAPIDAPI_KEY")

    # --- Telegram Config ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
    TELEGRAM_ID: str = "@zBET90"

    # --- Odds API Settings ---
    ODDS_API_BASE: str = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    ODDS_REGIONS: str = os.getenv("ODDS_REGIONS", "eu")
    ODDS_MARKETS: str = os.getenv("ODDS_MARKETS", "h2h,totals")
    ODDS_FORMAT: str = "decimal"

    # --- RapidAPI Settings (SofaScore) ---
    RAPIDAPI_HOST: str = os.getenv("RAPIDAPI_HOST", "sofascore.p.rapidapi.com")
    
    # --- Scan Window & Limits ---
    SCAN_WINDOW_HOURS: int = int(os.getenv("SCAN_WINDOW_HOURS", "2"))

    # --- Groq AI Settings ---
    GROQ_API_BASE: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.15"))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "4096"))

    # --- Telegram Rate Limiting ---
    TELEGRAM_DELAY: float = float(os.getenv("TELEGRAM_DELAY", "4.0"))
    TELEGRAM_MAX_RETRIES: int = int(os.getenv("TELEGRAM_MAX_RETRIES", "3"))

    # --- Persistence ---
    SENT_CACHE_FILE: str = os.getenv("SENT_CACHE_FILE", "sent_events.json")
    CACHE_EXPIRY_HOURS: int = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: HTTP SESSION WITH RETRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_http_session(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

http_session: requests.Session = create_http_session()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: MATHEMATICAL +EV CALCULATOR (PYTHON NATIVE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def remove_vig_and_find_value(odds_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculates Fair Odds by removing bookmaker margin (vig).
    This strictly handles the math so the AI doesn't have to.
    """
    valid_odds = [v for k, v in odds_dict.items() if v > 1.0]
    if len(valid_odds) < 2:
        return {"margin": 0, "fair_odds": odds_dict, "is_value": False}
        
    implied_probs = [1 / o for o in valid_odds]
    margin = sum(implied_probs)
    
    if margin <= 1.0 or margin > 1.3:
        return {"margin": round(margin, 3), "fair_odds": odds_dict, "is_value": False}
        
    fair_odds = {}
    for outcome, odd in odds_dict.items():
        if odd > 1.0:
            implied = 1 / odd
            fair_prob = implied / margin
            fair_odds[outcome] = round(1 / fair_prob, 2)
            
    return {
        "margin_percentage": round((margin - 1) * 100, 2),
        "fair_odds": fair_odds,
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: DEDUPLICATION / PERSISTENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SentCache:
    def __init__(self, filepath: str = Config.SENT_CACHE_FILE) -> None:
        self.filepath = Path(filepath)
        self._cache: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if not self.filepath.exists():
            return {}
        try:
            data = json.loads(self.filepath.read_text(encoding="utf-8"))
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=Config.CACHE_EXPIRY_HOURS)
            ).isoformat()
            return {k: v for k, v in data.items() if v > cutoff}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache file corrupted, resetting: {e}")
            return {}

    def _save(self) -> None:
        self.filepath.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    @staticmethod
    def make_key(event: dict[str, Any]) -> str:
        raw = f"{event.get('event_id', '')}-{event.get('pick', '')}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_sent(self, event: dict[str, Any]) -> bool:
        return self.make_key(event) in self._cache

    def mark_sent(self, event: dict[str, Any]) -> None:
        key = self.make_key(event)
        self._cache[key] = datetime.now(timezone.utc).isoformat()
        self._save()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7: RAPIDAPI SOFASCORE INTEGRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_sofascore_team_id(team_name: str) -> Optional[int]:
    """Search for a team ID using SofaScore API via RapidAPI."""
    url = f"https://{Config.RAPIDAPI_HOST}/teams/search"
    headers = {
        "x-rapidapi-key": Config.RAPIDAPI_KEY,
        "x-rapidapi-host": Config.RAPIDAPI_HOST
    }
    querystring = {"name": team_name}
    
    try:
        response = http_session.get(url, headers=headers, params=querystring, timeout=12)
        if response.status_code == 200:
            data = response.json()
            teams = data.get("data", [])
            if teams:
                # Return the ID of the top matched team
                return teams[0].get("id")
    except Exception as e:
        logger.error(f"SofaScore team lookup error for '{team_name}': {e}")
    return None

def get_sofascore_team_context(team_name: str) -> str:
    """Fetches performance stats & recent matches from SofaScore."""
    t_id = get_sofascore_team_id(team_name)
    if not t_id:
        return f"No statistical profile found in SofaScore for {team_name}."

    headers = {
        "x-rapidapi-key": Config.RAPIDAPI_KEY,
        "x-rapidapi-host": Config.RAPIDAPI_HOST
    }
    
    context_str = f"\nStats & Form for {team_name} (SofaScore ID: {t_id}):"
    
    # Fetch recent match statistics
    try:
        matches_url = f"https://{Config.RAPIDAPI_HOST}/teams/get-last-matches"
        querystring = {"teamId": str(t_id)}
        
        m_res = http_session.get(matches_url, headers=headers, params=querystring, timeout=10)
        if m_res.status_code == 200:
            m_data = m_res.json()
            events = m_data.get("data", {}).get("events", [])
            context_str += "\n- Recent Match Log:"
            
            for m in events[:5]:
                home = m.get("homeTeam", {}).get("name", "Home")
                away = m.get("awayTeam", {}).get("name", "Away")
                home_score = m.get("homeScore", {}).get("current", 0)
                away_score = m.get("awayScore", {}).get("current", 0)
                
                # Format timestamp safely
                ts = m.get("startTimestamp")
                date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else "N/A"
                
                context_str += f"\n  * [{date_str}] {home} {home_score} - {away_score} {away}"
    except Exception as e:
        logger.debug(f"Could not retrieve SofaScore match logs for team {t_id}: {e}")

    return context_str

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8: DATA COLLECTION (The Odds API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_next_events() -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=Config.SCAN_WINDOW_HOURS)

    params: dict[str, str] = {
        "apiKey": Config.ODDS_API_KEY,
        "regions": Config.ODDS_REGIONS,
        "markets": Config.ODDS_MARKETS,
        "oddsFormat": Config.ODDS_FORMAT,
    }

    try:
        response = http_session.get(Config.ODDS_API_BASE, params=params, timeout=30)
        if response.status_code != 200:
            logger.error(f"Odds API failure {response.status_code}: {response.text[:200]}")
            return []

        all_events: list[dict[str, Any]] = response.json()
        filtered: list[dict[str, Any]] = []
        
        for event in all_events:
            commence = event.get("commence_time", "")
            try:
                event_time = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                if now <= event_time <= window_end:
                    filtered.append(event)
            except (ValueError, TypeError):
                continue
        return filtered
    except requests.RequestException as e:
        logger.error(f"Network error querying Odds API: {e}")
        return []

def extract_best_odds(bookmakers: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    best: Dict[str, Dict[str, float]] = {"h2h": {}, "totals": {}}
    for b in bookmakers:
        for market in b.get("markets", []):
            m_key: str = market.get("key", "")
            if m_key not in ("h2h", "totals"):
                continue
            for outcome in market.get("outcomes", []):
                name: str = outcome.get("name", "Unknown")
                price: float = outcome.get("price", 0.0)
                if m_key == "totals" and "point" in outcome:
                    name = f"{name} {outcome['point']}"
                if price > best[m_key].get(name, 0.0):
                    best[m_key][name] = price
    return best

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9: AI ANALYSIS VIA GROQ (Qualitative Only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT: str = """You are an Elite Qualitative Sports Analyst.
The mathematical Expected Value (+EV) and fair odds have ALREADY been calculated by Python. 
Your ONLY job is to review the mathematical data and the SofaScore historical form, and generate a brief, professional logic summary explaining *why* a team has the edge qualitatively.

STRICT RULES:
1. Output ONLY a valid JSON array. Do NOT wrap inside markdown formatting blocks (`json ...`).
2. Do NOT perform your own math. Rely on the Python "Calculated Fair Odds" provided in the prompt.
3. Keep the "logic" field strictly brief (Max 2 sentences), focusing on recent form, historical performance, and why the bookmaker's odds offer value over the true fair odds.
4. Each object in the array MUST contain these exact keys:
   - "event_id": string
   - "sport": string
   - "sport_emoji": string
   - "home_team": string
   - "away_team": string
   - "home_flag": string
   - "away_flag": string
   - "commence_time": string
   - "pick": string
   - "pick_odds": number
   - "goals_pick": string
   - "goals_odds": number or "N/A"
   - "risk_level": "Low" | "Medium" | "High"
   - "logic": string (Qualitative reasoning based on form & python math)
"""

def build_analysis_prompt(events: List[Dict[str, Any]]) -> str:
    lines: list[str] = ["Analyze these events qualitatively based on the Python calculated math and Sofascore form:\n"]

    for i, event in enumerate(events, 1):
        best = extract_best_odds(event.get("bookmakers", []))
        
        # Calculate Math in Python!
        h2h_math = remove_vig_and_find_value(best["h2h"])
        h2h_str = ", ".join(f"{k}: {v:.2f}" for k, v in best["h2h"].items()) or "N/A"
        fair_h2h_str = ", ".join(f"{k}: {v:.2f}" for k, v in h2h_math.get("fair_odds", {}).items()) or "N/A"
        
        home = event.get("home_team", "Unknown")
        away = event.get("away_team", "Unknown")
        
        logger.info(f"Enriching event profile {i}/{len(events)} with SofaScore: {home} vs {away}")
        home_context = get_sofascore_team_context(home)
        away_context = get_sofascore_team_context(away)

        lines.append(f"--- Event {i} ---")
        lines.append(f"ID: {event.get('id', 'unknown')}")
        lines.append(f"Sport: {event.get('sport_title', 'Unknown')}")
        lines.append(f"Kickoff: {event.get('commence_time', 'Unknown')}")
        lines.append(f"Home Team: {home}")
        lines.append(f"Away Team: {away}")
        lines.append(f"Bookmaker H2H Odds: {h2h_str}")
        lines.append(f"Python Calculated True/Fair Odds (No Vig): {fair_h2h_str}")
        lines.append(f"SofaScore Form Context:\n{home_context}\n{away_context}")
        lines.append("")

    return "\n".join(lines)

def parse_ai_response(raw_text: str) -> List[Dict[str, Any]]:
    cleaned = raw_text.strip()
    t_ticks = "`" * 3
    if cleaned.startswith(t_ticks):
        cleaned = re.sub(r"^" + t_ticks + r"(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*" + t_ticks + r"$", "", cleaned)

    predictions: Optional[list[Any]] = None
    try:
        predictions = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                predictions = json.loads(match.group())
            except json.JSONDecodeError:
                return []

    if not isinstance(predictions, list):
        return []

    validated: list[dict[str, Any]] = []
    valid_risk_levels = {"Low", "Medium", "High"}

    for pred in predictions:
        if not isinstance(pred, dict) or not {"pick", "home_team", "away_team"}.issubset(pred.keys()):
            continue

        try:
            pred["pick_odds"] = float(pred.get("pick_odds", 0))
        except (ValueError, TypeError):
            pred["pick_odds"] = 0.0

        risk = str(pred.get("risk_level", "Medium")).strip().capitalize()
        pred["risk_level"] = risk if risk in valid_risk_levels else "Medium"
        pred.setdefault("sport_emoji", "⚽")
        pred.setdefault("home_flag", "🏠")
        pred.setdefault("away_flag", "✈️")
        pred.setdefault("goals_pick", "N/A")
        
        validated.append(pred)

    return validated

def analyze_with_groq(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []

    user_prompt = build_analysis_prompt(events)
    payload: dict[str, Any] = {
        "model": Config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": Config.GROQ_TEMPERATURE,
        "max_tokens": Config.GROQ_MAX_TOKENS,
    }

    headers = {
        "Authorization": f"Bearer {Config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = http_session.post(Config.GROQ_API_BASE, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            logger.error(f"Groq execution failure {response.status_code}")
            return []

        data = response.json()
        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return parse_ai_response(raw_content)
    except Exception as e:
        logger.error(f"Error querying AI model: {e}")
        return []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10: TELEGRAM FORMATTING & BROADCAST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK_EMOJI: dict[str, str] = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

def format_countdown(commence_time: str) -> str:
    try:
        event_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        diff = event_dt - datetime.now(timezone.utc)
        if diff.total_seconds() <= 0:
            return "🔴 LIVE"
        hours, remainder = divmod(int(diff.total_seconds()), 3600)
        minutes = remainder // 60
        return f"⏱ Starts in {hours}h {minutes}m" if hours > 0 else f"⏱ Starts in {minutes}m"
    except Exception:
        return "⏱ Kickoff Pending"

def format_prediction_message(pred: dict[str, Any]) -> str:
    sport_emoji = pred.get("sport_emoji", "⚽")
    sport = html.escape(pred.get("sport", "Sports"))
    home = html.escape(pred.get("home_team", "Home"))
    away = html.escape(pred.get("away_team", "Away"))
    home_flag = pred.get("home_flag", "🏠")
    away_flag = pred.get("away_flag", "✈️")
    countdown = format_countdown(pred.get("commence_time", ""))
    
    pick = html.escape(pred.get("pick", "N/A"))
    pick_odds = pred.get("pick_odds", 0)
    goals_pick = html.escape(str(pred.get("goals_pick", "N/A")))
    goals_odds = pred.get("goals_odds", "N/A")
    
    risk = pred.get("risk_level", "Medium")
    risk_emoji = RISK_EMOJI.get(risk, "🟠")
    
    logic_text = html.escape(pred.get("logic", ""))
    logic_formatted = re.sub(r"(\d+\.?\d*%?)", r"<code>\1</code>", logic_text)
    pick_odds_str = f"<code>{pick_odds:.2f}</code>" if pick_odds > 0 else "N/A"
    
    goals_line = ""
    if goals_pick != "N/A":
        goals_odds_str = f"<code>{float(goals_odds):.2f}</code>" if isinstance(goals_odds, (int, float, str)) and str(goals_odds).replace('.','',1).isdigit() else "N/A"
        goals_line = f"\n⚽ <b>Goals Pick:</b> {goals_pick} @ {goals_odds_str}"

    message = (
        f"{sport_emoji} <b>Sport:</b> {sport}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{home_flag} <b>{home}</b>  🆚  <b>{away}</b> {away_flag}\n"
        f"<i>{countdown}</i>\n\n"
        f"🎯 <b>Winner Pick:</b> <b>{pick}</b> @ {pick_odds_str}"
        f"{goals_line}\n\n"
        f"{risk_emoji} <b>Risk Level:</b> {risk}\n\n"
        f"💡 <b>Logic:</b> <i>{logic_formatted}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 {Config.TELEGRAM_ID}"
    )
    return message

def send_telegram_message(text: str) -> bool:
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        return False
        
    url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": Config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(1, Config.TELEGRAM_MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                return True
            if response.status_code == 429:
                retry_after = response.json().get("parameters", {}).get("retry_after", 5)
                time.sleep(retry_after)
                continue
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return False

def format_and_send_to_telegram(predictions: List[Dict[str, Any]], cache: SentCache) -> int:
    sent_count = 0
    predictions.sort(key=lambda p: datetime.fromisoformat(p.get('commence_time', '').replace("Z", "+00:00")) if p.get('commence_time') else datetime.now(timezone.utc))

    for i, pred in enumerate(predictions):
        if cache.is_sent(pred):
            continue

        message = format_prediction_message(pred)
        if send_telegram_message(message):
            cache.mark_sent(pred)
            sent_count += 1
            logger.info(f"Successfully broadcasted event: {pred.get('home_team')} vs {pred.get('away_team')}")
        
        if i < len(predictions) - 1:
            time.sleep(Config.TELEGRAM_DELAY)

    return sent_count

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 11: EXECUTION CORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    logger.info("🚀 Launching zBET90 +EV Predictor Engine v3...")
    
    events = get_next_events()
    if not events:
        logger.info("Scan complete. No upcoming events found.")
        return

    predictions = analyze_with_groq(events)
    if not predictions:
        logger.info("AI Analysis finished. No statistical edges broadcasted.")
        return

    cache = SentCache()
    sent = format_and_send_to_telegram(predictions, cache)
    logger.info(f"Run ended. Broadcasted {sent} new high-value opportunities.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal crash: {e}", exc_info=True)
        sys.exit(1)
