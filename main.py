#!/usr/bin/env python3
"""
zBET90 — Positive Expected Value (+EV) Sports Betting Predictor (v2)
================================================================
Fetches upcoming events, enriches them with historical form/stats from RapidAPI,
analyzes them with AI for genuine +EV opportunities, and broadcasts via Telegram.

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
from typing import Any, Optional
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
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY")

    # --- Telegram Config ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID")
    TELEGRAM_ID: str = "@zBET90"

    # --- Odds API Settings ---
    ODDS_API_BASE: str = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    ODDS_REGIONS: str = os.getenv("ODDS_REGIONS", "eu")
    ODDS_MARKETS: str = os.getenv("ODDS_MARKETS", "h2h,totals")
    ODDS_FORMAT: str = "decimal"

    # --- RapidAPI Settings (api-football186) ---
    RAPIDAPI_HOST: str = "api-football186.p.rapidapi.com"
    RAPIDAPI_BASE_URL: str = f"https://{RAPIDAPI_HOST}"

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

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        missing: list[str] = []
        if not cls.ODDS_API_KEY:
            missing.append("ODDS_API_KEY")
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not cls.RAPIDAPI_KEY:
            missing.append("RAPIDAPI_KEY")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

# Validate configuration on runtime
Config.validate()

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

# Global HTTP session
http_session: requests.Session = create_http_session()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: DEDUPLICATION / PERSISTENCE
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
        raw = f"{event.get('event_id', '')}-{event.get('pick', '')}-{event.get('goals_pick', '')}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_sent(self, event: dict[str, Any]) -> bool:
        return self.make_key(event) in self._cache

    def mark_sent(self, event: dict[str, Any]) -> None:
        key = self.make_key(event)
        self._cache[key] = datetime.now(timezone.utc).isoformat()
        self._save()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: RAPIDAPI DATA ENRICHMENT (api-football186)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_team_id(team_name: str) -> Optional[int]:
    """Looks up team ID by matching names from the /teams endpoint."""
    url = f"{Config.RAPIDAPI_BASE_URL}/teams"
    headers = {
        "x-rapidapi-key": Config.RAPIDAPI_KEY,
        "x-rapidapi-host": Config.RAPIDAPI_HOST,
        "Content-Type": "application/json"
    }
    
    try:
        response = http_session.get(url, headers=headers, timeout=12)
        if response.status_code == 200:
            teams_data = response.json()
            # If the response is a direct list or wrapped in a dict
            teams_list = teams_data if isinstance(teams_data, list) else teams_data.get("response", [])
            
            # Simple fuzzy/exact matching logic
            normalized_search = team_name.lower().strip()
            for team in teams_list:
                name = team.get("name", "").lower()
                t_id = team.get("id")
                if normalized_search in name or name in normalized_search:
                    return t_id
    except Exception as e:
        logger.error(f"Error resolving team ID for '{team_name}': {e}")
    return None

def get_team_context(team_name: str) -> str:
    """Fetches performance stats and recent match records for a team."""
    t_id = get_team_id(team_name)
    if not t_id:
        return f"No statistical profile found for {team_name}."

    headers = {
        "x-rapidapi-key": Config.RAPIDAPI_KEY,
        "x-rapidapi-host": Config.RAPIDAPI_HOST,
        "Content-Type": "application/json"
    }
    
    context_str = f"\nStats & Form for {team_name}:"
    
    # 1. Fetch Stats
    try:
        stats_url = f"{Config.RAPIDAPI_BASE_URL}/team/{t_id}/stats"
        s_res = http_session.get(stats_url, headers=headers, timeout=10)
        if s_res.status_code == 200:
            s_data = s_res.json()
            # Extract key performance parameters if available
            context_str += f"\n- General Stats: {json.dumps(s_data)[:180]}..."
    except Exception as e:
        logger.debug(f"Could not retrieve stats for team {t_id}: {e}")

    # 2. Fetch Recent Matches
    try:
        matches_url = f"{Config.RAPIDAPI_BASE_URL}/team/{t_id}/matches"
        m_res = http_session.get(matches_url, headers=headers, timeout=10)
        if m_res.status_code == 200:
            m_data = m_res.json()
            matches_list = m_data if isinstance(m_data, list) else m_data.get("response", [])
            context_str += "\n- Recent Match Log:"
            for m in matches_list[:4]:
                home = m.get("home_team", {}).get("name", "Home")
                away = m.get("away_team", {}).get("name", "Away")
                score = m.get("score", "N/A")
                date = m.get("date", "")[:10]
                context_str += f"\n  * [{date}] {home} vs {away} -> Score: {score}"
    except Exception as e:
        logger.debug(f"Could not retrieve match logs for team {t_id}: {e}")

    return context_str

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7: DATA COLLECTION (The Odds API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_next_events() -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=Config.SCAN_WINDOW_HOURS)

    params: dict[str, str] = {
        "apiKey": Config.ODDS_API_KEY,
        "regions": Config.ODDS_REGIONS,
        "markets": Config.ODDS_MARKETS,
        "oddsFormat": Config.ODDS_FORMAT,
    }

    logger.info(f"Scanning upcoming events starting within next {Config.SCAN_WINDOW_HOURS}h...")

    try:
        response = http_session.get(Config.ODDS_API_BASE, params=params, timeout=30)
        
        remaining = response.headers.get("x-requests-remaining", "unknown")
        used = response.headers.get("x-requests-used", "unknown")
        logger.info(f"Odds API Quota Usage — Used: {used}, Remaining: {remaining}")

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

        logger.info(f"Filtered down to {len(filtered)} events out of {len(all_events)} items.")
        return filtered

    except requests.RequestException as e:
        logger.error(f"Network error querying Odds API: {e}")
        return []

def extract_best_odds(bookmakers: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    best: dict[str, dict[str, float]] = {"h2h": {}, "totals": {}}
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
# SECTION 8: AI ANALYSIS VIA GROQ (Llama 3.3 70B)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT: str = """You are an Elite Quantitative Sports Analyst and Value Bettor.
Your objective is to find high-probability Positive Expected Value (+EV) opportunities by comparing bookmaker odds against factual team performance trends and match logs.

STRICT RULES:
1. Output ONLY a valid JSON array. Do NOT wrap inside markdown formatting blocks (`json ...`). No conversational filler.
2. Only include predictions where you identify clear mathematical value based on the historical logs provided.
3. Keep the "logic" field strictly brief, concise, and focused purely on numbers and historical data metrics. Avoid fluff.
4. Each object in the array MUST contain these exact keys:
   - "event_id": string
   - "sport": string
   - "sport_emoji": string (high-quality emoji)
   - "home_team": string
   - "away_team": string
   - "home_flag": string (country/team flag emoji)
   - "away_flag": string (country/team flag emoji)
   - "commence_time": string (ISO format)
   - "pick": string (Target winner or "Draw")
   - "pick_odds": number
   - "goals_pick": string (e.g., "Over 2.5", "Under 1.5", or "N/A")
   - "goals_odds": number or "N/A"
   - "risk_level": "Low" | "Medium" | "High"
   - "logic": string (Max 2 sentences containing short, precise statistical proof)

5. If no real value edge is spotted, return an empty array: []
"""

def build_analysis_prompt(events: list[dict[str, Any]]) -> str:
    lines: list[str] = ["Analyze these upcoming events with historical context for +EV indicators:\n"]

    for i, event in enumerate(events, 1):
        best = extract_best_odds(event.get("bookmakers", []))
        h2h_str = ", ".join(f"{k}: {v:.2f}" for k, v in best["h2h"].items()) or "N/A"
        totals_str = ", ".join(f"{k}: {v:.2f}" for k, v in best["totals"].items()) or "N/A"
        
        home = event.get("home_team", "Unknown")
        away = event.get("away_team", "Unknown")
        
        # Inject real-time structural stats enrichment
        logger.info(f"Enriching event profile {i}/{len(events)}: {home} vs {away}")
        home_context = get_team_context(home)
        away_context = get_team_context(away)

        lines.append(f"--- Event {i} ---")
        lines.append(f"ID: {event.get('id', 'unknown')}")
        lines.append(f"Sport: {event.get('sport_title', 'Unknown')}")
        lines.append(f"Kickoff: {event.get('commence_time', 'Unknown')}")
        lines.append(f"Home Team: {home}")
        lines.append(f"Away Team: {away}")
        lines.append(f"Best Available H2H Odds: {h2h_str}")
        lines.append(f"Best Available Totals Odds: {totals_str}")
        lines.append(f"Historical Form Context:\n{home_context}\n{away_context}")
        lines.append("")

    return "\n".join(lines)

def parse_ai_response(raw_text: str) -> list[dict[str, Any]]:
    cleaned = raw_text.strip()
    t_ticks = "`" * 3
    if cleaned.startswith(t_ticks):
        cleaned = re.sub(r"^" + t_ticks + r"(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*" + t_ticks + r"$", "", cleaned)

    predictions: Optional[list[Any]] = None
    try:
        predictions = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if predictions is None:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                predictions = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("JSON conversion failed on AI output.")
                return []

    if not isinstance(predictions, list):
        return []

    valid_risk_levels = {"Low", "Medium", "High"}
    validated: list[dict[str, Any]] = []

    for pred in predictions:
        if not isinstance(pred, dict) or not {"pick", "home_team", "away_team", "commence_time"}.issubset(pred.keys()):
            continue

        try:
            pred["pick_odds"] = float(pred.get("pick_odds", 0))
        except (ValueError, TypeError):
            pred["pick_odds"] = 0.0

        try:
            g_odds = pred.get("goals_odds", "N/A")
            pred["goals_odds"] = float(g_odds) if g_odds != "N/A" else "N/A"
        except (ValueError, TypeError):
            pred["goals_odds"] = "N/A"

        risk = pred.get("risk_level", "Medium")
        if isinstance(risk, str):
            risk = risk.strip().capitalize()
        pred["risk_level"] = risk if risk in valid_risk_levels else "Medium"

        pred.setdefault("sport_emoji", "⚽")
        pred.setdefault("home_flag", "🏠")
        pred.setdefault("away_flag", "✈️")
        pred.setdefault("goals_pick", "N/A")
        pred.setdefault("logic", "Data verified.")
        pred.setdefault("event_id", "unknown")
        pred.setdefault("sport", "Football")

        validated.append(pred)

    return validated

def analyze_with_groq(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        if not raw_content:
            return []

        return parse_ai_response(raw_content)
    except Exception as e:
        logger.error(f"Error querying AI model: {e}")
        return []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9: TELEGRAM FORMATTING & BROADCAST
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
    """Generates a highly readable, elegant, and professional layout with extra spacing."""
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
    
    # Highlight numerical metrics with code fonts for high contrast visual alignment
    logic_text = html.escape(pred.get("logic", ""))
    logic_formatted = re.sub(r"(\d+\.?\d*%?)", r"<code>\1</code>", logic_text)

    pick_odds_str = f"<code>{pick_odds:.2f}</code>" if pick_odds > 0 else "N/A"
    
    goals_line = ""
    if goals_pick != "N/A":
        goals_odds_str = f"<code>{goals_odds:.2f}</code>" if isinstance(goals_odds, (int, float)) else "N/A"
        goals_line = f"\n⚽ <b>Goals Pick:</b> {goals_pick} @ {goals_odds_str}"

    # Clean, modern layout with deliberate vertical padding ("تو دله هم نباشه")
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

def format_and_send_to_telegram(predictions: list[dict[str, Any]], cache: SentCache) -> int:
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.warning("Dry run active: Telegram credentials missing.")
        return 0

    sent_count = 0
    # Chronological sort (soonest games first)
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
# SECTION 10: EXECUTION CORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    logger.info("🚀 Launching zBET90 +EV Predictor Engine Engine v2...")
    
    events = get_next_events()
    if not events:
        logger.info("Scan complete. No upcoming events found.")
        return

    predictions = analyze_with_groq(events)
    if not predictions:
        logger.info("AI Analysis finished. No statistical edges found.")
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
