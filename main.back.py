#!/usr/bin/env python3
"""
zBET90 — Positive Expected Value (+EV) Sports Betting Predictor
================================================================
Fetches upcoming events, analyzes them with AI for +EV opportunities,
and broadcasts predictions via Telegram.

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
# SECTION 3: CONFIGURATION (externalized via environment variables)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Config:
    """Centralized configuration loaded from environment variables with defaults."""

    # --- API Keys (required) ---
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    # --- Telegram (optional for local testing) ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID")
    TELEGRAM_ID: str = "@zBET90"

    # --- Odds API settings ---
    ODDS_API_BASE: str = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    ODDS_REGIONS: str = os.getenv("ODDS_REGIONS", "eu")
    ODDS_MARKETS: str = os.getenv("ODDS_MARKETS", "h2h,totals")
    ODDS_FORMAT: str = "decimal"

    # --- Scan window ---
    SCAN_WINDOW_HOURS: int = int(os.getenv("SCAN_WINDOW_HOURS", "2"))

    # --- Groq AI settings ---
    GROQ_API_BASE: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.15"))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "4096"))

    # --- Telegram rate limiting ---
    TELEGRAM_DELAY: float = float(os.getenv("TELEGRAM_DELAY", "3.5"))
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
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


# Validate on import
Config.validate()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: HTTP SESSION WITH RETRY (exponential backoff)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_http_session(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """
    Create a requests session with automatic retry and exponential backoff.

    Args:
        retries: Maximum number of retries.
        backoff_factor: Multiplier for backoff between retries.
        status_forcelist: HTTP status codes that trigger a retry.

    Returns:
        Configured requests.Session.
    """
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


# Global session reused across all HTTP calls
http_session: requests.Session = create_http_session()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: DEDUPLICATION / PERSISTENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SentCache:
    """
    Simple file-based cache to track already-sent predictions.
    Prevents duplicate messages when the workflow runs multiple times.
    """

    def __init__(self, filepath: str = Config.SENT_CACHE_FILE) -> None:
        self.filepath = Path(filepath)
        self._cache: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        """Load cache from disk. Returns empty dict if file doesn't exist."""
        if not self.filepath.exists():
            return {}
        try:
            data = json.loads(self.filepath.read_text(encoding="utf-8"))
            # Prune expired entries
            cutoff = (
                datetime.now(timezone.utc)
                - timedelta(hours=Config.CACHE_EXPIRY_HOURS)
            ).isoformat()
            return {k: v for k, v in data.items() if v > cutoff}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache file corrupted, resetting: {e}")
            return {}

    def _save(self) -> None:
        """Persist cache to disk."""
        self.filepath.write_text(
            json.dumps(self._cache, indent=2), encoding="utf-8"
        )

    @staticmethod
    def make_key(event: dict[str, Any]) -> str:
        """
        Generate a unique key for a prediction based on event content.
        Uses event_id + pick combination to detect duplicates.
        """
        raw = f"{event.get('event_id', '')}-{event.get('pick', '')}-{event.get('goals_pick', '')}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_sent(self, event: dict[str, Any]) -> bool:
        """Check if this prediction was already sent."""
        return self.make_key(event) in self._cache

    def mark_sent(self, event: dict[str, Any]) -> None:
        """Mark a prediction as sent."""
        key = self.make_key(event)
        self._cache[key] = datetime.now(timezone.utc).isoformat()
        self._save()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: DATA COLLECTION (The Odds API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_next_events() -> list[dict[str, Any]]:
    """
    Fetch upcoming sports events starting within the configured scan window.

    Returns:
        List of event dicts from The Odds API, filtered by time window.
    """
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=Config.SCAN_WINDOW_HOURS)

    params: dict[str, str] = {
        "apiKey": Config.ODDS_API_KEY,
        "regions": Config.ODDS_REGIONS,
        "markets": Config.ODDS_MARKETS,
        "oddsFormat": Config.ODDS_FORMAT,
    }

    logger.info(
        f"Fetching events starting within next {Config.SCAN_WINDOW_HOURS}h..."
    )

    try:
        response = http_session.get(
            Config.ODDS_API_BASE, params=params, timeout=30
        )

        # Log remaining quota
        remaining = response.headers.get("x-requests-remaining", "unknown")
        used = response.headers.get("x-requests-used", "unknown")
        logger.info(f"Odds API quota — used: {used}, remaining: {remaining}")

        if response.status_code != 200:
            logger.error(
                f"Odds API error {response.status_code}: {response.text[:200]}"
            )
            return []

        all_events: list[dict[str, Any]] = response.json()

        # Filter by time window
        filtered: list[dict[str, Any]] = []
        for event in all_events:
            commence = event.get("commence_time", "")
            try:
                event_time = datetime.fromisoformat(
                    commence.replace("Z", "+00:00")
                )
                if now <= event_time <= window_end:
                    filtered.append(event)
            except (ValueError, TypeError):
                continue

        logger.info(
            f"Found {len(filtered)} events within {Config.SCAN_WINDOW_HOURS}h window "
            f"(out of {len(all_events)} total)."
        )
        return filtered

    except requests.RequestException as e:
        logger.error(f"Network error fetching odds: {e}")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7: HELPER — EXTRACT BEST ODDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def extract_best_odds(
    bookmakers: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Extract the best available odds across all bookmakers for H2H and Totals.

    Args:
        bookmakers: List of bookmaker dicts from the Odds API.

    Returns:
        Dict with 'h2h' and 'totals' keys, each containing outcome->best_odds mapping.
    """
    best: dict[str, dict[str, float]] = {"h2h": {}, "totals": {}}

    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            market_key: str = market.get("key", "")
            if market_key not in ("h2h", "totals"):
                continue

            for outcome in market.get("outcomes", []):
                name: str = outcome.get("name", "Unknown")
                price: float = outcome.get("price", 0.0)

                # For totals, include the point in the name
                if market_key == "totals" and "point" in outcome:
                    name = f"{name} {outcome['point']}"

                if price > best[market_key].get(name, 0.0):
                    best[market_key][name] = price

    return best


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8: AI ANALYSIS (Groq — Llama 3.3 70B)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT: str = """You are an Elite Quantitative Sports Analyst and Value Bettor.

STRICT RULES:
1. Output ONLY a valid JSON array. No text before or after. Do NOT use markdown formatting blocks.
2. Only include picks where you identify genuine +EV (positive expected value).
3. Confidence must be moderate to high — skip uncertain events.
4. Do NOT pick "Draw" for sports that don't have draws (tennis, basketball, etc.).
5. Each object in the array MUST have these exact fields:
   - "event_id": string (from input)
   - "sport": string
   - "sport_emoji": string (relevant emoji)
   - "home_team": string
   - "away_team": string
   - "home_flag": string (country/team flag emoji)
   - "away_flag": string (country/team flag emoji)
   - "commence_time": string (ISO format)
   - "pick": string (team name or "Draw")
   - "pick_odds": number
   - "goals_pick": string (e.g., "Over 2.5" or "Under 2.5" or "N/A")
   - "goals_odds": number or "N/A"
   - "risk_level": "Low" | "Medium" | "High"
   - "logic": string (2-3 sentences explaining the +EV reasoning with numbers)

6. If no events have +EV, return an empty array: []
"""


def build_analysis_prompt(events: list[dict[str, Any]]) -> str:
    """
    Build the user prompt containing event data for AI analysis.

    Args:
        events: List of event dicts from the Odds API.

    Returns:
        Formatted prompt string.
    """
    lines: list[str] = [
        f"Analyze these {len(events)} upcoming events for +EV opportunities:\n"
    ]

    for i, event in enumerate(events, 1):
        best = extract_best_odds(event.get("bookmakers", []))
        h2h_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in best["h2h"].items()
        ) or "N/A"
        totals_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in best["totals"].items()
        ) or "N/A"
        market_depth = len(event.get("bookmakers", []))

        lines.append(f"--- Event {i} ---")
        lines.append(f"ID: {event.get('id', 'unknown')}")
        lines.append(f"Sport: {event.get('sport_title', event.get('sport_key', 'Unknown'))}")
        lines.append(f"Kickoff: {event.get('commence_time', 'Unknown')}")
        lines.append(f"Home: {event.get('home_team', 'Unknown')}")
        lines.append(f"Away: {event.get('away_team', 'Unknown')}")
        lines.append(f"Best H2H Odds: {h2h_str}")
        lines.append(f"Best Totals Odds: {totals_str}")
        lines.append(f"Market Depth: {market_depth} bookmakers")
        lines.append("")

    return "\n".join(lines)


def parse_ai_response(raw_text: str) -> list[dict[str, Any]]:
    """
    Parse and validate the AI response into a list of predictions.

    Attempts direct JSON parsing first, then falls back to regex extraction.

    Args:
        raw_text: Raw text response from the AI.

    Returns:
        List of validated prediction dicts.
    """
    cleaned = raw_text.strip()

    # Safely remove markdown code block wrappers without using literal backticks in code
    t_ticks = "`" * 3
    if cleaned.startswith(t_ticks):
        cleaned = re.sub(r"^" + t_ticks + r"(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*" + t_ticks + r"$", "", cleaned)

    # Attempt 1: Direct parse
    predictions: Optional[list[Any]] = None
    try:
        predictions = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Find JSON array via regex
    if predictions is None:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                predictions = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Failed to parse AI response as JSON.")
                logger.debug(f"Raw response: {raw_text[:500]}")
                return []

    if not isinstance(predictions, list):
        logger.error("AI response is not a JSON array.")
        return []

    # Validate and clean each prediction
    valid_risk_levels = {"Low", "Medium", "High"}
    validated: list[dict[str, Any]] = []

    for pred in predictions:
        if not isinstance(pred, dict):
            continue

        # Ensure required fields exist
        required = {"pick", "home_team", "away_team", "commence_time"}
        if not required.issubset(pred.keys()):
            logger.warning(f"Skipping prediction missing required fields: {pred.get('home_team', '?')}")
            continue

        # Normalize odds
        try:
            pred["pick_odds"] = float(pred.get("pick_odds", 0))
        except (ValueError, TypeError):
            pred["pick_odds"] = 0.0

        try:
            goals_odds = pred.get("goals_odds", "N/A")
            pred["goals_odds"] = float(goals_odds) if goals_odds != "N/A" else "N/A"
        except (ValueError, TypeError):
            pred["goals_odds"] = "N/A"

        # Normalize risk level
        risk = pred.get("risk_level", "Medium")
        if isinstance(risk, str):
            risk = risk.strip().capitalize()
        pred["risk_level"] = risk if risk in valid_risk_levels else "Medium"

        # Default emojis/flags
        pred.setdefault("sport_emoji", "⚽")
        pred.setdefault("home_flag", "🏠")
        pred.setdefault("away_flag", "✈️")
        pred.setdefault("goals_pick", "N/A")
        pred.setdefault("logic", "No reasoning provided.")
        pred.setdefault("event_id", "unknown")
        pred.setdefault("sport", "Unknown")

        validated.append(pred)

    return validated


def analyze_with_groq(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Send events to Groq AI for +EV analysis.

    Args:
        events: List of event dicts from the Odds API.

    Returns:
        List of validated prediction dicts.
    """
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

    headers: dict[str, str] = {
        "Authorization": f"Bearer {Config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    logger.info(f"Sending {len(events)} events to Groq ({Config.GROQ_MODEL})...")

    try:
        response = http_session.post(
            Config.GROQ_API_BASE,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(
                f"Groq API error {response.status_code}: {response.text[:300]}"
            )
            return []

        data = response.json()
        raw_content: str = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not raw_content:
            logger.error("Empty response from Groq API.")
            return []

        # Log token usage
        usage = data.get("usage", {})
        logger.info(
            f"Groq tokens — prompt: {usage.get('prompt_tokens', '?')}, "
            f"completion: {usage.get('completion_tokens', '?')}, "
            f"total: {usage.get('total_tokens', '?')}"
        )

        predictions = parse_ai_response(raw_content)
        logger.info(f"AI returned {len(predictions)} valid predictions.")
        return predictions

    except requests.RequestException as e:
        logger.error(f"Network error calling Groq: {e}")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9: TELEGRAM BROADCASTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK_EMOJI: dict[str, str] = {
    "Low": "🟢",
    "Medium": "🟠",
    "High": "🔴",
}


def format_countdown(commence_time: str) -> str:
    """
    Format a human-readable countdown or status for an event.

    Args:
        commence_time: ISO format datetime string.

    Returns:
        Countdown string like "⏱ Starts in 1h 23m" or "🔴 LIVE".
    """
    try:
        event_dt = datetime.fromisoformat(
            commence_time.replace("Z", "+00:00")
        )
        now = datetime.now(timezone.utc)
        diff = event_dt - now

        if diff.total_seconds() <= 0:
            return "🔴 LIVE"

        hours, remainder = divmod(int(diff.total_seconds()), 3600)
        minutes = remainder // 60

        if hours > 0:
            return f"⏱ Starts in {hours}h {minutes}m"
        return f"⏱ Starts in {minutes}m"
    except (ValueError, TypeError):
        return "⏱ Time unknown"


def format_logic_numbers(logic: str) -> str:
    """
    Wrap numbers in the logic text with <code> tags for Telegram formatting.

    Args:
        logic: Raw logic string from AI.

    Returns:
        Formatted string with numbers highlighted.
    """
    # Match decimals like 1.85, percentages like 55%, and plain numbers
    return re.sub(
        r"(\d+\.?\d*%?)",
        r"<code>\1</code>",
        html.escape(logic),
    )


def format_prediction_message(pred: dict[str, Any]) -> str:
    """
    Format a single prediction into an HTML message for Telegram.

    Args:
        pred: Validated prediction dict.

    Returns:
        HTML-formatted message string.
    """
    sport_emoji = pred.get("sport_emoji", "⚽")
    sport = html.escape(pred.get("sport", "Unknown"))
    home = html.escape(pred.get("home_team", "Unknown"))
    away = html.escape(pred.get("away_team", "Unknown"))
    home_flag = pred.get("home_flag", "🏠")
    away_flag = pred.get("away_flag", "✈️")
    countdown = format_countdown(pred.get("commence_time", ""))
    pick = html.escape(pred.get("pick", "N/A"))
    pick_odds = pred.get("pick_odds", 0)
    goals_pick = html.escape(str(pred.get("goals_pick", "N/A")))
    goals_odds = pred.get("goals_odds", "N/A")
    risk = pred.get("risk_level", "Medium")
    risk_emoji = RISK_EMOJI.get(risk, "🟠")
    logic = format_logic_numbers(pred.get("logic", ""))

    # Format odds display
    pick_odds_str = f"<code>{pick_odds:.2f}</code>" if isinstance(pick_odds, (int, float)) and pick_odds > 0 else "N/A"
    goals_odds_str = f"<code>{goals_odds:.2f}</code>" if isinstance(goals_odds, (int, float)) and goals_odds != "N/A" else "N/A"

    # Build goals line only if available
    goals_line = ""
    if goals_pick != "N/A":
        goals_line = f"\n⚽ <b>Goals:</b> {goals_pick} @ {goals_odds_str}"

    message = (
        f"{sport_emoji} <b>{sport}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{home_flag} <b>{home}</b>  🆚  <b>{away}</b> {away_flag}\n"
        f"{countdown}\n"
        f"\n"
        f"🎯 <b>Pick:</b> {pick} @ {pick_odds_str}"
        f"{goals_line}\n"
        f"{risk_emoji} <b>Risk:</b> {risk}\n"
        f"\n"
        f"💡 <i>{logic}</i>\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 {Config.TELEGRAM_ID}"
    )

    return message


def send_telegram_message(text: str) -> bool:
    """
    Send a single message to Telegram with retry logic.

    Args:
        text: HTML-formatted message text.

    Returns:
        True if sent successfully, False otherwise.
    """
    url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload: dict[str, Any] = {
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
                # Rate limited — respect retry_after
                retry_after = response.json().get("parameters", {}).get(
                    "retry_after", 5
                )
                logger.warning(
                    f"Telegram rate limit hit. Waiting {retry_after}s..."
                )
                time.sleep(retry_after)
                continue

            logger.error(
                f"Telegram error {response.status_code} "
                f"(attempt {attempt}/{Config.TELEGRAM_MAX_RETRIES}): "
                f"{response.text[:200]}"
            )

        except requests.RequestException as e:
            logger.error(
                f"Telegram network error (attempt {attempt}/{Config.TELEGRAM_MAX_RETRIES}): {e}"
            )

        # Exponential backoff for retries
        backoff = 2**attempt
        logger.info(f"Retrying in {backoff}s...")
        time.sleep(backoff)

        return False


def format_and_send_to_telegram(
    predictions: list[dict[str, Any]], cache: SentCache
) -> int:
    """
    Format predictions and send them to Telegram, skipping duplicates.

    Args:
        predictions: List of validated prediction dicts.
        cache: SentCache instance for deduplication.

    Returns:
        Number of messages successfully sent.
    """
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set. Skipping broadcast.")
        for pred in predictions:
            logger.info(f"[DRY RUN] {pred.get('home_team')} vs {pred.get('away_team')} → {pred.get('pick')}")
        return 0

    sent_count = 0

    # مرتب سازی پیش‌بینی‌ها بر اساس زمان شروع (نزدیک‌ترین مسابقات در ابتدا)
    def get_time(p):
        time_str = p.get('commence_time', '')
        if time_str:
            try:
                return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)
        
    predictions.sort(key=get_time)

    for i, pred in enumerate(predictions):
        # Deduplication check
        if cache.is_sent(pred):
            logger.info(
                f"Skipping duplicate: {pred.get('home_team')} vs {pred.get('away_team')}"
            )
            continue

        message = format_prediction_message(pred)
        success = send_telegram_message(message)

        if success:
            cache.mark_sent(pred)
            sent_count += 1
            logger.info(
                f"✅ Sent ({sent_count}): {pred.get('home_team')} vs {pred.get('away_team')}"
            )
        else:
            logger.error(
                f"❌ Failed to send: {pred.get('home_team')} vs {pred.get('away_team')}"
            )

        # Rate limit delay between messages (skip after last)
        if i < len(predictions) - 1:
            time.sleep(Config.TELEGRAM_DELAY)

    return sent_count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10: MAIN EXECUTION FLOW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Main orchestration: fetch → analyze → broadcast."""
    logger.info("=" * 60)
    logger.info("🚀 zBET90 +EV Predictor — Starting run")
    logger.info(f"   Model: {Config.GROQ_MODEL}")
    logger.info(f"   Scan window: {Config.SCAN_WINDOW_HOURS}h")
    logger.info(f"   Regions: {Config.ODDS_REGIONS}")
    logger.info(f"   Markets: {Config.ODDS_MARKETS}")
    logger.info("=" * 60)

    start_time = time.time()

    # Step 1: Fetch events
    events = get_next_events()
    if not events:
        logger.info("No upcoming events found in scan window. Exiting.")
        return

    # Step 2: AI analysis
    predictions = analyze_with_groq(events)
    if not predictions:
        logger.info("AI found no +EV opportunities. Exiting.")
        return

    # Step 3: Broadcast
    cache = SentCache()
    sent = format_and_send_to_telegram(predictions, cache)

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(
        f"✅ Run complete in {elapsed:.1f}s — "
        f"{len(events)} events scanned, "
        f"{len(predictions)} predictions generated, "
        f"{sent} messages sent."
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⛔ Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
