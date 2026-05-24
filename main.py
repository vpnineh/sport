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
1. Output ONLY a valid JSON array. No text before or after.
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

    # Remove markdown code block wrappers if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^
http://googleusercontent.com/immersive_entry_chip/0
