#!/usr/bin/env python3
"""
zBET90 v4.0 — Professional +EV Sports Betting Intelligence System
================================================================================
✅ Correct +EV calculations with Kelly Criterion
✅ Multi-source data aggregation (The Odds API + SofaScore)
✅ Intelligent team name matching with fuzzy search
✅ Multi-bookmaker comparison for true value detection
✅ Async operations for performance
✅ Production-grade error handling
✅ Pydantic V2 compatible
✅ Async SQLite persistence with caching (aiosqlite)
✅ Smart rate limiting
✅ Telegram broadcasting

Author: @zBET90
License: MIT
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: IMPORTS & DEPENDENCIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import sys
import time
import json
import re
import logging
import hashlib
import asyncio
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import math

# Third-party imports
try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout
    from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
    import numpy as np
    from rapidfuzz import fuzz, process
    import aiosqlite
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install: pip install aiohttp pydantic numpy rapidfuzz aiosqlite")
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: LOGGING SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ColoredFormatter(logging.Formatter):
    """Colored console output for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m'
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname:8}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(log_file: Optional[str] = "zbet90.log") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    logger = logging.getLogger("zBET90")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")

    return logger


logger = setup_logging()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: CONFIGURATION (Direct from Environment)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RiskLevel(str, Enum):
    """Bet risk classification."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Config:
    """Centralized configuration loaded directly from environment variables."""
    
    # ═══ Core API Keys (Required) ═══
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # ═══ Odds API Settings ═══
    ODDS_API_BASE: str = "https://api.the-odds-api.com/v4"
    ODDS_REGIONS: str = os.getenv("ODDS_REGIONS", "eu,uk,us")
    ODDS_MARKETS: str = os.getenv("ODDS_MARKETS", "h2h,totals")
    ODDS_FORMAT: str = "decimal"
    
    # ═══ RapidAPI Settings (SofaScore) ═══
    RAPIDAPI_HOST: str = "sofascore.p.rapidapi.com"
    SOFASCORE_BASE: str = f"https://{RAPIDAPI_HOST}"
    
    # ═══ Scan Window & Limits ═══
    SCAN_WINDOW_HOURS: int = int(os.getenv("SCAN_WINDOW_HOURS", "3"))
    
    # ═══ Mathematical Thresholds ═══
    MIN_EV_PERCENTAGE: float = float(os.getenv("MIN_EV_PERCENTAGE", "3.0"))
    MAX_BOOKMAKER_MARGIN: float = float(os.getenv("MAX_BOOKMAKER_MARGIN", "8.0"))
    KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    MIN_ODDS: float = float(os.getenv("MIN_ODDS", "1.5"))
    MAX_ODDS: float = float(os.getenv("MAX_ODDS", "5.0"))
    
    # ═══ Groq AI Settings ═══
    GROQ_API_BASE: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "8000"))
    
    # ═══ Telegram Settings ═══
    TELEGRAM_DELAY: float = float(os.getenv("TELEGRAM_DELAY", "2.5"))
    TELEGRAM_ID: str = os.getenv("TELEGRAM_ID", "@zBET90")
    
    # ═══ Database & Cache ═══
    DB_PATH: Path = Path(os.getenv("DB_PATH", "zbet90.db"))
    CACHE_EXPIRY_HOURS: int = int(os.getenv("CACHE_EXPIRY_HOURS", "48"))
    
    # ═══ Feature Flags ═══
    ENABLE_POISSON: bool = os.getenv("ENABLE_POISSON", "true").lower() in ("true", "1", "yes")
    ENABLE_ELO: bool = os.getenv("ENABLE_ELO", "true").lower() in ("true", "1", "yes")
    ENABLE_FORM_ANALYSIS: bool = os.getenv("ENABLE_FORM_ANALYSIS", "true").lower() in ("true", "1", "yes")
    
    @classmethod
    def validate(cls):
        """Validate that all required API keys are present."""
        required_keys = {
            "ODDS_API_KEY": cls.ODDS_API_KEY,
            "GROQ_API_KEY": cls.GROQ_API_KEY,
            "RAPIDAPI_KEY": cls.RAPIDAPI_KEY,
            "TELEGRAM_BOT_TOKEN": cls.TELEGRAM_BOT_TOKEN,
            "TELEGRAM_CHAT_ID": cls.TELEGRAM_CHAT_ID,
        }
        
        missing = [key for key, value in required_keys.items() if not value]
        
        if missing:
            error_msg = f"❌ Missing required environment variables: {', '.join(missing)}"
            logger.critical(error_msg)
            raise ValueError(error_msg)
        
        logger.info("✅ All required API keys validated")


# Validate configuration on import
try:
    Config.validate()
except ValueError:
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: PYDANTIC MODELS (V2 Compatible)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TeamStats(BaseModel):
    class MatchContext(BaseModel):
    """Deep match context from SofaScore for AI Analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    match_id: Optional[int] = None
    ai_insights: List[str] = Field(default_factory=list)
    missing_players: List[str] = Field(default_factory=list)
    
    """Team statistical profile."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    team_id: Optional[int] = None
    team_name: str
    recent_form: List[str] = Field(default_factory=list)
    goals_scored_avg: float = 0.0
    goals_conceded_avg: float = 0.0
    win_rate: float = 0.0
    elo_rating: Optional[float] = None
    home_advantage: float = 0.0
    
    @field_validator('win_rate', 'goals_scored_avg', 'goals_conceded_avg')
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        """Ensure values are non-negative."""
        return max(0.0, v)


class ValueBet(BaseModel):
    """Validated value bet opportunity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    event_id: str
    sport: str
    sport_emoji: str
    home_team: str
    away_team: str
    commence_time: datetime
    
    pick: str
    market_type: str
    bookmaker_odds: float
    fair_odds: float
    ev_percentage: float
    kelly_stake: float
    
    risk_level: RiskLevel
    confidence_score: float = Field(ge=0, le=100)
    
    logic: str
    home_stats: Optional[TeamStats] = None
    away_stats: Optional[TeamStats] = None
    match_context: Optional[MatchContext] = None
    
    @field_validator('bookmaker_odds', 'fair_odds')
    @classmethod
    def validate_odds_range(cls, v: float) -> float:
        """Validate odds are within reasonable range."""
        if not (1.01 <= v <= 100):
            raise ValueError(f"Odds out of valid range: {v}")
        return v
    
    @field_validator('ev_percentage')
    @classmethod
    def validate_ev_positive(cls, v: float) -> float:
        """Ensure EV is positive."""
        if v < 0:
            raise ValueError("EV must be positive for value bets")
        return v


class BettingEvent(BaseModel):
    """Raw event from odds API."""
    model_config = ConfigDict(extra='allow')
    
    id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: List[Dict[str, Any]]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: DATABASE LAYER (aiosqlite)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Database:
    """Async SQLite database manager."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    async def init_db(self):
        """Initialize database schema asynchronously."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.executescript("""
                CREATE TABLE IF NOT EXISTS sent_bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_hash TEXT UNIQUE NOT NULL,
                    event_id TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    pick TEXT NOT NULL,
                    odds REAL NOT NULL,
                    ev_percentage REAL NOT NULL,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    commence_time TIMESTAMP NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_event_hash ON sent_bets(event_hash);
                CREATE INDEX IF NOT EXISTS idx_sent_at ON sent_bets(sent_at);
                
                CREATE TABLE IF NOT EXISTS team_cache (
                    team_name TEXT PRIMARY KEY,
                    team_id INTEGER NOT NULL,
                    matched_name TEXT,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    stats_json TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_cached_at ON team_cache(cached_at);
            """)
            await conn.commit()
        logger.info(f"✅ Database initialized: {self.db_path}")
    
    async def is_bet_sent(self, event_hash: str) -> bool:
        """Check if bet was already sent."""
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.execute(
                "SELECT 1 FROM sent_bets WHERE event_hash = ? LIMIT 1",
                (event_hash,)
            ) as cursor:
                return await cursor.fetchone() is not None
    
    async def mark_bet_sent(self, bet: ValueBet):
        """Mark bet as sent to Telegram."""
        event_hash = self._make_hash(bet)
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT OR IGNORE INTO sent_bets 
                (event_hash, event_id, home_team, away_team, pick, odds, ev_percentage, commence_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_hash,
                bet.event_id,
                bet.home_team,
                bet.away_team,
                bet.pick,
                bet.bookmaker_odds,
                bet.ev_percentage,
                bet.commence_time
            ))
            await conn.commit()
    
    async def clean_old_records(self, hours: int = 48):
        """Delete old records."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("DELETE FROM sent_bets WHERE sent_at < ?", (cutoff,))
            await conn.execute("DELETE FROM team_cache WHERE cached_at < ?", (cutoff,))
            await conn.commit()
            logger.info("🗑️  Cleaned old records from database")
    
    async def get_cached_team_id(self, team_name: str) -> Optional[int]:
        """Get cached team ID."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(
                "SELECT team_id FROM team_cache WHERE team_name = ? AND cached_at > ?",
                (team_name, cutoff)
            ) as cursor:
                row = await cursor.fetchone()
                return row['team_id'] if row else None
    
    async def cache_team_id(self, team_name: str, team_id: int, matched_name: Optional[str] = None, stats: Optional[dict] = None):
        """Cache team ID with matched name."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO team_cache (team_name, team_id, matched_name, stats_json, cached_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (team_name, team_id, matched_name, json.dumps(stats) if stats else None))
            await conn.commit()
    
    @staticmethod
    def _make_hash(bet: ValueBet) -> str:
        """Generate unique hash."""
        raw = f"{bet.event_id}-{bet.pick}-{bet.market_type}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


db = Database(Config.DB_PATH)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: RATE LIMITER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = defaultdict(float)
    
    async def acquire(self, key: str = "default"):
        """Wait if necessary."""
        now = time.time()
        elapsed = now - self.last_call[key]
        
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            await asyncio.sleep(wait_time)
        
        self.last_call[key] = time.time()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7: ASYNC HTTP CLIENT (Global Session)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class APIClient:
    """Async HTTP client with connection pooling and retry."""
    
    def __init__(self):
        self.timeout = ClientTimeout(total=30)
        self.rate_limiters = {
            'odds': RateLimiter(10),
            'sofascore': RateLimiter(20),
            'groq': RateLimiter(30),
        }
        self.session: Optional[ClientSession] = None
    
    async def get_session(self) -> ClientSession:
        """Get or create the global async session."""
        if self.session is None or self.session.closed:
            self.session = ClientSession(timeout=self.timeout)
        return self.session

    async def close(self):
        """Close the global async session cleanly."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_with_retry(
        self,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        rate_limit_key: str = "default",
        max_retries: int = 3
    ) -> Optional[Dict]:
        """GET with retry."""
        await self.rate_limiters.get(rate_limit_key, RateLimiter(10)).acquire()
        session = await self.get_session()
        
        for attempt in range(max_retries):
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                    elif response.status >= 500:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"HTTP {response.status}: {url}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    async def post_with_retry(
        self,
        url: str,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        rate_limit_key: str = "default",
        max_retries: int = 3
    ) -> Optional[Dict]:
        """POST with retry."""
        await self.rate_limiters.get(rate_limit_key, RateLimiter(10)).acquire()
        session = await self.get_session()
        
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=json_data) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                    elif response.status >= 500:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"HTTP {response.status}: {url}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"POST error: {e}")
                await asyncio.sleep(2 ** attempt)
        
        return None


api_client = APIClient()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8: MATHEMATICAL +EV ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EVCalculator:
    """Expected Value calculations with detailed logging."""
    
    @staticmethod
    def remove_vig(odds_dict: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Remove bookmaker margin."""
        if len(odds_dict) < 2:
            logger.debug(f"Not enough odds for vig removal: {odds_dict}")
            return 0.0, odds_dict
        
        implied_probs = {outcome: 1 / odd for outcome, odd in odds_dict.items()}
        total_prob = sum(implied_probs.values())
        
        margin_pct = (total_prob - 1) * 100
        
        logger.debug(f"Odds: {odds_dict}")
        logger.debug(f"Total implied probability: {total_prob:.4f} (Margin: {margin_pct:.2f}%)")
        
        if total_prob <= 1.0:
            logger.debug(f"No overround detected (total_prob={total_prob})")
            return margin_pct, odds_dict
            
        if margin_pct > Config.MAX_BOOKMAKER_MARGIN:
            logger.debug(f"Margin too high: {margin_pct:.2f}% > {Config.MAX_BOOKMAKER_MARGIN}%")
            return margin_pct, odds_dict
        
        fair_odds = {}
        for outcome, implied_prob in implied_probs.items():
            fair_prob = implied_prob / total_prob
            fair_odds[outcome] = round(1 / fair_prob, 2)
        
        logger.debug(f"Fair odds calculated: {fair_odds} (Margin: {margin_pct:.2f}%)")
        return margin_pct, fair_odds
    
    @staticmethod
    def calculate_ev(fair_odds: float, bookmaker_odds: float) -> float:
        """Calculate Expected Value percentage."""
        if fair_odds <= 1 or bookmaker_odds <= 1:
            return 0.0
        
        fair_prob = 1 / fair_odds
        profit_if_win = bookmaker_odds - 1
        loss_if_lose = 1.0
        
        ev = (fair_prob * profit_if_win) - ((1 - fair_prob) * loss_if_lose)
        ev_pct = ev * 100
        
        logger.debug(f"EV calc: fair={fair_odds:.2f}, book={bookmaker_odds:.2f} → EV={ev_pct:.2f}%")
        
        return round(ev_pct, 2)
    
    @staticmethod
    def kelly_criterion(fair_odds: float, bookmaker_odds: float) -> float:
        """Calculate Kelly stake."""
        if fair_odds <= 1 or bookmaker_odds <= 1:
            return 0.0
        
        p = 1 / fair_odds
        q = 1 - p
        b = bookmaker_odds - 1
        
        kelly = (b * p - q) / b
        
        if kelly <= 0:
            return 0.0
        
        kelly_stake = kelly * Config.KELLY_FRACTION * 100
        return min(round(kelly_stake, 2), 10.0)
    
    @staticmethod
    def find_value_bets(
        fair_odds: Dict[str, float],
        bookmaker_odds: Dict[str, float],
        min_ev: float = None
    ) -> List[Dict[str, Any]]:
        """Find value bets with detailed logging."""
        if min_ev is None:
            min_ev = Config.MIN_EV_PERCENTAGE
            
        value_bets = []
        
        logger.info(f"\n  🔍 Checking {len(fair_odds)} outcomes for value:")
        
        for outcome, fair_price in fair_odds.items():
            book_price = bookmaker_odds.get(outcome, 0)
            
            # Build complete status message
            status = f"     {outcome:30s} | Fair: {fair_price:5.2f} | Best: {book_price:5.2f}"
            
            if book_price <= fair_price:
                logger.info(f"{status} → ❌ No edge")
                continue
            
            ev_pct = EVCalculator.calculate_ev(fair_price, book_price)
            
            if ev_pct < min_ev:
                logger.info(f"{status} → EV: +{ev_pct:.2f}% (below {min_ev}%)")
                continue
            
            if not (Config.MIN_ODDS <= book_price <= Config.MAX_ODDS):
                logger.info(f"{status} → ⚠️  Odds {book_price:.2f} out of range")
                continue
            
            kelly = EVCalculator.kelly_criterion(fair_price, book_price)
            
            logger.info(f"{status} → ✅ +EV: {ev_pct:.2f}% | Kelly: {kelly:.2f}%")
            
            value_bets.append({
                'outcome': outcome,
                'fair_odds': fair_price,
                'bookmaker_odds': book_price,
                'ev_percentage': ev_pct,
                'kelly_stake': kelly,
                'value_ratio': book_price / fair_price
            })
        
        value_bets.sort(key=lambda x: x['ev_percentage'], reverse=True)
        return value_bets

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9: TEAM NAME NORMALIZATION & SOFASCORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TeamNameNormalizer:
    """Normalize and match team names across different APIs."""
    
    # Common name variations mapping
    TEAM_ALIASES = {
        # Premier League
        "Manchester United": ["Man United", "Man Utd", "Manchester Utd"],
        "Manchester City": ["Man City"],
        "Tottenham Hotspur": ["Tottenham", "Spurs"],
        "Newcastle United": ["Newcastle"],
        "Brighton and Hove Albion": ["Brighton", "Brighton & Hove Albion"],
        "Wolverhampton Wanderers": ["Wolves", "Wolverhampton"],
        "West Ham United": ["West Ham"],
        "Leicester City": ["Leicester"],
        "Nottingham Forest": ["Nott'm Forest", "Nottingham"],
        "Sheffield United": ["Sheffield Utd"],
        
        # La Liga
        "Athletic Club": ["Athletic Bilbao"],
        "Atletico Madrid": ["Atlético Madrid", "Atlético de Madrid"],
        
        # Serie A
        "Inter Milan": ["Inter", "Internazionale"],
        "AC Milan": ["Milan"],
        
        # Bundesliga
        "Bayern Munich": ["Bayern München", "FC Bayern"],
        "Borussia Dortmund": ["Dortmund", "BVB"],
        "Borussia Monchengladbach": ["Borussia M'gladbach", "Gladbach"],
        
        # Ligue 1
        "Paris Saint Germain": ["PSG", "Paris SG", "Paris Saint-Germain"],
        
        # Add more as needed
    }
    
    _REVERSE_MAP = {}
    
    @classmethod
    def _build_reverse_map(cls):
        """Build reverse lookup map."""
        if cls._REVERSE_MAP:
            return
        
        for canonical, aliases in cls.TEAM_ALIASES.items():
            cls._REVERSE_MAP[canonical.lower()] = canonical
            for alias in aliases:
                cls._REVERSE_MAP[alias.lower()] = canonical
    
    @classmethod
    def normalize(cls, team_name: str) -> str:
        """Normalize team name to canonical form."""
        cls._build_reverse_map()
        cleaned = team_name.strip()
        canonical = cls._REVERSE_MAP.get(cleaned.lower())
        if canonical:
            return canonical
        return cleaned
    
    @classmethod
    def get_search_variants(cls, team_name: str) -> List[str]:
        """Get all possible variants of a team name for searching."""
        variants = [team_name]
        
        # Add normalized version
        normalized = cls.normalize(team_name)
        if normalized != team_name:
            variants.append(normalized)
        
        # Add all aliases
        for canonical, aliases in cls.TEAM_ALIASES.items():
            if team_name.lower() == canonical.lower():
                variants.extend(aliases)
                break
            if team_name.lower() in [a.lower() for a in aliases]:
                variants.append(canonical)
                variants.extend(aliases)
                break
        
        # Add short version
        words = team_name.split()
        if len(words) > 1:
            variants.append(words[0])
        
        # Remove duplicates
        seen = set()
        unique_variants = []
        for v in variants:
            v_lower = v.lower()
            if v_lower not in seen:
                seen.add(v_lower)
                unique_variants.append(v)
        
        return unique_variants


class SofaScoreAPI:
    """SofaScore data enrichment with intelligent team matching."""
    
    @staticmethod
    def _get_headers() -> Dict[str, str]:
        return {
            'x-rapidapi-key': Config.RAPIDAPI_KEY,
            'x-rapidapi-host': Config.RAPIDAPI_HOST
        }
    
    async def search_team_with_variants(
        self, 
        team_name: str
    ) -> Optional[Tuple[int, str]]:
        """
        Search for team using multiple name variants.
        Returns: (team_id, matched_name) or None
        """
        variants = TeamNameNormalizer.get_search_variants(team_name)
        
        logger.debug(f"Searching for '{team_name}' with variants: {variants[:3]}...")
        
        url = f"{Config.SOFASCORE_BASE}/teams/search"
        
        best_match = None
        best_score = 0
        
        for variant in variants:
            params = {'name': variant}
            
            data = await api_client.get_with_retry(
                url,
                headers=self._get_headers(),
                params=params,
                rate_limit_key='sofascore'
            )
            await asyncio.sleep(0.5) # جلوگیری از بن شدن به دلیل ریکوئست‌های متوالی به RapidAPI
            
            if not data or 'data' not in data or not data['data']:
                continue
            
            # Check top results for best match
            for team in data['data'][:3]:
                team_id = team.get('id')
                api_name = team.get('name', '')
                
                if not team_id or not api_name:
                    continue
                
                # Calculate similarity
                score = fuzz.ratio(team_name.lower(), api_name.lower())
                variant_score = fuzz.ratio(variant.lower(), api_name.lower())
                score = max(score, variant_score)
                
                logger.debug(f"  Match candidate: '{api_name}' (ID: {team_id}, Score: {score})")
                
                if score > best_score:
                    best_score = score
                    best_match = (team_id, api_name)
                
                if score >= 95:
                    break
            
            if best_score >= 95:
                break
        
        if best_match and best_score >= 70:
            team_id, matched_name = best_match
            logger.info(f"✅ Matched '{team_name}' → '{matched_name}' (Score: {best_score})")
            return team_id, matched_name
        
        logger.warning(f"❌ No good match for '{team_name}' (best: {best_score})")
        return None
    
    async def get_team_id(self, team_name: str) -> Optional[int]:
        """Search team ID with caching and variant matching."""
        # Check cache
        cached_id = await db.get_cached_team_id(team_name)
        if cached_id:
            logger.debug(f"Cache hit for '{team_name}': {cached_id}")
            return cached_id
        
        # Search with variants
        result = await self.search_team_with_variants(team_name)
        
        if result:
            team_id, matched_name = result
            # Cache both names
            await db.cache_team_id(team_name, team_id, matched_name)
            if matched_name != team_name:
                await db.cache_team_id(matched_name, team_id, matched_name)
            return team_id
        
        return None
    async def get_match_id(self, home_team: str, away_team: str, commence_time: datetime) -> Optional[int]:
        """Find specific match ID using search."""
        # برای صرفه جویی، ابتدا تیم میزبان را سرچ میکنیم و آیدی آن را میگیریم
        team_id = await self.get_team_id(home_team)
        if not team_id:
            return None
            
        # سپس بازی های بعدی این تیم را میگیریم تا مسابقه مد نظر را پیدا کنیم
        url = f"{Config.SOFASCORE_BASE}/teams/get-next-matches"
        params = {'teamId': str(team_id)}
        
        data = await api_client.get_with_retry(
            url, headers=self._get_headers(), params=params, rate_limit_key='sofascore'
        )
        await asyncio.sleep(0.5)
        
        if not data or 'data' not in data:
            return None
            
        events = data['data'].get('events', [])
        
        # پیدا کردن مسابقه با تطبیق تیم روبرو یا تاریخ
        for event in events:
            api_away = event.get('awayTeam', {}).get('name', '')
            api_home = event.get('homeTeam', {}).get('name', '')
            
            # اگر تیم مهمان همخوانی داشت یا تاریخ نزدیک بود
            score_away = fuzz.ratio(away_team.lower(), api_away.lower())
            if score_away > 70 or fuzz.ratio(away_team.lower(), api_home.lower()) > 70:
                match_id = event.get('id')
                logger.info(f"🎯 Found Match ID: {match_id} for {home_team} vs {away_team}")
                return match_id
                
        return None

    async def get_match_context(self, home_team: str, away_team: str, commence_time: datetime) -> MatchContext:
        """Fetch deep match context (Insights & Lineups) saving API calls."""
        context = MatchContext()
        
        match_id = await self.get_match_id(home_team, away_team, commence_time)
        if not match_id:
            return context
            
        context.match_id = match_id
        
        # 1. گرفتن هوش مصنوعی سوفاسکور (Insights)
        insights_url = f"{Config.SOFASCORE_BASE}/matches/get-ai-insights"
        insights_data = await api_client.get_with_retry(
            insights_url, headers=self._get_headers(), params={'matchId': str(match_id)}, rate_limit_key='sofascore'
        )
        await asyncio.sleep(0.5)
        
        if insights_data and 'data' in insights_data:
            for item in insights_data['data']:
                text = item.get('text')
                if text:
                    context.ai_insights.append(text)
                    
        # 2. گرفتن مصدومان و غایبین از ترکیب (Lineups)
        lineups_url = f"{Config.SOFASCORE_BASE}/matches/get-lineups"
        lineups_data = await api_client.get_with_retry(
            lineups_url, headers=self._get_headers(), params={'matchId': str(match_id)}, rate_limit_key='sofascore'
        )
        await asyncio.sleep(0.5)
        
        if lineups_data and 'data' in lineups_data:
            missing = lineups_data['data'].get('missingPlayers', [])
            for player in missing:
                name = player.get('player', {}).get('name', 'Unknown')
                reason = player.get('reason', 'Missing')
                team_type = player.get('type', '') # 'home' or 'away'
                context.missing_players.append(f"{name} ({reason}) - {team_type} team")
                
        return context
        
    async def get_team_stats(self, team_name: str) -> Optional[TeamStats]:
        """Fetch team stats with intelligent name matching."""
        team_id = await self.get_team_id(team_name)
        if not team_id:
            return None
        
        url = f"{Config.SOFASCORE_BASE}/teams/get-last-matches"
        params = {'teamId': str(team_id)}
        
        data = await api_client.get_with_retry(
            url,
            headers=self._get_headers(),
            params=params,
            rate_limit_key='sofascore'
        )
        
        if not data or 'data' not in data:
            return None
        
        events = data['data'].get('events', [])[:10]
        
        if not events:
            return None
        
        recent_form = []
        goals_scored = []
        goals_conceded = []
        wins = 0
        
        for match in events:
            home_team = match.get('homeTeam', {})
            away_team = match.get('awayTeam', {})
            home_score = match.get('homeScore', {}).get('current', 0)
            away_score = match.get('awayScore', {}).get('current', 0)
            
            is_home = home_team.get('id') == team_id
            
            if is_home:
                goals_scored.append(home_score)
                goals_conceded.append(away_score)
                if home_score > away_score:
                    recent_form.append('W')
                    wins += 1
                elif home_score == away_score:
                    recent_form.append('D')
                else:
                    recent_form.append('L')
            else:
                goals_scored.append(away_score)
                goals_conceded.append(home_score)
                if away_score > home_score:
                    recent_form.append('W')
                    wins += 1
                elif away_score == home_score:
                    recent_form.append('D')
                else:
                    recent_form.append('L')
        
        return TeamStats(
            team_id=team_id,
            team_name=team_name,
            recent_form=recent_form[:5],
            goals_scored_avg=round(np.mean(goals_scored), 2) if goals_scored else 0.0,
            goals_conceded_avg=round(np.mean(goals_conceded), 2) if goals_conceded else 0.0,
            win_rate=round((wins / len(events)) * 100, 1) if events else 0.0
        )


sofascore = SofaScoreAPI()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10: ODDS AGGREGATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OddsAggregator:
    """Multi-source odds aggregation with proper bookmaker comparison."""
    
    async def fetch_the_odds_api(self) -> List[BettingEvent]:
        """Primary: The Odds API."""
        url = f"{Config.ODDS_API_BASE}/sports/upcoming/odds"
        params = {
            'apiKey': Config.ODDS_API_KEY,
            'regions': Config.ODDS_REGIONS,
            'markets': Config.ODDS_MARKETS,
            'oddsFormat': Config.ODDS_FORMAT
        }
        
        data = await api_client.get_with_retry(
            url,
            params=params,
            rate_limit_key='odds',
            max_retries=3
        )
        
        if not data:
            return []
        
        now = datetime.now(timezone.utc)
        window_end = now + timedelta(hours=Config.SCAN_WINDOW_HOURS)
        
        events = []
        for raw_event in data:
            try:
                commence = datetime.fromisoformat(
                    raw_event['commence_time'].replace('Z', '+00:00')
                )
                if now <= commence <= window_end:
                    event = BettingEvent(**raw_event)
                    events.append(event)
            except (ValueError, ValidationError, KeyError):
                continue
        
        logger.info(f"✅ Fetched {len(events)} events from Odds API")
        return events
    
    async def get_all_bookmaker_odds(self, bookmakers: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract ALL odds from ALL bookmakers for comparison.
        Returns: {market_type: [{bookmaker, outcome, price}, ...]}
        """
        all_odds = {'h2h': [], 'totals': []}
        
        for bookmaker in bookmakers:
            bookie_name = bookmaker.get('key', 'unknown')
            
            for market in bookmaker.get('markets', []):
                market_key = market.get('key')
                
                if market_key not in all_odds:
                    continue
                
                for outcome in market.get('outcomes', []):
                    name = outcome.get('name', 'Unknown')
                    price = outcome.get('price', 0.0)
                    
                    if market_key == 'totals' and 'point' in outcome:
                        name = f"{name} {outcome['point']}"
                    
                    all_odds[market_key].append({
                        'bookmaker': bookie_name,
                        'outcome': name,
                        'price': price
                    })
        
        return all_odds
    
    def calculate_fair_and_best_odds(
        self, 
        all_odds: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float], int]:
        """
        Calculate fair odds and best available odds.
        Returns: (fair_odds, best_odds, num_bookmakers)
        """
        if not all_odds:
            return {}, {}, 0
        
        # Group by outcome
        outcome_prices = defaultdict(list)
        outcome_bookmakers = defaultdict(set)
        
        for odd in all_odds:
            outcome = odd['outcome']
            outcome_prices[outcome].append(odd['price'])
            outcome_bookmakers[outcome].add(odd['bookmaker'])
        
        num_bookmakers = len(set(odd['bookmaker'] for odd in all_odds))
        
        # Calculate average odds (for fair odds calculation)
        avg_odds = {}
        for outcome, prices in outcome_prices.items():
            avg_odds[outcome] = round(sum(prices) / len(prices), 2)
        
        # Calculate best odds
        best_odds = {}
        for outcome, prices in outcome_prices.items():
            best_odds[outcome] = max(prices)
        
        # Remove vig from average to get fair odds
        margin, fair_odds = EVCalculator.remove_vig(avg_odds)
        
        return fair_odds, best_odds, num_bookmakers
    
    async def fetch_all_events(self) -> List[BettingEvent]:
        """Fetch all events."""
        events = await self.fetch_the_odds_api()
        return events


odds_aggregator = OddsAggregator()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 11: AI ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """You are an elite sports betting analyst specializing in qualitative assessment.

CRITICAL RULES:
1. You receive MATHEMATICALLY CALCULATED +EV data - DO NOT recalculate
2. Your role is QUALITATIVE ANALYSIS ONLY
3. Output ONLY valid JSON array (no markdown blocks)
4. Be concise - max 2 sentences per logic field
5. Focus on: form, head-to-head, injuries, motivation

REQUIRED JSON:
[
  {
    "event_id": "string",
    "sport": "string",
    "sport_emoji": "string",
    "home_team": "string",
    "away_team": "string",
    "commence_time": "ISO8601",
    "pick": "string",
    "market_type": "h2h|totals",
    "bookmaker_odds": number,
    "fair_odds": number,
    "ev_percentage": number,
    "kelly_stake": number,
    "confidence_score": number (0-100),
    "risk_level": "Low"|"Medium"|"High",
    "logic": "string"
  }
]"""


class GroqAnalyzer:
    """AI qualitative analysis."""
    
    async def analyze_value_bets(self, events_with_value: List[Dict[str, Any]], chunk_size: int = 5) -> List[ValueBet]:
        """Analyze with AI."""
        if not events_with_value:
            return []
        
        all_validated_bets = []
        
        for i in range(0, len(events_with_value), chunk_size):
            chunk = events_with_value[i:i + chunk_size]
            user_prompt = self._build_prompt(chunk)
            
            payload = {
                'model': Config.GROQ_MODEL,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_prompt}
                ],
                'temperature': Config.GROQ_TEMPERATURE,
                'max_tokens': Config.GROQ_MAX_TOKENS
            }
            
            headers = {
                'Authorization': f'Bearer {Config.GROQ_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            response = await api_client.post_with_retry(
                Config.GROQ_API_BASE,
                headers=headers,
                json_data=payload,
                rate_limit_key='groq'
            )
            
            if response:
                raw_content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                all_validated_bets.extend(self._parse_response(raw_content))
                
            await asyncio.sleep(1) # وقفه بین درخواست‌ها برای مدیریت Rate Limit
            
        return all_validated_bets
    
    def _build_prompt(self, events: List[Dict[str, Any]]) -> str:
        """Build prompt with deep context."""
        lines = ["# Value Bets - Qualitative Analysis Required\n"]
        
        for i, event_data in enumerate(events, 1):
            event = event_data['event']
            value_bet = event_data['value_bet']
            home_stats = event_data.get('home_stats')
            away_stats = event_data.get('away_stats')
            match_context = event_data.get('match_context')
            
            lines.append(f"## Event {i}")
            lines.append(f"**Sport:** {event.sport_title}")
            lines.append(f"**Match:** {event.home_team} vs {event.away_team}")
            lines.append("### Math (Python Calculated):")
            lines.append(f"- **Pick:** {value_bet['outcome']}")
            lines.append(f"- **EV%:** {value_bet['ev_percentage']}%")
            lines.append("")
            
            # تزریق اطلاعات طلایی کانتکست مسابقه
            if match_context:
                if match_context.ai_insights:
                    lines.append("### SofaScore AI Match Insights (Crucial):")
                    for insight in match_context.ai_insights:
                        lines.append(f"- {insight}")
                    lines.append("")
                    
                if match_context.missing_players:
                    lines.append("### Missing/Injured Players (Crucial):")
                    for mp in match_context.missing_players:
                        lines.append(f"- {mp}")
                    lines.append("")
            
            # اطلاعات فرم تیم‌ها
            if home_stats:
                lines.append(f"### {event.home_team} Recent Form: {' '.join(home_stats.recent_form)}")
            if away_stats:
                lines.append(f"### {event.away_team} Recent Form: {' '.join(away_stats.recent_form)}")
                
            lines.append("---\n")
        
        return "\n".join(lines)
    
    def _parse_response(self, raw_text: str) -> List[ValueBet]:
        """Parse AI response."""
        cleaned = raw_text.strip()
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        try:
            predictions = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if match:
                try:
                    predictions = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []
        
        if not isinstance(predictions, list):
            return []
        
        validated = []
        for pred in predictions:
            try:
                pred['commence_time'] = datetime.fromisoformat(
                    pred['commence_time'].replace('Z', '+00:00')
                )
                pred['sport_emoji'] = self._get_emoji(pred.get('sport', ''))
                
                bet = ValueBet(**pred)
                validated.append(bet)
            except (ValidationError, ValueError, KeyError) as e:
                logger.warning(f"Invalid prediction: {e}")
                continue
        
        return validated
    
    @staticmethod
    def _get_emoji(sport: str) -> str:
        """Get sport emoji."""
        sport_lower = sport.lower()
        if 'soccer' in sport_lower or 'football' in sport_lower:
            return '⚽'
        elif 'basketball' in sport_lower:
            return '🏀'
        elif 'tennis' in sport_lower:
            return '🎾'
        elif 'hockey' in sport_lower:
            return '🏒'
        elif 'baseball' in sport_lower:
            return '⚾'
        return '🎯'


groq = GroqAnalyzer()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 12: TELEGRAM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TelegramBot:
    """Telegram broadcaster."""
    
    RISK_EMOJI = {
        RiskLevel.LOW: '🟢',
        RiskLevel.MEDIUM: '🟠',
        RiskLevel.HIGH: '🔴'
    }
    
    @staticmethod
    def _escape_html(text: str) -> str:
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    @staticmethod
    def _format_countdown(commence_time: datetime) -> str:
        now = datetime.now(timezone.utc)
        diff = commence_time - now
        
        if diff.total_seconds() <= 0:
            return '🔴 <b>LIVE</b>'
        
        hours = int(diff.total_seconds() // 3600)
        minutes = int((diff.total_seconds() % 3600) // 60)
        
        if hours > 24:
            days = hours // 24
            return f'📅 <b>In {days}d {hours % 24}h</b>'
        elif hours > 0:
            return f'⏱ <b>In {hours}h {minutes}m</b>'
        else:
            return f'⏱ <b>In {minutes}m</b>'
    
    def format_message(self, bet: ValueBet) -> str:
        """Format message."""
        sport = self._escape_html(bet.sport)
        home = self._escape_html(bet.home_team)
        away = self._escape_html(bet.away_team)
        pick = self._escape_html(bet.pick)
        logic = self._escape_html(bet.logic)
        
        risk_emoji = self.RISK_EMOJI.get(bet.risk_level, '🟠')
        countdown = self._format_countdown(bet.commence_time)
        
        msg = f"""
{bet.sport_emoji} <b>{sport}</b>
━━━━━━━━━━━━━━━━━━━━━━━━
🏠 <b>{home}</b>
       🆚
✈️ <b>{away}</b>

{countdown}

━━━━━━━━━━━━━━━━━━━━━━━━
🎯 <b>PICK:</b> {pick}
💰 <b>Odds:</b> <code>{bet.bookmaker_odds:.2f}</code>

📊 <b>MATH EDGE:</b>
   • Fair Odds: <code>{bet.fair_odds:.2f}</code>
   • EV: <b>+{bet.ev_percentage}%</b>
   • Kelly: <code>{bet.kelly_stake}%</code>

{risk_emoji} <b>Risk:</b> {bet.risk_level.value}
📈 <b>Confidence:</b> {bet.confidence_score}/100

💡 <b>ANALYSIS:</b>
<blockquote expandable><i>{logic}</i></blockquote>
"""

        if bet.home_stats and bet.away_stats:
            msg += f"""
━━━━━━━━━━━━━━━━━━━━━━━━
📋 <b>FORM:</b>
🏠 {' '.join(bet.home_stats.recent_form)} ({bet.home_stats.win_rate}% wins)
✈️ {' '.join(bet.away_stats.recent_form)} ({bet.away_stats.win_rate}% wins)
"""
        
        msg += f"""
━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ <b>Bet responsibly</b>
📡 {Config.TELEGRAM_ID}
"""
        
        return msg.strip()
    
    async def send_message(self, text: str, retries: int = 3) -> bool:
        """Send to Telegram."""
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        
        payload = {
            'chat_id': Config.TELEGRAM_CHAT_ID,
            'text': text,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
        response = await api_client.post_with_retry(
            url,
            json_data=payload,
            rate_limit_key='default',
            max_retries=retries
        )
        
        return True if response and response.get('ok') else False
    
    async def broadcast_bets(self, bets: List[ValueBet]) -> int:
        """Broadcast all bets."""
        sent_count = 0
        bets.sort(key=lambda b: b.ev_percentage, reverse=True)
        
        for bet in bets:
            # Check async DB
            if await db.is_bet_sent(db._make_hash(bet)):
                continue
            
            message = self.format_message(bet)
            success = await self.send_message(message)
            
            if success:
                await db.mark_bet_sent(bet) # Update async DB
                sent_count += 1
                logger.info(f"✅ Sent: {bet.pick}")
            
            await asyncio.sleep(Config.TELEGRAM_DELAY)
        
        return sent_count


telegram = TelegramBot()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 13: MAIN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Pipeline:
    """Main execution pipeline with multi-bookmaker comparison."""
    
    async def process_events(self) -> List[ValueBet]:
        """Process all events."""
        logger.info("🚀 Starting pipeline...")
        
        events = await odds_aggregator.fetch_all_events()
        if not events:
            logger.warning("No events found")
            return []
        
        logger.info(f"📥 Processing {len(events)} events")
        
        events_with_value = []
        
        for event in events:
            try:
                # Get ALL bookmaker odds
                all_bookmaker_odds = await odds_aggregator.get_all_bookmaker_odds(event.bookmakers)
                
                if not all_bookmaker_odds['h2h']:
                    continue
                
                # Calculate fair vs best odds
                fair_odds, best_odds, num_bookies = odds_aggregator.calculate_fair_and_best_odds(
                    all_bookmaker_odds['h2h']
                )
                
                if not fair_odds or not best_odds:
                    continue
                
                logger.info(f"\n{'='*70}")
                logger.info(f"🏟️  {event.sport_title}: {event.home_team} vs {event.away_team}")
                logger.info(f"📅 {event.commence_time}")
                logger.info(f"📚 Bookmakers: {num_bookies}")
                logger.info(f"{'='*70}")
                
                # Find value bets
                value_bets = EVCalculator.find_value_bets(fair_odds, best_odds)
                
                if not value_bets:
                    continue
                
                # Fetch team stats AND match context
                home_stats_task = sofascore.get_team_stats(event.home_team)
                away_stats_task = sofascore.get_team_stats(event.away_team)
                match_context_task = sofascore.get_match_context(event.home_team, event.away_team, event.commence_time)
                
                home_stats, away_stats, match_context = await asyncio.gather(
                    home_stats_task, away_stats_task, match_context_task, return_exceptions=True
                )
                
                if isinstance(home_stats, Exception): home_stats = None
                if isinstance(away_stats, Exception): away_stats = None
                if isinstance(match_context, Exception): match_context = None
                
                for vb in value_bets:
                    events_with_value.append({
                        'event': event,
                        'value_bet': vb,
                        'home_stats': home_stats,
                        'away_stats': away_stats,
                        'match_context': match_context
                    })
            
            except Exception as e:
                logger.error(f"Error processing {event.id}: {e}")
                continue
        
        if not events_with_value:
            logger.info("\n" + "="*70)
            logger.info("📊 SUMMARY: No value bets found")
            logger.info("="*70)
            return []
        
        logger.info(f"\n💎 Found {len(events_with_value)} value opportunities")
        
        validated = await groq.analyze_value_bets(events_with_value)
        logger.info(f"✅ AI validated {len(validated)} bets")
        
        return validated
    
    async def run(self):
        """Execute pipeline."""
        try:
            await db.clean_old_records(Config.CACHE_EXPIRY_HOURS)
            
            value_bets = await self.process_events()
            
            if not value_bets:
                logger.info("✨ No bets to broadcast")
                return
            
            sent = await telegram.broadcast_bets(value_bets)
            logger.info(f"📢 Broadcast complete: {sent} bets sent")
            
        except Exception as e:
            logger.critical(f"Pipeline error: {e}", exc_info=True)
            raise

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 14: ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    """Main entry."""
    logger.info("=" * 80)
    logger.info(" zBET90 v4.0 - Professional +EV System")
    logger.info("=" * 80)
    
    # Initialize async database
    await db.init_db()
    
    try:
        pipeline = Pipeline()
        await pipeline.run()
    finally:
        # Close global session
        await api_client.close()
    
    logger.info("=" * 80)
    logger.info(" Completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"💥 Fatal: {e}", exc_info=True)
        sys.exit(1)
