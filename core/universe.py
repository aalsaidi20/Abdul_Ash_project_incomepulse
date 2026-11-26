"""Curated liquid universe for daily, weekly, and monthly income option strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TickerInfo:
    symbol: str
    name: str
    category: str
    cadence: str  # "Daily" or "Weekly"
    typical_price: float
    typical_iv: float
    plain_description: str


# Most liquid US ETFs and stocks for repetitive income strategies
INCOME_UNIVERSE: List[TickerInfo] = [
    TickerInfo("SPY", "SPDR S&P 500 ETF", "Index ETF", "Daily", 585.0, 0.16, "Tracks the 500 largest U.S. companies. Daily expirations, tightest option spreads."),
    TickerInfo("QQQ", "Invesco Nasdaq-100 ETF", "Index ETF", "Daily", 505.0, 0.21, "Tracks the top 100 tech & growth companies. Daily expirations, higher income premiums."),
    TickerInfo("IWM", "iShares Russell 2000 ETF", "Index ETF", "Daily", 225.0, 0.23, "Tracks 2,000 smaller U.S. companies. Daily expirations, lower share price for smaller accounts."),
    TickerInfo("AAPL", "Apple Inc.", "Mega-Cap Tech", "Weekly", 230.0, 0.24, "Steady mega-cap tech leader with ultra-liquid weekly options."),
    TickerInfo("MSFT", "Microsoft Corp.", "Mega-Cap Tech", "Weekly", 425.0, 0.23, "Enterprise software & cloud giant with calm, consistent weekly option volume."),
    TickerInfo("NVDA", "NVIDIA Corp.", "Semiconductors", "Weekly", 138.0, 0.48, "AI chip leader. High implied volatility creates large weekly option premiums (higher risk)."),
    TickerInfo("AMZN", "Amazon.com Inc.", "Consumer & Cloud", "Weekly", 205.0, 0.30, "E-commerce & AWS giant with deep weekly option liquidity."),
    TickerInfo("AMD", "Advanced Micro Devices", "Semiconductors", "Weekly", 142.0, 0.45, "Active semiconductor stock with strong weekly income premiums and moderate share price."),
    TickerInfo("TSLA", "Tesla Inc.", "EV & Energy", "Weekly", 330.0, 0.58, "Very high volatility = huge option income, but wide price swings require strict risk caps."),
    TickerInfo("META", "Meta Platforms", "Digital Ads & AI", "Weekly", 575.0, 0.33, "Strong cash-flow tech leader with active weekly option chains."),
    TickerInfo("PLTR", "Palantir Technologies", "AI & Software", "Weekly", 62.0, 0.62, "Lower share price with rich weekly premiums — popular for smaller Wheel accounts."),
    TickerInfo("SOFI", "SoFi Technologies", "Fintech", "Weekly", 15.0, 0.55, "Very affordable share price (~$15), making 100-share Wheel strategies accessible under $2,000."),
    TickerInfo("XLF", "Financial Select Sector ETF", "Sector ETF", "Weekly", 49.0, 0.17, "Diversified U.S. banks & financials. Calm, low share price, great for steady income."),
    TickerInfo("TLT", "20+ Year Treasury Bond ETF", "Bonds ETF", "Weekly", 92.0, 0.16, "Long-term U.S. Treasury bonds. Often moves opposite to stocks, useful for diversification."),
    TickerInfo("GLD", "SPDR Gold Shares", "Commodities ETF", "Weekly", 245.0, 0.16, "Tracks gold prices. Low correlation to tech stocks, smooth weekly option chains."),
]

TICKER_MAP: Dict[str, TickerInfo] = {t.symbol: t for t in INCOME_UNIVERSE}

CADENCE_CONFIG = {
    "Daily / Ultra-Short (1–4 Days)": {
        "min_dte": 1,
        "max_dte": 4,
        "target_dte": 2,
        "risk_note": "Fast cash turnover (several times per week), but sudden price moves leave very little time to recover. Use small position sizes and defined-risk spreads.",
        "preferred_symbols": ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "AMD", "TSLA", "PLTR", "SOFI", "XLF"],
    },
    "Weekly Income (5–10 Days)": {
        "min_dte": 5,
        "max_dte": 10,
        "target_dte": 7,
        "risk_note": "The sweet spot for repetitive weekly paychecks: fast time-decay (Theta) with enough room to collect meaningful cushion.",
        "preferred_symbols": ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "MSFT", "AMZN", "AMD", "TSLA", "PLTR", "SOFI", "XLF", "TLT", "GLD"],
    },
    "Bi-Weekly / Monthly (11–35 Days)": {
        "min_dte": 11,
        "max_dte": 35,
        "target_dte": 21,
        "risk_note": "Smoother ride and wider safety buffer for beginners, with lower stress and fewer daily adjustments.",
        "preferred_symbols": ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "MSFT", "AMZN", "AMD", "META", "PLTR", "SOFI", "XLF", "TLT", "GLD"],
    },
}


def get_ticker_info(symbol: str) -> TickerInfo:
    sym = symbol.upper().strip()
    if sym in TICKER_MAP:
        return TICKER_MAP[sym]
    return TickerInfo(
        symbol=sym,
        name=f"{sym} Equity",
        category="Custom Ticker",
        cadence="Weekly",
        typical_price=100.0,
        typical_iv=0.30,
        plain_description=f"Custom ticker {sym} loaded from live public market feeds.",
    )
