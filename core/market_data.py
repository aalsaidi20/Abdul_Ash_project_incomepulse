"""Zero-cost live public market data & option chain fetcher (Yahoo Finance + graceful fallback)."""
from __future__ import annotations

import datetime as dt
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.universe import get_ticker_info

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


@dataclass
class StockSnapshot:
    symbol: str
    name: str
    price: float
    prev_close: float
    change_pct: float
    day_high: float
    day_low: float
    volume: int
    iv_rank: float  # 0 to 100
    hist_vol_20d: float  # annualized decimal e.g. 0.25
    hist_vol_60d: float
    rsi_14: float
    sma_20: float
    sma_50: float
    next_earnings_days: Optional[int]
    history: pd.DataFrame
    is_live: bool = True
    source_note: str = "Live Yahoo Finance"
    fetched_at_et: str = ""
    last_bar_label: str = ""


@dataclass
class OptionChainData:
    symbol: str
    expiration: str
    dte: int
    calls: pd.DataFrame
    puts: pd.DataFrame
    is_live: bool = True
    available_expirations: List[str] = field(default_factory=list)


def _now_et_label() -> str:
    """Format current timestamp in US Eastern (Market) Time."""
    try:
        from zoneinfo import ZoneInfo

        now_et = dt.datetime.now(ZoneInfo("America/New_York"))
        return now_et.strftime("%b %d, %I:%M %p ET")
    except Exception:
        return dt.datetime.now().strftime("%b %d, %I:%M %p")


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _generate_fallback_history(symbol: str, days: int = 252) -> pd.DataFrame:
    """Deterministic realistic fallback OHLCV when network is temporarily unavailable."""
    info = get_ticker_info(symbol)
    seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=dt.date.today(), periods=days)
    daily_vol = info.typical_iv / math.sqrt(252)
    drift = 0.08 / 252
    shocks = rng.normal(drift, daily_vol, size=days)
    prices = info.typical_price * np.exp(np.cumsum(shocks))
    # Rescale so latest price sits near typical_price
    prices = prices * (info.typical_price / prices[-1])
    highs = prices * (1.0 + rng.uniform(0.003, 0.018, size=days))
    lows = prices * (1.0 - rng.uniform(0.003, 0.018, size=days))
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    volumes = rng.integers(5_000_000, 45_000_000, size=days)
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )


def _bs_price_vec(
    spot: float,
    strikes: np.ndarray,
    t_years: float,
    r: float,
    sigma: float,
    is_call: bool,
) -> np.ndarray:
    from scipy.stats import norm

    t = max(t_years, 1e-4)
    s = max(sigma, 0.05)
    d1 = (np.log(spot / strikes) + (r + 0.5 * s * s) * t) / (s * math.sqrt(t))
    d2 = d1 - s * math.sqrt(t)
    if is_call:
        return np.maximum(spot * norm.cdf(d1) - strikes * math.exp(-r * t) * norm.cdf(d2), 0.01)
    return np.maximum(strikes * math.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1), 0.01)


def _generate_fallback_chain(
    symbol: str,
    spot: float,
    iv: float,
    target_dte: int = 7,
) -> OptionChainData:
    today = dt.date.today()
    exp_date = today + dt.timedelta(days=max(1, target_dte))
    exp_str = exp_date.strftime("%Y-%m-%d")
    t_years = max(target_dte, 1) / 365.0
    step = 0.5 if spot < 30 else (1.0 if spot < 120 else (2.5 if spot < 300 else 5.0))
    center = round(spot / step) * step
    strikes = np.array([center + i * step for i in range(-14, 15)], dtype=float)
    strikes = strikes[strikes > 0]

    # Realistic volatility skew (OTM puts slightly higher IV)
    moneyness = (strikes - spot) / spot
    call_ivs = np.clip(iv * (1.0 - 0.15 * moneyness + 0.3 * moneyness**2), 0.08, 1.5)
    put_ivs = np.clip(iv * (1.0 - 0.25 * moneyness + 0.35 * moneyness**2), 0.08, 1.5)

    call_mids = np.array(
        [_bs_price_vec(spot, np.array([k]), t_years, 0.045, s, True)[0] for k, s in zip(strikes, call_ivs)]
    )
    put_mids = np.array(
        [_bs_price_vec(spot, np.array([k]), t_years, 0.045, s, False)[0] for k, s in zip(strikes, put_ivs)]
    )

    spread_factor = 0.03
    calls = pd.DataFrame(
        {
            "strike": strikes,
            "lastPrice": np.round(call_mids, 2),
            "bid": np.round(np.maximum(call_mids * (1 - spread_factor), 0.01), 2),
            "ask": np.round(call_mids * (1 + spread_factor) + 0.01, 2),
            "impliedVolatility": call_ivs,
            "volume": np.maximum(150, (2500 * np.exp(-20 * moneyness**2)).astype(int)),
            "openInterest": np.maximum(400, (8000 * np.exp(-15 * moneyness**2)).astype(int)),
        }
    )
    puts = pd.DataFrame(
        {
            "strike": strikes,
            "lastPrice": np.round(put_mids, 2),
            "bid": np.round(np.maximum(put_mids * (1 - spread_factor), 0.01), 2),
            "ask": np.round(put_mids * (1 + spread_factor) + 0.01, 2),
            "impliedVolatility": put_ivs,
            "volume": np.maximum(150, (2800 * np.exp(-20 * moneyness**2)).astype(int)),
            "openInterest": np.maximum(400, (9000 * np.exp(-15 * moneyness**2)).astype(int)),
        }
    )
    exps = [(today + dt.timedelta(days=d)).strftime("%Y-%m-%d") for d in (2, 7, 14, 21, 30)]
    return OptionChainData(
        symbol=symbol.upper(),
        expiration=exp_str,
        dte=max(1, target_dte),
        calls=calls,
        puts=puts,
        is_live=False,
        available_expirations=exps,
    )


def fetch_stock_snapshot(symbol: str, period: str = "1y") -> StockSnapshot:
    """Fetch live OHLCV, technicals, volatility rank, and upcoming earnings date via yfinance."""
    sym = symbol.upper().strip()
    info = get_ticker_info(sym)
    hist = pd.DataFrame()
    is_live = False
    name = info.name
    next_earnings_days: Optional[int] = None

    if yf is not None:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period=period, auto_adjust=True)
            if hist is not None and not hist.empty and len(hist) >= 30:
                is_live = True
                hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if getattr(hist.index, "tz", None) is not None:
                    hist.index = hist.index.tz_localize(None)
                try:
                    fast = ticker.fast_info
                    if hasattr(fast, "last_price") and fast.last_price:
                        hist.iloc[-1, hist.columns.get_loc("Close")] = float(fast.last_price)
                except Exception:
                    pass
                try:
                    cal = ticker.calendar
                    if isinstance(cal, dict) and "Earnings Date" in cal:
                        edates = cal["Earnings Date"]
                        if isinstance(edates, list) and edates:
                            ed = edates[0]
                            if isinstance(ed, dt.datetime):
                                ed = ed.date()
                            if isinstance(ed, dt.date):
                                diff = (ed - dt.date.today()).days
                                if -1 <= diff <= 90:
                                    next_earnings_days = diff
                except Exception:
                    pass
        except Exception:
            is_live = False

    if not is_live or hist.empty:
        hist = _generate_fallback_history(sym)

    close = hist["Close"]
    price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
    change_pct = ((price - prev_close) / prev_close) * 100.0 if prev_close else 0.0

    log_ret = np.log(close / close.shift(1)).dropna()
    hv_20_series = log_ret.rolling(20).std() * math.sqrt(252)
    hv_60_series = log_ret.rolling(60).std() * math.sqrt(252)
    hv_20 = float(hv_20_series.iloc[-1]) if not hv_20_series.dropna().empty else info.typical_iv
    hv_60 = float(hv_60_series.iloc[-1]) if not hv_60_series.dropna().empty else hv_20

    # Proxy IV Rank from 1-year rolling realized volatility range when historical IV isn't free
    hv_valid = hv_20_series.dropna()
    if len(hv_valid) > 10 and (hv_valid.max() - hv_valid.min()) > 1e-6:
        iv_rank = float(np.clip((hv_20 - hv_valid.min()) / (hv_valid.max() - hv_valid.min()) * 100.0, 5.0, 95.0))
    else:
        iv_rank = 50.0

    rsi_series = _compute_rsi(close, 14).dropna()
    rsi_14 = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    sma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma_20

    last_ts = pd.Timestamp(hist.index[-1])
    last_bar_label = last_ts.strftime("%b %d, %Y")

    return StockSnapshot(
        symbol=sym,
        name=name,
        price=round(price, 2),
        prev_close=round(prev_close, 2),
        change_pct=round(change_pct, 2),
        day_high=round(float(hist["High"].iloc[-1]), 2),
        day_low=round(float(hist["Low"].iloc[-1]), 2),
        volume=int(hist["Volume"].iloc[-1]),
        iv_rank=round(iv_rank, 1),
        hist_vol_20d=round(hv_20, 4),
        hist_vol_60d=round(hv_60, 4),
        rsi_14=round(rsi_14, 1),
        sma_20=round(sma_20, 2),
        sma_50=round(sma_50, 2),
        next_earnings_days=next_earnings_days,
        history=hist,
        is_live=is_live,
        source_note="Live Yahoo Finance (Free)" if is_live else "Offline Simulated Fallback",
        fetched_at_et=_now_et_label(),
        last_bar_label=last_bar_label,
    )


def fetch_option_chain(
    symbol: str,
    spot: float,
    fallback_iv: float = 0.28,
    min_dte: int = 1,
    max_dte: int = 14,
    target_dte: int = 7,
    chosen_expiration: Optional[str] = None,
) -> OptionChainData:
    """Fetch live free option chain closest to target_dte from Yahoo Finance."""
    sym = symbol.upper().strip()
    today = dt.date.today()

    if yf is not None:
        try:
            ticker = yf.Ticker(sym)
            exps = list(ticker.options or [])
            if exps:
                parsed: List[Tuple[str, int]] = []
                for e in exps:
                    try:
                        d = (dt.date.fromisoformat(e) - today).days
                        if d >= 1:
                            parsed.append((e, d))
                    except ValueError:
                        continue

                if parsed:
                    if chosen_expiration and chosen_expiration in exps:
                        sel_exp = chosen_expiration
                        sel_dte = max(1, (dt.date.fromisoformat(sel_exp) - today).days)
                    else:
                        in_window = [p for p in parsed if min_dte <= p[1] <= max_dte]
                        pool = in_window if in_window else parsed
                        sel_exp, sel_dte = min(pool, key=lambda x: abs(x[1] - target_dte))

                    chain = ticker.option_chain(sel_exp)
                    calls = chain.calls.copy()
                    puts = chain.puts.copy()
                    if not calls.empty and not puts.empty:
                        for df in (calls, puts):
                            for col in ("bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest"):
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                        return OptionChainData(
                            symbol=sym,
                            expiration=sel_exp,
                            dte=sel_dte,
                            calls=calls,
                            puts=puts,
                            is_live=True,
                            available_expirations=[p[0] for p in parsed[:12]],
                        )
        except Exception:
            pass

    return _generate_fallback_chain(sym, spot, max(fallback_iv, 0.14), target_dte=target_dte)
