"""Multi-ticker repetitive income opportunity scanner (Daily / Weekly / Monthly)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from core.macro_news import MacroEnvironment
from core.market_data import fetch_option_chain, fetch_stock_snapshot
from core.options_engine import OptionStrategyPlan, build_income_strategies
from core.predictor import MarketPrediction, predict_stock_movement
from core.universe import CADENCE_CONFIG, get_ticker_info


@dataclass
class ScanOpportunity:
    symbol: str
    name: str
    category: str
    cadence: str
    price: float
    change_pct: float
    iv_rank: float
    direction_label: str
    prob_up_pct: float
    best_strategy: str
    strategy_id: str
    expiration: str
    dte: int
    upfront_cash: float
    max_risk: float
    capital_required: float
    breakeven_low: Optional[float]
    take_profit_dollars: float
    stop_loss_dollars: float
    weekly_yield_pct: float
    annualized_yield_pct: float
    win_prob_pct: float
    safety_score: float
    risk_level: str
    fits_budget: bool
    earnings_soon: bool
    is_live: bool
    plain_pitch: str


def _scan_single(
    symbol: str,
    min_dte: int,
    max_dte: int,
    target_dte: int,
    budget: float,
    risk_profile: str,
    vix: float,
    risk_free_rate: float,
) -> Optional[ScanOpportunity]:
    try:
        snap = fetch_stock_snapshot(symbol, period="6mo")
        chain = fetch_option_chain(
            symbol,
            spot=snap.price,
            fallback_iv=snap.hist_vol_20d,
            min_dte=min_dte,
            max_dte=max_dte,
            target_dte=target_dte,
        )
        pred = predict_stock_movement(snap, horizon_days=chain.dte, news_sentiment=0.0, vix=vix)
        plans = build_income_strategies(
            snap,
            chain,
            pred,
            budget=budget,
            risk_profile=risk_profile,
            risk_free_rate=risk_free_rate,
        )
        if not plans:
            return None
        best: OptionStrategyPlan = plans[0]
        info = get_ticker_info(symbol)
        return ScanOpportunity(
            symbol=snap.symbol,
            name=info.name,
            category=info.category,
            cadence=info.cadence,
            price=snap.price,
            change_pct=snap.change_pct,
            iv_rank=snap.iv_rank,
            direction_label=pred.direction_label,
            prob_up_pct=round(pred.prob_up * 100.0, 0),
            best_strategy=best.strategy_name,
            strategy_id=best.strategy_id,
            expiration=best.expiration,
            dte=best.dte,
            upfront_cash=best.upfront_cash,
            max_risk=best.max_risk,
            capital_required=best.capital_required,
            breakeven_low=best.breakeven_low,
            take_profit_dollars=best.take_profit_dollars,
            stop_loss_dollars=best.stop_loss_dollars,
            weekly_yield_pct=best.weekly_yield_pct,
            annualized_yield_pct=best.annualized_yield_pct,
            win_prob_pct=best.prob_of_profit_pct,
            safety_score=best.safety_score,
            risk_level=best.risk_level,
            fits_budget=best.fits_budget,
            earnings_soon=bool(best.earnings_warning),
            is_live=snap.is_live and chain.is_live,
            plain_pitch=best.plain_tagline,
        )
    except Exception:
        return None


def run_income_scanner(
    cadence_label: str,
    budget: float,
    risk_profile: str,
    macro: MacroEnvironment,
    symbols: Optional[List[str]] = None,
) -> List[ScanOpportunity]:
    """Scan all high-liquidity daily/weekly tickers concurrently and rank top income setups."""
    cfg = CADENCE_CONFIG.get(cadence_label, list(CADENCE_CONFIG.values())[1])
    target_symbols = symbols or cfg["preferred_symbols"]
    rfr = max(0.01, macro.treasury_10y / 100.0)

    results: List[ScanOpportunity] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(
                _scan_single,
                sym,
                cfg["min_dte"],
                cfg["max_dte"],
                cfg["target_dte"],
                budget,
                risk_profile,
                macro.vix,
                rfr,
            ): sym
            for sym in target_symbols
        }
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                results.append(res)

    results.sort(key=lambda r: (r.fits_budget, not r.earnings_soon, r.safety_score, r.weekly_yield_pct), reverse=True)
    return results


def opportunities_to_dataframe(opps: List[ScanOpportunity]) -> pd.DataFrame:
    if not opps:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Ticker": o.symbol,
                "Company / ETF": o.name,
                "Price ($)": o.price,
                "Best Strategy": o.best_strategy,
                "Days": o.dte,
                "Entry Cash (+$)": o.upfront_cash,
                "Take-Profit (+$)": o.take_profit_dollars,
                "Breakeven ($)": o.breakeven_low or o.price,
                "Stop-Loss (-$)": o.stop_loss_dollars,
                "Max Loss (-$)": o.max_risk,
                "Weekly Yield (%)": o.weekly_yield_pct,
                "Win Odds (%)": o.win_prob_pct,
                "Forecast": o.direction_label,
                "Score": o.safety_score,
                "Alert": "⚠️ Earnings" if o.earnings_soon else ("✅ Fits Budget" if o.fits_budget else "💰 Higher Cap"),
            }
            for o in opps
        ]
    )
