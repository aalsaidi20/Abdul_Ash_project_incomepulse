"""Black-Scholes Greeks & Beginner-Friendly Repetitive Income Option Strategy Engine."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.market_data import OptionChainData, StockSnapshot
from core.predictor import MarketPrediction


@dataclass
class OptionStrategyPlan:
    strategy_id: str
    strategy_name: str
    short_label: str
    symbol: str
    expiration: str
    dte: int
    contracts: int
    # Core dollar numbers (scaled by `contracts`)
    upfront_cash: float          # Total credit received ($)
    max_profit: float            # Best-case profit ($)
    max_risk: float              # Worst-case loss ($)
    capital_required: float      # Collateral / buying power needed ($)
    return_on_capital_pct: float # Yield for this single cycle (%)
    weekly_yield_pct: float      # Normalized 7-day yield (%)
    annualized_yield_pct: float  # Annualized yield if repeated (%)
    prob_of_profit_pct: float    # Win probability (%)
    breakeven_low: Optional[float]
    breakeven_high: Optional[float]
    # Explicit Entry & Exit Points
    entry_stock_price: float
    entry_credit_per_share: float
    take_profit_buyback_per_share: float
    take_profit_dollars: float
    stop_loss_buyback_per_share: float
    stop_loss_dollars: float
    stop_loss_stock_trigger: float
    breakeven_cushion_pct: float
    # Greeks & risk
    net_delta: float
    daily_theta_dollars: float   # Daily cash decay working in your favor ($)
    safety_score: float          # 0-100 composite quality score
    risk_level: str              # "Low Risk", "Moderate Risk", "Higher Risk"
    risk_color: str
    fits_budget: bool
    is_recommended: bool
    # Plain-English beginner guide
    plain_tagline: str
    contract_summary: str
    what_you_do: str
    how_you_make_money: str
    what_is_your_risk: str
    repeatable_playbook: List[str]
    fit_reason: str
    earnings_warning: Optional[str]
    # Payoff diagram data
    payoff_prices: List[float] = field(default_factory=list)
    payoff_pnl: List[float] = field(default_factory=list)


def bs_greeks(
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    sigma: float,
    is_call: bool,
) -> Dict[str, float]:
    """Compute Black-Scholes price, Delta, and daily Theta."""
    t = max(t_years, 1e-4)
    s = max(sigma, 0.08)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(max(spot, 1e-4) / max(strike, 1e-4)) + (r + 0.5 * s * s) * t) / (s * sqrt_t)
    d2 = d1 - s * sqrt_t
    pdf_d1 = norm.pdf(d1)

    if is_call:
        price = spot * norm.cdf(d1) - strike * math.exp(-r * t) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta_annual = (-(spot * pdf_d1 * s) / (2 * sqrt_t)) - r * strike * math.exp(-r * t) * norm.cdf(d2)
    else:
        price = strike * math.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta_annual = (-(spot * pdf_d1 * s) / (2 * sqrt_t)) + r * strike * math.exp(-r * t) * norm.cdf(-d2)

    return {
        "price": max(0.01, float(price)),
        "delta": float(delta),
        "theta_daily": float(theta_annual / 365.0),
    }


def _mid_price(row: pd.Series) -> float:
    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)
    last = float(row.get("lastPrice", 0.0) or 0.0)
    if bid > 0 and ask > 0 and ask >= bid:
        return round((bid + ask) / 2.0, 2)
    if last > 0:
        return round(last, 2)
    return max(0.05, bid, ask)


def _row_iv(row: pd.Series, fallback: float) -> float:
    iv = float(row.get("impliedVolatility", 0.0) or 0.0)
    if 0.05 <= iv <= 2.5:
        return iv
    return max(0.12, fallback)


def _pick_strike_by_delta(
    df: pd.DataFrame,
    spot: float,
    t_years: float,
    r: float,
    fallback_iv: float,
    is_call: bool,
    target_abs_delta: float,
    otm_only: bool = True,
) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    sub = df.copy()
    if otm_only:
        sub = sub[sub["strike"] > spot * 1.002] if is_call else sub[sub["strike"] < spot * 0.998]
    if sub.empty:
        sub = df.copy()

    sub["mid"] = sub.apply(_mid_price, axis=1)
    sub = sub[sub["mid"] >= 0.03]
    if sub.empty:
        return None

    deltas = []
    for _, row in sub.iterrows():
        iv = _row_iv(row, fallback_iv)
        g = bs_greeks(spot, float(row["strike"]), t_years, r, iv, is_call)
        deltas.append(abs(g["delta"]))
    sub["abs_delta"] = deltas
    sub["delta_diff"] = (sub["abs_delta"] - target_abs_delta).abs()
    best_idx = sub["delta_diff"].idxmin()
    return sub.loc[best_idx]


def _pick_protection_wing(
    df: pd.DataFrame,
    short_strike: float,
    spot: float,
    is_call: bool,
    target_width_pct: float = 0.03,
) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    target_width = max(1.0, spot * target_width_pct)
    if is_call:
        cands = df[df["strike"] > short_strike].copy()
        if cands.empty:
            return None
        target_k = short_strike + target_width
    else:
        cands = df[df["strike"] < short_strike].copy()
        if cands.empty:
            return None
        target_k = max(0.5, short_strike - target_width)

    cands["dist"] = (cands["strike"] - target_k).abs()
    return cands.loc[cands["dist"].idxmin()]


def _risk_delta_target(risk_profile: str) -> float:
    if risk_profile == "Conservative (Safer)":
        return 0.16
    if risk_profile == "Growth (Higher Income)":
        return 0.30
    return 0.22


def _prob_above(spot: float, level: float, drift_pct: float, sigma: float, t_years: float) -> float:
    sd = max(sigma * math.sqrt(max(t_years, 1e-4)), 1e-4)
    expected = spot * (1.0 + drift_pct / 100.0)
    z = (math.log(max(level, 1e-4) / max(expected, 1e-4))) / sd
    return float(np.clip(1.0 - norm.cdf(z), 0.05, 0.97))


def _annualize(roc_pct: float, dte: int) -> tuple[float, float]:
    d = max(dte, 1)
    weekly = roc_pct * (7.0 / d)
    annual = min(roc_pct * (365.0 / d), 250.0)
    return round(weekly, 2), round(annual, 1)


def build_income_strategies(
    snapshot: StockSnapshot,
    chain: OptionChainData,
    prediction: MarketPrediction,
    budget: float = 5000.0,
    risk_profile: str = "Balanced",
    risk_free_rate: float = 0.045,
) -> List[OptionStrategyPlan]:
    """Build and rank the 4 core repetitive income option strategies with exact Entry/Exit/Breakeven numbers."""
    spot = snapshot.price
    dte = max(chain.dte, 1)
    t_years = dte / 365.0
    target_delta = _risk_delta_target(risk_profile)
    eff_vol = max(snapshot.hist_vol_20d, 0.14)
    drift_pct = prediction.expected_return_pct

    earnings_warning = None
    if snapshot.next_earnings_days is not None and snapshot.next_earnings_days <= dte + 2:
        earnings_warning = (
            f"Earnings in ~{snapshot.next_earnings_days} day(s) falls inside this trade! "
            "Beginners should usually skip or shrink size during earnings week."
        )

    price_grid = np.linspace(spot * 0.80, spot * 1.20, 100)
    plans: List[OptionStrategyPlan] = []

    # ------------------------------------------------------------------
    # 1. BULL PUT CREDIT SPREAD (Defined-Risk, Small-Account Friendly!)
    # ------------------------------------------------------------------
    short_put = _pick_strike_by_delta(chain.puts, spot, t_years, risk_free_rate, eff_vol, False, target_delta)
    if short_put is not None:
        long_put = _pick_protection_wing(chain.puts, float(short_put["strike"]), spot, is_call=False, target_width_pct=0.03)
        if long_put is not None:
            k_short = float(short_put["strike"])
            k_long = float(long_put["strike"])
            width = max(0.5, k_short - k_long)
            p_short = _mid_price(short_put)
            p_long = _mid_price(long_put)
            net_credit_ps = max(0.05, round(p_short - p_long, 2))
            if net_credit_ps >= width * 0.85:
                net_credit_ps = round(width * 0.25, 2)

            max_risk_per_contract = max(10.0, (width - net_credit_ps) * 100.0)
            credit_per_contract = net_credit_ps * 100.0
            max_affordable = int(budget // max_risk_per_contract)
            contracts = max(1, min(max_affordable, max(1, int((budget * 0.40) // max_risk_per_contract)))) if max_affordable >= 1 else 1
            fits_budget = max_risk_per_contract <= budget

            total_credit = round(credit_per_contract * contracts, 2)
            total_risk = round(max_risk_per_contract * contracts, 2)
            breakeven = round(k_short - net_credit_ps, 2)
            cushion_pct = round(((spot - breakeven) / spot) * 100.0, 2)

            # Clear Entry & Exit targets (50% early profit-take, 2.0x credit stop-loss capped at max_risk)
            tp_buyback = round(net_credit_ps * 0.50, 2)
            tp_dollars = round((net_credit_ps - tp_buyback) * 100.0 * contracts, 2)
            sl_buyback = round(min(width, net_credit_ps * 2.5), 2)
            sl_dollars = round(min(total_risk, (sl_buyback - net_credit_ps) * 100.0 * contracts), 2)
            sl_stock_trigger = round(k_short, 2)

            roc_pct = round((total_credit / total_risk) * 100.0, 2)
            weekly_pct, annual_pct = _annualize(roc_pct, dte)

            iv_s = _row_iv(short_put, eff_vol)
            iv_l = _row_iv(long_put, eff_vol)
            g_s = bs_greeks(spot, k_short, t_years, risk_free_rate, iv_s, False)
            g_l = bs_greeks(spot, k_long, t_years, risk_free_rate, iv_l, False)
            net_delta = round((-g_s["delta"] + g_l["delta"]) * contracts, 3)
            daily_theta = round((-g_s["theta_daily"] + g_l["theta_daily"]) * 100.0 * contracts, 2)

            pop = round(_prob_above(spot, breakeven, drift_pct, iv_s, t_years) * 100.0, 1)
            pnl_curve = [
                round((net_credit_ps - max(0.0, k_short - p) + max(0.0, k_long - p)) * 100.0 * contracts, 2)
                for p in price_grid
            ]
            dir_bonus = 10.0 if prediction.prob_up >= 0.52 else (-10.0 if prediction.prob_up < 0.44 else 2.0)
            safety_score = float(np.clip(pop * 0.72 + min(weekly_pct * 4.5, 22.0) + dir_bonus - (15.0 if earnings_warning else 0.0), 20.0, 96.0))

            plans.append(
                OptionStrategyPlan(
                    strategy_id="bull_put_spread",
                    strategy_name="Bull Put Credit Spread",
                    short_label="Defined-Risk Income",
                    symbol=snapshot.symbol,
                    expiration=chain.expiration,
                    dte=dte,
                    contracts=contracts,
                    upfront_cash=total_credit,
                    max_profit=total_credit,
                    max_risk=total_risk,
                    capital_required=total_risk,
                    return_on_capital_pct=roc_pct,
                    weekly_yield_pct=weekly_pct,
                    annualized_yield_pct=annual_pct,
                    prob_of_profit_pct=pop,
                    breakeven_low=breakeven,
                    breakeven_high=None,
                    entry_stock_price=spot,
                    entry_credit_per_share=net_credit_ps,
                    take_profit_buyback_per_share=tp_buyback,
                    take_profit_dollars=tp_dollars,
                    stop_loss_buyback_per_share=sl_buyback,
                    stop_loss_dollars=sl_dollars,
                    stop_loss_stock_trigger=sl_stock_trigger,
                    breakeven_cushion_pct=cushion_pct,
                    net_delta=net_delta,
                    daily_theta_dollars=max(0.5, daily_theta),
                    safety_score=round(safety_score, 1),
                    risk_level="Low Risk" if pop >= 76 else ("Moderate Risk" if pop >= 66 else "Higher Risk"),
                    risk_color="#00D4AA" if pop >= 76 else ("#F59E0B" if pop >= 66 else "#EF4444"),
                    fits_budget=fits_budget,
                    is_recommended=False,
                    plain_tagline=f"Collect ${total_credit:,.0f} upfront — win as long as {snapshot.symbol} stays above ${k_short:,.2f}.",
                    contract_summary=f"Sell {contracts}× ${k_short:,.2f} Put / Buy {contracts}× ${k_long:,.2f} Put (Exp {chain.expiration})",
                    what_you_do=(
                        f"Enter by selling {contracts} Put contract(s) at **${k_short:,.2f}** and buying a protective Put at "
                        f"**${k_long:,.2f}** for a net credit of **${net_credit_ps:.2f}/share (${total_credit:,.2f} total)** — expiring {chain.expiration}."
                    ),
                    how_you_make_money=(
                        f"You collect **${total_credit:,.2f} upfront**. {snapshot.symbol} can rise, stay flat, or drop up to **{cushion_pct:.1f}%** "
                        f"(to **${breakeven:,.2f}**) and you still avoid losing money! Close early at **${tp_buyback:.2f}/sh** to lock in **+${tp_dollars:,.2f}** fast, "
                        f"or hold above **${k_short:,.2f}** for full **+${total_credit:,.2f}**."
                    ),
                    what_is_your_risk=(
                        f"Your smart stop-loss exit is if {snapshot.symbol} hits **${sl_stock_trigger:,.2f}** (or option rises to **${sl_buyback:.2f}/sh**), "
                        f"limiting loss near **-${sl_dollars:,.2f}**. Even in a worst-case crash, your insurance Put caps max loss strictly at **-${total_risk:,.2f}**."
                    ),
                    repeatable_playbook=[
                        f"**Entry:** Sell spread for **${net_credit_ps:.2f}/sh** (+${total_credit:,.2f} cash upfront).",
                        f"**Take-Profit Exit (50% Rule):** Set a limit buy-to-close order at **${tp_buyback:.2f}/sh** to lock in **+${tp_dollars:,.2f}** automatically.",
                        f"**Stop-Loss Exit:** Exit early if {snapshot.symbol} drops to **${sl_stock_trigger:,.2f}** or spread hits **${sl_buyback:.2f}/sh** (~-${sl_dollars:,.2f} cap).",
                    ],
                    fit_reason=(
                        f"Best when the stock is steady or rising (odds of not dropping sharply: {prediction.prob_flat_or_up*100:.0f}%) "
                        f"and you want hard-capped risk with clear entry & exit prices."
                    ),
                    earnings_warning=earnings_warning,
                    payoff_prices=[round(float(p), 2) for p in price_grid],
                    payoff_pnl=pnl_curve,
                )
            )

    # ------------------------------------------------------------------
    # 2. IRON CONDOR (Neutral Range-Bound Income — Win When Stock Stays Calm!)
    # ------------------------------------------------------------------
    short_call = _pick_strike_by_delta(chain.calls, spot, t_years, risk_free_rate, eff_vol, True, target_delta)
    if short_put is not None and short_call is not None:
        long_put = _pick_protection_wing(chain.puts, float(short_put["strike"]), spot, is_call=False, target_width_pct=0.03)
        long_call = _pick_protection_wing(chain.calls, float(short_call["strike"]), spot, is_call=True, target_width_pct=0.03)
        if long_put is not None and long_call is not None:
            kp_s, kp_l = float(short_put["strike"]), float(long_put["strike"])
            kc_s, kc_l = float(short_call["strike"]), float(long_call["strike"])
            put_credit = max(0.03, _mid_price(short_put) - _mid_price(long_put))
            call_credit = max(0.03, _mid_price(short_call) - _mid_price(long_call))
            net_credit_ps = round(put_credit + call_credit, 2)
            max_wing = max(kp_s - kp_l, kc_l - kc_s, 1.0)
            if net_credit_ps >= max_wing * 0.85:
                net_credit_ps = round(max_wing * 0.35, 2)

            max_risk_per_contract = max(10.0, (max_wing - net_credit_ps) * 100.0)
            credit_per_contract = net_credit_ps * 100.0
            max_affordable = int(budget // max_risk_per_contract)
            contracts = max(1, min(max_affordable, max(1, int((budget * 0.35) // max_risk_per_contract)))) if max_affordable >= 1 else 1
            fits_budget = max_risk_per_contract <= budget

            total_credit = round(credit_per_contract * contracts, 2)
            total_risk = round(max_risk_per_contract * contracts, 2)
            be_low = round(kp_s - net_credit_ps, 2)
            be_high = round(kc_s + net_credit_ps, 2)
            cushion_pct = round(min((spot - be_low) / spot, (be_high - spot) / spot) * 100.0, 2)

            tp_buyback = round(net_credit_ps * 0.50, 2)
            tp_dollars = round((net_credit_ps - tp_buyback) * 100.0 * contracts, 2)
            sl_buyback = round(min(max_wing, net_credit_ps * 2.2), 2)
            sl_dollars = round(min(total_risk, (sl_buyback - net_credit_ps) * 100.0 * contracts), 2)
            sl_stock_trigger = round(kp_s, 2)

            roc_pct = round((total_credit / total_risk) * 100.0, 2)
            weekly_pct, annual_pct = _annualize(roc_pct, dte)

            iv_avg = (_row_iv(short_put, eff_vol) + _row_iv(short_call, eff_vol)) / 2.0
            p_above_low = _prob_above(spot, be_low, drift_pct, iv_avg, t_years)
            p_above_high = _prob_above(spot, be_high, drift_pct, iv_avg, t_years)
            pop = round(float(np.clip((p_above_low - p_above_high) * 100.0, 25.0, 92.0)), 1)

            g_ps = bs_greeks(spot, kp_s, t_years, risk_free_rate, iv_avg, False)
            g_pl = bs_greeks(spot, kp_l, t_years, risk_free_rate, iv_avg, False)
            g_cs = bs_greeks(spot, kc_s, t_years, risk_free_rate, iv_avg, True)
            g_cl = bs_greeks(spot, kc_l, t_years, risk_free_rate, iv_avg, True)
            daily_theta = round(((-g_ps["theta_daily"] + g_pl["theta_daily"] - g_cs["theta_daily"] + g_cl["theta_daily"]) * 100.0 * contracts), 2)
            net_delta = round((-g_ps["delta"] + g_pl["delta"] - g_cs["delta"] + g_cl["delta"]) * contracts, 3)

            pnl_curve = [
                round(
                    (
                        net_credit_ps
                        - max(0.0, kp_s - p)
                        + max(0.0, kp_l - p)
                        - max(0.0, p - kc_s)
                        + max(0.0, p - kc_l)
                    )
                    * 100.0
                    * contracts,
                    2,
                )
                for p in price_grid
            ]

            calm_bonus = 12.0 if 0.44 <= prediction.prob_up <= 0.57 else -4.0
            safety_score = float(np.clip(pop * 0.72 + min(weekly_pct * 4.0, 22.0) + calm_bonus - (18.0 if earnings_warning else 0.0), 20.0, 95.0))

            plans.append(
                OptionStrategyPlan(
                    strategy_id="iron_condor",
                    strategy_name="Iron Condor (Range Income)",
                    short_label="Sideways / Calm Income",
                    symbol=snapshot.symbol,
                    expiration=chain.expiration,
                    dte=dte,
                    contracts=contracts,
                    upfront_cash=total_credit,
                    max_profit=total_credit,
                    max_risk=total_risk,
                    capital_required=total_risk,
                    return_on_capital_pct=roc_pct,
                    weekly_yield_pct=weekly_pct,
                    annualized_yield_pct=annual_pct,
                    prob_of_profit_pct=pop,
                    breakeven_low=be_low,
                    breakeven_high=be_high,
                    entry_stock_price=spot,
                    entry_credit_per_share=net_credit_ps,
                    take_profit_buyback_per_share=tp_buyback,
                    take_profit_dollars=tp_dollars,
                    stop_loss_buyback_per_share=sl_buyback,
                    stop_loss_dollars=sl_dollars,
                    stop_loss_stock_trigger=sl_stock_trigger,
                    breakeven_cushion_pct=cushion_pct,
                    net_delta=net_delta,
                    daily_theta_dollars=max(0.5, daily_theta),
                    safety_score=round(safety_score, 1),
                    risk_level="Low Risk" if pop >= 72 else ("Moderate Risk" if pop >= 60 else "Higher Risk"),
                    risk_color="#00D4AA" if pop >= 72 else ("#F59E0B" if pop >= 60 else "#EF4444"),
                    fits_budget=fits_budget,
                    is_recommended=False,
                    plain_tagline=f"Collect ${total_credit:,.0f} if {snapshot.symbol} stays between ${be_low:,.2f} and ${be_high:,.2f}.",
                    contract_summary=f"Put Spread ${kp_l:,.0f}/${kp_s:,.0f} + Call Spread ${kc_s:,.0f}/${kc_l:,.0f} (Exp {chain.expiration})",
                    what_you_do=(
                        f"Enter by selling a Put at **${kp_s:,.2f}** and a Call at **${kc_s:,.2f}** (with outer insurance at ${kp_l:,.2f} & ${kc_l:,.2f}) "
                        f"for **${net_credit_ps:.2f}/sh (+${total_credit:,.2f} cash)**."
                    ),
                    how_you_make_money=(
                        f"Collect **${total_credit:,.2f} upfront**. Close early at **${tp_buyback:.2f}/sh** for **+${tp_dollars:,.2f}** profit, "
                        f"or keep 100% (**+${total_credit:,.2f}**) if {snapshot.symbol} stays between **${kp_s:,.2f}** and **${kc_s:,.2f}**."
                    ),
                    what_is_your_risk=(
                        f"Exit early if {snapshot.symbol} breaks below **${kp_s:,.2f}** or above **${kc_s:,.2f}** (condor reaches **${sl_buyback:.2f}/sh**, "
                        f"capping loss near **-${sl_dollars:,.2f}**). Absolute max loss outside **${be_low:,.2f}–${be_high:,.2f}** is **-${total_risk:,.2f}**."
                    ),
                    repeatable_playbook=[
                        f"**Entry:** Open both spreads together for **${net_credit_ps:.2f}/sh** (+${total_credit:,.2f}).",
                        f"**Take-Profit Exit (50% Rule):** Set automatic buy-to-close at **${tp_buyback:.2f}/sh** (+${tp_dollars:,.2f} profit).",
                        f"**Stop-Loss Exit:** Close if price breaches **${kp_s:,.2f}** or **${kc_s:,.2f}** (~-${sl_dollars:,.2f} loss cap).",
                    ],
                    fit_reason="Ideal when the stock is calm or range-bound and implied volatility pays generously on both sides.",
                    earnings_warning=earnings_warning,
                    payoff_prices=[round(float(p), 2) for p in price_grid],
                    payoff_pnl=pnl_curve,
                )
            )

    # ------------------------------------------------------------------
    # 3. CASH-SECURED PUT (The Wheel Part 1 — Get Paid to Wait for a Discount)
    # ------------------------------------------------------------------
    if short_put is not None:
        k_put = float(short_put["strike"])
        prem_ps = _mid_price(short_put)
        cap_per_contract = round(k_put * 100.0, 2)
        credit_per_contract = round(prem_ps * 100.0, 2)
        max_affordable = int(budget // cap_per_contract)
        contracts = max(1, max_affordable) if max_affordable >= 1 else 1
        fits_budget = cap_per_contract <= budget

        total_credit = round(credit_per_contract * contracts, 2)
        total_cap = round(cap_per_contract * contracts, 2)
        breakeven = round(k_put - prem_ps, 2)
        cushion_pct = round(((spot - breakeven) / spot) * 100.0, 2)
        total_max_risk = round(breakeven * 100.0 * contracts, 2)

        tp_buyback = round(prem_ps * 0.50, 2)
        tp_dollars = round((prem_ps - tp_buyback) * 100.0 * contracts, 2)
        sl_buyback = round(prem_ps * 2.5, 2)
        sl_dollars = round(min(total_max_risk, (sl_buyback - prem_ps) * 100.0 * contracts), 2)
        sl_stock_trigger = round(breakeven, 2)

        roc_pct = round((total_credit / total_cap) * 100.0, 2)
        weekly_pct, annual_pct = _annualize(roc_pct, dte)

        iv_p = _row_iv(short_put, eff_vol)
        g_p = bs_greeks(spot, k_put, t_years, risk_free_rate, iv_p, False)
        pop = round(_prob_above(spot, breakeven, drift_pct, iv_p, t_years) * 100.0, 1)

        pnl_curve = [
            round((prem_ps - max(0.0, k_put - p)) * 100.0 * contracts, 2)
            for p in price_grid
        ]
        budget_penalty = 0.0 if fits_budget else -14.0
        dir_bonus = 8.0 if prediction.prob_up >= 0.50 else -6.0
        safety_score = float(np.clip(pop * 0.78 + min(weekly_pct * 6.0, 18.0) + dir_bonus + budget_penalty - (12.0 if earnings_warning else 0.0), 18.0, 94.0))

        plans.append(
            OptionStrategyPlan(
                strategy_id="cash_secured_put",
                strategy_name="Cash-Secured Put (The Wheel)",
                short_label="Get Paid to Buy Dips",
                symbol=snapshot.symbol,
                expiration=chain.expiration,
                dte=dte,
                contracts=contracts,
                upfront_cash=total_credit,
                max_profit=total_credit,
                max_risk=total_max_risk,
                capital_required=total_cap,
                return_on_capital_pct=roc_pct,
                weekly_yield_pct=weekly_pct,
                annualized_yield_pct=annual_pct,
                prob_of_profit_pct=pop,
                breakeven_low=breakeven,
                breakeven_high=None,
                entry_stock_price=spot,
                entry_credit_per_share=prem_ps,
                take_profit_buyback_per_share=tp_buyback,
                take_profit_dollars=tp_dollars,
                stop_loss_buyback_per_share=sl_buyback,
                stop_loss_dollars=sl_dollars,
                stop_loss_stock_trigger=sl_stock_trigger,
                breakeven_cushion_pct=cushion_pct,
                net_delta=round(-g_p["delta"] * contracts, 3),
                daily_theta_dollars=round(max(0.5, -g_p["theta_daily"] * 100.0 * contracts), 2),
                safety_score=round(safety_score, 1),
                risk_level="Low Risk" if pop >= 78 else "Moderate Risk",
                risk_color="#00D4AA" if pop >= 78 else "#F59E0B",
                fits_budget=fits_budget,
                is_recommended=False,
                plain_tagline=f"Earn ${total_credit:,.0f} upfront — or buy {snapshot.symbol} at a {cushion_pct:.1f}% discount (${breakeven:,.2f}).",
                contract_summary=f"Sell {contracts}× ${k_put:,.2f} Put (Exp {chain.expiration}) backed by ${total_cap:,.0f} cash",
                what_you_do=(
                    f"Set aside **${total_cap:,.0f}** cash and sell {contracts} Put contract(s) at **${k_put:,.2f}** "
                    f"for **${prem_ps:.2f}/sh (+${total_credit:,.2f} upfront)**."
                ),
                how_you_make_money=(
                    f"Pocket **${total_credit:,.2f}** immediately. Close early at **${tp_buyback:.2f}/sh** for **+${tp_dollars:,.2f}**, "
                    f"or let it expire above **${k_put:,.2f}** to keep full **+${total_credit:,.2f}** and repeat!"
                ),
                what_is_your_risk=(
                    f"Breakeven is **${breakeven:,.2f}**. If {snapshot.symbol} drops below **${k_put:,.2f}**, you either buy back the Put near "
                    f"**${sl_buyback:.2f}/sh** (~-${sl_dollars:,.2f} exit) OR take 100 shares at net cost **${breakeven:,.2f}** and sell Covered Calls."
                ),
                repeatable_playbook=[
                    f"**Entry:** Sell ${k_put:,.2f} Put for **${prem_ps:.2f}/sh** (+${total_credit:,.2f} cash).",
                    f"**Take-Profit Exit:** Buy back at **${tp_buyback:.2f}/sh** (+${tp_dollars:,.2f}) or hold above **${k_put:,.2f}**.",
                    f"**Defense / Exit:** If price breaks **${breakeven:,.2f}**, exit at **${sl_buyback:.2f}/sh** (~-${sl_dollars:,.2f}) or take shares & sell Covered Calls.",
                ],
                fit_reason="Beginner favorite for building steady compounding wealth on stocks you'd genuinely like to own.",
                earnings_warning=earnings_warning,
                payoff_prices=[round(float(p), 2) for p in price_grid],
                payoff_pnl=pnl_curve,
            )
        )

    # ------------------------------------------------------------------
    # 4. COVERED CALL (Rent Out Shares You Own for Recurring Cash)
    # ------------------------------------------------------------------
    if short_call is not None:
        k_call = float(short_call["strike"])
        prem_ps = _mid_price(short_call)
        cap_per_contract = round(spot * 100.0, 2)
        credit_per_contract = round(prem_ps * 100.0, 2)
        max_affordable = int(budget // cap_per_contract)
        contracts = max(1, max_affordable) if max_affordable >= 1 else 1
        fits_budget = cap_per_contract <= budget

        total_credit = round(credit_per_contract * contracts, 2)
        total_cap = round(cap_per_contract * contracts, 2)
        upside_gain = max(0.0, (k_call - spot) * 100.0 * contracts)
        max_profit = round(total_credit + upside_gain, 2)
        breakeven = round(spot - prem_ps, 2)
        cushion_pct = round(((spot - breakeven) / spot) * 100.0, 2)
        total_max_risk = round(breakeven * 100.0 * contracts, 2)

        tp_buyback = round(prem_ps * 0.50, 2)
        tp_dollars = round((prem_ps - tp_buyback) * 100.0 * contracts, 2)
        # Practical stock stop-loss (e.g. 5% below breakeven)
        sl_stock_trigger = round(breakeven * 0.95, 2)
        sl_buyback = round(prem_ps * 2.0, 2)
        sl_dollars = round((breakeven - sl_stock_trigger) * 100.0 * contracts, 2)

        roc_pct = round((total_credit / total_cap) * 100.0, 2)
        weekly_pct, annual_pct = _annualize(roc_pct, dte)

        iv_c = _row_iv(short_call, eff_vol)
        g_c = bs_greeks(spot, k_call, t_years, risk_free_rate, iv_c, True)
        pop = round(_prob_above(spot, breakeven, drift_pct, iv_c, t_years) * 100.0, 1)

        pnl_curve = [
            round(((min(p, k_call) - spot) + prem_ps) * 100.0 * contracts, 2)
            for p in price_grid
        ]
        budget_penalty = 0.0 if fits_budget else -14.0
        safety_score = float(np.clip(pop * 0.72 + min(weekly_pct * 5.5, 18.0) + budget_penalty - (10.0 if earnings_warning else 0.0), 18.0, 92.0))

        plans.append(
            OptionStrategyPlan(
                strategy_id="covered_call",
                strategy_name="Covered Call (Share Rent)",
                short_label="Income on Owned Shares",
                symbol=snapshot.symbol,
                expiration=chain.expiration,
                dte=dte,
                contracts=contracts,
                upfront_cash=total_credit,
                max_profit=max_profit,
                max_risk=total_max_risk,
                capital_required=total_cap,
                return_on_capital_pct=roc_pct,
                weekly_yield_pct=weekly_pct,
                annualized_yield_pct=annual_pct,
                prob_of_profit_pct=pop,
                breakeven_low=breakeven,
                breakeven_high=k_call,
                entry_stock_price=spot,
                entry_credit_per_share=prem_ps,
                take_profit_buyback_per_share=tp_buyback,
                take_profit_dollars=tp_dollars,
                stop_loss_buyback_per_share=sl_buyback,
                stop_loss_dollars=sl_dollars,
                stop_loss_stock_trigger=sl_stock_trigger,
                breakeven_cushion_pct=cushion_pct,
                net_delta=round((1.0 - g_c["delta"]) * contracts, 3),
                daily_theta_dollars=round(max(0.5, -g_c["theta_daily"] * 100.0 * contracts), 2),
                safety_score=round(safety_score, 1),
                risk_level="Low Risk" if pop >= 65 else "Moderate Risk",
                risk_color="#00D4AA" if pop >= 65 else "#F59E0B",
                fits_budget=fits_budget,
                is_recommended=False,
                plain_tagline=f"Earn ${total_credit:,.0f} cash rent in {dte} days (up to ${max_profit:,.0f} if shares hit ${k_call:,.2f}).",
                contract_summary=f"Own {contracts*100} shares + Sell {contracts}× ${k_call:,.2f} Call (Exp {chain.expiration})",
                what_you_do=(
                    f"Hold {contracts*100} shares at **${spot:,.2f}** (~${total_cap:,.0f}) and sell {contracts} Call(s) "
                    f"at **${k_call:,.2f}** for **${prem_ps:.2f}/sh (+${total_credit:,.2f} cash)**."
                ),
                how_you_make_money=(
                    f"Keep **${total_credit:,.2f} cash** immediately (close Call early at **${tp_buyback:.2f}/sh** for **+${tp_dollars:,.2f}**), "
                    f"or earn up to **+${max_profit:,.2f}** total if {snapshot.symbol} rises to **${k_call:,.2f}**!"
                ),
                what_is_your_risk=(
                    f"Breakeven is **${breakeven:,.2f}**. If shares drop to the **${sl_stock_trigger:,.2f}** warning level, "
                    f"exiting limits loss near **-${sl_dollars:,.2f}**. Upside profit is capped above **${k_call:,.2f}**."
                ),
                repeatable_playbook=[
                    f"**Entry:** Hold shares & sell ${k_call:,.2f} Call for **${prem_ps:.2f}/sh** (+${total_credit:,.2f}).",
                    f"**Take-Profit Exit:** Buy back Call at **${tp_buyback:.2f}/sh** (+${tp_dollars:,.2f}) or get called away at **${k_call:,.2f}** (+${max_profit:,.2f}).",
                    f"**Stop-Loss Exit:** Protect shares if price falls below **${sl_stock_trigger:,.2f}** (~-${sl_dollars:,.2f} loss cap).",
                ],
                fit_reason="Best if you own shares and want a recurring weekly paycheck with clear profit & protection levels.",
                earnings_warning=earnings_warning,
                payoff_prices=[round(float(p), 2) for p in price_grid],
                payoff_pnl=pnl_curve,
            )
        )

    plans.sort(key=lambda p: (p.fits_budget, p.safety_score, p.weekly_yield_pct), reverse=True)
    if plans:
        plans[0].is_recommended = True
    return plans
