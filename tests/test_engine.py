"""Comprehensive verification suite for IncomePulse core engine & charts."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.macro_news import fetch_live_news_and_events, fetch_macro_environment, score_headline
from core.market_data import fetch_option_chain, fetch_stock_snapshot
from core.options_engine import bs_greeks, build_income_strategies
from core.predictor import predict_stock_movement
from core.scanner import opportunities_to_dataframe, run_income_scanner
from ui.charts import (
    build_forecast_chart,
    build_income_projection_chart,
    build_payoff_chart,
    build_scanner_scatter,
    build_sentiment_bar,
    build_vix_gauge,
)


def test_sentiment_scoring():
    pos_score, pos_label = score_headline("Apple beats earnings estimates and surges to record high")
    neg_score, neg_label = score_headline("Company plunges after downgrade and SEC investigation warning")
    assert pos_score > 0 and pos_label == "Positive", f"Expected Positive, got {pos_score}, {pos_label}"
    assert neg_score < 0 and neg_label == "Negative", f"Expected Negative, got {neg_score}, {neg_label}"


def test_black_scholes_greeks():
    g_call = bs_greeks(spot=100.0, strike=105.0, t_years=7 / 365.0, r=0.045, sigma=0.25, is_call=True)
    g_put = bs_greeks(spot=100.0, strike=95.0, t_years=7 / 365.0, r=0.045, sigma=0.25, is_call=False)
    assert 0.0 < g_call["delta"] < 0.5
    assert -0.5 < g_put["delta"] < 0.0
    assert g_call["theta_daily"] < 0 and g_put["theta_daily"] < 0


def test_full_pipeline_and_charts():
    macro = fetch_macro_environment()
    assert macro.vix > 0
    assert macro.weather_label

    snap = fetch_stock_snapshot("SPY", period="1y")
    assert snap.price > 0
    assert snap.fetched_at_et and snap.last_bar_label
    assert len(snap.history) >= 30

    chain = fetch_option_chain("SPY", spot=snap.price, fallback_iv=snap.hist_vol_20d, min_dte=1, max_dte=10, target_dte=7)
    assert not chain.calls.empty and not chain.puts.empty
    assert chain.dte >= 1

    news, avg_sent, _ = fetch_live_news_and_events("SPY", max_items=8)
    assert len(news) >= 1
    assert -1.0 <= avg_sent <= 1.0

    pred = predict_stock_movement(snap, horizon_days=chain.dte, news_sentiment=avg_sent, vix=macro.vix)
    assert pred.predicted_price > 0
    assert pred.expected_low_1sigma < pred.expected_high_1sigma
    assert 0.0 < pred.prob_up < 1.0
    assert pred.current_trend_label in ("Uptrend", "Sideways / Choppy", "Downtrend")
    assert len(pred.current_trend_dates) == len(pred.current_trend_line) >= 2
    assert len(pred.forecast_dates) == len(pred.forecast_mid) == len(pred.forecast_tech_only) == len(pred.forecast_upper) == len(pred.forecast_lower)

    plans = build_income_strategies(snap, chain, pred, budget=5000.0, risk_profile="Balanced")
    assert len(plans) == 4, f"Expected all 4 strategies, got {len(plans)}"
    assert sum(1 for p in plans if p.is_recommended) == 1
    for p in plans:
        assert p.upfront_cash > 0
        assert p.max_risk > 0
        assert p.entry_credit_per_share > 0
        assert 0 < p.take_profit_dollars <= p.upfront_cash
        assert 0 < p.stop_loss_dollars <= p.max_risk
        assert p.stop_loss_stock_trigger > 0
        assert not math.isnan(p.prob_of_profit_pct)
        assert len(p.payoff_prices) == len(p.payoff_pnl) == 100
        assert len(p.repeatable_playbook) >= 3

    # Verify all Plotly figures build cleanly with traces in BOTH English and Arabic
    for lang in ("en", "ar"):
        fig_forecast = build_forecast_chart(snap, pred, lang=lang, active_plan=plans[0])
        fig_payoff = build_payoff_chart(plans[0], snap.price, pred, lang=lang)
        fig_proj = build_income_projection_chart(plans[0], lang=lang)
        fig_vix = build_vix_gauge(macro, lang=lang)
        fig_sent = build_sentiment_bar(news, lang=lang)
        for fig in (fig_forecast, fig_payoff, fig_proj, fig_vix, fig_sent):
            assert len(fig.data) >= 1

    # Verify multi-ticker scanner on a 3-ticker subset
    opps = run_income_scanner("Weekly Income (5–10 Days)", budget=5000.0, risk_profile="Balanced", macro=macro, symbols=["SPY", "QQQ", "SOFI"])
    assert len(opps) == 3
    df = opportunities_to_dataframe(opps)
    assert len(df) == 3
    assert "Breakeven ($)" in df.columns and "Take-Profit (+$)" in df.columns and "Stop-Loss (-$)" in df.columns
    for lang in ("en", "ar"):
        fig_scan = build_scanner_scatter(opps, "SPY", lang=lang)
        assert len(fig_scan.data) == 3


if __name__ == "__main__":
    test_sentiment_scoring()
    test_black_scholes_greeks()
    test_full_pipeline_and_charts()
    print("ALL ENGINE & CHART TESTS PASSED!")
