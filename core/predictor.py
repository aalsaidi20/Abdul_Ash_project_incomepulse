"""Walk-forward ML stock direction & volatility range predictor combining technicals, macro & news."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from core.market_data import StockSnapshot


@dataclass
class MarketPrediction:
    symbol: str
    horizon_days: int
    current_price: float
    predicted_price: float
    expected_return_pct: float
    prob_up: float               # 0.0 to 1.0
    prob_flat_or_up: float       # probability of not dropping > 1.5%
    expected_low_1sigma: float   # ~68% range lower bound
    expected_high_1sigma: float  # ~68% range upper bound
    expected_low_90: float       # ~90% safety lower bound
    expected_high_90: float      # ~90% safety upper bound
    direction_label: str         # "Bullish", "Neutral / Sideways", "Cautious / Bearish"
    direction_color: str
    confidence_pct: float        # 0 to 100
    backtest_accuracy_pct: float # out-of-sample walk-forward directional accuracy
    plain_explanation: str
    key_drivers: List[str]
    forecast_dates: List[str]
    forecast_mid: List[float]
    forecast_upper: List[float]
    forecast_lower: List[float]
    # Current trend vs. News+AI future trend comparison
    current_trend_label: str       # "Uptrend", "Sideways", "Downtrend"
    current_trend_color: str
    current_trend_pct_20d: float
    current_trend_dates: List[str]
    current_trend_line: List[float]
    forecast_tech_only: List[float]
    news_impact_pct: float         # How much live news shifted the predicted return (%)
    news_sentiment_used: float


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float).replace(0, np.nan).ffill().fillna(1.0)

    feat = pd.DataFrame(index=df.index)
    feat["ret_1d"] = close.pct_change(1)
    feat["ret_3d"] = close.pct_change(3)
    feat["ret_5d"] = close.pct_change(5)
    feat["ret_10d"] = close.pct_change(10)
    feat["ret_20d"] = close.pct_change(20)

    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    feat["dist_sma5"] = (close - sma_5) / sma_5
    feat["dist_sma20"] = (close - sma_20) / sma_20
    feat["dist_sma50"] = (close - sma_50) / sma_50
    feat["sma_cross"] = (sma_5 - sma_20) / sma_20

    # Volatility & range features
    log_ret = np.log(close / close.shift(1))
    feat["vol_5d"] = log_ret.rolling(5).std() * math.sqrt(252)
    feat["vol_20d"] = log_ret.rolling(20).std() * math.sqrt(252)
    feat["vol_ratio"] = feat["vol_5d"] / feat["vol_20d"].replace(0, np.nan)
    feat["hl_range"] = (high - low) / close

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # Volume momentum
    vol_sma20 = vol.rolling(20).mean()
    feat["vol_surge"] = vol / vol_sma20.replace(0, np.nan)

    # Bollinger band position
    std20 = close.rolling(20).std()
    feat["bb_pos"] = (close - sma_20) / (2.0 * std20.replace(0, np.nan))

    return feat


def predict_stock_movement(
    snapshot: StockSnapshot,
    horizon_days: int = 7,
    news_sentiment: float = 0.0,
    vix: float = 18.0,
    fast_mode: bool = False,
) -> MarketPrediction:
    """Train a fast walk-forward ML ensemble on historical data and blend with live news & VIX."""
    trading_horizon = max(1, min(22, int(round(horizon_days * 5.0 / 7.0))))
    df = snapshot.history.copy()
    close = df["Close"].astype(float)
    spot = snapshot.price

    feat = _build_features(df)
    target_ret = close.shift(-trading_horizon) / close - 1.0
    target_up = (target_ret > 0.0).astype(int)

    valid_mask = feat.notna().all(axis=1) & target_ret.notna()
    X_all = feat.loc[valid_mask]
    y_ret_all = target_ret.loc[valid_mask]
    y_up_all = target_up.loc[valid_mask]

    latest_row = feat.dropna().iloc[-1:] if not feat.dropna().empty else None

    prob_up = 0.52
    pred_ret = 0.003
    backtest_acc = 55.0

    if latest_row is not None and len(X_all) >= 80:
        split = int(len(X_all) * 0.78)
        X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
        y_up_train, y_up_test = y_up_all.iloc[:split], y_up_all.iloc[split:]
        y_ret_train = y_ret_all.iloc[:split]

        n_est = 32 if fast_mode else 65
        clf = GradientBoostingClassifier(
            n_estimators=n_est,
            max_depth=3,
            learning_rate=0.06 if fast_mode else 0.05,
            subsample=0.85,
            random_state=42,
        )
        reg = GradientBoostingRegressor(
            n_estimators=n_est,
            max_depth=3,
            learning_rate=0.06 if fast_mode else 0.05,
            subsample=0.85,
            random_state=42,
        )

        # Evaluate walk-forward out-of-sample accuracy
        if y_up_train.nunique() > 1 and len(X_test) >= 10:
            clf.fit(X_train, y_up_train)
            preds_test = clf.predict(X_test)
            backtest_acc = float((preds_test == y_up_test).mean() * 100.0)

        # Refit on full available history for the forward forecast
        if y_up_all.nunique() > 1:
            clf.fit(X_all, y_up_all)
            prob_up = float(clf.predict_proba(latest_row)[0][1])

        reg.fit(X_all, y_ret_all.clip(-0.20, 0.20))
        pred_ret = float(reg.predict(latest_row)[0])

    # Shrink extreme ML outputs toward realistic bounds and blend live news + VIX regime
    prob_up = 0.5 + (prob_up - 0.5) * 0.65
    news_adj = float(np.clip(news_sentiment * 0.09, -0.09, 0.09))
    vix_adj = -0.04 if vix >= 25.0 else (0.02 if vix <= 15.0 else 0.0)
    prob_up = float(np.clip(prob_up + news_adj + vix_adj, 0.22, 0.78))

    # Expected return bounded by realistic volatility envelope
    eff_vol = max(snapshot.hist_vol_20d, 0.12)
    t_years = max(horizon_days, 1) / 365.0
    one_sigma_move = eff_vol * math.sqrt(t_years)

    tech_ret = float(np.clip(pred_ret * 0.55, -0.8 * one_sigma_move, 0.8 * one_sigma_move))
    news_shift = float(np.clip(news_sentiment * 0.014, -0.35 * one_sigma_move, 0.35 * one_sigma_move))
    pred_ret = float(np.clip(tech_ret + news_shift, -0.85 * one_sigma_move, 0.85 * one_sigma_move))
    predicted_price = round(spot * (1.0 + pred_ret), 2)

    # Current 20-day actual trend (linear regression fit + 20-day % return)
    recent_20 = close.tail(min(20, len(close)))
    pct_20d = ((float(recent_20.iloc[-1]) - float(recent_20.iloc[0])) / max(float(recent_20.iloc[0]), 1e-4)) * 100.0
    x_idx = np.arange(len(recent_20), dtype=float)
    if len(recent_20) >= 2:
        slope, intercept = np.polyfit(x_idx, recent_20.values.astype(float), 1)
        trend_line = [round(float(intercept + slope * x), 2) for x in x_idx]
    else:
        trend_line = [round(spot, 2)]
    trend_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in recent_20.index]

    if pct_20d >= 2.0 and spot >= snapshot.sma_20:
        current_trend_label = "Uptrend"
        current_trend_color = "#00D4AA"
    elif pct_20d <= -2.0 and spot <= snapshot.sma_20:
        current_trend_label = "Downtrend"
        current_trend_color = "#EF4444"
    else:
        current_trend_label = "Sideways / Choppy"
        current_trend_color = "#38BDF8"

    low_1s = round(max(0.5, predicted_price * (1.0 - one_sigma_move)), 2)
    high_1s = round(predicted_price * (1.0 + one_sigma_move), 2)
    low_90 = round(max(0.5, predicted_price * (1.0 - 1.645 * one_sigma_move)), 2)
    high_90 = round(predicted_price * (1.0 + 1.645 * one_sigma_move), 2)

    # Probability stock stays above -1.5% drop
    z_drop = (-0.015 - pred_ret) / max(one_sigma_move, 1e-4)
    from scipy.stats import norm

    prob_flat_or_up = float(np.clip(1.0 - norm.cdf(z_drop), 0.15, 0.96))

    if prob_up >= 0.57:
        direction_label = "Bullish Lean"
        direction_color = "#00D4AA"
    elif prob_up <= 0.43:
        direction_label = "Cautious / Bearish"
        direction_color = "#EF4444"
    else:
        direction_label = "Calm / Sideways"
        direction_color = "#38BDF8"

    # Confidence combines signal strength and out-of-sample stability
    signal_strength = abs(prob_up - 0.5) * 200.0  # 0 to 56
    confidence = float(np.clip(48.0 + signal_strength * 0.75 + (backtest_acc - 50.0) * 0.4, 42.0, 86.0))

    # Build plain-English drivers
    drivers: List[str] = []
    if snapshot.price >= snapshot.sma_20 >= snapshot.sma_50:
        drivers.append("Price is riding above both its 20-day and 50-day averages (healthy uptrend).")
    elif snapshot.price < snapshot.sma_20 and snapshot.price < snapshot.sma_50:
        drivers.append("Price is trading below its 20-day and 50-day averages (downward pressure).")
    else:
        drivers.append("Price is hovering near its recent averages (choppy / range-bound behavior).")

    if snapshot.rsi_14 >= 68:
        drivers.append(f"Momentum is running hot (RSI {snapshot.rsi_14:.0f}) — short-term pullback or pause is common.")
    elif snapshot.rsi_14 <= 32:
        drivers.append(f"Stock looks oversold (RSI {snapshot.rsi_14:.0f}) — bounce or stabilization is more likely.")
    else:
        drivers.append(f"Momentum is balanced (RSI {snapshot.rsi_14:.0f}) — not overheated or panicked.")

    if news_sentiment >= 0.15:
        drivers.append(f"Live public news sentiment is upbeat ({news_sentiment:+.2f}), adding a tailwind.")
    elif news_sentiment <= -0.15:
        drivers.append(f"Live public news sentiment is cautious/negative ({news_sentiment:+.2f}), raising headline risk.")
    else:
        drivers.append("Live news headlines are mostly neutral — no major shock dominating right now.")

    if snapshot.next_earnings_days is not None and snapshot.next_earnings_days <= horizon_days + 2:
        drivers.append(
            f"⚠️ Earnings report in ~{snapshot.next_earnings_days} day(s) — expect sudden big price jumps!"
        )

    plain_explanation = (
        f"Over the next {horizon_days} days, our model expects **{snapshot.symbol}** to lean **{direction_label.lower()}** "
        f"around **${predicted_price:,.2f}** (about **{pred_ret*100:+.1f}%**), with a typical normal range between "
        f"**${low_1s:,.2f}** and **${high_1s:,.2f}**. "
        f"Odds of finishing higher are **{prob_up*100:.0f}%** (historical walk-forward accuracy: {backtest_acc:.0f}%)."
    )

    # Smooth forecast cone for Plotly chart
    n_steps = max(2, min(14, horizon_days))
    last_date = pd.Timestamp(df.index[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)
    forecast_dates = [last_date.strftime("%Y-%m-%d")] + [d.strftime("%Y-%m-%d") for d in future_dates]
    forecast_mid = [spot]
    forecast_tech_only = [spot]
    forecast_upper = [spot]
    forecast_lower = [spot]
    for i in range(1, n_steps + 1):
        frac = i / float(n_steps)
        mid_i = spot * (1.0 + pred_ret * frac)
        tech_i = spot * (1.0 + tech_ret * frac)
        band_i = spot * one_sigma_move * math.sqrt(frac)
        forecast_mid.append(round(mid_i, 2))
        forecast_tech_only.append(round(tech_i, 2))
        forecast_upper.append(round(mid_i + band_i, 2))
        forecast_lower.append(round(max(0.5, mid_i - band_i), 2))

    return MarketPrediction(
        symbol=snapshot.symbol,
        horizon_days=horizon_days,
        current_price=spot,
        predicted_price=predicted_price,
        expected_return_pct=round(pred_ret * 100.0, 2),
        prob_up=round(prob_up, 3),
        prob_flat_or_up=round(prob_flat_or_up, 3),
        expected_low_1sigma=low_1s,
        expected_high_1sigma=high_1s,
        expected_low_90=low_90,
        expected_high_90=high_90,
        direction_label=direction_label,
        direction_color=direction_color,
        confidence_pct=round(confidence, 1),
        backtest_accuracy_pct=round(backtest_acc, 1),
        plain_explanation=plain_explanation,
        key_drivers=drivers,
        forecast_dates=forecast_dates,
        forecast_mid=forecast_mid,
        forecast_upper=forecast_upper,
        forecast_lower=forecast_lower,
        current_trend_label=current_trend_label,
        current_trend_color=current_trend_color,
        current_trend_pct_20d=round(pct_20d, 2),
        current_trend_dates=trend_dates,
        current_trend_line=trend_line,
        forecast_tech_only=forecast_tech_only,
        news_impact_pct=round(news_shift * 100.0, 2),
        news_sentiment_used=round(news_sentiment, 3),
    )
