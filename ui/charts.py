"""Interactive, smooth Plotly visualizations with full English & Arabic support."""
from __future__ import annotations

from typing import List

import numpy as np
import plotly.graph_objects as go

from core.macro_news import MacroEnvironment, NewsItem
from core.market_data import StockSnapshot
from core.options_engine import OptionStrategyPlan
from core.predictor import MarketPrediction
from core.scanner import ScanOpportunity
from ui.i18n import STRATEGY_NAMES_AR

PLOT_BG = "rgba(15, 22, 38, 0.0)"
PAPER_BG = "rgba(15, 22, 38, 0.0)"
GRID_COLOR = "rgba(148, 163, 184, 0.11)"
FONT_COLOR = "#E2E8F0"


def _base_layout(title: str = "", height: int = 410, lang: str = "en") -> dict:
    font_family = "Cairo, Inter, sans-serif" if lang == "ar" else "Inter, sans-serif"
    return dict(
        title=dict(text=title, font=dict(size=16, color="#F8FAFC", family=font_family)),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family=font_family),
        height=height,
        margin=dict(l=16, r=18, t=48 if title else 24, b=24),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#131D31", bordercolor="#334155", font=dict(color="#F8FAFC", size=12, family=font_family)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
    )


def build_scanner_scatter(opps: List[ScanOpportunity], selected_symbol: str, lang: str = "en") -> go.Figure:
    """Interactive bubble chart: Win Probability vs Weekly Income Yield (size = Safety Score)."""
    is_ar = lang == "ar"
    fig = go.Figure()
    if not opps:
        fig.update_layout(**_base_layout("أفضل فرص الدخل" if is_ar else "Top Income Opportunities", height=360, lang=lang))
        return fig

    for o in opps:
        color = "#00D4AA" if o.symbol == selected_symbol else ("#38BDF8" if o.fits_budget else "#94A3B8")
        if o.earnings_soon:
            color = "#F59E0B"
        strat_disp = STRATEGY_NAMES_AR.get(o.strategy_id, (o.best_strategy,))[0] if is_ar else o.best_strategy
        if is_ar:
            hover = (
                f"<b>{o.symbol} — {o.name}</b><br>"
                f"الاستراتيجية: {strat_disp}<br>"
                f"احتمال النجاح: <b>{o.win_prob_pct:.1f}%</b><br>"
                f"العائد الأسبوعي: <b>{o.weekly_yield_pct:.2f}%</b> (~${o.upfront_cash:,.0f} كاش)<br>"
                f"أقصى مخاطرة: ${o.max_risk:,.0f} | المدة: {o.dte} يوم<br>"
                f"درجة الأمان: <b>{o.safety_score:.0f}/100</b><extra></extra>"
            )
        else:
            hover = (
                f"<b>{o.symbol} — {o.name}</b><br>"
                f"Strategy: {strat_disp}<br>"
                f"Win Odds: <b>{o.win_prob_pct:.1f}%</b><br>"
                f"Weekly Yield: <b>{o.weekly_yield_pct:.2f}%</b> (~${o.upfront_cash:,.0f} upfront)<br>"
                f"Max Risk: ${o.max_risk:,.0f} | Days: {o.dte}d<br>"
                f"Safety Score: <b>{o.safety_score:.0f}/100</b><extra></extra>"
            )

        fig.add_trace(
            go.Scatter(
                x=[o.win_prob_pct],
                y=[o.weekly_yield_pct],
                mode="markers+text",
                name=o.symbol,
                text=[o.symbol],
                textposition="top center",
                textfont=dict(size=12, color="#F8FAFC"),
                marker=dict(
                    size=max(14, o.safety_score * 0.42),
                    color=color,
                    opacity=0.92 if o.symbol == selected_symbol else 0.78,
                    line=dict(width=2.5 if o.symbol == selected_symbol else 1.2, color="#FFFFFF" if o.symbol == selected_symbol else "#0B0F19"),
                ),
                hovertemplate=hover,
                showlegend=False,
            )
        )

    fig.add_vrect(
        x0=72,
        x1=96,
        fillcolor="rgba(0, 212, 170, 0.05)",
        line_width=0,
        annotation_text="منطقة الأمان الأعلى" if is_ar else "Higher Safety Zone",
        annotation_position="top right",
        annotation_font_color="#34E8C5",
    )

    title = "خريطة الدخل مقابل الأمان (أعلى اليمين = التوازن الأفضل)" if is_ar else "Income vs. Safety Map (Top-Right = Best Balance)"
    layout = _base_layout(title, height=370, lang=lang)
    layout["hovermode"] = "closest"
    layout["xaxis"]["title"] = "احتمال النجاح التقديري (%)" if is_ar else "Estimated Win Probability (%)"
    layout["yaxis"]["title"] = "العائد النقدي الأسبوعي (%)" if is_ar else "Weekly Cash Yield (%)"
    fig.update_layout(**layout)
    return fig


def build_forecast_chart(
    snapshot: StockSnapshot,
    prediction: MarketPrediction,
    lang: str = "en",
    active_plan: OptionStrategyPlan | None = None,
) -> go.Figure:
    """Interactive Current Trend vs. Predicted Trend (AI + Live News) plot with Entry/Breakeven/Stop levels."""
    is_ar = lang == "ar"
    hist = snapshot.history.tail(45).copy()
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    closes = hist["Close"].round(2).tolist()
    today_date = dates[-1] if dates else prediction.forecast_dates[0]

    fig = go.Figure()

    # 1. Actual recent stock price
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=closes,
            name="سعر السهم الفعلي" if is_ar else "Actual Stock Price",
            mode="lines",
            line=dict(color="#38BDF8", width=2.4),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.08)",
            hovertemplate=("التاريخ: %{x}<br>السعر: <b>$%{y:,.2f}</b><extra></extra>" if is_ar else "Date: %{x}<br>Price: <b>$%{y:,.2f}</b><extra></extra>"),
        )
    )

    # 2. Current 20-day actual trendline (linear slope)
    if prediction.current_trend_dates and prediction.current_trend_line:
        fig.add_trace(
            go.Scatter(
                x=prediction.current_trend_dates,
                y=prediction.current_trend_line,
                name=(
                    f"الاتجاه الحالي ({prediction.current_trend_pct_20d:+.1f}%)"
                    if is_ar
                    else f"Current 20d Trend ({prediction.current_trend_pct_20d:+.1f}%)"
                ),
                mode="lines",
                line=dict(color="#A78BFA", width=3.0),
                hovertemplate=("خط الاتجاه الحالي: <b>$%{y:,.2f}</b><extra></extra>" if is_ar else "Current Trend Line: <b>$%{y:,.2f}</b><extra></extra>"),
            )
        )

    # 3. Forecast confidence range cone (~68%)
    fig.add_trace(
        go.Scatter(
            x=prediction.forecast_dates,
            y=prediction.forecast_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction.forecast_dates,
            y=prediction.forecast_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 212, 170, 0.16)",
            name="النطاق المتوقع (~68%)" if is_ar else "Expected Range (~68%)",
            hovertemplate=("الحد الأدنى المتوقع: $%{y:,.2f}<extra></extra>" if is_ar else "Range Low: $%{y:,.2f}<extra></extra>"),
        )
    )

    # 4. Technical-only baseline forecast (without news shift)
    if prediction.forecast_tech_only and len(prediction.forecast_tech_only) == len(prediction.forecast_dates):
        fig.add_trace(
            go.Scatter(
                x=prediction.forecast_dates,
                y=prediction.forecast_tech_only,
                name="توقع السعر فقط (بدون الأخبار)" if is_ar else "Price-Only Forecast (No News)",
                mode="lines",
                line=dict(color="#94A3B8", width=1.8, dash="dot"),
                hovertemplate=("السعر فقط: $%{y:,.2f}<extra></extra>" if is_ar else "Price-Only: $%{y:,.2f}<extra></extra>"),
            )
        )

    # 5. Bold Predicted Future Trend (AI + Live News)
    pred_color = prediction.direction_color or "#00D4AA"
    fig.add_trace(
        go.Scatter(
            x=prediction.forecast_dates,
            y=prediction.forecast_mid,
            name=(
                f"الاتجاه المتوقع — AI + الأخبار ({prediction.expected_return_pct:+.1f}%)"
                if is_ar
                else f"Predicted Trend — AI + News ({prediction.expected_return_pct:+.1f}%)"
            ),
            mode="lines+markers",
            line=dict(color=pred_color, width=3.4),
            marker=dict(size=6, color=pred_color),
            hovertemplate=("التوقع (AI + الأخبار): <b>$%{y:,.2f}</b><extra></extra>" if is_ar else "Predicted (AI + News): <b>$%{y:,.2f}</b><extra></extra>"),
        )
    )

    # Vertical "Today" divider separating Current Trend vs Future Prediction
    fig.add_vline(
        x=today_date,
        line_color="rgba(226, 232, 240, 0.45)",
        line_dash="dash",
        line_width=1.4,
    )
    fig.add_annotation(
        x=today_date,
        y=1.03,
        yref="paper",
        text=("اليوم ◀ التوقع" if is_ar else "Today ▶ Forecast"),
        showarrow=False,
        font=dict(size=11, color="#CBD5E1"),
    )

    # End-of-forecast target price callout
    if prediction.forecast_dates and prediction.forecast_mid:
        fig.add_annotation(
            x=prediction.forecast_dates[-1],
            y=prediction.forecast_mid[-1],
            text=f"<b>${prediction.predicted_price:,.2f}</b> ({prediction.expected_return_pct:+.1f}%)",
            showarrow=True,
            arrowhead=2,
            arrowcolor=pred_color,
            ax=18,
            ay=-26,
            bgcolor="#131D31",
            bordercolor=pred_color,
            borderwidth=1.2,
            font=dict(size=11, color=pred_color),
        )

    # Overlay Entry, Breakeven & Stop-Loss price levels from the active strategy!
    if active_plan is not None:
        if active_plan.breakeven_low is not None:
            fig.add_hline(
                y=active_plan.breakeven_low,
                line_color="#FBBF24",
                line_dash="dash",
                line_width=1.7,
                annotation_text=(f"🟡 التعادل ${active_plan.breakeven_low:,.2f}" if is_ar else f"🟡 Breakeven ${active_plan.breakeven_low:,.2f}"),
                annotation_position="bottom left",
                annotation_font_color="#FBBF24",
            )
        if active_plan.breakeven_high is not None and active_plan.strategy_id == "iron_condor":
            fig.add_hline(
                y=active_plan.breakeven_high,
                line_color="#FBBF24",
                line_dash="dash",
                line_width=1.7,
                annotation_text=(f"🟡 التعادل العلوي ${active_plan.breakeven_high:,.2f}" if is_ar else f"🟡 Upper BE ${active_plan.breakeven_high:,.2f}"),
                annotation_position="top right",
                annotation_font_color="#FBBF24",
            )
        if active_plan.stop_loss_stock_trigger and abs(active_plan.stop_loss_stock_trigger - (active_plan.breakeven_low or 0)) > 0.25:
            fig.add_hline(
                y=active_plan.stop_loss_stock_trigger,
                line_color="#FB923C",
                line_dash="dot",
                line_width=1.5,
                annotation_text=(f"🟠 وقف الخسارة ${active_plan.stop_loss_stock_trigger:,.2f}" if is_ar else f"🟠 Stop-Loss ${active_plan.stop_loss_stock_trigger:,.2f}"),
                annotation_position="bottom right",
                annotation_font_color="#FB923C",
            )

    title = (
        f"الاتجاه الحالي مقابل المتوقع (AI + الأخبار الحية) — {snapshot.symbol} ({prediction.horizon_days} يوم)"
        if is_ar
        else f"{snapshot.symbol}: Current Trend vs. Predicted Trend (AI + Live News, {prediction.horizon_days}d)"
    )
    layout = _base_layout(title, height=430, lang=lang)
    layout["yaxis"]["tickprefix"] = "$"
    fig.update_layout(**layout)
    return fig


def build_payoff_chart(plan: OptionStrategyPlan, spot: float, prediction: MarketPrediction, lang: str = "en") -> go.Figure:
    """Crystal-clear beginner payoff chart highlighting Entry, Take-Profit, Max Profit, Breakeven & Stop-Loss."""
    is_ar = lang == "ar"
    prices = np.array(plan.payoff_prices, dtype=float)
    pnls = np.array(plan.payoff_pnl, dtype=float)
    profit_y = np.where(pnls >= 0, pnls, 0.0)
    loss_y = np.where(pnls < 0, pnls, 0.0)

    fig = go.Figure()

    fig.add_vrect(
        x0=prediction.expected_low_1sigma,
        x1=prediction.expected_high_1sigma,
        fillcolor="rgba(56, 189, 248, 0.07)",
        line_width=0,
        annotation_text="النطاق المتوقع" if is_ar else "Expected Zone",
        annotation_position="top left",
        annotation_font_color="#7DD3FC",
        annotation_font_size=11,
    )

    fig.add_trace(
        go.Scatter(
            x=prices,
            y=profit_y,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.25)",
            name="منطقة الربح (+)" if is_ar else "Profit Zone (+)",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=prices,
            y=loss_y,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.25)",
            name="منطقة الخسارة (-)" if is_ar else "Loss Zone (-)",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=prices,
            y=pnls,
            mode="lines",
            name="الربح / الخسارة عند الانتهاء" if is_ar else "Profit / Loss at Expiration",
            line=dict(color="#F8FAFC", width=2.9),
            hovertemplate=(
                f"إذا أغلق {plan.symbol} عند <b>$%{{x:,.2f}}</b><br>ربحك / خسارتك: <b>$%{{y:+,.2f}}</b><extra></extra>"
                if is_ar
                else f"If {plan.symbol} ends at <b>$%{{x:,.2f}}</b><br>Your Profit/Loss: <b>$%{{y:+,.2f}}</b><extra></extra>"
            ),
        )
    )

    # Zero line
    fig.add_hline(y=0, line_color="rgba(226, 232, 240, 0.55)", line_width=1.4)

    # Highlight Take-Profit Target (+50%) & Smart Stop-Loss (-$) horizontal levels
    fig.add_hline(
        y=plan.take_profit_dollars,
        line_color="#00D4AA",
        line_dash="dash",
        line_width=1.6,
        annotation_text=(f"🎯 خروج جني الربح +${plan.take_profit_dollars:,.0f}" if is_ar else f"🎯 Take-Profit Exit +${plan.take_profit_dollars:,.0f}"),
        annotation_position="top right",
        annotation_font_color="#34E8C5",
    )
    fig.add_hline(
        y=-plan.stop_loss_dollars,
        line_color="#FB923C",
        line_dash="dash",
        line_width=1.6,
        annotation_text=(f"🟠 وقف الخسارة -${plan.stop_loss_dollars:,.0f}" if is_ar else f"🟠 Stop-Loss Exit -${plan.stop_loss_dollars:,.0f}"),
        annotation_position="bottom right",
        annotation_font_color="#FB923C",
    )

    # Entry Stock Price vertical marker
    fig.add_vline(
        x=spot,
        line_color="#38BDF8",
        line_dash="dash",
        line_width=2.0,
        annotation_text=(f"🔵 الدخول ${spot:,.2f}" if is_ar else f"🔵 Entry ${spot:,.2f}"),
        annotation_position="bottom right",
        annotation_font_color="#38BDF8",
    )

    # Breakeven vertical marker(s)
    if plan.breakeven_low is not None:
        fig.add_vline(
            x=plan.breakeven_low,
            line_color="#FBBF24",
            line_dash="dot",
            line_width=2.0,
            annotation_text=(f"🟡 التعادل ${plan.breakeven_low:,.2f}" if is_ar else f"🟡 Breakeven ${plan.breakeven_low:,.2f}"),
            annotation_position="top right",
            annotation_font_color="#FBBF24",
        )
    if plan.breakeven_high is not None and plan.strategy_id == "iron_condor":
        fig.add_vline(
            x=plan.breakeven_high,
            line_color="#FBBF24",
            line_dash="dot",
            line_width=2.0,
            annotation_text=(f"🟡 التعادل العلوي ${plan.breakeven_high:,.2f}" if is_ar else f"🟡 Upper BE ${plan.breakeven_high:,.2f}"),
            annotation_position="top left",
            annotation_font_color="#FBBF24",
        )

    strat_disp = STRATEGY_NAMES_AR.get(plan.strategy_id, (plan.strategy_name,))[0] if is_ar else plan.strategy_name
    title = (f"خريطة الدخول والخروج والربح والخسارة — {strat_disp}" if is_ar else f"Entry, Exit, Breakeven & Profit/Loss Map — {strat_disp}")
    layout = _base_layout(title, height=420, lang=lang)
    layout["xaxis"]["title"] = (
        f"سعر {plan.symbol} يوم الانتهاء ({plan.expiration})"
        if is_ar
        else f"{plan.symbol} Price on Expiration Day ({plan.expiration})"
    )
    layout["xaxis"]["tickprefix"] = "$"
    layout["yaxis"]["title"] = "ربحك أو خسارتك بالدولار ($)" if is_ar else "Your Dollar Profit / Loss ($)"
    layout["yaxis"]["tickprefix"] = "$"
    fig.update_layout(**layout)
    return fig


def build_income_projection_chart(plan: OptionStrategyPlan, lang: str = "en") -> go.Figure:
    """Visualizes realistic repetitive income potential over 12 cycles (accounting for occasional losses)."""
    is_ar = lang == "ar"
    cycles = list(range(1, 13))
    win_rate = plan.prob_of_profit_pct / 100.0
    avg_win = plan.upfront_cash * 0.65
    avg_loss = min(plan.max_risk, plan.upfront_cash * 1.5)
    expected_per_cycle = win_rate * avg_win - (1.0 - win_rate) * avg_loss

    expected_cum = [round(expected_per_cycle * c, 2) for c in cycles]
    optimistic_cum = [round(plan.upfront_cash * 0.75 * c, 2) for c in cycles]
    cautious_cum = [round((expected_per_cycle * 0.5) * c, 2) for c in cycles]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=optimistic_cum,
            name="سلسلة ناجحة (دون خسائر)" if is_ar else "Smooth Streak (No Losses)",
            mode="lines",
            line=dict(color="#38BDF8", width=1.8, dash="dot"),
            hovertemplate=("الدورة %{x}: $%{y:,.0f}<extra></extra>" if is_ar else "Cycle %{x}: $%{y:,.0f}<extra></extra>"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=expected_cum,
            name="المسار الواقعي المتوقع (مع الخسائر)" if is_ar else "Realistic Expected Path (Includes Losses)",
            mode="lines+markers",
            line=dict(color="#00D4AA", width=3.0),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.14)",
            hovertemplate=("بعد %{x} تكرارات: <b>$%{y:+,.0f}</b><extra></extra>" if is_ar else "After %{x} Repeats: <b>$%{y:+,.0f}</b><extra></extra>"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=cautious_cum,
            name="سوق متقلب / متحفظ" if is_ar else "Conservative / Choppy Market",
            mode="lines",
            line=dict(color="#F59E0B", width=1.8, dash="dash"),
            hovertemplate=("الدورة %{x}: $%{y:+,.0f}<extra></extra>" if is_ar else "Cycle %{x}: $%{y:+,.0f}<extra></extra>"),
        )
    )

    title = (
        f"تقدير تكرار صفقة الـ {plan.dte} أيام 12 مرة"
        if is_ar
        else f"What Repeating This {plan.dte}-Day Trade 12 Times Could Look Like"
    )
    layout = _base_layout(title, height=320, lang=lang)
    layout["xaxis"]["title"] = (f"عدد مرات تكرار صفقة ({plan.dte} يوم)" if is_ar else f"Number of Repeated {plan.dte}-Day Trades")
    layout["yaxis"]["title"] = "الربح التراكمي التقديري ($)" if is_ar else "Cumulative Estimated Profit ($)"
    layout["yaxis"]["tickprefix"] = "$"
    fig.update_layout(**layout)
    return fig


def build_vix_gauge(macro: MacroEnvironment, lang: str = "en") -> go.Figure:
    """Intuitive Market Stress / Fear Meter (VIX)."""
    is_ar = lang == "ar"
    font_family = "Cairo, Inter, sans-serif" if is_ar else "Inter, sans-serif"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=macro.vix,
            title=dict(text="مقياس خوف السوق (VIX)" if is_ar else "Market Fear Gauge (VIX)", font=dict(size=14, color="#E2E8F0", family=font_family)),
            number=dict(font=dict(size=32, color=macro.weather_color)),
            gauge=dict(
                axis=dict(range=[8, 42], tickcolor="#94A3B8"),
                bar=dict(color=macro.weather_color, thickness=0.28),
                bgcolor="rgba(19, 26, 42, 0.6)",
                borderwidth=0,
                steps=[
                    dict(range=[8, 16.5], color="rgba(0, 212, 170, 0.22)"),
                    dict(range=[16.5, 24], color="rgba(245, 158, 11, 0.22)"),
                    dict(range=[24, 42], color="rgba(239, 68, 68, 0.25)"),
                ],
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family=font_family),
        height=230,
        margin=dict(l=18, r=18, t=38, b=10),
    )
    return fig


def build_sentiment_bar(news_items: List[NewsItem], lang: str = "en") -> go.Figure:
    """Shows live public news mood breakdown in one glance."""
    is_ar = lang == "ar"
    pos = sum(1 for n in news_items if n.sentiment_label == "Positive")
    neu = sum(1 for n in news_items if n.sentiment_label == "Neutral")
    neg = sum(1 for n in news_items if n.sentiment_label == "Negative")

    labels = ["إيجابي", "محايد", "سلبي"] if is_ar else ["Positive", "Neutral", "Negative"]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[pos, neu, neg],
            marker_color=["#00D4AA", "#64748B", "#EF4444"],
            text=[pos, neu, neg],
            textposition="auto",
            hovertemplate=("%{x}: <b>%{y}</b> عناوين<extra></extra>" if is_ar else "%{x} Headlines: <b>%{y}</b><extra></extra>"),
        )
    )
    layout = _base_layout("توزيع مزاج الأخبار الحية" if is_ar else "Live News Mood Breakdown", height=230, lang=lang)
    layout["yaxis"]["title"] = "عدد العناوين" if is_ar else "Headlines"
    fig.update_layout(**layout)
    return fig
