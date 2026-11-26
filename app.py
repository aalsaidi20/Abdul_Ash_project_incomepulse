"""IncomePulse — Bilingual (Arabic / English) AI Stock Prediction & Repetitive Income Options Dashboard."""
from __future__ import annotations

import html
import re
from typing import List

import pandas as pd
import streamlit as st

from core.macro_news import MacroEnvironment, NewsItem, fetch_live_news_and_events, fetch_macro_environment
from core.market_data import OptionChainData, StockSnapshot, fetch_option_chain, fetch_stock_snapshot
from core.options_engine import OptionStrategyPlan, build_income_strategies
from core.predictor import MarketPrediction, predict_stock_movement
from core.scanner import ScanOpportunity, opportunities_to_dataframe, run_income_scanner
from core.universe import CADENCE_CONFIG, INCOME_UNIVERSE, get_ticker_info
from ui.charts import (
    build_forecast_chart,
    build_income_projection_chart,
    build_payoff_chart,
    build_scanner_scatter,
    build_sentiment_bar,
    build_vix_gauge,
)
from ui.i18n import (
    CATEGORY_AR,
    DIRECTION_AR,
    SENTIMENT_AR,
    STRATEGY_NAMES_AR,
    tr,
    tr_cadence,
    tr_cadence_note,
    tr_prediction,
    tr_risk_profile,
    tr_strategy,
    tr_ticker_desc,
    tr_weather,
)
from ui.styles import get_theme_css

st.set_page_config(
    page_title="IncomePulse — AI Stock & Options Income Studio",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached Zero-Cost Live Data Loaders (keeps the dashboard silky fast!)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_macro() -> MacroEnvironment:
    return fetch_macro_environment()


@st.cache_data(ttl=240, show_spinner=False)
def load_scanner(cadence: str, budget: float, risk_profile: str) -> List[ScanOpportunity]:
    macro = load_macro()
    return run_income_scanner(cadence, budget, risk_profile, macro)


@st.cache_data(ttl=180, show_spinner=False)
def load_stock(symbol: str) -> StockSnapshot:
    return fetch_stock_snapshot(symbol, period="1y")


@st.cache_data(ttl=180, show_spinner=False)
def load_chain(symbol: str, spot: float, iv: float, min_dte: int, max_dte: int, target_dte: int, exp: str = "") -> OptionChainData:
    return fetch_option_chain(
        symbol,
        spot=spot,
        fallback_iv=iv,
        min_dte=min_dte,
        max_dte=max_dte,
        target_dte=target_dte,
        chosen_expiration=exp or None,
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_news(symbol: str) -> tuple[List[NewsItem], float, bool]:
    return fetch_live_news_and_events(symbol, max_items=12)


def _md_bold_to_html(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def _kpi_card(label: str, value: str, sub: str, value_color: str = "#F8FAFC") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{html.escape(label)}</div>
        <div class="kpi-value" style="color:{value_color};" dir="ltr">{html.escape(value)}</div>
        <div class="kpi-delta">{html.escape(sub)}</div>
    </div>
    """


def _level_card(css_cls: str, tag: str, big_value: str, sub: str, color: str) -> str:
    return f"""
    <div class="level-card {css_cls}">
        <div class="level-tag" style="color:{color};">{html.escape(tag)}</div>
        <div class="level-big" style="color:{color};" dir="ltr">{html.escape(big_value)}</div>
        <div class="level-sub">{html.escape(sub)}</div>
    </div>
    """


def _open_blueprint(sym: str, strat_id: str) -> None:
    st.session_state["selected_symbol"] = sym
    st.session_state["selected_strategy_id"] = strat_id
    st.session_state["active_view"] = "blueprint"


def _switch_view(view_key: str) -> None:
    st.session_state["active_view"] = view_key


def _localized_scanner_df(opps: List[ScanOpportunity], lang: str) -> pd.DataFrame:
    if lang != "ar":
        return opportunities_to_dataframe(opps)
    return pd.DataFrame(
        [
            {
                "الرمز": o.symbol,
                "الشركة / الصندوق": o.name,
                "السعر ($)": o.price,
                "أفضل استراتيجية": STRATEGY_NAMES_AR.get(o.strategy_id, (o.best_strategy,))[0],
                "الأيام": o.dte,
                "كاش الدخول (+$)": o.upfront_cash,
                "جني الربح (+$)": o.take_profit_dollars,
                "التعادل ($)": o.breakeven_low or o.price,
                "وقف الخسارة (-$)": o.stop_loss_dollars,
                "أقصى خسارة (-$)": o.max_risk,
                "العائد الأسبوعي (%)": o.weekly_yield_pct,
                "نسبة النجاح (%)": o.win_prob_pct,
                "التقييم": o.safety_score,
                "الحالة": "⚠️ أرباح قريبة" if o.earnings_soon else ("✅ يناسب الميزانية" if o.fits_budget else "💰 رأس مال أكبر"),
            }
            for o in opps
        ]
    )


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "selected_symbol" not in st.session_state:
    st.session_state["selected_symbol"] = "SPY"
if "selected_strategy_id" not in st.session_state:
    st.session_state["selected_strategy_id"] = None
if "active_view" not in st.session_state:
    st.session_state["active_view"] = "scanner"
if "lang" not in st.session_state:
    st.session_state["lang"] = "ar"


# ---------------------------------------------------------------------------
# Sidebar — Language Switcher & Simple, Jargon-Free Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    lang_choice = st.radio(
        "🌐 اللغة / Language",
        options=["ar", "en"],
        format_func=lambda code: "العربية (Arabic)" if code == "ar" else "English",
        horizontal=True,
        key="lang",
    )
    lang = lang_choice or "ar"

# Inject LTR or RTL theme CSS immediately after reading language state
st.markdown(get_theme_css(lang), unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### {tr(lang, 'sidebar_title')}")
    st.caption(tr(lang, "sidebar_caption"))

    budget = st.slider(
        tr(lang, "budget_label"),
        min_value=500,
        max_value=100000,
        value=5000,
        step=500,
        format="$%d",
        help=tr(lang, "budget_help"),
    )

    risk_options = ["Conservative (Safer)", "Balanced", "Growth (Higher Income)"]
    risk_profile = st.select_slider(
        tr(lang, "risk_label"),
        options=risk_options,
        value="Balanced",
        format_func=lambda r: tr_risk_profile(r, lang),
        help=tr(lang, "risk_help"),
    )

    cadence_label = st.radio(
        tr(lang, "cadence_label"),
        options=list(CADENCE_CONFIG.keys()),
        index=1,
        format_func=lambda c: tr_cadence(c, lang),
        help=tr(lang, "cadence_help"),
    )
    cadence_cfg = CADENCE_CONFIG[cadence_label]
    st.info(f"**{tr(lang, 'rhythm_tip')}:** {tr_cadence_note(cadence_label, cadence_cfg['risk_note'], lang)}")

    st.divider()

    st.markdown(tr(lang, "inspect_heading"))
    universe_symbols = [t.symbol for t in INCOME_UNIVERSE]
    current_sym = st.session_state["selected_symbol"]
    if current_sym not in universe_symbols:
        universe_symbols = [current_sym] + universe_symbols

    picked = st.selectbox(
        tr(lang, "popular_tickers"),
        options=universe_symbols,
        index=universe_symbols.index(current_sym),
        format_func=lambda s: f"{s} — {get_ticker_info(s).name}",
    )
    if picked != st.session_state["selected_symbol"]:
        st.session_state["selected_symbol"] = picked
        st.session_state["selected_strategy_id"] = None

    custom_ticker = (
        st.text_input(tr(lang, "custom_ticker"), placeholder=tr(lang, "custom_placeholder"))
        .strip()
        .upper()
    )
    if custom_ticker and custom_ticker.isalpha() and len(custom_ticker) <= 6:
        if custom_ticker != st.session_state["selected_symbol"]:
            st.session_state["selected_symbol"] = custom_ticker
            st.session_state["selected_strategy_id"] = None

    st.divider()
    if st.button(tr(lang, "refresh_btn"), width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        f"""
        <div style="margin-top:0.8rem; font-size:0.78rem; color:#94A3B8; line-height:1.6;">
        <strong>{tr(lang, 'free_sources_title')}</strong><br>
        {tr(lang, 'free_sources_list')}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load Macro + Active Ticker Data
# ---------------------------------------------------------------------------
symbol = st.session_state["selected_symbol"]
macro = load_macro()
weather_t = tr_weather(macro, lang)

with st.spinner(tr(lang, "spinner_ticker", symbol=symbol)):
    snapshot = load_stock(symbol)
    news_items, news_sentiment, news_is_live = load_news(symbol)
    chain = load_chain(
        symbol,
        spot=snapshot.price,
        iv=snapshot.hist_vol_20d,
        min_dte=cadence_cfg["min_dte"],
        max_dte=cadence_cfg["max_dte"],
        target_dte=cadence_cfg["target_dte"],
    )
    prediction = predict_stock_movement(
        snapshot,
        horizon_days=chain.dte,
        news_sentiment=news_sentiment,
        vix=macro.vix,
    )
    pred_t = tr_prediction(snapshot, prediction, news_sentiment, lang)
    strategies = build_income_strategies(
        snapshot,
        chain,
        prediction,
        budget=float(budget),
        risk_profile=risk_profile,
        risk_free_rate=max(0.01, macro.treasury_10y / 100.0),
    )


# ---------------------------------------------------------------------------
# Hero Header + Live Market Weather Strip
# ---------------------------------------------------------------------------
live_badge = (
    f'<span class="pill pill-live">{html.escape(tr(lang, "live_badge"))}</span>'
    if (snapshot.is_live or macro.is_live)
    else f'<span class="pill pill-warn">{html.escape(tr(lang, "offline_badge"))}</span>'
)
weather_badge = (
    f'<span class="pill pill-info">{macro.weather_emoji} '
    f'{html.escape(tr(lang, "market_weather_pill"))}: {html.escape(weather_t["label"])}</span>'
)

hero_sub_html = tr(
    lang,
    "hero_sub",
    symbol=html.escape(snapshot.symbol),
    price=snapshot.price,
    change=snapshot.change_pct,
    budget=budget,
    risk=html.escape(tr_risk_profile(risk_profile, lang)),
)

st.markdown(
    f"""
    <div class="hero-banner">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:0.45rem;">
            <div class="hero-title">{html.escape(tr(lang, "app_title"))}</div>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">{live_badge}{weather_badge}</div>
        </div>
        <p class="hero-sub">{hero_sub_html}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Macro Weather Summary Bar
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(_kpi_card(tr(lang, "kpi_vix"), f"{macro.vix:.1f}", weather_t["label"], macro.weather_color), unsafe_allow_html=True)
with m2:
    st.markdown(_kpi_card(tr(lang, "kpi_10y"), f"{macro.treasury_10y:.2f}%", tr(lang, "kpi_fed_sub", fed=macro.fed_funds_rate)), unsafe_allow_html=True)
with m3:
    st.markdown(_kpi_card(tr(lang, "kpi_cpi"), f"{macro.cpi_yoy_pct:.1f}%", tr(lang, "kpi_unemp_sub", unemp=macro.unemployment_rate)), unsafe_allow_html=True)
with m4:
    sent_label = tr(lang, "mood_upbeat") if news_sentiment >= 0.15 else (tr(lang, "mood_cautious") if news_sentiment <= -0.15 else tr(lang, "mood_balanced"))
    sent_color = "#00D4AA" if news_sentiment >= 0.15 else ("#EF4444" if news_sentiment <= -0.15 else "#38BDF8")
    st.markdown(_kpi_card(tr(lang, "kpi_news_mood", symbol=symbol), sent_label, tr(lang, "kpi_news_sub", score=news_sentiment), sent_color), unsafe_allow_html=True)
with m5:
    st.markdown(
        _kpi_card(
            tr(lang, "kpi_ai_forecast", symbol=symbol, dte=chain.dte),
            f"${prediction.predicted_price:,.2f}",
            tr(lang, "kpi_up_odds", dir_label=str(pred_t["direction"]), odds=prediction.prob_up * 100.0),
            prediction.direction_color,
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Interactive View Switcher (1-click smooth jump from Scanner -> Blueprint)
# ---------------------------------------------------------------------------
VIEW_OPTIONS = [
    ("scanner", tr(lang, "nav_scanner")),
    ("blueprint", tr(lang, "nav_blueprint", symbol=symbol)),
    ("forecast", tr(lang, "nav_forecast", symbol=symbol)),
    ("news", tr(lang, "nav_news")),
]
if st.session_state["active_view"] not in ("scanner", "blueprint", "forecast", "news"):
    st.session_state["active_view"] = "scanner"

nav_cols = st.columns(len(VIEW_OPTIONS))
for col, (vkey, vlabel) in zip(nav_cols, VIEW_OPTIONS):
    with col:
        st.button(
            vlabel,
            key=f"nav_{vkey}",
            width="stretch",
            type="primary" if st.session_state["active_view"] == vkey else "secondary",
            on_click=_switch_view,
            args=(vkey,),
        )

st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)
active_view = st.session_state["active_view"]


# ===========================================================================
# VIEW 1: MULTI-TICKER REPETITIVE INCOME SCANNER
# ===========================================================================
if active_view == "scanner":
    st.markdown(tr(lang, "scanner_title"))
    st.caption(
        tr(
            lang,
            "scanner_caption",
            cadence=tr_cadence(cadence_label, lang),
            budget=budget,
            risk=tr_risk_profile(risk_profile, lang),
        )
    )

    with st.spinner(tr(lang, "spinner_scanner")):
        opportunities = load_scanner(cadence_label, float(budget), risk_profile)

    if opportunities:
        top3 = opportunities[:3]
        cols = st.columns(3)
        medals = [tr(lang, "medal_1"), tr(lang, "medal_2"), tr(lang, "medal_3")]
        for idx, opp in enumerate(top3):
            with cols[idx]:
                warn_html = (
                    f'<span class="pill pill-warn">{html.escape(tr(lang, "pill_earnings"))}</span>'
                    if opp.earnings_soon
                    else (
                        f'<span class="pill pill-live">{html.escape(tr(lang, "pill_fits"))}</span>'
                        if opp.fits_budget
                        else f'<span class="pill pill-info">{html.escape(tr(lang, "pill_larger"))}</span>'
                    )
                )
                strat_name_disp = STRATEGY_NAMES_AR.get(opp.strategy_id, (opp.best_strategy,))[0] if lang == "ar" else opp.best_strategy
                pitch_disp = (
                    f"اقبض ${opp.upfront_cash:,.0f} كاش مقدماً خلال {opp.dte} يوم — نسبة النجاح التقديرية {opp.win_prob_pct:.0f}% وبمخاطرة قصوى ${opp.max_risk:,.0f}."
                    if lang == "ar"
                    else opp.plain_pitch
                )
                st.markdown(
                    f"""
                    <div class="kpi-card" style="padding:1.15rem 1.25rem; margin-bottom:0.65rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <span style="font-size:0.78rem; font-weight:700; color:#38BDF8;">{html.escape(medals[idx])}</span>
                            {warn_html}
                        </div>
                        <div style="font-size:1.35rem; font-weight:800; color:#F8FAFC;">
                            {html.escape(opp.symbol)} <span style="font-size:0.9rem; font-weight:500; color:#94A3B8;" dir="ltr">${opp.price:,.2f}</span>
                        </div>
                        <div style="font-size:0.86rem; color:#00D4AA; font-weight:600; margin:0.2rem 0 0.5rem 0;">
                            {html.escape(strat_name_disp)} ({opp.dte}{tr(lang, "days_short")})
                        </div>
                        <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.5; margin-bottom:0.65rem;">
                            {html.escape(pitch_disp)}
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#94A3B8; border-top:1px solid rgba(148,163,184,0.14); padding-top:0.55rem; gap:6px; flex-wrap:wrap;">
                            <span>{html.escape(tr(lang, "card_tp"))}: <strong style="color:#00D4AA;" dir="ltr">+${opp.take_profit_dollars:,.0f}</strong></span>
                            <span>{html.escape(tr(lang, "card_breakeven"))}: <strong style="color:#FBBF24;" dir="ltr">${(opp.breakeven_low or opp.price):,.2f}</strong></span>
                            <span>{html.escape(tr(lang, "card_stop"))}: <strong style="color:#FB923C;" dir="ltr">-${opp.stop_loss_dollars:,.0f}</strong></span>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94A3B8; margin-top:0.35rem;">
                            <span>{html.escape(tr(lang, "card_cash"))}: <strong style="color:#38BDF8;" dir="ltr">+${opp.upfront_cash:,.0f}</strong></span>
                            <span>{html.escape(tr(lang, "card_win_odds"))}: <strong style="color:#F8FAFC;" dir="ltr">{opp.win_prob_pct:.0f}%</strong></span>
                            <span>{html.escape(tr(lang, "card_weekly"))}: <strong style="color:#7CEFD7;" dir="ltr">{opp.weekly_yield_pct:.1f}%</strong></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    tr(lang, "inspect_btn", symbol=opp.symbol),
                    key=f"pick_{opp.symbol}_{idx}",
                    width="stretch",
                    on_click=_open_blueprint,
                    args=(opp.symbol, opp.strategy_id),
                )

        st.plotly_chart(build_scanner_scatter(opportunities, symbol, lang=lang), use_container_width=True)

        st.markdown(tr(lang, "all_scanned_heading"))
        df_scan = _localized_scanner_df(opportunities, lang)
        if lang == "ar":
            col_cfg = {
                "السعر ($)": st.column_config.NumberColumn(format="$%.2f"),
                "كاش الدخول (+$)": st.column_config.NumberColumn(format="+$%.0f"),
                "جني الربح (+$)": st.column_config.NumberColumn(format="+$%.0f"),
                "التعادل ($)": st.column_config.NumberColumn(format="$%.2f"),
                "وقف الخسارة (-$)": st.column_config.NumberColumn(format="-$%.0f"),
                "أقصى خسارة (-$)": st.column_config.NumberColumn(format="-$%.0f"),
                "العائد الأسبوعي (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "نسبة النجاح (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                "التقييم": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            }
        else:
            col_cfg = {
                "Price ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Entry Cash (+$)": st.column_config.NumberColumn(format="+$%.0f"),
                "Take-Profit (+$)": st.column_config.NumberColumn(format="+$%.0f"),
                "Breakeven ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Stop-Loss (-$)": st.column_config.NumberColumn(format="-$%.0f"),
                "Max Loss (-$)": st.column_config.NumberColumn(format="-$%.0f"),
                "Weekly Yield (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Win Odds (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            }
        st.dataframe(df_scan, width="stretch", hide_index=True, column_config=col_cfg)
    else:
        st.warning(tr(lang, "no_scanner_warn"))


# ===========================================================================
# VIEW 2: PLAIN-ENGLISH / ARABIC STRATEGY BLUEPRINT (HOW YOU MAKE MONEY & RISK)
# ===========================================================================
elif active_view == "blueprint":
    if not strategies:
        st.error(tr(lang, "no_strats_err", symbol=symbol))
    else:
        top_left, top_right = st.columns([2.2, 1.0])
        with top_left:
            st.markdown(tr(lang, "blueprint_title", symbol=snapshot.symbol, name=snapshot.name))
            st.caption(tr_ticker_desc(snapshot.symbol, get_ticker_info(snapshot.symbol).plain_description, lang))
        with top_right:
            if chain.available_expirations:
                exp_idx = chain.available_expirations.index(chain.expiration) if chain.expiration in chain.available_expirations else 0
                chosen_exp = st.selectbox(
                    tr(lang, "exp_date_label"),
                    options=chain.available_expirations,
                    index=exp_idx,
                    help=tr(lang, "exp_date_help"),
                )
                if chosen_exp != chain.expiration:
                    chain = load_chain(
                        symbol,
                        spot=snapshot.price,
                        iv=snapshot.hist_vol_20d,
                        min_dte=cadence_cfg["min_dte"],
                        max_dte=cadence_cfg["max_dte"],
                        target_dte=cadence_cfg["target_dte"],
                        exp=chosen_exp,
                    )
                    prediction = predict_stock_movement(snapshot, horizon_days=chain.dte, news_sentiment=news_sentiment, vix=macro.vix)
                    pred_t = tr_prediction(snapshot, prediction, news_sentiment, lang)
                    strategies = build_income_strategies(
                        snapshot, chain, prediction, budget=float(budget), risk_profile=risk_profile, risk_free_rate=max(0.01, macro.treasury_10y / 100.0)
                    )

        strat_names = []
        for s in strategies:
            st_disp = STRATEGY_NAMES_AR.get(s.strategy_id, (s.strategy_name,))[0] if lang == "ar" else s.strategy_name
            strat_names.append(
                f"{'⭐ ' if s.is_recommended else ''}{st_disp} — ${s.upfront_cash:,.0f} {tr(lang, 'cash_short')} ({s.prob_of_profit_pct:.0f}% {tr(lang, 'win_short')})"
            )

        default_idx = 0
        if st.session_state.get("selected_strategy_id"):
            for i, s in enumerate(strategies):
                if s.strategy_id == st.session_state["selected_strategy_id"]:
                    default_idx = i
                    break

        chosen_idx = st.radio(
            tr(lang, "choose_strat_label"),
            options=list(range(len(strategies))),
            index=default_idx,
            format_func=lambda i: strat_names[i],
            horizontal=True,
        )
        plan: OptionStrategyPlan = strategies[chosen_idx]
        plan_t = tr_strategy(plan, snapshot.price, prediction, lang)

        if plan_t["earnings_warning"]:
            st.warning(f"{tr(lang, 'earnings_alert_prefix')} {plan_t['earnings_warning']}")
        if not plan.fits_budget:
            st.info(
                tr(
                    lang,
                    "budget_headsup",
                    strat=str(plan_t["name"]),
                    symbol=plan.symbol,
                    cap=plan.capital_required,
                    budget=budget,
                )
            )

        rec_tag = (
            f'<span class="rec-badge">{html.escape(tr(lang, "best_fit_badge"))}</span>'
            if plan.is_recommended
            else f'<span class="pill pill-info">{html.escape(str(plan_t["short_label"]))}</span>'
        )
        st.markdown(
            f"""
            <div class="weather-card" style="border-color: rgba(0, 212, 170, 0.28);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:0.45rem;">
                    <div style="font-size:1.22rem; font-weight:800; color:#F8FAFC;">{html.escape(str(plan_t["name"]))}</div>
                    <div style="display:flex; gap:8px; align-items:center;">
                        {rec_tag}
                        <span class="pill" style="background:rgba(148,163,184,0.14); color:{plan.risk_color}; border:1px solid {plan.risk_color};">{html.escape(str(plan_t["risk_level"]))}</span>
                    </div>
                </div>
                <div style="font-size:1.0rem; color:#7CEFD7; font-weight:600; margin-bottom:0.35rem;">{html.escape(str(plan_t["tagline"]))}</div>
                <div style="font-size:0.86rem; color:#94A3B8;">
                    <strong>{html.escape(tr(lang, "exact_setup"))}:</strong> <span dir="ltr">{html.escape(str(plan_t["contract_summary"]))}</span>
                    &nbsp;|&nbsp;
                    <strong>{html.escape(tr(lang, "daily_theta"))}:</strong> <span dir="ltr">~${plan.daily_theta_dollars:,.2f}</span>{html.escape(tr(lang, "per_day"))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 5 High-Contrast Highlight Cards: ENTER -> TAKE PROFIT -> BREAKEVEN -> STOP-LOSS -> MAX LOSS
        st.markdown(tr(lang, "levels_heading"))
        st.caption(tr(lang, "levels_sub"))

        be_value = (
            f"${plan.breakeven_low:,.2f} – ${plan.breakeven_high:,.2f}"
            if (plan.strategy_id == "iron_condor" and plan.breakeven_low and plan.breakeven_high)
            else f"${(plan.breakeven_low or snapshot.price):,.2f}"
        )
        be_sub = (
            tr(lang, "lvl_be_ic_sub", low=plan.breakeven_low, high=plan.breakeven_high, cushion=plan.breakeven_cushion_pct)
            if (plan.strategy_id == "iron_condor" and plan.breakeven_low and plan.breakeven_high)
            else tr(lang, "lvl_be_sub", cushion=plan.breakeven_cushion_pct)
        )

        l1, l2, l3, l4, l5 = st.columns(5)
        with l1:
            st.markdown(
                _level_card(
                    "level-entry",
                    tr(lang, "lvl_entry_tag"),
                    f"+${plan.upfront_cash:,.2f}",
                    tr(lang, "lvl_entry_sub", spot=plan.entry_stock_price, cred=plan.entry_credit_per_share, cash=plan.upfront_cash),
                    "#38BDF8",
                ),
                unsafe_allow_html=True,
            )
        with l2:
            st.markdown(
                _level_card(
                    "level-profit",
                    tr(lang, "lvl_tp_tag"),
                    f"+${plan.take_profit_dollars:,.2f}",
                    tr(lang, "lvl_tp_sub", buyback=plan.take_profit_buyback_per_share, max_p=plan.max_profit),
                    "#00D4AA",
                ),
                unsafe_allow_html=True,
            )
        with l3:
            st.markdown(
                _level_card(
                    "level-breakeven",
                    tr(lang, "lvl_be_tag"),
                    be_value,
                    be_sub,
                    "#FBBF24",
                ),
                unsafe_allow_html=True,
            )
        with l4:
            st.markdown(
                _level_card(
                    "level-stop",
                    tr(lang, "lvl_stop_tag"),
                    f"-${plan.stop_loss_dollars:,.2f}",
                    tr(lang, "lvl_stop_sub", trig=plan.stop_loss_stock_trigger, buyback=plan.stop_loss_buyback_per_share),
                    "#FB923C",
                ),
                unsafe_allow_html=True,
            )
        with l5:
            st.markdown(
                _level_card(
                    "level-maxloss",
                    tr(lang, "lvl_maxloss_tag"),
                    f"-${plan.max_risk:,.2f}",
                    tr(lang, "lvl_maxloss_sub", cap=plan.capital_required, pop=plan.prob_of_profit_pct),
                    "#EF4444",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)

        # Side-by-Side Plain-English/Arabic "HOW YOU MAKE MONEY" vs "WHAT IS YOUR RISK"
        c_money, c_risk = st.columns(2)
        with c_money:
            st.markdown(
                f"""
                <div class="money-card">
                    <div class="card-heading" style="color:#34E8C5;">{html.escape(tr(lang, "how_money_heading"))}</div>
                    <div class="card-body">
                        <p><strong>{html.escape(tr(lang, "step_action"))}</strong> {_md_bold_to_html(str(plan_t["what_you_do"]))}</p>
                        <p><strong>{html.escape(tr(lang, "step_payout"))}</strong> {_md_bold_to_html(str(plan_t["how_you_make_money"]))}</p>
                        <p style="margin-bottom:0;"><strong>{html.escape(tr(lang, "step_why"))}</strong> {html.escape(str(plan_t["fit_reason"]))}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_risk:
            rules_html = "".join(f'<li style="margin-bottom:0.35rem;">{_md_bold_to_html(str(rule))}</li>' for rule in plan_t["playbook"])
            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="card-heading" style="color:#FCA5A5;">{html.escape(tr(lang, "risk_heading"))}</div>
                    <div class="card-body">
                        <p><strong>{html.escape(tr(lang, "worst_case"))}</strong> {_md_bold_to_html(str(plan_t["what_is_your_risk"]))}</p>
                        <p style="margin-bottom:0.35rem;"><strong>{html.escape(tr(lang, "safety_rules"))}</strong></p>
                        <ul style="margin:0;">{rules_html}</ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

        p_col1, p_col2 = st.columns([1.35, 1.0])
        with p_col1:
            st.plotly_chart(build_payoff_chart(plan, snapshot.price, prediction, lang=lang), use_container_width=True)
        with p_col2:
            st.plotly_chart(build_income_projection_chart(plan, lang=lang), use_container_width=True)

        with st.expander(tr(lang, "compare_expander")):
            if lang == "ar":
                comp_df = pd.DataFrame(
                    [
                        {
                            "الاستراتيجية": ("⭐ " if s.is_recommended else "") + STRATEGY_NAMES_AR.get(s.strategy_id, (s.strategy_name,))[0],
                            "كاش الدخول (+$)": s.upfront_cash,
                            "جني الربح 50% (+$)": s.take_profit_dollars,
                            "أقصى ربح (+$)": s.max_profit,
                            "نقطة التعادل ($)": s.breakeven_low or snapshot.price,
                            "وقف الخسارة (-$)": s.stop_loss_dollars,
                            "أقصى خسارة (-$)": s.max_risk,
                            "احتمال النجاح (%)": s.prob_of_profit_pct,
                            "يناسب الميزانية": tr(lang, "yes_budget") if s.fits_budget else tr(lang, "no_budget"),
                        }
                        for s in strategies
                    ]
                )
            else:
                comp_df = pd.DataFrame(
                    [
                        {
                            "Strategy": ("⭐ " if s.is_recommended else "") + s.strategy_name,
                            "Entry Cash (+$)": s.upfront_cash,
                            "Take-Profit 50% (+$)": s.take_profit_dollars,
                            "Max Profit (+$)": s.max_profit,
                            "Breakeven ($)": s.breakeven_low or snapshot.price,
                            "Stop-Loss (-$)": s.stop_loss_dollars,
                            "Max Loss (-$)": s.max_risk,
                            "Win Odds (%)": s.prob_of_profit_pct,
                            "Fits Budget": tr(lang, "yes_budget") if s.fits_budget else tr(lang, "no_budget"),
                        }
                        for s in strategies
                    ]
                )
            st.dataframe(comp_df, width="stretch", hide_index=True)


# ===========================================================================
# VIEW 3: AI STOCK PREDICTION & INTERACTIVE PRICE CHARTS
# ===========================================================================
elif active_view == "forecast":
    st.markdown(tr(lang, "forecast_title", symbol=snapshot.symbol))
    st.markdown(str(pred_t["explanation"]))

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(_kpi_card(tr(lang, "kpi_current_price"), f"${snapshot.price:,.2f}", tr(lang, "kpi_today_chg", chg=snapshot.change_pct)), unsafe_allow_html=True)
    with f2:
        st.markdown(
            _kpi_card(
                tr(lang, "kpi_exp_range", dte=prediction.horizon_days),
                f"${prediction.expected_low_1sigma:,.0f} – ${prediction.expected_high_1sigma:,.0f}",
                tr(lang, "kpi_range_sub"),
                "#38BDF8",
            ),
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            _kpi_card(
                tr(lang, "kpi_confidence"),
                f"{prediction.confidence_pct:.0f}%",
                tr(lang, "kpi_wf_acc", acc=prediction.backtest_accuracy_pct),
                "#00D4AA",
            ),
            unsafe_allow_html=True,
        )
    with f4:
        iv_msg = tr(lang, "iv_rich") if snapshot.iv_rank >= 55 else (tr(lang, "iv_normal") if snapshot.iv_rank >= 30 else tr(lang, "iv_calm"))
        st.markdown(_kpi_card(tr(lang, "kpi_iv_rank"), f"{snapshot.iv_rank:.0f} / 100", iv_msg, "#FBBF24"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    top_plan = strategies[0] if strategies else None
    st.plotly_chart(build_forecast_chart(snapshot, prediction, lang=lang, active_plan=top_plan), use_container_width=True)

    st.markdown(tr(lang, "why_ai_heading"))
    for drv in pred_t["drivers"]:
        st.markdown(f"- {drv}")

    with st.expander(tr(lang, "chain_expander", symbol=snapshot.symbol, exp=chain.expiration, dte=chain.dte)):
        oc1, oc2 = st.columns(2)
        cols_keep = [c for c in ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest"] if c in chain.puts.columns]
        with oc1:
            st.markdown(tr(lang, "puts_heading"))
            puts_near = chain.puts[(chain.puts["strike"] >= snapshot.price * 0.88) & (chain.puts["strike"] <= snapshot.price * 1.05)][cols_keep]
            st.dataframe(puts_near, width="stretch", hide_index=True)
        with oc2:
            st.markdown(tr(lang, "calls_heading"))
            calls_near = chain.calls[(chain.calls["strike"] >= snapshot.price * 0.95) & (chain.calls["strike"] <= snapshot.price * 1.12)][cols_keep]
            st.dataframe(calls_near, width="stretch", hide_index=True)


# ===========================================================================
# VIEW 4: LIVE NEWS, MACRO ENVIRONMENT & BEGINNER'S GUIDE
# ===========================================================================
elif active_view == "news":
    st.markdown(tr(lang, "news_title"))

    st.markdown(
        f"""
        <div class="weather-card">
            <div class="card-heading" style="color:{macro.weather_color};">{macro.weather_emoji} {html.escape(tr(lang, "weather_heading"))}: {html.escape(weather_t["label"])}</div>
            <div class="card-body">
                <p style="margin-bottom:0.45rem;">{html.escape(weather_t["summary"])}</p>
                <p style="margin-bottom:0;"><strong>{html.escape(tr(lang, "income_guidance"))}</strong> {html.escape(weather_t["advice"])}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(build_vix_gauge(macro, lang=lang), use_container_width=True)
    with g2:
        st.plotly_chart(build_sentiment_bar(news_items, lang=lang), use_container_width=True)

    st.markdown(tr(lang, "headlines_heading", symbol=symbol))
    for item in news_items:
        badge_cls = "pill-live" if item.sentiment_label == "Positive" else ("pill-danger" if item.sentiment_label == "Negative" else "pill-info")
        sent_disp = SENTIMENT_AR.get(item.sentiment_label, item.sentiment_label) if lang == "ar" else item.sentiment_label
        cat_disp = CATEGORY_AR.get(item.category, item.category) if lang == "ar" else item.category
        safe_title = html.escape(item.title)
        title_html = f'<a href="{html.escape(item.link)}" target="_blank" dir="auto">{safe_title} ↗</a>' if item.link else f'<span dir="auto">{safe_title}</span>'
        st.markdown(
            f"""
            <div class="news-item">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">
                    <div class="news-title">{title_html}</div>
                    <span class="pill {badge_cls}" style="white-space:nowrap;">{html.escape(sent_disp)} ({item.sentiment_score:+.2f})</span>
                </div>
                <div class="news-meta">{html.escape(cat_disp)} • {html.escape(item.source)} • {html.escape(item.published)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(tr(lang, "guide_title"))
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(tr(lang, "guide_col1"))
    with b2:
        st.markdown(tr(lang, "guide_col2"))

st.markdown(
    f"""
    <div style="margin-top:2rem; padding-top:0.9rem; border-top:1px solid rgba(148,163,184,0.14); font-size:0.8rem; color:#64748B; text-align:center;">
        {tr(lang, "disclaimer")}
    </div>
    """,
    unsafe_allow_html=True,
)
