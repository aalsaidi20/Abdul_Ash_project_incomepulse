"""Zero-cost live macro environment, public RSS news, SEC EDGAR 8-K events, and local sentiment."""
from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests

try:
    import feedparser
except ImportError:  # pragma: no cover
    feedparser = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


HEADERS = {
    "User-Agent": "AshIncomeLab/1.0 (Personal Financial Research Tool; contact@localhost)",
    "Accept": "application/json, text/csv, application/xml, */*",
}


@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    link: str
    sentiment_score: float  # -1.0 to +1.0
    sentiment_label: str    # "Positive", "Neutral", "Negative"
    category: str = "Market"  # "Ticker", "Macro", "SEC Filing"


@dataclass
class MacroEnvironment:
    vix: float
    treasury_10y: float
    fed_funds_rate: float
    yield_curve_10y_2y: float
    unemployment_rate: float
    cpi_yoy_pct: float
    weather_label: str       # "Calm & Sunny", "Partly Cloudy", "Stormy / High Alert"
    weather_emoji: str
    weather_color: str
    plain_summary: str
    income_advice: str
    is_live: bool = True
    updated_at: str = field(default_factory=lambda: dt.datetime.now().strftime("%Y-%m-%d %H:%M"))


# Financial-domain lexicon additions to boost VADER accuracy on market headlines
FINANCE_LEXICON_BOOST = {
    "beats": 2.1,
    "beat": 1.8,
    "surge": 2.2,
    "surges": 2.2,
    "rally": 2.0,
    "rallies": 2.0,
    "upgrade": 1.9,
    "upgraded": 1.9,
    "record": 1.2,
    "guidance": 0.4,
    "bullish": 2.2,
    "dividend": 1.2,
    "buyback": 1.5,
    "misses": -2.2,
    "miss": -1.8,
    "plunge": -2.6,
    "plunges": -2.6,
    "crash": -3.0,
    "slump": -2.1,
    "downgrade": -2.1,
    "downgraded": -2.1,
    "layoffs": -1.8,
    "lawsuit": -1.9,
    "probe": -1.8,
    "investigation": -1.7,
    "tariff": -1.4,
    "tariffs": -1.4,
    "recession": -2.5,
    "bearish": -2.2,
    "inflation": -0.8,
    "hike": -1.0,
    "cut": 0.6,
    "cuts": 0.6,
    "default": -2.8,
    "warning": -2.0,
    "selloff": -2.4,
    "sell-off": -2.4,
}


def _get_analyzer():
    if SentimentIntensityAnalyzer is None:
        return None
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(FINANCE_LEXICON_BOOST)
    return analyzer


def score_headline(text: str) -> tuple[float, str]:
    """Score a news headline locally at zero API cost (-1.0 bearish to +1.0 bullish)."""
    analyzer = _get_analyzer()
    if analyzer is not None:
        score = float(analyzer.polarity_scores(text)["compound"])
    else:
        lower = text.lower()
        pos = sum(1 for k, v in FINANCE_LEXICON_BOOST.items() if v > 0 and k in lower)
        neg = sum(1 for k, v in FINANCE_LEXICON_BOOST.items() if v < 0 and k in lower)
        score = (pos - neg) / max(pos + neg, 1)

    if score >= 0.15:
        label = "Positive"
    elif score <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"
    return round(score, 3), label


def _fetch_fred_latest(series_id: str, timeout: int = 6) -> Optional[float]:
    """Fetch latest value from FRED's completely free, no-API-key public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200 or "DATE" not in resp.text[:40].upper():
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        val_col = df.columns[1]
        vals = pd.to_numeric(df[val_col], errors="coerce").dropna()
        if vals.empty:
            return None
        return float(vals.iloc[-1])
    except Exception:
        return None


def _fetch_fred_yoy(series_id: str, timeout: int = 6) -> Optional[float]:
    """Compute Year-over-Year % change for monthly FRED series like CPIAUCSL."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        vals = pd.to_numeric(df[df.columns[1]], errors="coerce").dropna()
        if len(vals) < 13:
            return None
        return round(((float(vals.iloc[-1]) - float(vals.iloc[-13])) / float(vals.iloc[-13])) * 100.0, 2)
    except Exception:
        return None


def fetch_macro_environment() -> MacroEnvironment:
    """Gather free live macroeconomic & volatility indicators (VIX, Yields, FRED)."""
    is_live = False
    vix = 17.5
    t10y = 4.25

    if yf is not None:
        try:
            vix_hist = yf.Ticker("^VIX").history(period="5d")
            if vix_hist is not None and not vix_hist.empty:
                vix = round(float(vix_hist["Close"].iloc[-1]), 2)
                is_live = True
        except Exception:
            pass
        try:
            tnx_hist = yf.Ticker("^TNX").history(period="5d")
            if tnx_hist is not None and not tnx_hist.empty:
                t10y = round(float(tnx_hist["Close"].iloc[-1]), 2)
                is_live = True
        except Exception:
            pass

    fed_funds = _fetch_fred_latest("DFF")
    yc_10_2 = _fetch_fred_latest("T10Y2Y")
    unemp = _fetch_fred_latest("UNRATE")
    cpi_yoy = _fetch_fred_yoy("CPIAUCSL")

    if any(x is not None for x in (fed_funds, yc_10_2, unemp, cpi_yoy)):
        is_live = True

    fed_funds_val = round(fed_funds, 2) if fed_funds is not None else 4.50
    yc_val = round(yc_10_2, 2) if yc_10_2 is not None else 0.15
    unemp_val = round(unemp, 1) if unemp is not None else 4.1
    cpi_val = round(cpi_yoy, 2) if cpi_yoy is not None else 2.7

    # Translate raw macro indicators into beginner-friendly "Market Weather"
    if vix < 16.5 and yc_val >= -0.1:
        weather_label = "Calm & Sunny"
        weather_emoji = "☀️"
        weather_color = "#00D4AA"
        plain_summary = (
            f"Market fear (VIX {vix:.1f}) is low and interest rates are steady (10Y Treasury {t10y:.2f}%). "
            "Stock prices tend to drift smoothly without violent overnight shocks."
        )
        income_advice = (
            "Great environment for steady Covered Calls, Cash-Secured Puts, and Iron Condors. "
            "Premiums are moderate, so focus on consistent weekly singles rather than stretching for risky strikes."
        )
    elif vix < 24.0:
        weather_label = "Partly Cloudy (Rich Income)"
        weather_emoji = "⛅"
        weather_color = "#F59E0B"
        plain_summary = (
            f"Market nerves are moderately elevated (VIX {vix:.1f}, inflation ~{cpi_val:.1f}%). "
            "Because traders are paying more for protection, option sellers earn noticeably bigger cash payouts."
        )
        income_advice = (
            "Prime zone for defined-risk Bull Put Credit Spreads and wider-buffer Cash-Secured Puts. "
            "Keep position sizes modest and pick strikes further away from the current price."
        )
    else:
        weather_label = "Stormy / High Alert"
        weather_emoji = "⛈️"
        weather_color = "#EF4444"
        plain_summary = (
            f"Volatility is high (VIX {vix:.1f}) — headlines and macro fears are causing large daily swings. "
            "Option payouts look tempting, but prices can blow past normal safety levels fast."
        )
        income_advice = (
            "Protect your capital first! Stick strictly to small, defined-risk Credit Spreads with wide safety cushions, "
            "or wait for the storm to calm before selling short-dated options."
        )

    return MacroEnvironment(
        vix=vix,
        treasury_10y=t10y,
        fed_funds_rate=fed_funds_val,
        yield_curve_10y_2y=yc_val,
        unemployment_rate=unemp_val,
        cpi_yoy_pct=cpi_val,
        weather_label=weather_label,
        weather_emoji=weather_emoji,
        weather_color=weather_color,
        plain_summary=plain_summary,
        income_advice=income_advice,
        is_live=is_live,
    )


def _parse_rss(url: str, source_name: str, category: str, limit: int = 6) -> List[NewsItem]:
    items: List[NewsItem] = []
    if feedparser is None:
        return items
    try:
        resp = requests.get(url, headers=HEADERS, timeout=6)
        if resp.status_code != 200:
            return items
        feed = feedparser.parse(resp.content)
        for entry in feed.entries[:limit]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            link = entry.get("link") or ""
            published = entry.get("published") or entry.get("updated") or "Recent"
            if len(published) > 25:
                published = published[:22] + "..."
            src = source_name
            if hasattr(entry, "source") and isinstance(entry.source, dict) and entry.source.get("title"):
                src = entry.source["title"]
            score, label = score_headline(title)
            items.append(
                NewsItem(
                    title=title,
                    source=src,
                    published=published,
                    link=link,
                    sentiment_score=score,
                    sentiment_label=label,
                    category=category,
                )
            )
    except Exception:
        pass
    return items


def fetch_live_news_and_events(symbol: str, max_items: int = 12) -> tuple[List[NewsItem], float, bool]:
    """Pull 100% free public headlines for ticker + macro environment + SEC 8-K live feed."""
    sym = symbol.upper().strip()
    all_items: List[NewsItem] = []

    # 1. Google News RSS for ticker (100% free, live)
    q_ticker = quote_plus(f"{sym} stock options market")
    all_items.extend(
        _parse_rss(
            f"https://news.google.com/rss/search?q={q_ticker}&hl=en-US&gl=US&ceid=US:en",
            "Google News",
            "Ticker",
            limit=6,
        )
    )

    # 2. Yahoo Finance RSS for ticker (100% free)
    all_items.extend(
        _parse_rss(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US",
            "Yahoo Finance",
            "Ticker",
            limit=5,
        )
    )

    # 3. Broader market / Fed / economy live news RSS
    q_macro = quote_plus("Federal Reserve stock market economy inflation")
    all_items.extend(
        _parse_rss(
            f"https://news.google.com/rss/search?q={q_macro}&hl=en-US&gl=US&ceid=US:en",
            "Macro News",
            "Macro",
            limit=4,
        )
    )

    # 4. SEC EDGAR live 8-K current reports RSS (public corporate material events)
    all_items.extend(
        _parse_rss(
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=8-K&company=&dateb=&owner=include&start=0&count=10&output=atom",
            "SEC EDGAR 8-K",
            "SEC Filing",
            limit=3,
        )
    )

    # Deduplicate by normalized title prefix
    seen = set()
    unique: List[NewsItem] = []
    for item in all_items:
        key = item.title[:55].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    is_live = len(unique) > 0

    if not unique:
        # Clean offline fallback notice so UI never breaks
        unique = [
            NewsItem(
                title=f"Live RSS news feeds are currently offline — showing baseline neutral sentiment for {sym}.",
                source="Local Fallback",
                published="Offline",
                link="",
                sentiment_score=0.0,
                sentiment_label="Neutral",
                category="Macro",
            )
        ]

    # Weight ticker headlines higher than general macro headlines
    ticker_scores = [i.sentiment_score for i in unique if i.category == "Ticker"]
    macro_scores = [i.sentiment_score for i in unique if i.category != "Ticker"]
    if ticker_scores and macro_scores:
        avg_sent = 0.7 * (sum(ticker_scores) / len(ticker_scores)) + 0.3 * (sum(macro_scores) / len(macro_scores))
    elif ticker_scores:
        avg_sent = sum(ticker_scores) / len(ticker_scores)
    else:
        avg_sent = sum(macro_scores) / max(len(macro_scores), 1)

    return unique[:max_items], round(float(avg_sent), 3), is_live
