"""Silky, modern fintech CSS styling with full English (LTR) & Arabic (RTL) support."""
from __future__ import annotations

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Cairo', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Smooth page layout */
.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    max-width: 1420px;
}

/* Glassmorphic Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(16, 30, 54, 0.96) 0%, rgba(11, 22, 39, 0.98) 55%, rgba(0, 88, 76, 0.35) 100%);
    border: 1px solid rgba(0, 212, 170, 0.22);
    border-radius: 18px;
    padding: 1.45rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.32);
}

.hero-title {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    background: linear-gradient(90deg, #FFFFFF 0%, #7CEFD7 55%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}

.hero-sub {
    color: #94A3B8;
    font-size: 0.96rem;
    margin: 0;
    line-height: 1.6;
}

/* Live Status Pill */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-live {
    background: rgba(0, 212, 170, 0.14);
    color: #34E8C5;
    border: 1px solid rgba(0, 212, 170, 0.35);
}
.pill-warn {
    background: rgba(245, 158, 11, 0.14);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.pill-info {
    background: rgba(56, 189, 248, 0.14);
    color: #7DD3FC;
    border: 1px solid rgba(56, 189, 248, 0.35);
}
.pill-danger {
    background: rgba(239, 68, 68, 0.15);
    color: #FCA5A5;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

/* Sleek Metric Cards */
.kpi-card {
    background: linear-gradient(180deg, rgba(22, 31, 50, 0.92) 0%, rgba(16, 23, 38, 0.96) 100%);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 14px;
    padding: 1rem 1.15rem;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 212, 170, 0.38);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
}
.kpi-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94A3B8;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.52rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.2;
}
.kpi-delta {
    font-size: 0.82rem;
    margin-top: 0.32rem;
    color: #94A3B8;
}

/* Plain-English Explainer Cards */
.money-card {
    background: linear-gradient(155deg, rgba(0, 212, 170, 0.10) 0%, rgba(16, 25, 42, 0.96) 70%);
    border: 1px solid rgba(0, 212, 170, 0.32);
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    height: 100%;
}
.risk-card {
    background: linear-gradient(155deg, rgba(239, 68, 68, 0.10) 0%, rgba(16, 25, 42, 0.96) 70%);
    border: 1px solid rgba(239, 68, 68, 0.32);
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    height: 100%;
}
.weather-card {
    background: linear-gradient(135deg, rgba(19, 28, 46, 0.96) 0%, rgba(26, 38, 62, 0.94) 100%);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-heading {
    font-size: 1.08rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-body {
    color: #CBD5E1;
    font-size: 0.95rem;
    line-height: 1.68;
}

/* News Feed Card */
.news-item {
    background: rgba(19, 26, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.news-item:hover {
    border-color: rgba(56, 189, 248, 0.35);
    background: rgba(24, 34, 54, 0.95);
}
.news-title a {
    color: #F1F5F9 !important;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.93rem;
}
.news-title a:hover {
    color: #38BDF8 !important;
}
.news-meta {
    color: #94A3B8;
    font-size: 0.78rem;
    margin-top: 0.3rem;
}

/* Strategy Selector Card Highlight */
.rec-badge {
    display: inline-block;
    background: linear-gradient(90deg, #00D4AA, #38BDF8);
    color: #06101C;
    font-weight: 800;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 999px;
}

/* High-Contrast Entry / Exit / Breakeven Action Cards */
.level-card {
    background: linear-gradient(180deg, rgba(20, 30, 48, 0.96) 0%, rgba(13, 20, 34, 0.98) 100%);
    border-radius: 15px;
    padding: 1rem 1.05rem;
    height: 100%;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.28);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.level-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 26px rgba(0, 0, 0, 0.36);
}
.level-entry {
    border: 1px solid rgba(56, 189, 248, 0.42);
    border-top: 4px solid #38BDF8;
}
.level-profit {
    border: 1px solid rgba(0, 212, 170, 0.45);
    border-top: 4px solid #00D4AA;
}
.level-breakeven {
    border: 1px solid rgba(251, 191, 36, 0.48);
    border-top: 4px solid #FBBF24;
}
.level-stop {
    border: 1px solid rgba(251, 146, 60, 0.48);
    border-top: 4px solid #FB923C;
}
.level-maxloss {
    border: 1px solid rgba(239, 68, 68, 0.48);
    border-top: 4px solid #EF4444;
}
.level-tag {
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}
.level-big {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.35rem;
}
.level-sub {
    font-size: 0.8rem;
    color: #CBD5E1;
    line-height: 1.45;
}

/* Sidebar polish */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1525 0%, #0A0F1C 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

/* Mobile / iPhone Responsive Polish */
@media (max-width: 768px) {
    .block-container {
        padding-top: 0.75rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    .hero-banner {
        padding: 1rem 1.05rem;
        border-radius: 14px;
    }
    .hero-title {
        font-size: 1.32rem;
    }
    .hero-sub {
        font-size: 0.86rem;
    }
    .kpi-card, .level-card, .money-card, .risk-card, .weather-card {
        margin-bottom: 0.65rem;
        padding: 0.9rem 1rem;
    }
    .kpi-value, .level-big {
        font-size: 1.32rem;
    }
}
</style>
"""

RTL_CSS = """
<style>
html, body, [class*="css"], .stMarkdown, .hero-banner, .kpi-card, .money-card, .risk-card, .weather-card, .news-item {
    font-family: 'Cairo', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.block-container, [data-testid="stSidebarContent"], .hero-banner, .kpi-card, .money-card, .risk-card, .weather-card, .news-item {
    direction: rtl;
    text-align: right;
}
.card-body ul {
    padding-right: 1.25rem !important;
    padding-left: 0 !important;
}
/* Keep Plotly charts & numbers crisp LTR inside their canvas so axes and tooltips never mirror */
.js-plotly-plot, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    direction: ltr;
}
</style>
"""

# Backwards-compatible alias
CUSTOM_CSS = BASE_CSS


def get_theme_css(lang: str = "en") -> str:
    return BASE_CSS + (RTL_CSS if lang == "ar" else "")
