"""Bilingual (Arabic / English) localization & beginner-friendly financial translator."""
from __future__ import annotations

from typing import Dict, List

from core.macro_news import MacroEnvironment
from core.market_data import StockSnapshot
from core.options_engine import OptionStrategyPlan
from core.predictor import MarketPrediction

STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "IncomePulse — AI Stock & Options Income Studio",
        "sidebar_title": "💎 IncomePulse Controls",
        "sidebar_caption": "Adjust your budget, risk comfort, and income rhythm — everything updates live.",
        "lang_label": "🌐 Language / اللغة",
        "budget_label": "💰 Your Trading Budget ($)",
        "budget_help": "Used to size safe option contracts and show exact dollar income & risk.",
        "risk_label": "🛡️ Comfort With Risk",
        "risk_help": "Safer picks strikes further away for higher win odds. Growth collects bigger payouts with tighter room.",
        "cadence_label": "🔁 How Often Do You Want to Repeat?",
        "cadence_help": "Daily/Ultra-Short turns over fastest; Weekly is the classic income sweet spot; Bi-Weekly/Monthly is smoothest.",
        "rhythm_tip": "Rhythm Tip",
        "inspect_heading": "#### 🔍 Inspect Any Stock or ETF",
        "popular_tickers": "Popular Liquid Income Tickers",
        "custom_ticker": "Or type any custom ticker (e.g. COIN, NFLX)",
        "custom_placeholder": "Enter symbol & press Enter",
        "refresh_btn": "🔄 Refresh All Live Free Data",
        "free_sources_title": "100% Free Public Data Sources",
        "free_sources_list": (
            "• Yahoo Finance (Prices &amp; Option Chains)<br>"
            "• Federal Reserve FRED (Macro CSVs)<br>"
            "• Google &amp; Yahoo Public News RSS<br>"
            "• SEC EDGAR 8-K Public Filings<br>"
            "• Local AI Forecast &amp; Sentiment Engine"
        ),
        "spinner_ticker": "Loading live public market data, news & option chains for {symbol}...",
        "spinner_scanner": "Scanning liquid daily & weekly option markets...",
        "live_badge": "● LIVE FREE PUBLIC DATA",
        "offline_badge": "● OFFLINE SIMULATED MODE",
        "market_weather_pill": "Market Weather",
        "hero_sub": (
            "Zero-jargon repetitive option income planner powered by 100% free public market, macro, and live news feeds. "
            "Currently inspecting <strong>{symbol} (${price:,.2f}, {change:+.2f}%)</strong> "
            "with a <strong>${budget:,.0f}</strong> budget ({risk})."
        ),
        "kpi_vix": "Market Fear (VIX)",
        "kpi_10y": "10Y Treasury Yield",
        "kpi_fed_sub": "Fed Funds ~{fed:.2f}%",
        "kpi_cpi": "U.S. Inflation (CPI)",
        "kpi_unemp_sub": "Unemployment {unemp:.1f}%",
        "kpi_news_mood": "{symbol} News Mood",
        "kpi_news_sub": "Score {score:+.2f} (Free RSS)",
        "kpi_ai_forecast": "{symbol} AI {dte}d Forecast",
        "kpi_up_odds": "{dir_label} ({odds:.0f}% up odds)",
        "mood_upbeat": "Upbeat",
        "mood_cautious": "Cautious",
        "mood_balanced": "Balanced",
        "nav_scanner": "🎯 Best Income Setups Now",
        "nav_blueprint": "💡 {symbol} Trade Blueprint",
        "nav_forecast": "📈 {symbol} AI Prediction",
        "nav_news": "🌍 Live News & Guide",
        "levels_heading": "### 🎯 Exact Entry, Exit, Breakeven & Risk Numbers",
        "levels_sub": "Your clear roadmap before placing any trade — know exactly where you enter, take profit, break even, and exit to protect your cash.",
        "lvl_entry_tag": "🔵 ENTER TRADE",
        "lvl_entry_sub": "Stock @ ${spot:,.2f} • Collect ${cred:.2f}/sh (+${cash:,.2f} cash upfront)",
        "lvl_tp_tag": "🟢 TAKE-PROFIT EXIT",
        "lvl_tp_sub": "Buy back option @ ${buyback:.2f}/sh (50% rule) • Max profit at expiry: +${max_p:,.2f}",
        "lvl_be_tag": "🟡 BREAKEVEN PRICE",
        "lvl_be_sub": "{cushion:.1f}% safety cushion from today • Above this price you do NOT lose",
        "lvl_be_ic_sub": "Safe zone between ${low:,.2f} and ${high:,.2f} ({cushion:.1f}% cushion)",
        "lvl_stop_tag": "🟠 SMART STOP-LOSS EXIT",
        "lvl_stop_sub": "Exit if stock hits ${trig:,.2f} or option hits ${buyback:.2f}/sh to protect capital",
        "lvl_maxloss_tag": "🔴 ABSOLUTE MAX LOSS",
        "lvl_maxloss_sub": "Worst-case cap at expiry • Capital reserved: ${cap:,.0f} • Win odds: {pop:.0f}%",
        "card_breakeven": "Breakeven",
        "card_tp": "Take-Profit",
        "card_stop": "Stop-Loss",
        "scanner_title": "### 🏆 Live Scanner: Top Repetitive Income Opportunities",
        "scanner_caption": (
            "Scanning high-liquidity **{cadence}** stocks & ETFs that match your **${budget:,.0f}** budget and **{risk}** style. "
            "Click any top pick below to jump straight into its plain-English Profit & Risk breakdown."
        ),
        "medal_1": "🥇 #1 Top Opportunity",
        "medal_2": "🥈 #2 Runner-Up",
        "medal_3": "🥉 #3 Strong Setup",
        "pill_earnings": "⚠️ Earnings Soon",
        "pill_fits": "✅ Fits Budget",
        "pill_larger": "💰 Larger Account",
        "card_cash": "Cash",
        "card_win_odds": "Win Odds",
        "card_weekly": "Weekly",
        "inspect_btn": "Inspect {symbol} Blueprint →",
        "all_scanned_heading": "#### 📋 All Scanned Stocks & ETFs (Ranked by Safety + Income)",
        "no_scanner_warn": "No scanner results returned right now — click Refresh Live Data in the sidebar.",
        "no_strats_err": "Could not construct option strategies for {symbol}. Try another liquid ticker like SPY, QQQ, AAPL, or NVDA.",
        "blueprint_title": "### 💡 Step-by-Step Income Plan for **{symbol}** ({name})",
        "exp_date_label": "📅 Expiration Date",
        "exp_date_help": "Pick a shorter expiration for faster repetition or a longer one for more breathing room.",
        "choose_strat_label": "Choose an Income Strategy to Explore:",
        "earnings_alert_prefix": "⚠️ **Earnings Event Alert:**",
        "budget_headsup": (
            "💡 **Budget Heads-Up:** One contract of **{strat}** on {symbol} needs **${cap:,.0f}** reserved cash "
            "(above your current ${budget:,.0f} slider). For smaller budgets, try a **Bull Put Credit Spread**, **Iron Condor**, "
            "or lower-priced tickers like **SOFI**, **XLF**, or **PLTR**."
        ),
        "best_fit_badge": "⭐ Best Fit Right Now",
        "exact_setup": "Exact Setup",
        "daily_theta": "Daily Time-Decay Paying You",
        "per_day": "/day",
        "kpi_cash_upfront": "Cash Paid Upfront",
        "kpi_max_profit_sub": "Max Profit: ${max_p:,.2f} ({n}× contract)",
        "kpi_max_risk": "Maximum Possible Risk",
        "kpi_cap_sub": "Capital Reserved: ${cap:,.0f}",
        "kpi_win_odds": "Estimated Win Odds",
        "kpi_safety_sub": "Safety Score: {score:.0f}/100",
        "kpi_income_yield": "Income Yield",
        "kpi_yield_sub": "~{wk:.2f}% weekly ({yr:.0f}%/yr pace)",
        "how_money_heading": "💰 How You Make Money",
        "step_action": "1. The Action:",
        "step_payout": "2. Your Payout:",
        "step_why": "3. Why It Fits Now:",
        "risk_heading": "🛡️ What Is Your Risk (Honest Breakdown)",
        "worst_case": "Worst-Case Scenario:",
        "safety_rules": "Your Repeatable Safety Rules:",
        "compare_expander": "📊 Compare All 4 Income Strategies Side-by-Side",
        "forecast_title": "### 📈 AI Market Forecast & Price Range — **{symbol}**",
        "kpi_current_price": "Current Price",
        "kpi_today_chg": "Today: {chg:+.2f}%",
        "kpi_exp_range": "Expected {dte}-Day Range",
        "kpi_range_sub": "Typical ~68% normal window",
        "kpi_confidence": "Model Confidence",
        "kpi_wf_acc": "Walk-Forward Accuracy: {acc:.0f}%",
        "kpi_iv_rank": "Volatility Rank",
        "iv_rich": "Rich Payouts (Great for Selling)",
        "iv_normal": "Normal Payouts",
        "iv_calm": "Calm / Smaller Payouts",
        "why_ai_heading": "#### 🧠 Why the AI Model Sees This (Plain English)",
        "chain_expander": "🔎 Inspect Live Free Option Chain ({symbol} — Exp {exp}, {dte}d)",
        "puts_heading": "**Put Options (Used for Credit Spreads & The Wheel)**",
        "calls_heading": "**Call Options (Used for Covered Calls & Iron Condors)**",
        "news_title": "### 🌍 Live Public Environment, Economy & News Pulse",
        "weather_heading": "Today's Market Weather",
        "income_guidance": "Income Strategy Guidance:",
        "headlines_heading": "#### 📰 Live Free Public Headlines & SEC Filings ({symbol} + Market)",
        "guide_title": "### 🎓 Zero-Jargon Beginner's Guide to Options Income",
        "guide_col1": (
            "**Think of Selling Options Like Being an Insurance Company:**\n"
            "- Traders pay you **cash upfront** (called *premium*) to insure their stocks against sudden drops or jumps.\n"
            "- Just like car insurance, most short-term policies expire unused — meaning **you keep the cash** and can repeat the process every few days or every week!\n"
            "- **Bull Put Credit Spread:** Best starter strategy for smaller accounts. You sell insurance at one price and buy cheaper backup insurance below it, so your max loss is **100% capped**.\n"
            "- **Iron Condor:** You collect cash from BOTH sides when you expect a stock to stay relatively calm inside a range."
        ),
        "guide_col2": (
            "**3 Golden Safety Rules for Beginners:**\n"
            "1. **Never Bet the Farm:** Keep any single short-dated trade to a small slice of your account so one bad week never wipes out months of steady wins.\n"
            "2. **Take 50% Profits Early:** If a weekly trade already earned half its cash payout early, close it! Locking in fast wins dramatically lowers surprise risk.\n"
            "3. **Respect Earnings & Big News:** Companies reporting quarterly earnings can jump ±10% overnight — watch for the ⚠️ **Earnings Alert** badge and skip those weeks."
        ),
        "disclaimer": (
            "Educational &amp; analytical software using 100% free public data (Yahoo Finance, FRED, public RSS, SEC EDGAR). "
            "Market forecasts are probabilistic estimates, not financial advice or guaranteed returns. Always practice with paper trading first."
        ),
        "days_short": "d",
        "cash_short": "cash",
        "win_short": "win",
        "yes_budget": "✅ Yes",
        "no_budget": "❌ Needs More",
    },
    "ar": {
        "app_title": "IncomePulse — منصة الذكاء الاصطناعي لتوقع الأسهم واستراتيجيات دخل الخيارات",
        "sidebar_title": "💎 إعدادات المنصة",
        "sidebar_caption": "عدّل ميزانيتك، مستوى المخاطرة، وسرعة التكرار — كل الأرقام والرسوم تتحدث فوراً.",
        "lang_label": "🌐 اللغة / Language",
        "budget_label": "💰 ميزانية التداول ($)",
        "budget_help": "تُستخدم لحساب عدد العقود الآمن وعرض أرباحك ومخاطرك بالدولار بدقة.",
        "risk_label": "🛡️ مستوى راحتك مع المخاطرة",
        "risk_help": "الأكثر أماناً يختار أسعاراً أبعد بنسبة نجاح أعلى. النمو يحقق دخلاً أكبر لكن بهامش أمان أضيق.",
        "cadence_label": "🔁 كم مرة تريد تكرار الصفقات؟",
        "cadence_help": "اليومي/القصير جداً أسرع في تكرار الدخل؛ الأسبوعي هو التوازن المثالي؛ نصف الشهري/الشهري هو الأكثر هدوءاً.",
        "rhythm_tip": "نصيحة الإيقاع",
        "inspect_heading": "#### 🔍 افحص أي سهم أو صندوق ETF",
        "popular_tickers": "أشهر الأسهم والصناديق عالية السيولة",
        "custom_ticker": "أو اكتب رمز أي سهم (مثل COIN أو NFLX)",
        "custom_placeholder": "اكتب الرمز واضغط Enter",
        "refresh_btn": "🔄 تحديث كل البيانات الحية المجانية",
        "free_sources_title": "مصادر بيانات عامة مجانية 100%",
        "free_sources_list": (
            "• Yahoo Finance (الأسعار وسلاسل الخيارات)<br>"
            "• بيانات الاحتياطي الفيدرالي FRED (الاقتصاد الكلي)<br>"
            "• أخبار Google و Yahoo الحية (RSS)<br>"
            "• إفصاحات هيئة الأوراق المالية SEC EDGAR 8-K<br>"
            "• محرك توقعات ومشاعر أخبار محلي بالذكاء الاصطناعي"
        ),
        "spinner_ticker": "جاري تحميل بيانات السوق الحية والأخبار وسلسلة الخيارات لـ {symbol}...",
        "spinner_scanner": "جاري فحص أفضل فرص الدخل المتكرر في الأسواق عالية السيولة...",
        "live_badge": "● بيانات عامة حية ومجانية",
        "offline_badge": "● وضع المحاكاة (بدون إنترنت)",
        "market_weather_pill": "طقس السوق",
        "hero_sub": (
            "مخطط دخل متكرر من الخيارات بلغة بسيطة بدون تعقيد، يعمل بالكامل على بيانات عامة ومجانية 100%. "
            "تفحص الآن <strong>{symbol} (${price:,.2f}، {change:+.2f}%)</strong> "
            "بميزانية <strong>${budget:,.0f}</strong> ({risk})."
        ),
        "kpi_vix": "مؤشر خوف السوق (VIX)",
        "kpi_10y": "عائد سندات 10 سنوات",
        "kpi_fed_sub": "فائدة الفيدرالي ~{fed:.2f}%",
        "kpi_cpi": "التضخم الأمريكي (CPI)",
        "kpi_unemp_sub": "البطالة {unemp:.1f}%",
        "kpi_news_mood": "مزاج أخبار {symbol}",
        "kpi_news_sub": "الدرجة {score:+.2f} (أخبار حية)",
        "kpi_ai_forecast": "توقع الذكاء الاصطناعي لـ {symbol} ({dte} يوم)",
        "kpi_up_odds": "{dir_label} (احتمال صعود {odds:.0f}%)",
        "mood_upbeat": "إيجابي ومتفائل",
        "mood_cautious": "حذر / سلبي",
        "mood_balanced": "متوازن ومحايد",
        "nav_scanner": "🎯 أفضل فرص الدخل الآن",
        "nav_blueprint": "💡 خطة صفقة {symbol} (الربح والمخاطرة)",
        "nav_forecast": "📈 توقع الذكاء الاصطناعي لـ {symbol}",
        "nav_news": "🌍 الأخبار الحية ودليل المبتدئين",
        "levels_heading": "### 🎯 أرقام الدخول والخروج ونقطة التعادل والربح والخسارة",
        "levels_sub": "خريطتك الواضحة قبل فتح أي صفقة — اعرف بدقة أين تدخل، وأين تجني الربح، وأين نقطة التعادل، وأين تخرج لحماية أموالك.",
        "lvl_entry_tag": "🔵 نقطة الدخول",
        "lvl_entry_sub": "السهم @ ${spot:,.2f} • اقبض ${cred:.2f}/سهم (+${cash:,.2f} كاش فوري)",
        "lvl_tp_tag": "🟢 خروج جني الربح",
        "lvl_tp_sub": "أغلق العقد عند ${buyback:.2f}/سهم (قاعدة 50%) • أقصى ربح عند الانتهاء: +${max_p:,.2f}",
        "lvl_be_tag": "🟡 نقطة التعادل (Breakeven)",
        "lvl_be_sub": "هامش أمان {cushion:.1f}% من سعر اليوم • فوق هذا السعر لا تخسر",
        "lvl_be_ic_sub": "نطاق الأمان بين ${low:,.2f} و ${high:,.2f} (هامش {cushion:.1f}%)",
        "lvl_stop_tag": "🟠 خروج وقف الخسارة الذكي",
        "lvl_stop_sub": "اخرج إذا وصل السهم ${trig:,.2f} أو وصل العقد ${buyback:.2f}/سهم لحماية رأس مالك",
        "lvl_maxloss_tag": "🔴 أقصى خسارة مطلقة",
        "lvl_maxloss_sub": "الحد الأقصى للخسارة عند الانتهاء • المبلغ المحجوز: ${cap:,.0f} • نسبة النجاح: {pop:.0f}%",
        "card_breakeven": "التعادل",
        "card_tp": "جني الربح",
        "card_stop": "وقف الخسارة",
        "scanner_title": "### 🏆 الماسح الحي: أفضل فرص الدخل المتكرر الآن",
        "scanner_caption": (
            "فحص آلي لأكثر الأسهم والصناديق سيولة لفئة **{cadence}** بما يناسب ميزانيتك **${budget:,.0f}** وأسلوب **{risk}**. "
            "اضغط على أي فرصة بالأسفل لرؤية شرح مبسط لكيفية الربح وحجم المخاطرة بالدولار."
        ),
        "medal_1": "🥇 #1 أفضل فرصة حالياً",
        "medal_2": "🥈 #2 المركز الثاني",
        "medal_3": "🥉 #3 فرصة قوية",
        "pill_earnings": "⚠️ أرباح شركة قريبة",
        "pill_fits": "✅ يناسب ميزانيتك",
        "pill_larger": "💰 يحتاج رأس مال أكبر",
        "card_cash": "الكاش الفوري",
        "card_win_odds": "نسبة النجاح",
        "card_weekly": "العائد الأسبوعي",
        "inspect_btn": "افتح خطة {symbol} ←",
        "all_scanned_heading": "#### 📋 جميع الأسهم والصناديق المفحوصة (مرتبة حسب الأمان + الدخل)",
        "no_scanner_warn": "لا توجد نتائج فحص حالياً — اضغط على زر تحديث البيانات الحية في القائمة الجانبية.",
        "no_strats_err": "تعذر بناء استراتيجيات خيارات لـ {symbol}. جرّب سهماً عالي السيولة مثل SPY أو QQQ أو AAPL أو NVDA.",
        "blueprint_title": "### 💡 خطة الدخل خطوة بخطوة لسهم **{symbol}** ({name})",
        "exp_date_label": "📅 تاريخ الانتهاء",
        "exp_date_help": "اختر مدة أقصر لتكرار أسرع، أو مدة أطول لمزيد من الهدوء وهامش الأمان.",
        "choose_strat_label": "اختر استراتيجية دخل لاستكشافها:",
        "earnings_alert_prefix": "⚠️ **تنبيه إعلان أرباح:**",
        "budget_headsup": (
            "💡 **ملاحظة الميزانية:** عقد واحد من **{strat}** على {symbol} يحتاج حجز **${cap:,.0f}** "
            "(أكبر من ميزانيتك الحالية ${budget:,.0f}). للميزانيات الأصغر، جرّب **سبريد البيع الصاعد (Bull Put Spread)** أو **الكوندور الحديدي (Iron Condor)** "
            "أو أسهماً بسعر أقل مثل **SOFI** أو **XLF** أو **PLTR**."
        ),
        "best_fit_badge": "⭐ الأنسب لك الآن",
        "exact_setup": "تفاصيل العقد",
        "daily_theta": "تآكل الوقت اليومي لصالحك",
        "per_day": "/يوم",
        "kpi_cash_upfront": "الكاش الفوري لحسابك",
        "kpi_max_profit_sub": "أقصى ربح: ${max_p:,.2f} ({n}× عقد)",
        "kpi_max_risk": "أقصى خسارة محتملة",
        "kpi_cap_sub": "رأس المال المحجوز: ${cap:,.0f}",
        "kpi_win_odds": "احتمال نجاح الصفقة",
        "kpi_safety_sub": "درجة الأمان: {score:.0f}/100",
        "kpi_income_yield": "عائد الدخل",
        "kpi_yield_sub": "~{wk:.2f}% أسبوعياً (بمعدل {yr:.0f}% سنوياً)",
        "how_money_heading": "💰 كيف تربح المال من هذه الصفقة؟",
        "step_action": "1. ما الذي تفعله:",
        "step_payout": "2. كيف يأتي الربح:",
        "step_why": "3. لماذا تناسب الوضع الحالي:",
        "risk_heading": "🛡️ ما هي مخاطرك؟ (شرح صريح وواضح)",
        "worst_case": "أسوأ سيناريو محتمل:",
        "safety_rules": "قواعد الأمان لتكرار الصفقة:",
        "compare_expander": "📊 قارن جميع الاستراتيجيات الـ 4 جنباً إلى جنب",
        "forecast_title": "### 📈 توقعات الذكاء الاصطناعي والنطاق السعري — **{symbol}**",
        "kpi_current_price": "السعر الحالي",
        "kpi_today_chg": "تغير اليوم: {chg:+.2f}%",
        "kpi_exp_range": "النطاق المتوقع خلال {dte} يوم",
        "kpi_range_sub": "النطاق الطبيعي (~68% احتمال)",
        "kpi_confidence": "ثقة النموذج",
        "kpi_wf_acc": "الدقة التاريخية المختبرة: {acc:.0f}%",
        "kpi_iv_rank": "مؤشر التقلب (IV Rank)",
        "iv_rich": "عوائد ممتازة (مناسب لبيع الخيارات)",
        "iv_normal": "عوائد طبيعية",
        "iv_calm": "سوق هادئ / عوائد أصغر",
        "why_ai_heading": "#### 🧠 لماذا يتوقع الذكاء الاصطناعي هذا الاتجاه؟ (بلغة بسيطة)",
        "chain_expander": "🔎 استعراض سلسلة الخيارات الحية المجانية ({symbol} — انتهاء {exp}، {dte} يوم)",
        "puts_heading": "**عقود البيع Put (تُستخدم لسبريد الدخل وعجلة الدخل)**",
        "calls_heading": "**عقود الشراء Call (تُستخدم للكول المغطى والكوندور الحديدي)**",
        "news_title": "### 🌍 نبض الأخبار الحية وبيئة الاقتصاد والسوق",
        "weather_heading": "طقس السوق اليوم",
        "income_guidance": "نصيحة استراتيجية الدخل:",
        "headlines_heading": "#### 📰 أحدث الأخبار العامة وإفصاحات SEC الحية ({symbol} + السوق)",
        "guide_title": "### 🎓 دليل المبتدئين المبسط لدخل الخيارات (بدون مصطلحات معقدة)",
        "guide_col1": (
            "**فكّر في بيع الخيارات وكأنك شركة تأمين:**\n"
            "- يدفع لك المتداولون **مبلغاً نقدياً مقدماً** (يُسمى القسط) لتأمين أسهمهم ضد الهبوط أو الصعود المفاجئ.\n"
            "- تماماً مثل تأمين السيارات، أغلب العقود قصيرة الأجل تنتهي دون استخدام — مما يعني أنك **تحتفظ بالكاش كاملاً** وتكرر العملية كل بضعة أيام أو كل أسبوع!\n"
            "- **سبريد البيع الصاعد (Bull Put Credit Spread):** أفضل استراتيجية للبداية برأس مال صغير. تبيع تأميناً عند سعر معين وتشتري تأميناً احتياطياً أرخص تحته، فتكون أقصى خسارة لديك **محدودة ومقفولة 100%**.\n"
            "- **الكوندور الحديدي (Iron Condor):** تجمع كاش من الجانبين معاً عندما تتوقع أن يبقى السهم هادئاً داخل نطاق سعري محدد."
        ),
        "guide_col2": (
            "**3 قواعد ذهبية لحماية أموالك كمبتدئ:**\n"
            "1. **لا تضع كل أموالك في صفقة واحدة:** اجعل كل صفقة قصيرة الأجل جزءاً صغيراً من حسابك حتى لا يمحو أسبوع سيء أرباح شهور.\n"
            "2. **اجنِ 50% من الربح مبكراً:** إذا حققت صفقة أسبوعية نصف ربحها المتوقع قبل موعد الانتهاء، أغلقها فوراً! تأمين الأرباح السريعة يقلل خطر المفاجآت بشكل هائل.\n"
            "3. **احذر أيام إعلان الأرباح والأخبار الكبرى:** الشركات التي تعلن نتائجها الربعية قد تقفز أو تهبط ±10% فجأة — راقب علامة ⚠️ **تنبيه الأرباح** وتجنب تلك الأسابيع."
        ),
        "disclaimer": (
            "أداة تعليمية وتحليلية تعتمد بالكامل على بيانات عامة ومجانية 100% (Yahoo Finance، FRED، RSS، SEC EDGAR). "
            "توقعات السوق هي تقديرات احتمالية وليست نصيحة مالية أو أرباحاً مضمونة. ابدأ دائماً بالتداول التجريبي (Paper Trading) أولاً."
        ),
        "days_short": "ي",
        "cash_short": "كاش",
        "win_short": "نجاح",
        "yes_budget": "✅ نعم",
        "no_budget": "❌ يحتاج أكثر",
    },
}


RISK_PROFILE_LABELS = {
    "Conservative (Safer)": {"en": "Conservative (Safer)", "ar": "محافظ (أكثر أماناً)"},
    "Balanced": {"en": "Balanced", "ar": "متوازن (الأفضل للغالبية)"},
    "Growth (Higher Income)": {"en": "Growth (Higher Income)", "ar": "نمو (دخل أعلى ومخاطرة أكبر)"},
}

CADENCE_LABELS = {
    "Daily / Ultra-Short (1–4 Days)": {
        "en": "Daily / Ultra-Short (1–4 Days)",
        "ar": "يومي / قصير جداً (1–4 أيام)",
        "note_ar": "دوران نقدي سريع جداً (عدة مرات أسبوعياً)، لكن التحركات المفاجئة لا تترك وقتاً للتصحيح. استخدم أحجاماً صغيرة واستراتيجيات محدودة المخاطر فقط.",
    },
    "Weekly Income (5–10 Days)": {
        "en": "Weekly Income (5–10 Days)",
        "ar": "دخل أسبوعي متكرر (5–10 أيام)",
        "note_ar": "المنطقة الذهبية للدخل الأسبوعي المتكرر: تآكل زمني سريع لصالحك مع هامش أمان مريح يمتص التذبذب اليومي.",
    },
    "Bi-Weekly / Monthly (11–35 Days)": {
        "en": "Bi-Weekly / Monthly (11–35 Days)",
        "ar": "نصف شهري / شهري (11–35 يوماً)",
        "note_ar": "حركة أكثر سلاسة وهامش أمان أوسع للمبتدئين، مع توتر أقل ومتابعة يومية أخف.",
    },
}

TICKER_AR_DESC = {
    "SPY": "يتتبع أكبر 500 شركة أمريكية. عقود يومية وأعلى سيولة وأضيق فروق أسعار في السوق.",
    "QQQ": "يتتبع أكبر 100 شركة تكنولوجيا ونمو. عقود يومية مع عوائد دخل خيارات أعلى.",
    "IWM": "يتتبع 2000 شركة أمريكية صغيرة. عقود يومية وسعر سهم أقل يناسب الحسابات الأصغر.",
    "AAPL": "شركة أبل العملاقة — حركة مستقرة وسيولة ممتازة لعقود الخيارات الأسبوعية.",
    "MSFT": "شركة مايكروسوفت — عملاق السحابة والبرمجيات بحجم تداول أسبوعي مستقر.",
    "NVDA": "شركة إنفيديا الرائدة في رقائق الذكاء الاصطناعي — تقلب عالٍ يعني عوائد أسبوعية كبيرة (ومخاطرة أعلى).",
    "AMZN": "شركة أمازون للتجارة والسحابة — سيولة عميقة في الخيارات الأسبوعية.",
    "AMD": "شركة AMD للرقائق — عوائد خيارات أسبوعية جذابة وسعر سهم متوسط.",
    "TSLA": "شركة تسلا — تقلب مرتفع جداً يمنح عوائد ضخمة، لكن يتطلب ضبطاً صارماً للمخاطر.",
    "META": "شركة ميتا — تدفقات نقدية قوية ونشاط مرتفع في الخيارات الأسبوعية.",
    "PLTR": "شركة بالانتير للذكاء الاصطناعي — سعر سهم منخفض نسبياً بعوائد أسبوعية غنية، شائعة للحسابات المتوسطة.",
    "SOFI": "شركة SoFi المالية — سعر سهم منخفض جداً (~15$) مما يتيح استراتيجية العجلة بأقل من 2000$.",
    "XLF": "صندوق قطاع البنوك والمال الأمريكي — حركة هادئة وسعر منخفض ومثالي للدخل الثابت.",
    "TLT": "صندوق سندات الخزانة الأمريكية طويلة الأجل — يتحرك غالباً بعكس الأسهم ويفيد لتنويع المحفظة.",
    "GLD": "صندوق الذهب العالمي — ارتباط ضعيف بأسهم التكنولوجيا وخيارات أسبوعية سلسة.",
}

STRATEGY_NAMES_AR = {
    "bull_put_spread": ("سبريد البيع الصاعد (Bull Put Spread)", "دخل محدود المخاطرة"),
    "iron_condor": ("الكوندور الحديدي (Iron Condor)", "دخل النطاق الهادئ"),
    "cash_secured_put": ("بيع Put مغطى بالكاش (استراتيجية العجلة)", "اربح وأنت تنتظر خصماً"),
    "covered_call": ("الكول المغطى (Covered Call)", "إيجار أسبوعي لأسهمك"),
}

RISK_LEVEL_AR = {
    "Low Risk": "مخاطرة منخفضة",
    "Moderate Risk": "مخاطرة متوسطة",
    "Higher Risk": "مخاطرة أعلى",
}

DIRECTION_AR = {
    "Bullish Lean": "ميل صاعد",
    "Calm / Sideways": "هادئ / عرضي",
    "Cautious / Bearish": "حذر / هابط",
}

WEATHER_AR = {
    "Calm & Sunny": "هادئ ومشمس",
    "Partly Cloudy (Rich Income)": "غائم جزئياً (دخل غني)",
    "Stormy / High Alert": "عاصف / حذر شديد",
}

SENTIMENT_AR = {
    "Positive": "إيجابي",
    "Neutral": "محايد",
    "Negative": "سلبي",
}

CATEGORY_AR = {
    "Ticker": "خبر السهم",
    "Macro": "الاقتصاد الكلي",
    "SEC Filing": "إفصاح رسمي SEC",
}


def tr(lang: str, key: str, **kwargs) -> str:
    text = STRINGS.get(lang, STRINGS["en"]).get(key, STRINGS["en"].get(key, key))
    return text.format(**kwargs) if kwargs else text


def tr_risk_profile(value: str, lang: str) -> str:
    return RISK_PROFILE_LABELS.get(value, {}).get(lang, value)


def tr_cadence(value: str, lang: str) -> str:
    return CADENCE_LABELS.get(value, {}).get(lang, value)


def tr_cadence_note(value: str, fallback_en: str, lang: str) -> str:
    if lang == "ar":
        return CADENCE_LABELS.get(value, {}).get("note_ar", fallback_en)
    return fallback_en


def tr_ticker_desc(symbol: str, fallback_en: str, lang: str) -> str:
    if lang == "ar":
        return TICKER_AR_DESC.get(symbol.upper(), f"سهم {symbol.upper()} تم تحميله من البيانات العامة الحية.")
    return fallback_en


def tr_weather(macro: MacroEnvironment, lang: str) -> Dict[str, str]:
    if lang != "ar":
        return {
            "label": macro.weather_label,
            "summary": macro.plain_summary,
            "advice": macro.income_advice,
        }
    label = WEATHER_AR.get(macro.weather_label, macro.weather_label)
    if macro.vix < 16.5 and macro.yield_curve_10y_2y >= -0.1:
        summary = (
            f"خوف السوق منخفض (مؤشر VIX عند {macro.vix:.1f}) وأسعار الفائدة مستقرة (سندات 10 سنوات عند {macro.treasury_10y:.2f}%). "
            "تميل أسعار الأسهم للتحرك بهدوء دون صدمات مفاجئة."
        )
        advice = (
            "بيئة ممتازة للكول المغطى (Covered Calls)، وبيع Put المغطى بالكاش، والكوندور الحديدي. "
            "العوائد معتدلة، لذا ركّز على أرباح أسبوعية ثابتة دون المبالغة في المخاطرة."
        )
    elif macro.vix < 24.0:
        summary = (
            f"توتر السوق متوسط (مؤشر VIX عند {macro.vix:.1f} والتضخم ~{macro.cpi_yoy_pct:.1f}%). "
            "لأن المتداولين يدفعون أكثر للتأمين، يحصل بائعو الخيارات على عوائد نقدية أكبر بوضوح."
        )
        advice = (
            "منطقة مثالية لسبريد البيع الصاعد محدود المخاطرة (Bull Put Spread) وبيع Put بهامش أمان واسع. "
            "حافظ على حجم صفقات معتدل واختر أسعار تنفيذ بعيدة عن السعر الحالي."
        )
    else:
        summary = (
            f"التقلبات مرتفعة حالياً (مؤشر VIX عند {macro.vix:.1f}) — الأخبار والمخاوف الاقتصادية تسبب تذبذبات يومية قوية. "
            "عوائد الخيارات مغرية، لكن الأسعار قد تتحرك بسرعة كبيرة."
        )
        advice = (
            "احمِ رأس مالك أولاً! التزم بصفقات صغيرة محدودة المخاطرة (Credit Spreads) مع هامش أمان واسع جداً، "
            "أو انتظر هدوء العاصفة قبل بيع خيارات قصيرة الأجل."
        )
    return {"label": label, "summary": summary, "advice": advice}


def tr_prediction(snapshot: StockSnapshot, pred: MarketPrediction, news_sentiment: float, lang: str) -> Dict[str, object]:
    if lang != "ar":
        return {
            "direction": pred.direction_label,
            "explanation": pred.plain_explanation,
            "drivers": pred.key_drivers,
        }
    dir_ar = DIRECTION_AR.get(pred.direction_label, pred.direction_label)
    explanation = (
        f"خلال الأيام الـ **{pred.horizon_days}** القادمة، يتوقع النموذج أن يميل سهم **{snapshot.symbol}** نحو **{dir_ar}** "
        f"قرب **${pred.predicted_price:,.2f}** (حوالي **{pred.expected_return_pct:+.1f}%**)، مع نطاق حركة طبيعي بين "
        f"**${pred.expected_low_1sigma:,.2f}** و **${pred.expected_high_1sigma:,.2f}**. "
        f"احتمال الإغلاق على ارتفاع هو **{pred.prob_up*100:.0f}%** (الدقة التاريخية المختبرة: {pred.backtest_accuracy_pct:.0f}%)."
    )
    drivers: List[str] = []
    if snapshot.price >= snapshot.sma_20 >= snapshot.sma_50:
        drivers.append("السعر يتحرك فوق متوسطه لـ 20 يوماً و 50 يوماً (اتجاه صاعد صحي).")
    elif snapshot.price < snapshot.sma_20 and snapshot.price < snapshot.sma_50:
        drivers.append("السعر يتداول أسفل متوسطه لـ 20 يوماً و 50 يوماً (ضغط بيعي حالي).")
    else:
        drivers.append("السعر يحوم قرب متوسطاته الأخيرة (حركة عرضية / متذبذبة).")

    if snapshot.rsi_14 >= 68:
        drivers.append(f"الزخم مرتفع جداً (مؤشر RSI عند {snapshot.rsi_14:.0f}) — غالباً ما يتبعه هدوء أو تصحيح بسيط.")
    elif snapshot.rsi_14 <= 32:
        drivers.append(f"السهم في منطقة تشبع بيعي (مؤشر RSI عند {snapshot.rsi_14:.0f}) — يزداد احتمال الارتداد أو الاستقرار.")
    else:
        drivers.append(f"الزخم متوازن (مؤشر RSI عند {snapshot.rsi_14:.0f}) — لا يوجد ذعر أو تضخم مفرط.")

    if news_sentiment >= 0.15:
        drivers.append(f"مزاج الأخبار العامة الحية إيجابي ({news_sentiment:+.2f})، مما يمنح السهم دعماً إضافياً.")
    elif news_sentiment <= -0.15:
        drivers.append(f"مزاج الأخبار العامة الحية سلبي/حذر ({news_sentiment:+.2f})، مما يرفع مخاطر العناوين المفاجئة.")
    else:
        drivers.append("الأخبار العامة الحية محايدة في الغالب — لا توجد صدمة خبرية مسيطرة حالياً.")

    if snapshot.next_earnings_days is not None and snapshot.next_earnings_days <= pred.horizon_days + 2:
        drivers.append(f"⚠️ إعلان أرباح الشركة بعد ~{snapshot.next_earnings_days} يوم — توقّع قفزات سعرية حادة!")

    return {"direction": dir_ar, "explanation": explanation, "drivers": drivers}


def tr_strategy(plan: OptionStrategyPlan, spot: float, prediction: MarketPrediction, lang: str) -> Dict[str, object]:
    if lang != "ar":
        return {
            "name": plan.strategy_name,
            "short_label": plan.short_label,
            "risk_level": plan.risk_level,
            "tagline": plan.plain_tagline,
            "contract_summary": plan.contract_summary,
            "what_you_do": plan.what_you_do,
            "how_you_make_money": plan.how_you_make_money,
            "what_is_your_risk": plan.what_is_your_risk,
            "playbook": plan.repeatable_playbook,
            "fit_reason": plan.fit_reason,
            "earnings_warning": plan.earnings_warning,
        }

    name_ar, short_ar = STRATEGY_NAMES_AR.get(plan.strategy_id, (plan.strategy_name, plan.short_label))
    risk_ar = RISK_LEVEL_AR.get(plan.risk_level, plan.risk_level)
    earn_ar = None
    if plan.earnings_warning:
        earn_ar = (
            "موعد إعلان أرباح الشركة يقع خلال فترة هذه الصفقة! "
            "يُنصح للمبتدئين بتجنب الصفقة أو تقليل حجمها خلال أسبوع الأرباح."
        )

    be_low = plan.breakeven_low or spot
    be_high = plan.breakeven_high or spot
    cushion_pct = max(0.0, ((spot - be_low) / spot) * 100.0)

    if plan.strategy_id == "bull_put_spread":
        tagline = f"اقبض ${plan.upfront_cash:,.0f} كاش مقدماً — تربح طالما بقي {plan.symbol} فوق ${be_low:,.2f}."
        what_do = (
            f"تبيع {plan.contracts} عقد Put للحصول على دخل فوري، وتشتري معه عقد Put أرخص تحته كتأمين وحماية — "
            f"تنتهي خلال **{plan.dte} يوم** ({plan.expiration})."
        )
        how_money = (
            f"تستلم **${plan.upfront_cash:,.2f} كاش فوري** في حسابك. يمكن لسهم {plan.symbol} أن يصعد أو يثبت أو حتى يهبط بنسبة "
            f"**{cushion_pct:.1f}%** (حتى ${be_low:,.2f}) وتظل تحتفظ بـ 100% من **${plan.upfront_cash:,.2f}**! "
            f"هذا عائد **{plan.return_on_capital_pct:.1f}%** على مبلغ الضمان (${plan.max_risk:,.0f}) في {plan.dte} يوم فقط."
        )
        risk_txt = (
            f"بفضل عقد التأمين الذي اشتريته، أقصى خسارة ممكنة **محدودة ومقفولة عند ${plan.max_risk:,.2f} فقط** "
            f"(حتى لو انهار السهم تماماً). تبدأ الخسارة فقط إذا أغلق {plan.symbol} تحت **${be_low:,.2f}** عند الانتهاء."
        )
        playbook = [
            f"**قاعدة الـ 50%:** إذا حققت نصف الربح (~${plan.upfront_cash*0.5:,.0f}) قبل موعد الانتهاء، أغلق الصفقة مبكراً واحفظ أرباحك.",
            f"**خروج الأمان:** إذا هبط السهم قرب ${be_low:,.2f} أو وصلت الخسارة إلى 1.5× الكاش (~${plan.upfront_cash*1.5:,.0f})، أغلق الصفقة لحماية رأس المال.",
            f"**التكرار المنظم:** بعد الإغلاق، افتح صفقة جديدة لمدة {plan.dte} أيام مع إبقاء كل صفقة أقل من 20–40% من حسابك.",
        ]
        fit = (
            f"الأنسب عندما يكون السهم مستقراً أو مائلاً للصعود (احتمال عدم الهبوط الحاد: {prediction.prob_flat_or_up*100:.0f}%) "
            "وتريد مخاطرة محدودة وميزانية صغيرة."
        )
    elif plan.strategy_id == "iron_condor":
        tagline = f"اقبض ${plan.upfront_cash:,.0f} إذا بقي {plan.symbol} داخل منطقة الأمان (${be_low:,.2f} – ${be_high:,.2f})."
        what_do = (
            f"ترسم 'سياج أمان' حول سعر {plan.symbol} — تبيع Put أسفل السعر و Call أعلى السعر مع عقود حماية خارجية من الجهتين "
            f"لمدة **{plan.dte} يوم** ({plan.expiration})."
        )
        how_money = (
            f"تجمع **${plan.upfront_cash:,.2f} كاش مقدماً** من الجانبين معاً. لا تحتاج لتخمين هل سيصعد السهم أم سيهبط — "
            f"طالما أغلق {plan.symbol} داخل النطاق (**${be_low:,.2f}** إلى **${be_high:,.2f}**)، تحتفظ بكامل الربح (**{plan.return_on_capital_pct:.1f}%**)."
        )
        risk_txt = (
            f"إذا قفز {plan.symbol} بقوة فوق **${be_high:,.2f}** أو هبط تحت **${be_low:,.2f}** تحدث خسارة، "
            f"لكنها **محدودة ومقفولة بحد أقصى ${plan.max_risk:,.2f}**."
        )
        playbook = [
            f"**قاعدة الـ 50%:** أغلق الصفقة عند تحقيق ~${plan.upfront_cash*0.5:,.0f} ربح لتجنب تقلبات اليوم الأخير.",
            "**تجنّب الأخبار الكبرى:** لا تترك الكوندور الحديدي مفتوحاً أثناء إعلان الأرباح أو قرارات الفيدرالي المفاجئة.",
            f"**تآكل زمني مزدوج:** كلا الجانبين يعمل لصالحك يومياً (~${plan.daily_theta_dollars:,.2f}/يوم طالما السعر داخل النطاق).",
        ]
        fit = "مثالية عندما يتوقع النموذج حركة هادئة أو عرضية وتكون علاوة التقلب مجزية على الجانبين."
    elif plan.strategy_id == "cash_secured_put":
        tagline = f"اربح ${plan.upfront_cash:,.0f} كاش مقدماً — أو اشترِ {plan.symbol} بخصم {cushion_pct:.1f}% (${be_low:,.2f})."
        what_do = (
            f"تحجز **${plan.capital_required:,.0f}** نقداً في حسابك وتتعهد بشراء {plan.contracts*100} سهم من {plan.symbol} "
            f"فقط إذا هبط السعر تحت سعر التنفيذ خلال {plan.dte} يوم ({plan.expiration})."
        )
        how_money = (
            f"تحصل على **${plan.upfront_cash:,.2f} كاش فوراً** ({plan.return_on_capital_pct:.2f}% خلال {plan.dte} أيام ≈ {plan.annualized_yield_pct:.0f}% سنوياً عند التكرار). "
            f"إذا بقي {plan.symbol} مرتفعاً، تحتفظ بالكاش وبمبلغك الكامل (${plan.capital_required:,.0f}) وتكرر الأسبوع القادم!"
        )
        risk_txt = (
            f"إذا هبط {plan.symbol} أسفل سعر التنفيذ، تشتري {plan.contracts*100} سهم بتكلفة صافية **${be_low:,.2f}** للسهم. "
            f"تماماً كامتلاك السهم، تخسر إذا واصل الهبوط تحت ${be_low:,.2f} — لكن يمكنك فوراً بيع Covered Calls عليها لجني دخل أسبوعي أثناء انتظار التعافي!"
        )
        playbook = [
            f"**اختر أسهم جودة فقط:** استخدم استراتيجية العجلة فقط على أسهم/صناديق مثل {plan.symbol} لا تمانع امتلاكها فعلاً.",
            "**إذا هبط السهم واشتريته:** لا تقلق — انتقل مباشرة لبيع Covered Call أسبوعي فوق سعر تكلفتك.",
            f"**ملاحظة رأس المال:** تحتاج **${plan.capital_required:,.0f}** لكل 100 سهم" + ("" if plan.fits_budget else " (أعلى من ميزانيتك الحالية — جرّب SOFI أو XLF أو سبريد البيع الصاعد)."),
        ]
        fit = "المفضلة لدى المبتدئين لبناء دخل تراكمي هادئ بدون ديون هامش، وبنتيجتين واضحتين وسهلتين."
    else:  # covered_call
        tagline = f"اربح ${plan.upfront_cash:,.0f} كإيجار خلال {plan.dte} يوم (وحتى ${plan.max_profit:,.0f} إذا ارتفع السهم)."
        what_do = (
            f"تمتلك {plan.contracts*100} سهم من {plan.symbol} (~${plan.capital_required:,.0f}) وتبيع {plan.contracts} عقد Call "
            f"فوق السعر الحالي لمدة {plan.dte} يوم ({plan.expiration})."
        )
        how_money = (
            f"تقبض **${plan.upfront_cash:,.2f} كاش فوراً** ({plan.return_on_capital_pct:.2f}% خلال {plan.dte} أيام). "
            f"إذا بقي السهم تحت سعر التنفيذ تحتفظ بالأسهم وبالكاش معاً وتكررها! وإذا صعد السهم، تبيع أسهمك بربح إجمالي **${plan.max_profit:,.2f}**."
        )
        risk_txt = (
            f"مخاطرتك الأساسية هي هبوط السهم نفسه تحت **${be_low:,.2f}** (رغم أن الكاش ${plan.upfront_cash:,.2f} يخفف أول جزء من الهبوط). "
            f"كما أن صعود السهم بقوة كبيرة يحد ربحك الأقصى عند **${plan.max_profit:,.2f}**."
        )
        playbook = [
            f"**شيك إيجار متكرر:** بِع عقد Call جديداً فوق سعر شرائك كل {plan.dte} يوم لخفض متوسط تكلفتك تدريجياً.",
            "**لا تبع تحت تكلفتك:** اختر دائماً سعر تنفيذ مساوياً أو أعلى من سعر شرائك للأسهم.",
            f"**ملاحظة رأس المال:** تتطلب امتلاك 100 سهم (~${plan.capital_required:,.0f})" + ("" if plan.fits_budget else " — أعلى من ميزانيتك الحالية."),
        ]
        fit = "الأفضل إذا كنت تمتلك الأسهم (أو ترغب بامتلاكها) وتريد تحويل السهم المستقر إلى مصدر دخل دوري."

    return {
        "name": name_ar,
        "short_label": short_ar,
        "risk_level": risk_ar,
        "tagline": tagline,
        "contract_summary": plan.contract_summary,
        "what_you_do": what_do,
        "how_you_make_money": how_money,
        "what_is_your_risk": risk_txt,
        "playbook": playbook,
        "fit_reason": fit,
        "earnings_warning": earn_ar,
    }
