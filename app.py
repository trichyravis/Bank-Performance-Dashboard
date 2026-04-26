"""
=========================================================================
Indian Bank Stock Analyzer — Streamlit Dashboard
Financial Analytics | MBA ZG517 / PDBA / PDFI / PDFT
Prof. V. Ravichandran

A live, ICICI-style annual report dashboard for ANY Indian bank.

Run
---
    pip install streamlit yfinance pandas numpy plotly
    streamlit run bank_dashboard_app.py

Layout (mirrors the ICICI infographic):
    Header banner ─ Bank name + one-line thesis
    KPI strip     ─ Net Profit / ROE / NIM / Loan Growth / CAR
    Row A         ─ Profitability | Asset Quality | Loan Growth (pie)
    Row B         ─ Capital | Efficiency | Risks
    Scorecard     ─ Star ratings + "What stood out"
    Banner        ─ Overall fundamental rating

Data sources
    1. Live    : yfinance (price, market cap, P/B, P/E, financials)
    2. Curated : Bank-specific ratios (NIM, NPAs, CASA, segment mix)
                 that yfinance does NOT carry — pre-populated from FY25
                 filings for the major Indian banks. User can override
                 every value in the sidebar.

All curated numbers are illustrative for classroom use. Verify against
the latest annual report before any investment decision.
=========================================================================
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================================================================
# 1. THEME
# =========================================================================
NAVY        = "#102A43"
NAVY_LIGHT  = "#1F4E79"
GOLD        = "#E07B00"
GREEN       = "#2D8E47"
RED         = "#C0392B"
SLATE       = "#4A5568"
BG          = "#F7FAFC"
CARD        = "#FFFFFF"
DIVIDER     = "#E2E8F0"

PALETTE = [NAVY_LIGHT, GOLD, GREEN, "#7B2CBF", "#0E7C66", "#B23A48"]

st.set_page_config(
    page_title="Indian Bank Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light styling overrides — keep it elegant
st.markdown(f"""
    <style>
    .main {{ background-color: {BG}; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }}

    .hero {{
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
        color: white; border-radius: 14px; padding: 24px 32px;
        margin-bottom: 18px;
    }}
    .hero h1 {{ font-size: 2.0rem; margin: 0; font-weight: 700; }}
    .hero p  {{ font-size: 1.0rem; margin: 6px 0 0 0; opacity: 0.92; }}

    .kpi-card {{
        background: {CARD}; border: 1px solid {DIVIDER};
        border-radius: 12px; padding: 16px 18px; height: 100%;
    }}
    .kpi-label {{ color: {SLATE}; font-size: 0.80rem; letter-spacing: 0.5px; text-transform: uppercase; }}
    .kpi-value {{ color: {NAVY}; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
    .kpi-sub   {{ color: {SLATE}; font-size: 0.85rem; margin-top: 2px; }}

    .card {{
        background: {CARD}; border: 1px solid {DIVIDER};
        border-radius: 12px; padding: 18px; height: 100%;
    }}
    .card h3 {{
        color: {NAVY}; font-size: 1.05rem; margin: 0 0 12px 0;
        padding-bottom: 8px; border-bottom: 2px solid {DIVIDER};
    }}
    .takeaway {{
        background: #F0F9FF; border-left: 4px solid {NAVY_LIGHT};
        padding: 10px 14px; border-radius: 6px; margin-top: 10px;
        font-size: 0.92rem; color: {NAVY};
    }}

    .verdict {{
        background: {GREEN}; color: white;
        padding: 22px 28px; border-radius: 12px;
        text-align: center; font-size: 1.6rem; font-weight: 700;
        letter-spacing: 1px;
    }}
    .verdict.amber  {{ background: {GOLD}; }}
    .verdict.red    {{ background: {RED}; }}
    .verdict.green  {{ background: {GREEN}; }}

    .stMetric > div {{ background: transparent; }}
    </style>
""", unsafe_allow_html=True)


# =========================================================================
# 2. CURATED BANK DATA (illustrative FY25 / Q4 FY26 figures)
# =========================================================================
BANKS: Dict[str, Dict[str, Any]] = {
    "HDFC Bank": {
        "ticker": "HDFCBANK.NS",
        "fy25": dict(net_profit_cr=67200, yoy_pat=15.0, roe=16.8, nim=3.55,
                     loan_growth=7.2, car=19.30, gross_npa=1.42, net_npa=0.46,
                     casa=38.0, cost_income=39.5, fy24_npa=1.24,
                     mix=dict(Retail=55, SME=12, Corporate=22, Others=11)),
    },
    "ICICI Bank": {
        "ticker": "ICICIBANK.NS",
        "fy25": dict(net_profit_cr=51470, yoy_pat=15.6, roe=19.0, nim=4.40,
                     loan_growth=15.8, car=17.20, gross_npa=1.67, net_npa=0.39,
                     casa=39.0, cost_income=39.0, fy24_npa=2.26,
                     mix=dict(Retail=54, SME=11, Corporate=28, Others=7)),
    },
    "State Bank of India": {
        "ticker": "SBIN.NS",
        "fy25": dict(net_profit_cr=71000, yoy_pat=14.5, roe=17.4, nim=3.15,
                     loan_growth=14.0, car=14.30, gross_npa=1.82, net_npa=0.47,
                     casa=38.0, cost_income=51.5, fy24_npa=2.24,
                     mix=dict(Retail=42, SME=15, Corporate=29, Others=14)),
    },
    "Kotak Mahindra Bank": {
        "ticker": "KOTAKBANK.NS",
        "fy25": dict(net_profit_cr=16400, yoy_pat=10.5, roe=14.0, nim=4.95,
                     loan_growth=17.0, car=22.30, gross_npa=1.42, net_npa=0.34,
                     casa=42.0, cost_income=46.0, fy24_npa=1.39,
                     mix=dict(Retail=46, SME=14, Corporate=30, Others=10)),
    },
    "Axis Bank": {
        "ticker": "AXISBANK.NS",
        "fy25": dict(net_profit_cr=26000, yoy_pat=20.0, roe=17.0, nim=4.05,
                     loan_growth=10.0, car=17.00, gross_npa=1.43, net_npa=0.31,
                     casa=41.0, cost_income=49.0, fy24_npa=1.43,
                     mix=dict(Retail=58, SME=11, Corporate=24, Others=7)),
    },
    "IndusInd Bank": {
        "ticker": "INDUSINDBK.NS",
        "fy25": dict(net_profit_cr=7500, yoy_pat=-12.0, roe=10.0, nim=3.95,
                     loan_growth=6.0, car=16.50, gross_npa=2.10, net_npa=0.60,
                     casa=38.0, cost_income=48.0, fy24_npa=1.92,
                     mix=dict(Retail=55, SME=10, Corporate=27, Others=8)),
    },
    "Bank of Baroda": {
        "ticker": "BANKBARODA.NS",
        "fy25": dict(net_profit_cr=19300, yoy_pat=12.0, roe=16.0, nim=3.05,
                     loan_growth=12.0, car=17.20, gross_npa=2.26, net_npa=0.59,
                     casa=38.0, cost_income=49.0, fy24_npa=2.92,
                     mix=dict(Retail=27, SME=15, Corporate=44, Others=14)),
    },
    "Punjab National Bank": {
        "ticker": "PNB.NS",
        "fy25": dict(net_profit_cr=17000, yoy_pat=85.0, roe=13.0, nim=3.00,
                     loan_growth=13.0, car=16.00, gross_npa=3.95, net_npa=0.40,
                     casa=41.0, cost_income=53.0, fy24_npa=5.73,
                     mix=dict(Retail=27, SME=15, Corporate=44, Others=14)),
    },
    "Federal Bank": {
        "ticker": "FEDERALBNK.NS",
        "fy25": dict(net_profit_cr=4000, yoy_pat=8.0, roe=13.0, nim=3.10,
                     loan_growth=16.0, car=16.40, gross_npa=2.10, net_npa=0.50,
                     casa=30.0, cost_income=53.0, fy24_npa=2.13,
                     mix=dict(Retail=53, SME=10, Corporate=28, Others=9)),
    },
    "IDFC First Bank": {
        "ticker": "IDFCFIRSTB.NS",
        "fy25": dict(net_profit_cr=2900, yoy_pat=-12.0, roe=9.0, nim=6.40,
                     loan_growth=22.0, car=16.50, gross_npa=1.90, net_npa=0.60,
                     casa=47.0, cost_income=72.0, fy24_npa=1.88,
                     mix=dict(Retail=63, SME=10, Corporate=22, Others=5)),
    },
    "AU Small Finance Bank": {
        "ticker": "AUBANK.NS",
        "fy25": dict(net_profit_cr=2100, yoy_pat=18.0, roe=14.0, nim=5.50,
                     loan_growth=25.0, car=18.50, gross_npa=1.78, net_npa=0.55,
                     casa=33.0, cost_income=63.0, fy24_npa=1.67,
                     mix=dict(Retail=60, SME=14, Corporate=18, Others=8)),
    },
}


# =========================================================================
# 3. HELPERS
# =========================================================================
def fmt_cr(v: float) -> str:
    """Format ₹ in lakh crore / crore."""
    if v is None or pd.isna(v):
        return "—"
    if abs(v) >= 1_00_000:
        return f"₹{v/1_00_000:.2f} L Cr"
    return f"₹{v:,.0f} Cr"


def fmt_pct(v: float, dec: int = 1, with_sign: bool = False) -> str:
    if v is None or pd.isna(v):
        return "—"
    s = f"{v:+.{dec}f}%" if with_sign else f"{v:.{dec}f}%"
    return s


def stars(n: int) -> str:
    n = max(1, min(5, int(round(n))))
    return "★" * n + "☆" * (5 - n)


def rate(value: float, thresholds: list, reverse: bool = False) -> int:
    """
    Return integer 1-5 based on value vs thresholds.
    thresholds is sorted in DESCENDING quality order (best -> worst):
        e.g. growth: [20, 15, 10, 5]   -> 5★ if >=20, 4★ if >=15, ...
    reverse=True for "lower is better" metrics (NPA, cost/income, P/B):
        e.g. NPA:    [1.5, 2.0, 3.0, 4.0] -> 5★ if <=1.5, 4★ if <=2, ...
    """
    if value is None or pd.isna(value):
        return 3
    if reverse:
        for i, t in enumerate(thresholds):
            if value <= t:
                return 5 - i
        return 1
    else:
        for i, t in enumerate(thresholds):
            if value >= t:
                return 5 - i
        return 1


@st.cache_data(ttl=600, show_spinner=False)
def fetch_market_data(ticker: str) -> Dict[str, Any]:
    """Pull a robust subset of yfinance data; squeeze multi-index DataFrames."""
    out: Dict[str, Any] = {}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        out["info"] = info
        out["price"] = info.get("currentPrice") or info.get("regularMarketPrice")
        out["market_cap"] = info.get("marketCap")
        out["pe"] = info.get("trailingPE")
        out["pb"] = info.get("priceToBook")
        out["dy"] = info.get("dividendYield")
        out["52w_high"] = info.get("fiftyTwoWeekHigh")
        out["52w_low"]  = info.get("fiftyTwoWeekLow")
        out["beta"]     = info.get("beta")
        out["currency"] = info.get("currency", "INR")

        hist = t.history(period="2y", auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        out["history"] = hist

        # Annual financials (last 4 years)
        try:
            fin = t.financials
            bs  = t.balance_sheet
            out["financials"] = fin
            out["balance_sheet"] = bs
        except Exception:
            out["financials"] = None
            out["balance_sheet"] = None
    except Exception as e:
        out["error"] = str(e)
    return out


def make_kpi(col, label: str, value: str, sub: str = ""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def card_open(title: str):
    st.markdown(f'<div class="card"><h3>{title}</h3>', unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# 4. SIDEBAR — selection + overrides
# =========================================================================
st.sidebar.markdown(f"## 🏦 Indian Bank Analyzer")
st.sidebar.caption("Live market data + curated fundamentals")

bank_name = st.sidebar.selectbox(
    "Choose a bank",
    list(BANKS.keys()),
    index=1,                       # default to ICICI to mirror your infographic
)

cfg = BANKS[bank_name].copy()
ticker_default = cfg["ticker"]

custom_ticker = st.sidebar.text_input(
    "or override the NSE ticker",
    value=ticker_default,
    help="e.g., HDFCBANK.NS, SBIN.NS, KOTAKBANK.NS",
)

st.sidebar.divider()
st.sidebar.markdown("**Override fundamentals (FY25)**")
with st.sidebar.expander("Edit ratios", expanded=False):
    f = cfg["fy25"]
    f["net_profit_cr"] = st.number_input("Net Profit (₹ Cr)", value=float(f["net_profit_cr"]), step=100.0)
    f["yoy_pat"]       = st.number_input("Net Profit YoY (%)", value=float(f["yoy_pat"]), step=0.5)
    f["roe"]           = st.number_input("ROE (%)",   value=float(f["roe"]),   step=0.1)
    f["nim"]           = st.number_input("NIM (%)",   value=float(f["nim"]),   step=0.05)
    f["loan_growth"]   = st.number_input("Loan Growth YoY (%)", value=float(f["loan_growth"]), step=0.5)
    f["car"]           = st.number_input("CAR (%)",   value=float(f["car"]),   step=0.1)
    f["gross_npa"]     = st.number_input("Gross NPA (%)", value=float(f["gross_npa"]), step=0.05)
    f["net_npa"]       = st.number_input("Net NPA (%)",   value=float(f["net_npa"]),   step=0.05)
    f["casa"]          = st.number_input("CASA (%)",  value=float(f["casa"]),  step=0.5)
    f["cost_income"]   = st.number_input("Cost-to-Income (%)", value=float(f["cost_income"]), step=0.5)
    f["fy24_npa"]      = st.number_input("FY24 Gross NPA (%) — for trend", value=float(f["fy24_npa"]), step=0.05)

st.sidebar.divider()
st.sidebar.markdown("**Position size (for VaR)**")
position_cr = st.sidebar.number_input("Position (₹ Cr)", value=2.0, min_value=0.1, step=0.5)

st.sidebar.divider()
refresh = st.sidebar.button("🔄 Refresh live data")
if refresh:
    fetch_market_data.clear()


# =========================================================================
# 5. FETCH LIVE DATA
# =========================================================================
md = fetch_market_data(custom_ticker)
hist = md.get("history")
price = md.get("price")
mcap  = md.get("market_cap")
pe    = md.get("pe")
pb    = md.get("pb")
dy    = (md.get("dy") or 0) * 100 if md.get("dy") is not None else None
hi52  = md.get("52w_high")
lo52  = md.get("52w_low")
beta  = md.get("beta")


# =========================================================================
# 6. HEADER
# =========================================================================
narrative_map = {
    range(0, 4):  ("⚠️ Stress signals", RED),
    range(4, 7):  ("Steady franchise", GOLD),
    range(7, 10): ("Quality compounder", GREEN),
}

st.markdown(f"""
<div class="hero">
    <h1>🏦 {bank_name} — Annual Report Analytics ({datetime.now().strftime('%b %Y')})</h1>
    <p>Live market data via Yahoo Finance • Curated fundamentals from FY25 / Q4 FY26 disclosures</p>
</div>
""", unsafe_allow_html=True)

if "error" in md or price is None:
    st.warning(f"Could not fetch live data for **{custom_ticker}**. Showing curated fundamentals only. "
               f"({md.get('error', 'no price returned')})")


# =========================================================================
# 7. KPI STRIP
# =========================================================================
f = cfg["fy25"]
k1, k2, k3, k4, k5 = st.columns(5)
make_kpi(k1, "Net Profit (FY25)", fmt_cr(f["net_profit_cr"]), f"YoY {fmt_pct(f['yoy_pat'], 1, True)}")
make_kpi(k2, "ROE",                fmt_pct(f["roe"]),          "Robust & sustainable" if f["roe"] >= 15 else "Below peer median")
make_kpi(k3, "NIM",                fmt_pct(f["nim"], 2),       "Among strongest" if f["nim"] >= 4 else "Sector average")
make_kpi(k4, "Loan Growth",        fmt_pct(f["loan_growth"]),  "Diversified" if f["loan_growth"] >= 12 else "Moderating")
make_kpi(k5, "CAR",                fmt_pct(f["car"], 2),       "Above regulatory")

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================================
# 8. ROW A — Profitability | Asset Quality | Loan Mix
# =========================================================================
a1, a2, a3 = st.columns(3)

with a1:
    card_open("📈 1. Profitability Momentum")
    st.markdown(f"""
- **PAT growth:** {fmt_pct(f['yoy_pat'], 1, True)} YoY — driven by loan expansion and operating leverage
- **ROE:** {fmt_pct(f['roe'])} — {'top-tier' if f['roe'] >= 17 else 'peer-median'}
- **NIM:** {fmt_pct(f['nim'], 2)} — {'best-in-class' if f['nim'] >= 4 else 'in line'}
""")
    st.markdown(f"""<div class="takeaway"><b>Takeaway:</b> {
        "Compounding franchise — earnings quality is structural." if f['yoy_pat'] >= 12 and f['roe'] >= 15
        else "Earnings under pressure — needs catalyst to re-rate."
    }</div>""", unsafe_allow_html=True)
    card_close()

with a2:
    card_open("🛡️ 2. Asset Quality")
    npa_df = pd.DataFrame({
        "Metric": ["Gross NPA", "Net NPA"],
        "FY24":   [f["fy24_npa"],            f["net_npa"] + (f["fy24_npa"] - f["gross_npa"])],
        "FY25":   [f["gross_npa"],           f["net_npa"]],
    })
    st.dataframe(
        npa_df.style.format({"FY24": "{:.2f}%", "FY25": "{:.2f}%"})
                    .set_properties(**{"font-size": "0.92rem"}),
        hide_index=True, use_container_width=True,
    )
    delta = f["fy24_npa"] - f["gross_npa"]
    st.markdown(f"""
- **Gross NPA**: improved by {delta:+.2f} pp YoY
- **Net NPA**: {fmt_pct(f['net_npa'], 2)} — {'best-in-class' if f['net_npa'] <= 0.5 else 'sector average'}
- Conservative provisioning continues
""")
    st.markdown(f"""<div class="takeaway"><b>Impact:</b> Higher earnings quality and lower credit-cost risk.</div>""",
                unsafe_allow_html=True)
    card_close()

with a3:
    card_open("🌐 3. Loan Mix & Growth")
    mix = pd.DataFrame(list(f["mix"].items()), columns=["Segment", "Weight"])
    fig = px.pie(
        mix, values="Weight", names="Segment", hole=0.55,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textinfo="label+percent", textfont_size=11)
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=10), height=240,
        showlegend=False, paper_bgcolor=CARD,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Loan growth (Q4 FY26):** {fmt_pct(f['loan_growth'], 1, True)} YoY")
    st.caption("Diversified across segments; deposit franchise supports growth.")
    card_close()

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================================
# 9. ROW B — Capital | Efficiency | Risks
# =========================================================================
b1, b2, b3 = st.columns(3)

with b1:
    card_open("🏛️ 4. Capital Position")
    st.markdown(f"""
- **CAR:** {fmt_pct(f['car'], 2)} — well above regulatory minimums
- Provides room for: future growth, dividends, shock absorption
- **CASA:** {fmt_pct(f['casa'])} — {'low-cost deposit base' if f['casa'] >= 38 else 'mix could improve'}
""")
    st.markdown(f"""<div class="takeaway"><b>Interpretation:</b> Balance-sheet strength is a key long-term advantage.</div>""",
                unsafe_allow_html=True)
    card_close()

with b2:
    card_open("⚙️ 5. Efficiency")
    st.markdown(f"""
- **Cost-to-Income:** {fmt_pct(f['cost_income'], 1)}
- {'Improving steadily — digital tailwind' if f['cost_income'] <= 45 else 'Room to optimise via digital lending'}
- Drivers: lower OPEX growth, better customer experience, higher scalability
""")
    st.markdown(f"""<div class="takeaway"><b>Read:</b> Increasingly a technology-led bank, not just a traditional lender.</div>""",
                unsafe_allow_html=True)
    card_close()

with b3:
    card_open("⚠️ Risks / Concerns")
    risks = []
    if pb is not None and pb > 2.5:
        risks.append(f"**Valuation rich:** P/B {pb:.1f}x — quality already priced in")
    if f["nim"] >= 4.0:
        risks.append("**Margin pressure:** rate cuts could compress NIM")
    if f["gross_npa"] >= 2.0:
        risks.append("**Credit-cycle risk:** asset quality is cyclical")
    if f["loan_growth"] >= 18:
        risks.append("**Growth quality:** monitor unsecured retail seasoning")
    if f["cost_income"] >= 55:
        risks.append("**Operating leverage:** elevated cost base needs to come down")
    if not risks:
        risks.append("No material red flags — monitor macro shocks and rate trajectory")
    for r in risks:
        st.markdown(f"- {r}")
    card_close()

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================================
# 10. PRICE & VOLATILITY (live)
# =========================================================================
if hist is not None and not hist.empty:
    p1, p2 = st.columns([2, 1])

    with p1:
        card_open(f"📊 Price action — last 24 months ({custom_ticker})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Close"], mode="lines",
            line=dict(color=NAVY_LIGHT, width=2), name="Close",
        ))
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor=DIVIDER, title="₹"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    with p2:
        card_open("Risk metrics")
        # Returns + volatility + 1-day 99% VaR on user position
        r = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        sigma_d = float(r.std(ddof=1))
        sigma_a = sigma_d * math.sqrt(252)
        from math import sqrt
        z = 2.326           # 99% one-tailed
        var_pct = z * sigma_d
        var_inr = var_pct * (position_cr * 1_00_00_000)
        max_dd = float((hist["Close"] / hist["Close"].cummax() - 1).min())

        st.metric("Daily σ",          f"{sigma_d:.2%}")
        st.metric("Annualised σ",     f"{sigma_a:.1%}")
        st.metric("1-day 99% VaR",    f"{var_pct:.2%}",
                  f"≈ ₹{var_inr/1_00_000:.1f} L on ₹{position_cr:.1f} Cr")
        st.metric("Max drawdown (2Y)", f"{max_dd:.1%}")
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)


# =========================================================================
# 11. MARKET SNAPSHOT
# =========================================================================
m1, m2, m3, m4, m5, m6 = st.columns(6)
make_kpi(m1, "Price",         f"₹{price:,.1f}" if price else "—")
make_kpi(m2, "Market Cap",    fmt_cr(mcap / 1_00_00_000) if mcap else "—")
make_kpi(m3, "P/E (TTM)",     f"{pe:.1f}x" if pe else "—")
make_kpi(m4, "P/B",           f"{pb:.1f}x" if pb else "—")
make_kpi(m5, "Dividend Yield", fmt_pct(dy, 2) if dy else "—")
make_kpi(m6, "Beta",          f"{beta:.2f}" if beta else "—")
if hi52 and lo52:
    st.caption(f"52-week range: ₹{lo52:,.0f} – ₹{hi52:,.0f}   |   Currency: {md.get('currency','INR')}")

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================================
# 12. FUNDAMENTAL SCORECARD
# =========================================================================
score = {
    "Profit Growth":   (rate(f["yoy_pat"],     [20, 15, 10, 5]),                f"{fmt_pct(f['yoy_pat'], 1, True)} YoY"),
    "Asset Quality":   (rate(f["gross_npa"],   [1.5, 2.0, 3.0, 4.0], reverse=True), f"Gross NPA {fmt_pct(f['gross_npa'], 2)}"),
    "Loan Growth":     (rate(f["loan_growth"], [18, 14, 10, 6]),                f"{fmt_pct(f['loan_growth'])}"),
    "Capital Strength":(rate(f["car"],         [18, 16, 14, 12]),               f"CAR {fmt_pct(f['car'], 2)}"),
    "Efficiency":      (rate(f["cost_income"], [40, 45, 50, 55], reverse=True), f"C/I {fmt_pct(f['cost_income'], 1)}"),
    "Valuation":       (rate(pb if pb else 999, [1.5, 2.0, 2.5, 3.5], reverse=True),
                        f"P/B {pb:.1f}x" if pb else "P/B —"),
}

s1, s2 = st.columns([3, 2])
with s1:
    card_open("📋 Fundamental Scorecard")
    sc_df = pd.DataFrame([
        {"Metric": k, "Rating": stars(v[0]), "Comment": v[1], "Score": v[0]}
        for k, v in score.items()
    ])
    st.dataframe(
        sc_df[["Metric", "Rating", "Comment"]],
        hide_index=True, use_container_width=True,
    )
    avg = np.mean([v[0] for v in score.values()])
    st.caption(f"Average score: **{avg:.2f} / 5**")
    card_close()

with s2:
    card_open("✨ What stood out")
    highlights = []
    if f["gross_npa"] < f["fy24_npa"]:
        highlights.append("**Asset quality improvement is structural, not temporary.**")
    if f["roe"] >= 15 and f["yoy_pat"] >= 10:
        highlights.append("**Profitability remains durable.**")
    if f["car"] >= 16 and f["loan_growth"] >= 10:
        highlights.append("**Capital + growth combination is attractive.**")
    if f["cost_income"] <= 45:
        highlights.append("**Digital-led efficiency is becoming a moat.**")
    if not highlights:
        highlights.append("Mixed quarter — wait for next two prints to confirm trend.")
    for i, h in enumerate(highlights, 1):
        st.markdown(f"**{i}.** {h}")

    st.markdown(f"""<div class="takeaway"><b>Investor Takeaway</b><br>
    Long-term (5+ years): {'Constructive' if avg >= 3.5 else 'Cautious'}<br>
    Value investor: {'May wait for better entry points' if (pb or 0) > 2.5 else 'Risk/reward is balanced'}<br>
    Dividend investor: {'Decent, not primary attraction' if (dy or 0) < 2 else 'Attractive yield'}<br>
    Quality compounder investor: {'Very interesting' if avg >= 4.0 else 'Selective'}
    </div>""", unsafe_allow_html=True)
    card_close()

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================================
# 13. OVERALL VERDICT
# =========================================================================
if avg >= 4.3:
    verdict = ("🐂 BULLISH / HIGH-QUALITY COMPOUNDER", "green")
elif avg >= 3.5:
    verdict = ("CONSTRUCTIVE / SOLID FRANCHISE", "green")
elif avg >= 2.5:
    verdict = ("NEUTRAL / WAIT FOR CATALYSTS", "amber")
else:
    verdict = ("⚠️ CAUTIOUS / STRESS SIGNALS", "red")

st.markdown(
    f'<div class="verdict {verdict[1]}">OVERALL RATING: {verdict[0]}</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "**Disclaimer.** Curated fundamentals are illustrative figures based on FY25 / Q4 FY26 "
    "disclosures and are intended for classroom and analytical use only. Verify against the latest "
    "annual report and exchange filings before any investment decision. Live market data via "
    "Yahoo Finance — subject to data quality and feed availability."
)
