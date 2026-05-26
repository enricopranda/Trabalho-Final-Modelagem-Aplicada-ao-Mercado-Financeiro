"""
============================================================
VaR PROFESSIONAL CALCULATOR — Streamlit App
Modelagem Aplicada ao Mercado Financeiro
============================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import norm, kurtosis, skew
from datetime import date, timedelta
import warnings, time, random
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VaR Terminal · Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --bg:#0a0c0f; --bg2:#0f1318; --card:#141920; --card2:#1a2030;
  --border:#1e2a3a; --blue:#00a8ff; --green:#00e676; --red:#ff3d57;
  --amber:#ffab00; --purple:#c77dff; --text:#e8edf5; --muted:#7a8fa6; --dim:#3d5068;
}
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--text);}
.main .block-container{background:var(--bg)!important;padding-top:0.8rem!important;max-width:1500px;}
section[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border)!important;}

/* Sidebar section labels */
section[data-testid="stSidebar"] h3{
  color:var(--blue)!important;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
  letter-spacing:0.18em;text-transform:uppercase;border-bottom:1px solid var(--border);
  padding-bottom:0.35rem;margin:1rem 0 0.6rem!important;}

/* Metrics */
[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:6px!important;padding:0.9rem!important;}
[data-testid="metric-container"] label{color:var(--muted)!important;font-family:'IBM Plex Mono',monospace!important;font-size:0.62rem!important;letter-spacing:0.12em;text-transform:uppercase;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--blue)!important;font-family:'IBM Plex Mono',monospace!important;font-size:1.25rem!important;}
[data-testid="stMetricDelta"]{font-family:'IBM Plex Mono',monospace!important;font-size:0.7rem!important;}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#0057a8,#0086d4)!important;color:#fff!important;
  border:none!important;border-radius:4px!important;font-family:'IBM Plex Mono',monospace!important;
  font-size:0.75rem!important;letter-spacing:0.1em;text-transform:uppercase;padding:0.65rem 1.5rem!important;transition:all 0.2s;}
.stButton>button:hover{background:linear-gradient(135deg,#0070cc,#00a8ff)!important;box-shadow:0 0 20px rgba(0,168,255,0.3)!important;}

/* Inputs */
.stTextInput>div>div>input,.stNumberInput>div>div>input{
  background:var(--card2)!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:4px!important;font-family:'IBM Plex Mono',monospace!important;font-size:0.8rem!important;}
.stSelectbox>div>div>div{background:var(--card2)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:4px!important;}
.stSlider [data-baseweb="slider"] div{background:var(--blue)!important;}
.stMultiSelect [data-baseweb="tag"]{background:rgba(0,168,255,0.2)!important;border:1px solid var(--blue)!important;}
.stMultiSelect>div>div{background:var(--card2)!important;border:1px solid var(--border)!important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--card)!important;border-bottom:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;
  font-family:'IBM Plex Mono',monospace!important;font-size:0.68rem!important;letter-spacing:0.1em;text-transform:uppercase;border-radius:0!important;}
.stTabs [aria-selected="true"]{background:var(--card2)!important;color:var(--blue)!important;border-bottom:2px solid var(--blue)!important;}

/* Dataframe */
.stDataFrame{border:1px solid var(--border)!important;border-radius:6px!important;}

/* Custom divs */
.sec-hdr{font-family:'IBM Plex Mono',monospace;font-size:0.6rem;letter-spacing:0.22em;text-transform:uppercase;
  color:var(--blue);border-left:3px solid var(--blue);padding-left:0.7rem;margin:1.3rem 0 0.7rem;}
.info-box{background:rgba(0,168,255,0.07);border:1px solid rgba(0,168,255,0.25);border-left:3px solid var(--blue);
  border-radius:4px;padding:0.7rem 0.9rem;font-size:0.8rem;margin:0.6rem 0;}
.warn-box{background:rgba(255,61,87,0.07);border:1px solid rgba(255,61,87,0.25);border-left:3px solid var(--red);
  border-radius:4px;padding:0.7rem 0.9rem;font-size:0.8rem;margin:0.6rem 0;}
.ok-box{background:rgba(0,230,118,0.07);border:1px solid rgba(0,230,118,0.25);border-left:3px solid var(--green);
  border-radius:4px;padding:0.7rem 0.9rem;font-size:0.8rem;margin:0.6rem 0;}

/* stock cards */
.stock-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1rem 1.1rem;margin-bottom:0.7rem;}
.stock-ticker{font-family:'IBM Plex Mono',monospace;font-size:0.9rem;font-weight:600;color:var(--blue);}
.stock-name{font-size:0.75rem;color:var(--muted);margin-bottom:0.6rem;}
.stock-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.4rem;}
.stock-item{background:var(--card2);border-radius:4px;padding:0.4rem 0.6rem;}
.stock-item-label{font-family:'IBM Plex Mono',monospace;font-size:0.55rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--dim);}
.stock-item-val{font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:var(--text);margin-top:0.1rem;}

/* Welcome hero */
.hero{background:linear-gradient(135deg,#0a0c0f 0%,#0d1a2a 60%,#0a0c0f 100%);
  border:1px solid var(--border);border-radius:10px;padding:3.5rem 3rem;text-align:center;margin-bottom:1.5rem;}

/* Expander */
.streamlit-expanderHeader{background:var(--card)!important;border:1px solid var(--border)!important;
  border-radius:4px!important;font-family:'IBM Plex Mono',monospace!important;font-size:0.72rem!important;color:var(--muted)!important;}

#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DEFAULTS
# ─────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#7a8fa6", size=10),
    title_font=dict(family="IBM Plex Sans", color="#e8edf5", size=13),
    xaxis=dict(gridcolor="#1e2a3a", linecolor="#1e2a3a", tickcolor="#3d5068"),
    yaxis=dict(gridcolor="#1e2a3a", linecolor="#1e2a3a", tickcolor="#3d5068"),
    legend=dict(bgcolor="rgba(15,19,24,0.85)", bordercolor="#1e2a3a", borderwidth=1),
    margin=dict(l=55, r=25, t=50, b=40),
)
COLORS = ["#00a8ff", "#ffab00", "#00e676", "#c77dff", "#ff7c43", "#ff3d57"]

# ─────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────
def bs(S, K, T, r, sigma, tipo="call"):
    if T <= 0:
        return max(S-K,0) if tipo=="call" else max(K-S,0)
    if sigma <= 0:
        return (max(S-K*np.exp(-r*T),0) if tipo=="call" else max(K*np.exp(-r*T)-S,0))
    d1=(np.log(S/K)+(r+.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    if tipo=="call": return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def gregas(S, K, T, r, sigma, tipo="call"):
    if T<=0 or sigma<=0: return dict(delta=0,gamma=0,vega=0,theta=0,rho=0)
    d1=(np.log(S/K)+(r+.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    delta = norm.cdf(d1) if tipo=="call" else norm.cdf(d1)-1
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega  = S*norm.pdf(d1)*np.sqrt(T)/100
    if tipo=="call":
        theta=(-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/252
        rho=K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        theta=(-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/252
        rho=-K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    return dict(delta=delta,gamma=gamma,vega=vega,theta=theta,rho=rho)

# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (realistic fallback)
# ─────────────────────────────────────────────
TICKER_PARAMS = {
    # BR stocks
    "PETR4.SA": {"name":"Petrobras PN","S0":38.5,"mu":0.12,"sigma":0.38,"sector":"Energia"},
    "VALE3.SA": {"name":"Vale ON","S0":62.0,"mu":0.08,"sigma":0.35,"sector":"Mineração"},
    "ITUB4.SA": {"name":"Itaú Unibanco PN","S0":35.0,"mu":0.14,"sigma":0.28,"sector":"Financeiro"},
    "BBDC4.SA": {"name":"Bradesco PN","S0":14.5,"mu":0.09,"sigma":0.30,"sector":"Financeiro"},
    "BBAS3.SA": {"name":"Banco do Brasil ON","S0":28.0,"mu":0.15,"sigma":0.29,"sector":"Financeiro"},
    "MGLU3.SA": {"name":"Magazine Luiza ON","S0":8.5,"mu":-0.05,"sigma":0.65,"sector":"Varejo"},
    "WEGE3.SA": {"name":"WEG ON","S0":52.0,"mu":0.18,"sigma":0.30,"sector":"Industrial"},
    "RENT3.SA": {"name":"Localiza ON","S0":42.0,"mu":0.10,"sigma":0.32,"sector":"Serviços"},
    "LREN3.SA": {"name":"Lojas Renner ON","S0":18.0,"mu":0.05,"sigma":0.38,"sector":"Varejo"},
    "ABEV3.SA": {"name":"Ambev ON","S0":13.0,"mu":0.07,"sigma":0.22,"sector":"Consumo"},
    "JBSS3.SA": {"name":"JBS ON","S0":32.0,"mu":0.11,"sigma":0.33,"sector":"Alimentos"},
    "SBSP3.SA": {"name":"Sabesp ON","S0":78.0,"mu":0.09,"sigma":0.26,"sector":"Saneamento"},
    "EGIE3.SA": {"name":"Engie Brasil ON","S0":44.0,"mu":0.10,"sigma":0.20,"sector":"Utilities"},
    "SUZB3.SA": {"name":"Suzano ON","S0":56.0,"mu":0.12,"sigma":0.34,"sector":"Papel & Celulose"},
    "RADL3.SA": {"name":"Raia Drogasil ON","S0":30.0,"mu":0.15,"sigma":0.25,"sector":"Saúde"},
    # US stocks
    "AAPL":     {"name":"Apple Inc.","S0":185.0,"mu":0.18,"sigma":0.26,"sector":"Technology"},
    "MSFT":     {"name":"Microsoft Corp.","S0":415.0,"mu":0.22,"sigma":0.24,"sector":"Technology"},
    "GOOGL":    {"name":"Alphabet Inc.","S0":175.0,"mu":0.16,"sigma":0.27,"sector":"Technology"},
    "AMZN":     {"name":"Amazon.com Inc.","S0":195.0,"mu":0.20,"sigma":0.30,"sector":"E-Commerce"},
    "NVDA":     {"name":"NVIDIA Corp.","S0":875.0,"mu":0.45,"sigma":0.55,"sector":"Semiconductors"},
    "TSLA":     {"name":"Tesla Inc.","S0":175.0,"mu":0.15,"sigma":0.65,"sector":"EV"},
    "JPM":      {"name":"JPMorgan Chase","S0":205.0,"mu":0.14,"sigma":0.24,"sector":"Banking"},
    "BRK-B":    {"name":"Berkshire Hathaway B","S0":375.0,"mu":0.12,"sigma":0.18,"sector":"Conglomerate"},
    "SPY":      {"name":"SPDR S&P 500 ETF","S0":520.0,"mu":0.12,"sigma":0.16,"sector":"ETF"},
}

def get_synthetic_data(ticker, start_date, seed_offset=0):
    params = TICKER_PARAMS.get(ticker, {"name":ticker,"S0":50.0,"mu":0.10,"sigma":0.30,"sector":"—"})
    # Use ticker hash as seed for reproducibility across sessions
    seed = abs(hash(ticker)) % 100000 + seed_offset
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, date.today(), freq="B")
    n = len(dates)
    if n < 30:
        dates = pd.date_range(date.today()-timedelta(days=500), date.today(), freq="B")
        n = len(dates)
    mu_d = params["mu"]/252; sigma_d = params["sigma"]/np.sqrt(252)
    # Add jump component for realism
    jumps = rng.binomial(1, 0.02, n) * rng.normal(0, 0.04, n)
    returns = rng.normal(mu_d, sigma_d, n) + jumps
    prices = params["S0"] * np.cumprod(1+returns)
    return pd.Series(prices, index=dates, name=ticker), params

@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(tickers, start_str):
    """Try real Yahoo Finance first; fall back to realistic synthetic data."""
    start = date.fromisoformat(start_str)
    all_closes = {}
    all_meta   = {}
    used_real  = {}

    for t in tickers:
        try:
            raw = yf.download(t, start=start_str, auto_adjust=True, progress=False)
            if raw.empty or len(raw) < 30:
                raise ValueError("empty")
            closes = raw["Close"] if "Close" in raw.columns else raw.iloc[:,0]
            if isinstance(closes, pd.DataFrame): closes = closes.iloc[:,0]
            closes = closes.dropna()
            # Try to get info
            meta = {"name":t,"sector":"—"}
            try:
                info = yf.Ticker(t).fast_info
                meta["name"] = getattr(info,"display_name",t) or t
            except: pass
            all_closes[t] = closes
            all_meta[t]   = meta
            used_real[t]  = True
        except:
            synth, params = get_synthetic_data(t, start)
            all_closes[t] = synth
            all_meta[t]   = {"name":params["name"],"sector":params["sector"]}
            used_real[t]  = False

    # Align on common dates
    combined = pd.DataFrame(all_closes).dropna(how="all").ffill().dropna()
    return combined, all_meta, used_real

# ─────────────────────────────────────────────
# SUGGESTED TICKERS
# ─────────────────────────────────────────────
BR_TICKERS = ["PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","BBAS3.SA",
               "WEGE3.SA","ABEV3.SA","JBSS3.SA","MGLU3.SA","LREN3.SA",
               "RENT3.SA","SBSP3.SA","EGIE3.SA","SUZB3.SA","RADL3.SA"]
US_TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","JPM","BRK-B","SPY"]

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "results" not in st.session_state:
    st.session_state.results = None
if "tickers_selected" not in st.session_state:
    st.session_state.tickers_selected = ["PETR4.SA","VALE3.SA","ITUB4.SA"]

# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
def render_home():
    st.markdown("""
<div class="hero">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:0.3em;text-transform:uppercase;color:#00a8ff;margin-bottom:0.8rem;">
    ◈ &nbsp; RISK ANALYTICS TERMINAL &nbsp; ◈
  </div>
  <h1 style="font-size:2.8rem;font-weight:700;color:#e8edf5;line-height:1.05;margin-bottom:0.6rem;">
    Value at Risk<br><span style="color:#00a8ff;">Professional Calculator</span>
  </h1>
  <p style="color:#7a8fa6;font-size:1rem;max-width:600px;margin:0 auto 2rem;line-height:1.7;">
    Três metodologias de VaR integradas — Paramétrico, Histórico e Full Valuation — 
    com precificação Black-Scholes, análise de gregas e stress testing para carteiras 
    com ações e opções europeias.
  </p>
  <div style="display:flex;justify-content:center;gap:1.2rem;flex-wrap:wrap;margin-bottom:2rem;">
    <div style="background:rgba(0,168,255,0.1);border:1px solid rgba(0,168,255,0.3);border-radius:6px;padding:0.6rem 1.2rem;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#00a8ff;">VaR Paramétrico</div>
    <div style="background:rgba(0,230,118,0.1);border:1px solid rgba(0,230,118,0.3);border-radius:6px;padding:0.6rem 1.2rem;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#00e676;">VaR Histórico</div>
    <div style="background:rgba(255,171,0,0.1);border:1px solid rgba(255,171,0,0.3);border-radius:6px;padding:0.6rem 1.2rem;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#ffab00;">Full Valuation</div>
    <div style="background:rgba(199,125,255,0.1);border:1px solid rgba(199,125,255,0.3);border-radius:6px;padding:0.6rem 1.2rem;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#c77dff;">Black-Scholes</div>
    <div style="background:rgba(255,124,67,0.1);border:1px solid rgba(255,124,67,0.3);border-radius:6px;padding:0.6rem 1.2rem;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#ff7c43;">Stress Testing</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Feature cards
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""
<div style="background:#141920;border:1px solid #1e2a3a;border-radius:8px;padding:1.5rem;height:100%;">
  <div style="font-size:1.5rem;margin-bottom:0.8rem;">📊</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#00a8ff;margin-bottom:0.5rem;">Três Metodologias</div>
  <div style="color:#7a8fa6;font-size:0.83rem;line-height:1.6;">VaR Paramétrico com distribuição normal, VaR Histórico sem hipótese distribucional, e Full Valuation com reprecificação completa por cenário.</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div style="background:#141920;border:1px solid #1e2a3a;border-radius:8px;padding:1.5rem;height:100%;">
  <div style="font-size:1.5rem;margin-bottom:0.8rem;">⚙️</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#00a8ff;margin-bottom:0.5rem;">Opções Europeias</div>
  <div style="color:#7a8fa6;font-size:0.83rem;line-height:1.6;">Precificação Black-Scholes com cálculo completo das gregas (Δ, Γ, ν, Θ, ρ) e análise de sensibilidade à volatilidade e ao preço do ativo.</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div style="background:#141920;border:1px solid #1e2a3a;border-radius:8px;padding:1.5rem;height:100%;">
  <div style="font-size:1.5rem;margin-bottom:0.8rem;">🔬</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#00a8ff;margin-bottom:0.5rem;">Stress Testing</div>
  <div style="color:#7a8fa6;font-size:0.83rem;line-height:1.6;">Cenários de choque pré-definidos e personalizados com análise de P&L, comparação com limites de VaR e Expected Shortfall.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn = st.columns([2,1,2])[1]
    with col_btn:
        if st.button("▶  INICIAR ANÁLISE", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()

    st.markdown("""
<div style="margin-top:2rem;padding:1rem;background:#141920;border:1px solid #1e2a3a;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3d5068;text-align:center;letter-spacing:0.1em;">
  MODELAGEM APLICADA AO MERCADO FINANCEIRO · VaR PARAMÉTRICO · HISTÓRICO · FULL VALUATION · BLACK-SCHOLES · EXPECTED SHORTFALL
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
<div style="padding:0.8rem 0 0.5rem;border-bottom:1px solid #1e2a3a;margin-bottom:0.5rem;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;letter-spacing:0.25em;color:#00a8ff;text-transform:uppercase;">◈ Risk Terminal</div>
  <div style="font-size:1rem;font-weight:700;color:#e8edf5;margin-top:0.2rem;">VaR Calculator</div>
</div>""", unsafe_allow_html=True)

        if st.session_state.page == "app":
            if st.button("← Tela inicial", use_container_width=False):
                st.session_state.page = "home"
                st.rerun()

        st.markdown("### CARTEIRA DE AÇÕES")

        # Suggested tickers dropdown
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#7a8fa6;margin-bottom:0.3rem;">SUGESTÕES RÁPIDAS (clique para adicionar)</div>', unsafe_allow_html=True)

        c1b, c2b = st.columns(2)
        with c1b:
            st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.58rem;color:#3d5068;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.2rem;">Brasil</div>', unsafe_allow_html=True)
            for t in BR_TICKERS[:8]:
                selected = t in st.session_state.tickers_selected
                if st.button(
                    f"{'✓ ' if selected else ''}{t}",
                    key=f"btn_{t}",
                    use_container_width=True,
                    type="primary" if selected else "secondary"
                ):
                    if selected:
                        if len(st.session_state.tickers_selected) > 1:
                            st.session_state.tickers_selected.remove(t)
                    else:
                        if len(st.session_state.tickers_selected) < 6:
                            st.session_state.tickers_selected.append(t)
                    st.rerun()
        with c2b:
            st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.58rem;color:#3d5068;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.2rem;">EUA / ETF</div>', unsafe_allow_html=True)
            for t in US_TICKERS[:8]:
                selected = t in st.session_state.tickers_selected
                if st.button(
                    f"{'✓ ' if selected else ''}{t}",
                    key=f"btn_{t}",
                    use_container_width=True,
                    type="primary" if selected else "secondary"
                ):
                    if selected:
                        if len(st.session_state.tickers_selected) > 1:
                            st.session_state.tickers_selected.remove(t)
                    else:
                        if len(st.session_state.tickers_selected) < 6:
                            st.session_state.tickers_selected.append(t)
                    st.rerun()

        # Custom ticker input
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#7a8fa6;margin-top:0.8rem;margin-bottom:0.25rem;">ADICIONAR TICKER PERSONALIZADO</div>', unsafe_allow_html=True)
        custom_col1, custom_col2 = st.columns([3,1])
        with custom_col1:
            custom_t = st.text_input("", placeholder="Ex: EMBR3.SA", label_visibility="collapsed")
        with custom_col2:
            if st.button("＋", use_container_width=True) and custom_t.strip():
                t2 = custom_t.strip().upper()
                if t2 not in st.session_state.tickers_selected and len(st.session_state.tickers_selected)<6:
                    st.session_state.tickers_selected.append(t2)
                    st.rerun()

        # Show selected tickers
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#7a8fa6;margin-top:0.7rem;margin-bottom:0.3rem;">SELECIONADOS ({}/6)</div>'.format(len(st.session_state.tickers_selected)), unsafe_allow_html=True)
        for t in st.session_state.tickers_selected:
            params = TICKER_PARAMS.get(t, {})
            name = params.get("name", t)
            st.markdown(f'<div style="background:#1a2030;border:1px solid #1e2a3a;border-radius:4px;padding:0.35rem 0.6rem;margin-bottom:0.25rem;font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;color:#00a8ff;">◈ {t} <span style="color:#7a8fa6;font-size:0.65rem;">— {name}</span></div>', unsafe_allow_html=True)

        # Quantities
        st.markdown("### QUANTIDADES")
        quantities = {}
        for t in st.session_state.tickers_selected:
            quantities[t] = st.number_input(f"{t}", min_value=0, value=1000, step=100, key=f"qty_{t}")

        # Risk params
        st.markdown("### PARÂMETROS DE RISCO")
        conf = st.select_slider("Nível de Confiança", [0.90,0.95,0.975,0.99], value=0.95,
                                 format_func=lambda x: f"{x*100:.1f}%")
        horiz = st.select_slider("Horizonte", [1,5,10,21], value=1, format_func=lambda x: f"{x}d")
        start_date = st.date_input("Início do histórico", value=date(2022,1,1),
                                    max_value=date.today()-timedelta(days=90))

        # Option
        st.markdown("### OPÇÃO EUROPEIA")
        usar_opcao = st.checkbox("Incluir opção na carteira", value=True)

        if usar_opcao and st.session_state.tickers_selected:
            ativo_op = st.selectbox("Ativo objeto", st.session_state.tickers_selected)
            tipo_op  = st.radio("Tipo", ["call","put"], horizontal=True)
            qtd_op   = st.number_input("Qtd de contratos", 0, value=1000, step=100)
            strike   = st.number_input("Strike (K)", 0.01, value=40.0, step=0.5, format="%.2f")
            taxa_rf  = st.number_input("Taxa livre de risco (% a.a.)", 0.0, value=10.5, step=0.5)/100
            venc     = st.number_input("Vencimento (anos)", 0.01, value=0.25, step=0.05, format="%.2f")
        else:
            ativo_op = st.session_state.tickers_selected[0] if st.session_state.tickers_selected else ""
            tipo_op,qtd_op,strike,taxa_rf,venc = "call",0,40.0,0.105,0.25

        st.markdown("---")
        run = st.button("▶  CALCULAR VaR", use_container_width=True)

    return dict(tickers=st.session_state.tickers_selected, quantities=quantities,
                conf=conf, horiz=horiz, start_date=str(start_date),
                usar_opcao=usar_opcao, ativo_op=ativo_op, tipo_op=tipo_op,
                qtd_op=qtd_op, strike=strike, taxa_rf=taxa_rf, venc=venc, run=run)


# ─────────────────────────────────────────────
# CALCULATION ENGINE
# ─────────────────────────────────────────────
def calculate(cfg):
    tickers   = cfg["tickers"]
    quantities= cfg["quantities"]
    conf      = cfg["conf"]
    horiz     = cfg["horiz"]

    with st.spinner("🔄 Carregando dados de mercado..."):
        precos_df, meta, used_real = fetch_data(tickers, cfg["start_date"])

    tickers = [t for t in tickers if t in precos_df.columns]
    if not tickers:
        st.markdown('<div class="warn-box">⚠ Nenhum ticker válido encontrado.</div>', unsafe_allow_html=True)
        return None

    rets_df = precos_df.pct_change().dropna()
    ult     = precos_df.iloc[-1]

    # Checks data source
    synthetic = [t for t in tickers if not used_real.get(t, False)]
    real      = [t for t in tickers if used_real.get(t, False)]
    if real:
        st.markdown(f'<div class="ok-box">✓ Dados reais obtidos: {", ".join(real)}</div>', unsafe_allow_html=True)
    if synthetic:
        st.markdown(f'<div class="info-box">ℹ Dados sintéticos (realistas) utilizados para: {", ".join(synthetic)} — Yahoo Finance indisponível no ambiente. Execute localmente para dados reais.</div>', unsafe_allow_html=True)

    val_acoes = sum(quantities.get(t,0)*ult[t] for t in tickers)
    pesos     = np.array([quantities.get(t,0)*ult[t]/val_acoes for t in tickers])
    ret_cart  = rets_df[tickers].dot(pesos)

    mu_c, sig_c = ret_cart.mean(), ret_cart.std()
    vol_anual_port = sig_c*np.sqrt(252)

    # Option
    usar_op = cfg["usar_opcao"] and cfg["qtd_op"]>0
    ao = cfg["ativo_op"] if cfg["ativo_op"] in tickers else tickers[0]
    S0 = ult[ao]
    vol_op = rets_df[ao].std()*np.sqrt(252)
    preco_op = bs(S0,cfg["strike"],cfg["venc"],cfg["taxa_rf"],vol_op,cfg["tipo_op"]) if usar_op else 0
    val_op   = cfg["qtd_op"]*preco_op if usar_op else 0
    gk       = gregas(S0,cfg["strike"],cfg["venc"],cfg["taxa_rf"],vol_op,cfg["tipo_op"]) if usar_op else {}
    val_total= val_acoes+val_op

    # VaR Paramétrico
    z = norm.ppf(1-conf)
    var_param = -(mu_c*horiz+z*sig_c*np.sqrt(horiz))*val_acoes
    tail_p    = ret_cart[ret_cart<=np.percentile(ret_cart,(1-conf)*100)]
    es_param  = -tail_p.mean()*val_acoes if len(tail_p)>0 else var_param*1.2

    # VaR Histórico
    rh = ret_cart*np.sqrt(horiz) if horiz>1 else ret_cart
    var_hist = -np.percentile(rh,(1-conf)*100)*val_acoes
    tail_h   = rh[rh<=np.percentile(rh,(1-conf)*100)]
    es_hist  = -tail_h.mean()*val_acoes if len(tail_h)>0 else var_hist*1.2

    # Full Valuation
    pnl_fv = []
    T_cen  = max(cfg["venc"]-horiz/252,0)
    for i in range(len(rets_df)):
        row = rets_df[tickers].iloc[i]
        novos = ult[tickers]*(1+row)
        na = sum(quantities.get(t,0)*novos[t] for t in tickers)
        no = cfg["qtd_op"]*bs(novos[ao],cfg["strike"],T_cen,cfg["taxa_rf"],vol_op,cfg["tipo_op"]) if usar_op else 0
        pnl_fv.append((na+no)-val_total)
    pnl_fv = np.array(pnl_fv)
    var_fv = -np.percentile(pnl_fv,(1-conf)*100)
    tail_f = pnl_fv[pnl_fv<=np.percentile(pnl_fv,(1-conf)*100)]
    es_fv  = -tail_f.mean() if len(tail_f)>0 else var_fv*1.2

    # Per-ticker stats
    ticker_stats = {}
    for t in tickers:
        r = rets_df[t]
        p = precos_df[t]
        ticker_stats[t] = {
            "price": ult[t],
            "vol_anual": r.std()*np.sqrt(252),
            "ret_ytd": (ult[t]/p[p.index.year==p.index[-1].year].iloc[0]-1) if len(p[p.index.year==p.index[-1].year])>0 else 0,
            "ret_total": ult[t]/p.iloc[0]-1,
            "ret_1m": (ult[t]/p.iloc[-22]-1) if len(p)>22 else 0,
            "max": p.max(), "min": p.min(),
            "sharpe": (r.mean()*252)/(r.std()*np.sqrt(252)) if r.std()>0 else 0,
            "skew": float(skew(r)), "kurt": float(kurtosis(r)),
            "beta": np.cov(r,ret_cart)[0,1]/np.var(ret_cart) if np.var(ret_cart)>0 else 1.0,
            "var_indiv": -np.percentile(r,(1-conf)*100)*ult[t]*quantities.get(t,0),
            "name": meta.get(t,{}).get("name",t),
            "sector": meta.get(t,{}).get("sector","—"),
            "real": used_real.get(t,False),
        }

    return dict(
        tickers=tickers, quantities=quantities, precos=precos_df, rets=rets_df,
        ult=ult, meta=meta, used_real=used_real,
        val_acoes=val_acoes, val_op=val_op, val_total=val_total,
        pesos=pesos, ret_cart=ret_cart, mu_c=mu_c, sig_c=sig_c, vol_anual_port=vol_anual_port,
        usar_op=usar_op, S0=S0, vol_op=vol_op, preco_op=preco_op, gk=gk,
        ao=ao, cfg=cfg,
        var_param=var_param, es_param=es_param,
        var_hist=var_hist, es_hist=es_hist,
        var_fv=var_fv, es_fv=es_fv, pnl_fv=pnl_fv,
        ticker_stats=ticker_stats,
        conf=conf, horiz=horiz,
    )


# ─────────────────────────────────────────────
# RENDER RESULTS
# ─────────────────────────────────────────────
def render_results(r):
    cfg = r["cfg"]
    pct_p = r["var_param"]/r["val_total"]*100
    pct_h = r["var_hist"]/r["val_total"]*100
    pct_f = r["var_fv"]/r["val_total"]*100

    # ── HEADER ──────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#0a0c0f,#0d1520);border:1px solid #1e2a3a;border-radius:8px;padding:1.2rem 1.8rem;margin-bottom:1.2rem;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;letter-spacing:0.25em;text-transform:uppercase;color:#00a8ff;margin-bottom:0.25rem;">◈ RISK ANALYTICS TERMINAL</div>
  <div style="font-size:1.4rem;font-weight:700;color:#e8edf5;">Value at Risk — Relatório de Risco</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#7a8fa6;margin-top:0.2rem;">
    Confiança: {:.0f}% · Horizonte: {}d · {} ativos · {} a {}
  </div>
</div>""".format(r["conf"]*100, r["horiz"], len(r["tickers"]),
                 r["precos"].index[0].date(), r["precos"].index[-1].date()), unsafe_allow_html=True)

    # ── KPIs ──────────────────────
    st.markdown('<div class="sec-hdr">// Visão Geral da Carteira</div>', unsafe_allow_html=True)
    k = st.columns(5)
    k[0].metric("Valor Total", f"R$ {r['val_total']:,.0f}")
    k[1].metric("Ações", f"R$ {r['val_acoes']:,.0f}")
    k[2].metric("Opções", f"R$ {r['val_op']:,.0f}" if r["usar_op"] else "—")
    k[3].metric("Vol. Diária", f"{r['sig_c']*100:.2f}%")
    k[4].metric("Vol. Anual", f"{r['vol_anual_port']*100:.1f}%")

    # ── VAR CARDS ──────────────────────
    st.markdown(f'<div class="sec-hdr">// Resultados VaR — {int(r["conf"]*100)}% | {r["horiz"]}d</div>', unsafe_allow_html=True)
    v1,v2,v3 = st.columns(3)
    with v1:
        st.metric("VaR Paramétrico", f"R$ {r['var_param']:,.0f}", f"-{pct_p:.2f}% do portfólio")
        st.metric("Expected Shortfall (ES)", f"R$ {r['es_param']:,.0f}")
    with v2:
        st.metric("VaR Histórico", f"R$ {r['var_hist']:,.0f}", f"-{pct_h:.2f}% do portfólio")
        st.metric("Expected Shortfall (ES)", f"R$ {r['es_hist']:,.0f}")
    with v3:
        lbl = "VaR Full Valuation (Ações+Opção)" if r["usar_op"] else "VaR Full Valuation"
        st.metric(lbl, f"R$ {r['var_fv']:,.0f}", f"-{pct_f:.2f}% do portfólio")
        st.metric("Expected Shortfall (ES)", f"R$ {r['es_fv']:,.0f}")

    # ── TABS ──────────────────────
    tabs = st.tabs(["📊 Distribuições","📈 Mercado & Retornos","🏢 Ativos","⚙ Opção","🔬 Stress Test","📋 Relatório"])

    # ─ TAB 0: DISTRIBUIÇÕES ─
    with tabs[0]:
        ret = r["ret_cart"]
        cut_p = np.percentile(ret,(1-r["conf"])*100)

        fig = make_subplots(rows=1,cols=2,subplot_titles=["Distribuição dos Retornos (Carteira)","Comparação de Métodos"])
        counts,edges = np.histogram(ret,bins=55)
        mids = (edges[:-1]+edges[1:])/2
        bar_colors = ["rgba(255,61,87,0.65)" if m<cut_p else "rgba(0,168,255,0.45)" for m in mids]
        fig.add_trace(go.Bar(x=mids,y=counts,marker_color=bar_colors,name="Retornos",showlegend=False),1,1)
        x_n = np.linspace(ret.min(),ret.max(),200)
        pdf_n = norm.pdf(x_n,ret.mean(),ret.std())*len(ret)*(edges[1]-edges[0])
        fig.add_trace(go.Scatter(x=x_n,y=pdf_n,mode="lines",line=dict(color="#00a8ff",width=1.5,dash="dot"),name="Normal"),1,1)
        fig.add_vline(x=cut_p,line_color="#ff3d57",line_width=2,row=1,col=1)

        vars_  = [r["var_param"],r["var_hist"],r["var_fv"]]
        ess_   = [r["es_param"],r["es_hist"],r["es_fv"]]
        lbls   = ["Paramétrico","Histórico","Full Valuation"]
        fig.add_trace(go.Bar(name="VaR",x=lbls,y=vars_,marker_color=["#00a8ff","#00e676","#ffab00"],opacity=0.8,
                             text=[f"R${v:,.0f}" for v in vars_],textposition="outside",textfont_color="#e8edf5"),1,2)
        fig.add_trace(go.Bar(name="ES",x=lbls,y=ess_,marker_color=["rgba(255,61,87,0.5)"]*3,opacity=0.7,
                             text=[f"R${v:,.0f}" for v in ess_],textposition="outside",textfont_color="#e8edf5"),1,2)
        fig.update_layout(**LAYOUT,height=380,barmode="group")
        fig.update_xaxes(gridcolor="#1e2a3a",linecolor="#1e2a3a")
        fig.update_yaxes(gridcolor="#1e2a3a",linecolor="#1e2a3a")
        st.plotly_chart(fig,use_container_width=True)

        if r["usar_op"]:
            st.markdown('<div class="sec-hdr">// P&L Full Valuation (Ações + Opção)</div>', unsafe_allow_html=True)
            pnl = r["pnl_fv"]
            cut_f = np.percentile(pnl,(1-r["conf"])*100)
            cnt2,edg2=np.histogram(pnl,bins=55)
            mid2=(edg2[:-1]+edg2[1:])/2
            col2=["rgba(255,61,87,0.65)" if m<cut_f else "rgba(0,230,118,0.4)" for m in mid2]
            fig2=go.Figure()
            fig2.add_trace(go.Bar(x=mid2,y=cnt2,marker_color=col2,name="P&L"))
            fig2.add_vline(x=cut_f,line_color="#ff3d57",line_width=2,
                           annotation_text=f"VaR FV: R${r['var_fv']:,.0f}",annotation_font_color="#ff3d57")
            fig2.add_vline(x=0,line_color="#3d5068",line_width=1,line_dash="dot")
            fig2.update_layout(**LAYOUT,title="P&L Full Valuation por Cenário Histórico",height=300)
            st.plotly_chart(fig2,use_container_width=True)

        # Stats
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.markdown('<div class="sec-hdr">// Estatísticas dos Retornos da Carteira</div>', unsafe_allow_html=True)
            stats_data = {
                "Métrica": ["Média Diária","Vol. Diária","Vol. Anual (σ√252)","Skewness","Kurtosis Excesso","Mín. Histórico","Máx. Histórico","Obs."],
                "Valor":   [f"{ret.mean()*100:.4f}%",f"{ret.std()*100:.4f}%",f"{ret.std()*np.sqrt(252)*100:.2f}%",
                            f"{float(skew(ret)):.4f}",f"{float(kurtosis(ret)):.4f}",
                            f"{ret.min()*100:.2f}%",f"{ret.max()*100:.2f}%",f"{len(ret):,}"]
            }
            st.dataframe(pd.DataFrame(stats_data),hide_index=True,use_container_width=True)
        with col_st2:
            st.markdown('<div class="sec-hdr">// QQ-Plot de Normalidade</div>', unsafe_allow_html=True)
            sorted_r = np.sort(ret)
            n_obs = len(sorted_r)
            theo_q = norm.ppf(np.linspace(0.01,0.99,n_obs),ret.mean(),ret.std())
            fig_qq=go.Figure()
            fig_qq.add_trace(go.Scatter(x=theo_q,y=sorted_r,mode="markers",
                                         marker=dict(color="#00a8ff",size=3,opacity=0.5),name="Empírico vs. Normal"))
            mn_q=min(theo_q.min(),sorted_r.min()); mx_q=max(theo_q.max(),sorted_r.max())
            fig_qq.add_trace(go.Scatter(x=[mn_q,mx_q],y=[mn_q,mx_q],mode="lines",
                                         line=dict(color="#ff3d57",width=1.5,dash="dot"),name="Linha 45°"))
            fig_qq.update_layout(**LAYOUT,title="QQ-Plot (Normal vs. Empírico)",height=280,
                                   xaxis_title="Quantis Teóricos",yaxis_title="Quantis Observados")
            st.plotly_chart(fig_qq,use_container_width=True)

    # ─ TAB 1: MERCADO ─
    with tabs[1]:
        st.markdown('<div class="sec-hdr">// Preços Normalizados (Base 100)</div>', unsafe_allow_html=True)
        norm_p = r["precos"]/r["precos"].iloc[0]*100
        fig_p=go.Figure()
        for i,t in enumerate(r["tickers"]):
            fig_p.add_trace(go.Scatter(x=norm_p.index,y=norm_p[t],mode="lines",name=t,
                                        line=dict(color=COLORS[i%len(COLORS)],width=1.5)))
        fig_p.update_layout(**LAYOUT,title="Performance Relativa dos Ativos",height=350)
        st.plotly_chart(fig_p,use_container_width=True)

        c1r,c2r = st.columns(2)
        with c1r:
            st.markdown('<div class="sec-hdr">// Volatilidade Móvel Anualizada (21d)</div>', unsafe_allow_html=True)
            fig_v=go.Figure()
            for i,t in enumerate(r["tickers"]):
                vr = r["rets"][t].rolling(21).std()*np.sqrt(252)*100
                fig_v.add_trace(go.Scatter(x=vr.index,y=vr,mode="lines",name=t,
                                            line=dict(color=COLORS[i%len(COLORS)],width=1.3)))
            fig_v.update_layout(**LAYOUT,title="Vol. Anualizada Móvel (%)",height=300,yaxis_ticksuffix="%")
            st.plotly_chart(fig_v,use_container_width=True)
        with c2r:
            st.markdown('<div class="sec-hdr">// Drawdown Histórico da Carteira</div>', unsafe_allow_html=True)
            cum=(1+r["ret_cart"]).cumprod(); dd=(cum/cum.cummax()-1)*100
            fig_dd=go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd.index,y=dd.values,fill="tozeroy",mode="lines",
                                         fillcolor="rgba(255,61,87,0.13)",line=dict(color="#ff3d57",width=1.2),name="Drawdown"))
            fig_dd.update_layout(**LAYOUT,title="Drawdown (%)",height=300,yaxis_ticksuffix="%")
            st.plotly_chart(fig_dd,use_container_width=True)

        st.markdown('<div class="sec-hdr">// Matriz de Correlação</div>', unsafe_allow_html=True)
        corr = r["rets"][r["tickers"]].corr()
        fig_c=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,
            colorscale=[[0,"#ff3d57"],[0.5,"#141920"],[1,"#00a8ff"]],
            zmin=-1,zmax=1,text=np.round(corr.values,2),texttemplate="%{text}",showscale=True))
        fig_c.update_layout(**LAYOUT,title="Correlação dos Retornos Diários",height=300+50*len(r["tickers"]))
        st.plotly_chart(fig_c,use_container_width=True)

    # ─ TAB 2: ATIVOS INDIVIDUAIS ─
    with tabs[2]:
        st.markdown('<div class="sec-hdr">// Informações por Ativo</div>', unsafe_allow_html=True)

        for t in r["tickers"]:
            ts = r["ticker_stats"][t]
            real_lbl = "✓ Dados reais" if ts["real"] else "≈ Dados sintéticos"
            real_col = "color:#00e676" if ts["real"] else "color:#ffab00"
            st.markdown(f"""
<div class="stock-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.6rem;">
    <div>
      <span class="stock-ticker">{t}</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3d5068;margin-left:0.7rem;">
        {ts['sector']}
      </span>
    </div>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;{real_col}">{real_lbl}</span>
  </div>
  <div class="stock-name">{ts['name']}</div>
  <div class="stock-grid">
    <div class="stock-item"><div class="stock-item-label">Preço Atual</div><div class="stock-item-val">R$ {ts['price']:.2f}</div></div>
    <div class="stock-item"><div class="stock-item-label">Ret. 1 Mês</div><div class="stock-item-val" style="color:{'#00e676' if ts['ret_1m']>=0 else '#ff3d57'}">{ts['ret_1m']*100:+.1f}%</div></div>
    <div class="stock-item"><div class="stock-item-label">Ret. Total Período</div><div class="stock-item-val" style="color:{'#00e676' if ts['ret_total']>=0 else '#ff3d57'}">{ts['ret_total']*100:+.1f}%</div></div>
    <div class="stock-item"><div class="stock-item-label">Vol. Anual</div><div class="stock-item-val">{ts['vol_anual']*100:.1f}%</div></div>
    <div class="stock-item"><div class="stock-item-label">Sharpe (anual)</div><div class="stock-item-val" style="color:{'#00e676' if ts['sharpe']>1 else '#ffab00' if ts['sharpe']>0 else '#ff3d57'}">{ts['sharpe']:.2f}</div></div>
    <div class="stock-item"><div class="stock-item-label">Beta (vs. carteira)</div><div class="stock-item-val">{ts['beta']:.2f}</div></div>
    <div class="stock-item"><div class="stock-item-label">Skewness</div><div class="stock-item-val">{ts['skew']:.3f}</div></div>
    <div class="stock-item"><div class="stock-item-label">Kurtosis Exc.</div><div class="stock-item-val">{ts['kurt']:.3f}</div></div>
    <div class="stock-item"><div class="stock-item-label">VaR Indiv. ({int(r['conf']*100)}%)</div><div class="stock-item-val" style="color:#ff3d57">R$ {ts['var_indiv']:,.0f}</div></div>
    <div class="stock-item"><div class="stock-item-label">Máx. Histórico</div><div class="stock-item-val">{ts['max']:.2f}</div></div>
    <div class="stock-item"><div class="stock-item-label">Mín. Histórico</div><div class="stock-item-val">{ts['min']:.2f}</div></div>
    <div class="stock-item"><div class="stock-item-label">Peso na Carteira</div><div class="stock-item-val">{r['pesos'][r['tickers'].index(t)]*100:.1f}%</div></div>
  </div>
</div>""", unsafe_allow_html=True)

            # Price chart per ticker
            with st.expander(f"📈 Ver gráfico de preços — {t}"):
                fig_ti = go.Figure()
                pi = r["precos"][t]
                fig_ti.add_trace(go.Scatter(x=pi.index, y=pi.values, mode="lines",
                    fill="tozeroy", fillcolor="rgba(0,168,255,0.07)",
                    line=dict(color=COLORS[r['tickers'].index(t)%len(COLORS)],width=1.5), name=t))
                fig_ti.update_layout(**LAYOUT, title=f"{t} — Série de Preços", height=260)
                st.plotly_chart(fig_ti, use_container_width=True)

    # ─ TAB 3: OPÇÃO ─
    with tabs[3]:
        if not r["usar_op"]:
            st.markdown('<div class="info-box">ℹ Ative a opção na barra lateral para visualizar esta análise.</div>', unsafe_allow_html=True)
        else:
            gk = r["gk"]; cfg_op = r["cfg"]
            st.markdown('<div class="sec-hdr">// Resumo da Opção</div>', unsafe_allow_html=True)
            o1,o2,o3,o4 = st.columns(4)
            o1.metric(f"{cfg_op['tipo_op'].upper()} — {r['ao']}", f"R$ {r['preco_op']:.4f}")
            o2.metric("Spot S₀", f"R$ {r['S0']:.2f}")
            o3.metric("Strike K", f"R$ {cfg_op['strike']:.2f}")
            o4.metric("Vol. Implícita", f"{r['vol_op']*100:.1f}%")
            st.markdown('<div class="sec-hdr">// Gregas</div>', unsafe_allow_html=True)
            g1,g2,g3,g4,g5 = st.columns(5)
            g1.metric("Δ Delta",  f"{gk['delta']:.4f}")
            g2.metric("Γ Gamma",  f"{gk['gamma']:.6f}")
            g3.metric("ν Vega (×1%vol)", f"{gk['vega']:.4f}")
            g4.metric("Θ Theta (diário)", f"{gk['theta']:.4f}")
            g5.metric("ρ Rho (×1%r)", f"{gk['rho']:.4f}")

            c_op1, c_op2 = st.columns(2)
            with c_op1:
                st.markdown('<div class="sec-hdr">// Perfil de Payoff no Vencimento</div>', unsafe_allow_html=True)
                sr = np.linspace(r["S0"]*0.5, r["S0"]*1.5, 250)
                payoff = [max(s-cfg_op["strike"],0) if cfg_op["tipo_op"]=="call" else max(cfg_op["strike"]-s,0) for s in sr]
                bsp    = [bs(s,cfg_op["strike"],cfg_op["venc"],cfg_op["taxa_rf"],r["vol_op"],cfg_op["tipo_op"]) for s in sr]
                fig_po = go.Figure()
                fig_po.add_trace(go.Scatter(x=sr,y=payoff,mode="lines",name="Payoff vencimento",
                                             line=dict(color="#ffab00",width=2,dash="dash")))
                fig_po.add_trace(go.Scatter(x=sr,y=bsp,mode="lines",name="Preço BS atual",
                                             line=dict(color="#00a8ff",width=2)))
                fig_po.add_vline(x=r["S0"],line_color="#00e676",line_width=1.5,line_dash="dot",
                                  annotation_text="S₀",annotation_font_color="#00e676")
                fig_po.add_vline(x=cfg_op["strike"],line_color="#7a8fa6",line_width=1,line_dash="dot",
                                  annotation_text="K",annotation_font_color="#7a8fa6")
                fig_po.update_layout(**LAYOUT,title="Payoff & Valor BS",height=330,
                                      xaxis_title="S (R$)",yaxis_title="Valor (R$)")
                st.plotly_chart(fig_po,use_container_width=True)
            with c_op2:
                st.markdown('<div class="sec-hdr">// Sensibilidade à Volatilidade (Vega)</div>', unsafe_allow_html=True)
                vols = np.linspace(0.05, 0.90, 120)
                bsv  = [bs(r["S0"],cfg_op["strike"],cfg_op["venc"],cfg_op["taxa_rf"],v,cfg_op["tipo_op"]) for v in vols]
                fig_vg=go.Figure()
                fig_vg.add_trace(go.Scatter(x=vols*100,y=bsv,mode="lines",
                                             line=dict(color="#c77dff",width=2),name="Preço BS"))
                fig_vg.add_vline(x=r["vol_op"]*100,line_color="#ffab00",line_width=1.5,line_dash="dot",
                                  annotation_text=f"σ atual {r['vol_op']*100:.1f}%",annotation_font_color="#ffab00")
                fig_vg.update_layout(**LAYOUT,title="Preço da Opção vs. Volatilidade",height=330,
                                      xaxis_title="Volatilidade (%)",yaxis_title="Preço (R$)")
                st.plotly_chart(fig_vg,use_container_width=True)

    # ─ TAB 4: STRESS ─
    with tabs[4]:
        st.markdown('<div class="sec-hdr">// Stress Test — Cenários de Choque</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Simule o impacto de choques hipotéticos na carteira e compare com os limites de VaR.</div>', unsafe_allow_html=True)

        cs_col, cv_col = st.columns(2)
        with cs_col:
            choque_s = st.slider("Choque nas ações (%)", -60, 30, -10)
        with cv_col:
            choque_v = st.slider("Choque na volatilidade implícita (p.p.)", -20, 40, 10) if r["usar_op"] else 0

        CENS = [
            ("Personalizado",  choque_s/100,  choque_v/100),
            ("Base (atual)",   0.0,  0.0),
            ("Crash Leve −5%", -0.05, 0.05),
            ("Crash −10%",    -0.10, 0.10),
            ("Crash Severo −20%", -0.20, 0.20),
            ("Crash 2008 (−40%)", -0.40, 0.30),
            ("Rally +5%",     +0.05,-0.02),
            ("Rally +10%",    +0.10,-0.05),
        ]
        rows_st = []
        pnl_vals = []
        for nome, cs, cv in CENS:
            na = sum(r["quantities"].get(t,0)*r["ult"][t]*(1+cs) for t in r["tickers"])
            no = 0
            if r["usar_op"]:
                Sc = r["S0"]*(1+cs)
                vs = max(r["vol_op"]+cv, 0.01)
                no = cfg["qtd_op"]*bs(Sc,cfg["strike"],cfg["venc"],cfg["taxa_rf"],vs,cfg["tipo_op"])
            novo = na+no; pnl=novo-r["val_total"]; pct=pnl/r["val_total"]*100
            rows_st.append({"Cenário":nome,"P&L (R$)":f"R$ {pnl:,.0f}","P&L (%)":f"{pct:+.2f}%",
                            "Novo Valor":f"R$ {novo:,.0f}","vs VaR Param.":"⚠ Excede" if pnl<-r["var_param"] else "✓ Dentro"})
            pnl_vals.append(pnl)

        st.dataframe(pd.DataFrame(rows_st), hide_index=True, use_container_width=True)

        fig_st=go.Figure()
        fig_st.add_trace(go.Bar(
            x=[c[0] for c in CENS], y=pnl_vals,
            marker_color=["rgba(0,230,118,0.75)" if v>=0 else "rgba(255,61,87,0.75)" for v in pnl_vals],
            text=[f"R${v:,.0f}" for v in pnl_vals], textposition="outside",textfont_color="#e8edf5",
            name="P&L"
        ))
        fig_st.add_hline(y=-r["var_param"],line_color="#ff3d57",line_width=1.5,line_dash="dot",
                          annotation_text="VaR Param.",annotation_font_color="#ff3d57")
        fig_st.update_layout(**LAYOUT,title="P&L por Cenário de Stress",height=380,yaxis_title="P&L (R$)")
        st.plotly_chart(fig_st, use_container_width=True)

    # ─ TAB 5: RELATÓRIO ─
    with tabs[5]:
        st.markdown('<div class="sec-hdr">// Relatório Executivo de Risco</div>', unsafe_allow_html=True)
        rp1, rp2 = st.columns([3,2])
        with rp1:
            st.markdown(f"""
<div style="background:#141920;border:1px solid #1e2a3a;border-radius:6px;padding:1.4rem 1.8rem;font-size:0.82rem;line-height:2;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;letter-spacing:0.22em;color:#00a8ff;text-transform:uppercase;margin-bottom:0.8rem;">◈ Relatório de Risco de Mercado</div>
  <div style="color:#e8edf5;font-weight:600;">Carteira Analisada</div>
  <div style="color:#7a8fa6;">Ativos: <b style="color:#e8edf5;">{', '.join(r['tickers'])}</b></div>
  <div style="color:#7a8fa6;">Período: <b style="color:#e8edf5;">{r['precos'].index[0].date()} a {r['precos'].index[-1].date()}</b></div>
  <div style="color:#7a8fa6;">Valor em ações: <b style="color:#e8edf5;">R$ {r['val_acoes']:,.2f}</b></div>
  <div style="color:#7a8fa6;">Valor em opções: <b style="color:#e8edf5;">{'R$ '+f'{r[chr(118)+chr(97)+chr(108)+"_op"]:,.2f}' if r['usar_op'] else '—'}</b></div>
  <div style="color:#7a8fa6;">Valor total: <b style="color:#e8edf5;">R$ {r['val_total']:,.2f}</b></div>
  <br/>
  <div style="color:#e8edf5;font-weight:600;">Parâmetros</div>
  <div style="color:#7a8fa6;">Nível de confiança: <b style="color:#e8edf5;">{int(r['conf']*100)}%</b></div>
  <div style="color:#7a8fa6;">Horizonte: <b style="color:#e8edf5;">{r['horiz']} dia(s)</b></div>
  <div style="color:#7a8fa6;">Vol. anual da carteira: <b style="color:#e8edf5;">{r['vol_anual_port']*100:.1f}%</b></div>
  <br/>
  <div style="color:#e8edf5;font-weight:600;">Resultados</div>
  <div style="color:#00a8ff;">VaR Paramétrico: <b>R$ {r['var_param']:,.2f} ({pct_p:.2f}%)</b> · ES: R$ {r['es_param']:,.2f}</div>
  <div style="color:#00e676;">VaR Histórico: <b>R$ {r['var_hist']:,.2f} ({pct_h:.2f}%)</b> · ES: R$ {r['es_hist']:,.2f}</div>
  <div style="color:#ffab00;">VaR Full Valuation: <b>R$ {r['var_fv']:,.2f} ({pct_f:.2f}%)</b> · ES: R$ {r['es_fv']:,.2f}</div>
</div>""", unsafe_allow_html=True)
        with rp2:
            st.markdown("""
<div style="background:#141920;border:1px solid #1e2a3a;border-radius:6px;padding:1.4rem 1.8rem;font-size:0.8rem;line-height:1.9;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;letter-spacing:0.22em;color:#00a8ff;text-transform:uppercase;margin-bottom:0.8rem;">◈ Interpretação dos Métodos</div>
  <div style="color:#e8edf5;font-weight:600;">Paramétrico</div>
  <div style="color:#7a8fa6;font-size:0.76rem;">Assume retornos normais. Rápido para carteiras lineares, mas subestima caudas gordas.</div>
  <div style="color:#e8edf5;font-weight:600;margin-top:0.7rem;">Histórico</div>
  <div style="color:#7a8fa6;font-size:0.76rem;">Usa a distribuição empírica real. Sem hipótese distribucional, mas limitado pela janela histórica.</div>
  <div style="color:#e8edf5;font-weight:600;margin-top:0.7rem;">Full Valuation</div>
  <div style="color:#7a8fa6;font-size:0.76rem;">Reprecifica a carteira completa em cada cenário. Captura a não-linearidade de opções. Mais robusto.</div>
  <div style="color:#e8edf5;font-weight:600;margin-top:0.7rem;">Expected Shortfall</div>
  <div style="color:#7a8fa6;font-size:0.76rem;">CVaR — média das perdas além do VaR. Responde o que o VaR não diz sobre a magnitude da cauda.</div>
</div>""", unsafe_allow_html=True)

        # Composition table
        st.markdown('<div class="sec-hdr">// Composição da Carteira</div>', unsafe_allow_html=True)
        comp_rows = []
        for t in r["tickers"]:
            ts = r["ticker_stats"][t]
            comp_rows.append({"Ativo":t,"Nome":ts["name"],"Setor":ts["sector"],
                               "Preço":f"R$ {ts['price']:.2f}","Qtd":r["quantities"].get(t,0),
                               "Valor":f"R$ {ts['price']*r['quantities'].get(t,0):,.0f}",
                               "Peso":f"{r['pesos'][r['tickers'].index(t)]*100:.1f}%",
                               "Vol. Anual":f"{ts['vol_anual']*100:.1f}%","Sharpe":f"{ts['sharpe']:.2f}",
                               "VaR Indiv.":f"R$ {ts['var_indiv']:,.0f}"})
        st.dataframe(pd.DataFrame(comp_rows),hide_index=True,use_container_width=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    cfg = render_sidebar()

    if st.session_state.page == "home":
        render_home()
        return

    # App page
    if cfg["run"]:
        result = calculate(cfg)
        if result:
            st.session_state.results = result

    if st.session_state.results:
        render_results(st.session_state.results)
    else:
        st.markdown("""
<div style="text-align:center;padding:4rem 2rem;color:#3d5068;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:2.5rem;opacity:0.15;margin-bottom:1rem;">◈</div>
  <div style="color:#7a8fa6;font-size:0.95rem;">Configure a carteira na barra lateral e clique em <b style="color:#00a8ff;">CALCULAR VaR</b></div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#3d5068;margin-top:1rem;letter-spacing:0.15em;">
    VaR PARAMÉTRICO · HISTÓRICO · FULL VALUATION · BLACK-SCHOLES · STRESS TEST
  </div>
</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
