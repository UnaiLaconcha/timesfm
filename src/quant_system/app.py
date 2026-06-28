import sys
import os
import json
import datetime
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from data_loader import BinanceLoader
from model_inference import TimesFMPredictor
from backtester import BacktestEngine

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TimesFM Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — st.markdown with <style> tags works in 1.58 for CSS injection
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0A0E1A;
    color: #E8EAF6;
}

/* Remove default Streamlit top padding and move everything up */
header[data-testid="stHeader"] {
    background: transparent !important;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
}

/* Sidebar Compact Styling */
section[data-testid="stSidebar"] {
    background: #0D1321;
    border-right: 1px solid #1E2A45;
}
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
    padding-top: 0.8rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #7C8CF8;
    font-size: 1.25rem !important;
    margin-top: 0 !important;
    margin-bottom: 0.4rem !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7C8CF8;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.2rem !important;
    letter-spacing: 0.02em;
}
section[data-testid="stSidebar"] hr {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    border-color: #1E2A45 !important;
}
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stCheckbox {
    margin-bottom: 0.2rem !important;
}
section[data-testid="stSidebar"] label {
    font-size: 0.78rem !important;
    color: #A0AEC0 !important;
}

/* Header banner */
.quant-header {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    border: 1px solid #1E3A5F;
    position: relative;
    overflow: hidden;
}
.quant-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(0,255,163,0.07) 0%, transparent 60%);
    animation: pulse 6s ease-in-out infinite alternate;
}
@keyframes pulse {
    0%   { opacity: 0.4; transform: scale(1); }
    100% { opacity: 1;   transform: scale(1.1); }
}
.quant-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.5px;
}
.quant-header p {
    color: #8DA4C4;
    margin: 4px 0 0;
    font-size: 0.88rem;
}
.badge {
    display: inline-block;
    background: rgba(0,255,163,0.12);
    color: #00FFA3;
    border: 1px solid #00FFA3;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 10px;
    vertical-align: middle;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 16px 0;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1E2A45;
    border-radius: 12px;
    padding: 16px 18px;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    border-color: #3B4FCF;
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.75rem;
    color: #6B7B9A;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.kpi-value.positive { color: #00FFA3; }
.kpi-value.negative { color: #FF4560; }
.kpi-value.neutral  { color: #7C8CF8; }
.kpi-value.white    { color: #FFFFFF; }
.kpi-sub {
    font-size: 0.72rem;
    color: #4B5A72;
    margin-top: 4px;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #6B7B9A !important;
    font-size: 0.88rem;
    font-weight: 500;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00FFA3 !important;
    border-bottom: 2px solid #00FFA3 !important;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4B5A72;
    margin: 16px 0 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E2A45;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1E2A45;
    border-radius: 8px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #4B5A72;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 16px; }
.empty-state .title {
    font-size: 1.1rem; font-weight: 600;
    color: #6B7B9A; margin-bottom: 8px;
}
.empty-state .desc { font-size: 0.85rem; }
.empty-state .accent { color: #00FFA3; }

/* Trade stats */
.trade-stats {
    display: flex; gap: 24px; margin-top: 12px; flex-wrap: wrap;
}
.trade-stats .stat-label {
    color: #6B7B9A; font-size: 0.75rem;
}
.trade-stats .stat-value {
    font-size: 1.2rem; font-weight: 600;
}
.trade-stats .stat-value.green  { color: #00FFA3; }
.trade-stats .stat-value.gold   { color: #FFD700; }
.trade-stats .stat-value.cyan   { color: #00E5FF; }
.trade-stats .stat-value.purple { color: #A855F7; }
.trade-stats .stat-value.red    { color: #FF4560; }
.trade-stats .stat-value.white  { color: #FFF; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="quant-header">
  <h1>📈 TimesFM Quant Dashboard <span class="badge">BACKTESTING</span></h1>
  <p>Motor de inversión algorítmica · Google TimesFM 2.5 × Binance · Estrategia Long/Short & Cartera Multi-Activo</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY MANAGEMENT — helpers
# ─────────────────────────────────────────────────────────────────────────────
STRATEGIES_DIR = os.path.join(os.path.dirname(__file__), "saved_strategies")

def _ensure_strategies_dir():
    os.makedirs(STRATEGIES_DIR, exist_ok=True)

def _list_saved_strategies():
    """Return sorted list of saved strategy names (without .json extension)."""
    _ensure_strategies_dir()
    files = [f[:-5] for f in os.listdir(STRATEGIES_DIR) if f.endswith(".json")]
    return sorted(files)

def _load_strategy(name):
    """Load a strategy JSON and return its dict, or None on error."""
    path = os.path.join(STRATEGIES_DIR, f"{name}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def _save_strategy(name, params):
    """Persist strategy params as JSON."""
    _ensure_strategies_dir()
    path = os.path.join(STRATEGIES_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, ensure_ascii=False, default=str)
    return path

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY LOAD — apply loaded values into session_state BEFORE widgets render
# ─────────────────────────────────────────────────────────────────────────────

def _apply_strategy_to_session_state(strategy_data):
    """Write strategy values into session_state keys that match widget keys."""
    key_map = {
        "analysis_mode":          "cfg_analysis_mode",
        "symbol":                 "cfg_symbol",
        "portfolio_symbols":      "cfg_portfolio_symbols",
        "custom_portfolio_symbols": "cfg_custom_portfolio_symbols",
        "interval":               "cfg_interval",
        "start_date":             "cfg_start_date",
        "end_date":               "cfg_end_date",
        "context_len":            "cfg_context_len",
        "horizon_len":            "cfg_horizon_len",
        "step_size":              "cfg_step_size",
        "stop_loss_pct":          "cfg_stop_loss_pct",
        "exit_mode":              "cfg_exit_mode",
        "take_profit_pct":        "cfg_take_profit_pct",
        "trailing_sl_pct":        "cfg_trailing_sl_pct",
        "break_even":             "cfg_break_even",
        "break_even_trigger_pct": "cfg_break_even_trigger_pct",
        "threshold_pct":          "cfg_threshold_pct",
        "initial_capital":        "cfg_initial_capital",
        "overlapping":            "cfg_overlapping",
        "max_positions":          "cfg_max_positions",
        "dynamic_sizing":         "cfg_dynamic_sizing",
        "confidence_multiplier":  "cfg_confidence_multiplier",
        "uncertainty_filter":     "cfg_uncertainty_filter",
        "max_uncertainty_pct":    "cfg_max_uncertainty_pct",
        "adaptive_sl":            "cfg_adaptive_sl",
        "volatility_multiplier":  "cfg_volatility_multiplier",
    }
    for json_key, widget_key in key_map.items():
        if json_key in strategy_data:
            val = strategy_data[json_key]
            if json_key in ("start_date", "end_date") and isinstance(val, str):
                val = datetime.date.fromisoformat(val)
            st.session_state[widget_key] = val

# Check if a strategy load was requested on the previous run
if st.session_state.get("_pending_strategy_load"):
    strategy_name = st.session_state["_pending_strategy_load"]
    data = _load_strategy(strategy_name)
    if data:
        _apply_strategy_to_session_state(data)
    st.session_state["_pending_strategy_load"] = None

# Initialize session state backtest history for benchmarking
if "backtest_history" not in st.session_state:
    st.session_state["backtest_history"] = []

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (COMPACT & OPTIMIZED LAYOUT)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    # ── Strategy loader ──
    st.markdown("### 📂 Estrategia Guardada")
    saved_names = _list_saved_strategies()
    strategy_options = ["Ninguna"] + saved_names

    selected_strategy = st.selectbox(
        "Cargar Estrategia Guardada",
        strategy_options,
        key="cfg_load_strategy",
        help="Selecciona una estrategia guardada para restaurar todos los parámetros.",
    )
    if selected_strategy != "Ninguna" and selected_strategy != st.session_state.get("_last_loaded_strategy"):
        st.session_state["_pending_strategy_load"] = selected_strategy
        st.session_state["_last_loaded_strategy"] = selected_strategy
        st.rerun()

    st.markdown("---")

    st.markdown("### 📡 Activo, Cartera & Capital")
    analysis_mode = st.radio("Modo de Análisis", ["Activo Único", "Cartera Multi-Activo"], key="cfg_analysis_mode")
    
    col_cap1, col_cap2 = st.columns(2)
    initial_capital = col_cap1.number_input(
        "Capital Total ($)",
        min_value=10.0, max_value=10000000.0, value=1000.0, step=100.0,
        key="cfg_initial_capital",
        help="Monto total en dólares a invertir en el activo único o a repartir proporcionalmente entre la cartera."
    )
    interval = col_cap2.selectbox("Intervalo", ["1h", "4h", "1d"], key="cfg_interval")

    if analysis_mode == "Activo Único":
        symbol = st.text_input("Par Trading (ej. BTCUSDT, ETHUSDT, PEPEUSDT)", value="BTCUSDT", key="cfg_symbol", help="Escribe cualquier par de Binance.").strip().upper()
        portfolio_symbols = [symbol]
    else:
        preset_options = [
            "BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT",
            "AVAXUSDT", "SHIBUSDT", "LINKUSDT", "DOTUSDT", "NEARUSDT", "PEPEUSDT",
            "FETUSDT", "RENDERUSDT", "SUIUSDT", "APTUSDT", "ATOMUSDT", "ICPUSDT", "BCHUSDT",
            "XLMUSDT", "FILUSDT", "ARBUSDT", "OPUSDT", "WIFUSDT", "FLOKIUSDT", "TIAUSDT",
            "INJUSDT", "RUNEUSDT", "FTMUSDT", "GALAUSDT", "SANDUSDT", "MANAUSDT", "ALGOUSDT"
        ]
        selected_preset = st.multiselect(
            "Seleccionar Criptomonedas Destacadas",
            preset_options,
            default=["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"],
            key="cfg_portfolio_symbols",
            help="Selecciona criptoactivos de la lista rápida."
        )
        custom_input = st.text_input(
            "Añadir criptomonedas personalizadas (comas)",
            placeholder="Ej: PNUTUSDT, BONKUSDT, RENDERUSDT",
            key="cfg_custom_portfolio_symbols",
            help="Escribe cualquier criptomoneda adicional de Binance separada por comas."
        )
        custom_list = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
        
        portfolio_symbols = []
        for s in selected_preset + custom_list:
            if s not in portfolio_symbols:
                portfolio_symbols.append(s)

        if not portfolio_symbols:
            st.warning("⚠️ Selecciona o escribe al menos un par para la cartera.")
            st.stop()
        symbol = portfolio_symbols[0]

    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Fecha Inicio", value=datetime.date(2026, 1, 1), key="cfg_start_date")
    end_date = col_d2.date_input("Fecha Fin", value=datetime.date.today(), key="cfg_end_date")

    st.markdown("---")

    st.markdown("### 🧠 Modelo TimesFM")
    context_len = st.slider("Context Length (velas)", 128, 1024, 512, 128,
                             help="Número de velas históricas que recibe el modelo como contexto.",
                             key="cfg_context_len")
    col_m1, col_m2 = st.columns(2)
    horizon_len = col_m1.slider("Horizonte", 1, 96, 24, 1, key="cfg_horizon_len")
    step_size = col_m2.slider("Paso Eval.", 1, 24, 6, 1, key="cfg_step_size")

    st.markdown("---")

    st.markdown("### 🛡️ Gestión de Riesgo & Salida")
    col_r1, col_r2 = st.columns(2)
    stop_loss_pct = col_r1.number_input("Stop Loss (%)", min_value=0.5, max_value=15.0, value=2.0, step=0.5, key="cfg_stop_loss_pct") / 100.0
    threshold_pct = col_r2.number_input("Umbral (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="cfg_threshold_pct") / 100.0

    exit_mode = st.radio(
        "Modo Salida",
        ["Take Profit Fijo", "Trailing Stop Loss (Dinámico)"],
        key="cfg_exit_mode"
    )

    if exit_mode == "Trailing Stop Loss (Dinámico)":
        trailing_sl = True
        trailing_sl_pct = st.number_input("Distancia Trailing SL (%)", min_value=0.5, max_value=15.0, value=2.0, step=0.5, key="cfg_trailing_sl_pct") / 100.0
        take_profit_pct = 0.0
    else:
        trailing_sl = False
        trailing_sl_pct = 0.02
        take_profit_pct = st.number_input("Take Profit (%)", min_value=0.0, max_value=30.0, value=4.0, step=0.5, key="cfg_take_profit_pct") / 100.0

    st.markdown("---")

    st.markdown("### 🔬 Optimizaciones Avanzadas")
    col_c1, col_c2 = st.columns(2)
    adaptive_sl = col_c1.checkbox("SL Adaptativo", value=False, key="cfg_adaptive_sl")
    break_even = col_c2.checkbox("Break-Even", value=False, key="cfg_break_even")

    if adaptive_sl:
        volatility_multiplier = st.slider("Mult. Volatilidad (k·σ)", 1.0, 4.0, 2.0, 0.5, key="cfg_volatility_multiplier")
    else:
        volatility_multiplier = 2.0

    if break_even:
        break_even_trigger_pct = st.number_input("Gatillo BE (%)", min_value=0.1, max_value=15.0, value=1.5, step=0.1, key="cfg_break_even_trigger_pct") / 100.0
    else:
        break_even_trigger_pct = 0.015

    col_c3, col_c4 = st.columns(2)
    dynamic_sizing = col_c3.checkbox("Quantile Sizing", value=False, key="cfg_dynamic_sizing")
    uncertainty_filter = col_c4.checkbox("Filtro Incertidumbre", value=False, key="cfg_uncertainty_filter")

    if dynamic_sizing:
        confidence_multiplier = st.slider("Mult. Alta Confianza", 1.1, 3.0, 1.5, 0.1, key="cfg_confidence_multiplier")
    else:
        confidence_multiplier = 1.5

    if uncertainty_filter:
        max_uncertainty_pct = st.slider("Máx. Incertidumbre (%)", 1.0, 15.0, 5.0, 0.5, key="cfg_max_uncertainty_pct") / 100.0
    else:
        max_uncertainty_pct = 0.05

    st.markdown("---")

    st.markdown("### ⚡ Modo de Ejecución")
    overlapping = st.checkbox("Operaciones Simultáneas", value=True, key="cfg_overlapping")
    if overlapping:
        max_positions = st.slider("Máx. Posiciones Simultáneas", 2, 10, 5, 1, key="cfg_max_positions")
    else:
        max_positions = 1

    st.markdown("---")
    run_btn = st.button("🚀 Ejecutar Backtest", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0A0E1A",
    plot_bgcolor="#111827",
    font=dict(family="Inter, sans-serif", color="#A0AEC0", size=12),
    xaxis=dict(gridcolor="#1E2A45", zerolinecolor="#1E2A45"),
    yaxis=dict(gridcolor="#1E2A45", zerolinecolor="#1E2A45"),
    margin=dict(l=0, r=0, t=40, b=0),
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_color(value, kind="return"):
    if kind == "return":
        return "positive" if value >= 0 else "negative"
    if kind == "drawdown":
        return "negative" if value < -0.05 else "neutral"
    if kind == "sharpe":
        return "positive" if value >= 1 else ("neutral" if value >= 0 else "negative")
    return "white"

def render_kpis(metrics, initial_cap, final_eq):
    total_r = metrics['total_return']
    dd      = metrics['max_drawdown']
    sharpe  = metrics['sharpe_ratio']
    sortino = metrics.get('sortino_ratio', sharpe * 1.2)
    win_rate = metrics.get('win_rate', 0.0)
    pf      = metrics.get('profit_factor', 0.0)
    expect  = metrics.get('expectancy', 0.0)
    ntrades = metrics['total_trades']
    pnl     = final_eq - initial_cap

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Retorno Total</div>
        <div class="kpi-value {kpi_color(total_r, 'return')}">{total_r*100:+.2f}%</div>
        <div class="kpi-sub">P&L: {'+' if pnl>=0 else ''}{pnl:.2f} USD</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value {kpi_color(dd, 'drawdown')}">{dd*100:.2f}%</div>
        <div class="kpi-sub">Pérdida máxima desde el pico</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value {kpi_color(sharpe, 'sharpe')}">{sharpe:.2f}</div>
        <div class="kpi-sub">{'Excelente' if sharpe>2 else 'Bueno' if sharpe>1 else 'Mejorable'}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Win Rate (%)</div>
        <div class="kpi-value {'positive' if win_rate>=0.5 else 'neutral'}">{win_rate*100:.1f}%</div>
        <div class="kpi-sub">Porcentaje de trades ganadores</div>
      </div>
    </div>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Profit Factor</div>
        <div class="kpi-value {'positive' if pf>=1.5 else 'neutral'}">{pf:.2f}</div>
        <div class="kpi-sub">Ganancia Bruta / Pérdida Bruta</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Sortino Ratio</div>
        <div class="kpi-value {'positive' if sortino>=1.5 else 'neutral'}">{sortino:.2f}</div>
        <div class="kpi-sub">Retorno sobre riesgo a la baja</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Expectancia / Trade</div>
        <div class="kpi-value {'positive' if expect>=0 else 'negative'}">${expect:+.2f}</div>
        <div class="kpi-sub">Beneficio medio por operación</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total Operaciones</div>
        <div class="kpi-value white">{ntrades}</div>
        <div class="kpi-sub">Señales ejecutadas por el bot</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def build_equity_chart(results, df_dict, ctx_len, init_cap):
    equity_df = results['equity_df']
    is_portfolio = 'asset_results' in results
    
    fig = go.Figure()
    
    # Portfolio or Single Strategy Main Curve
    main_name = "Cartera TimesFM (Consolidada)" if is_portfolio else "Estrategia TimesFM (Actual)"
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df['equity'],
        name=main_name,
        line=dict(color="#00FFA3", width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0,255,163,0.06)',
    ))

    # Buy & Hold Benchmark
    if not is_portfolio:
        sym = list(df_dict.keys())[0]
        df = df_dict[sym]
        start_price = df['close'].iloc[ctx_len]
        bnh_equity = init_cap * (df['close'].iloc[ctx_len:] / start_price)
        fig.add_trace(go.Scatter(
            x=bnh_equity.index, y=bnh_equity.values,
            name="Buy & Hold",
            line=dict(color="#7C8CF8", width=1.5, dash='dash'),
        ))
    else:
        # Plot breakdown of individual assets in portfolio
        asset_colors = ["#FFD700", "#00E5FF", "#A855F7", "#FF8C00", "#F7B731", "#E056FD", "#00FFA3", "#FF4560"]
        for idx, (sym, res_a) in enumerate(results['asset_results'].items()):
            a_eq = res_a['equity_df']['equity']
            fig.add_trace(go.Scatter(
                x=a_eq.index, y=a_eq.values,
                name=f"Sub-Activo: {sym}",
                line=dict(color=asset_colors[idx % len(asset_colors)], width=1.2, dash='dot'),
            ))

    # Add benchmark runs from session state
    history = st.session_state.get("backtest_history", [])
    colors = ["#E2E8F0", "#CBD5E1", "#94A3B8"]
    for idx, run in enumerate(history[-3:]):
        h_df = run['equity_df']
        label = run['name']
        fig.add_trace(go.Scatter(
            x=h_df.index, y=h_df['equity'],
            name=f"Historial {idx+1}: {label}",
            line=dict(color=colors[idx % len(colors)], width=1, dash='dashdot'),
            visible='legendonly'
        ))

    chart_title = "Evolución del Capital Consolidado de la Cartera" if is_portfolio else "Equity Curve vs. Buy & Hold"
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text=chart_title, font=dict(size=14, color="#E8EAF6")),
        yaxis_title="Capital (USD)",
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
        hovermode="x unified",
    )
    return fig


def build_advanced_chart(df, trades_df, equity_df):
    cummax = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - cummax) / cummax

    vol_colors = ['#00FFA3' if c >= o else '#FF4560'
                  for o, c in zip(df['open'], df['close'])]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("Precio & Señales de Trading", "Drawdown Subacuático", "Volumen"),
    )

    # ROW 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="OHLC",
        increasing_line_color="#00FFA3", decreasing_line_color="#FF4560",
        increasing_fillcolor="#00FFA3", decreasing_fillcolor="#FF4560",
        showlegend=False,
    ), row=1, col=1)

    # Markers
    if not trades_df.empty:
        trade_map = {
            'ENTER_LONG':      dict(sym='triangle-up',   color='#00FFA3', size=12, label='Long'),
            'ENTER_SHORT':     dict(sym='triangle-down', color='#FF4560', size=12, label='Short'),
            'CLOSE_LONG_SL':   dict(sym='x',             color='#FF8C00', size=10, label='SL Long'),
            'CLOSE_SHORT_SL':  dict(sym='x',             color='#FF8C00', size=10, label='SL Short'),
            'CLOSE_LONG_BE':   dict(sym='diamond',       color='#FFD700', size=9,  label='BE Long'),
            'CLOSE_SHORT_BE':  dict(sym='diamond',       color='#FFD700', size=9,  label='BE Short'),
            'CLOSE_LONG_TSL':  dict(sym='star',          color='#00E5FF', size=10, label='TSL Long'),
            'CLOSE_SHORT_TSL': dict(sym='star',          color='#00E5FF', size=10, label='TSL Short'),
            'CLOSE_LONG_TP':   dict(sym='circle',        color='#00FFA3', size=8,  label='TP Long'),
            'CLOSE_SHORT_TP':  dict(sym='circle',        color='#00FFA3', size=8,  label='TP Short'),
            'CLOSE_LONG':      dict(sym='circle',        color='#7C8CF8', size=8,  label='Close L'),
            'CLOSE_SHORT':     dict(sym='circle',        color='#F7B731', size=8,  label='Close S'),
            'CLOSE_LONG_END':  dict(sym='circle-open',   color='#A0AEC0', size=8,  label='End L'),
            'CLOSE_SHORT_END': dict(sym='circle-open',   color='#A0AEC0', size=8,  label='End S'),
        }
        added = set()
        for trade_type, cfg in trade_map.items():
            subset = trades_df[trades_df['type'] == trade_type]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset['time'], y=subset['price'],
                mode='markers',
                marker=dict(symbol=cfg['sym'], color=cfg['color'], size=cfg['size'],
                            line=dict(width=1, color='#0A0E1A')),
                name=cfg['label'],
                showlegend=trade_type not in added,
                legendgroup=trade_type,
            ), row=1, col=1)
            added.add(trade_type)

    # ROW 2: Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        name="Drawdown %", fill='tozeroy',
        fillcolor='rgba(255,69,96,0.2)',
        line=dict(color='#FF4560', width=1.5),
        showlegend=False,
    ), row=2, col=1)

    # ROW 3: Volume
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'], name="Volumen",
        marker_color=vol_colors, showlegend=False, opacity=0.7,
    ), row=3, col=1)

    fig.update_layout(
        **PLOTLY_DARK, height=760,
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="#1E2A45",
                    borderwidth=1, x=0.01, y=0.99, font=dict(size=10)),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1, tickprefix="$")
    fig.update_yaxes(title_text="DD %", row=2, col=1, ticksuffix="%")
    fig.update_yaxes(title_text="Vol", row=3, col=1)
    return fig


def build_monthly_heatmap(equity_df):
    eq = equity_df['equity'].copy()
    daily = eq.resample('D').last().ffill()
    monthly_ret = daily.resample('ME').last().pct_change() * 100
    monthly_ret = monthly_ret.dropna()

    months_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

    mr_df = monthly_ret.to_frame(name='ret')
    mr_df['year'] = mr_df.index.year
    mr_df['month'] = mr_df.index.month

    if mr_df.empty:
        return None

    pivot = mr_df.pivot_table(index='year', columns='month', values='ret')
    pivot.columns = [months_es[m - 1] for m in pivot.columns]

    z = pivot.values
    x = list(pivot.columns)
    y = [str(yr) for yr in pivot.index]
    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, text=text, texttemplate="%{text}",
        colorscale='RdYlGn', zmid=0,
        colorbar=dict(title="%", ticksuffix="%", thickness=14),
        hovertemplate="<b>%{y} %{x}</b><br>Retorno: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text="Retorno Mensual de la Estrategia / Cartera (%)", font=dict(size=14)),
        height=max(200, 80 * len(y) + 80),
    )
    return fig


def style_trades(df):
    """Format trades for display without pandas Styler (avoids jinja2 dependency)."""
    display_df = df.copy()
    if 'price' in display_df.columns:
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
    if 'capital' in display_df.columns:
        display_df['capital'] = display_df['capital'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
    if 'pnl_usd' in display_df.columns:
        display_df['pnl_usd'] = display_df['pnl_usd'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else "")
    if 'pnl_pct' in display_df.columns:
        display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x*100:+.2f}%" if isinstance(x, (int, float)) else "")
    return display_df


def _on_save_clicked(an_mode, sym, port_syms, custom_port_syms, intv, s_date, e_date, c_len, h_len, s_size, sl_pct, ex_mode, tp_pct, tsl_pct, be, be_trig, th_pct, init_cap, overl, max_pos, dyn_size, conf_mult, uncert_filt, max_uncert, adapt_sl, vol_mult):
    name = st.session_state.get("strategy_name_input_key", "").strip()
    if not name:
        st.session_state["save_status"] = ("warning", "⚠️ Escribe un nombre para la estrategia/cartera antes de guardar.")
    else:
        clean_name = name.replace(" ", "_")
        params_to_save = {
            "analysis_mode": an_mode,
            "symbol": sym,
            "portfolio_symbols": port_syms,
            "custom_portfolio_symbols": custom_port_syms,
            "interval": intv,
            "start_date": s_date.isoformat(),
            "end_date": e_date.isoformat(),
            "context_len": c_len,
            "horizon_len": h_len,
            "step_size": s_size,
            "stop_loss_pct": sl_pct * 100,
            "exit_mode": ex_mode,
            "take_profit_pct": tp_pct * 100,
            "trailing_sl_pct": tsl_pct * 100,
            "break_even": be,
            "break_even_trigger_pct": be_trig * 100,
            "threshold_pct": th_pct * 100,
            "initial_capital": init_cap,
            "overlapping": overl,
            "max_positions": max_pos,
            "dynamic_sizing": dyn_size,
            "confidence_multiplier": conf_mult,
            "uncertainty_filter": uncert_filt,
            "max_uncertainty_pct": max_uncert * 100,
            "adaptive_sl": adapt_sl,
            "volatility_multiplier": vol_mult,
        }
        _save_strategy(clean_name, params_to_save)
        st.session_state["save_status"] = ("success", f"✅ Estrategia/Cartera **{clean_name}** guardada correctamente.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    # LOAD DATA
    df_dict = {}
    with st.spinner("📡 Descargando datos históricos de Binance..."):
        try:
            if interval == "1h":
                delta = datetime.timedelta(hours=int(context_len))
            elif interval == "4h":
                delta = datetime.timedelta(hours=int(context_len) * 4)
            else:  # "1d"
                delta = datetime.timedelta(days=int(context_len))
            
            fetch_start = start_date - delta
            start_str = fetch_start.strftime("%Y-%m-%d %H:%M:%S")
            end_str = (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            
            loader = BinanceLoader()
            for s in portfolio_symbols:
                df_dict[s] = loader.fetch_historical_data(s, interval, start_str, end_str)
        except Exception as e:
            st.error(f"Error al descargar datos de Binance: {e}")
            st.stop()

    min_candles = min(len(df_dict[s]) for s in portfolio_symbols)
    st.markdown('<div class="section-label">✅ DATOS CARGADOS</div>', unsafe_allow_html=True)
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Modo de Análisis", f"{analysis_mode} ({len(portfolio_symbols)} activos)")
    col_info2.metric("Mín. Velas descargadas", min_candles)
    col_info3.metric("Capital Inicial", f"${initial_capital:,.2f} USD")

    if min_candles < context_len + horizon_len:
        st.error(f"⚠️ Se necesitan al menos **{context_len + horizon_len}** velas. Tienes {min_candles}.")
        st.stop()

    # LOAD MODEL
    with st.spinner("🧠 Cargando modelo TimesFM 2.5 (primera carga ~30s)..."):
        try:
            predictor = TimesFMPredictor(context_len=context_len, horizon_len=horizon_len)
        except Exception as e:
            st.error(f"Error al cargar el modelo TimesFM: {e}")
            st.stop()

    # RUN BACKTEST (SINGLE OR PORTFOLIO)
    with st.spinner("⚡ Ejecutando simulación ultra-rápida (Batch Inferences)..."):
        try:
            engine = BacktestEngine(initial_capital=float(initial_capital), fee_rate=0.0004)
            if analysis_mode == "Activo Único":
                results = engine.run_backtest(
                    df_dict[symbol], predictor, horizon_len, stop_loss_pct, threshold_pct,
                    step_size=step_size, take_profit_pct=take_profit_pct,
                    overlapping=overlapping, max_positions=max_positions,
                    trailing_sl=trailing_sl, trailing_sl_pct=trailing_sl_pct,
                    break_even=break_even, break_even_trigger_pct=break_even_trigger_pct,
                    dynamic_sizing=dynamic_sizing, confidence_multiplier=confidence_multiplier,
                    uncertainty_filter=uncertainty_filter, max_uncertainty_pct=max_uncertainty_pct,
                    adaptive_sl=adaptive_sl, volatility_multiplier=volatility_multiplier,
                )
            else:
                results = engine.run_portfolio_backtest(
                    df_dict, predictor, horizon_len, stop_loss_pct, threshold_pct,
                    step_size=step_size, take_profit_pct=take_profit_pct,
                    overlapping=overlapping, max_positions=max_positions,
                    trailing_sl=trailing_sl, trailing_sl_pct=trailing_sl_pct,
                    break_even=break_even, break_even_trigger_pct=break_even_trigger_pct,
                    dynamic_sizing=dynamic_sizing, confidence_multiplier=confidence_multiplier,
                    uncertainty_filter=uncertainty_filter, max_uncertainty_pct=max_uncertainty_pct,
                    adaptive_sl=adaptive_sl, volatility_multiplier=volatility_multiplier,
                )
        except Exception as e:
            st.error(f"Error en el backtesting: {e}")
            st.stop()

    equity_df    = results['equity_df']
    trades_df    = results['trades']
    metrics      = results['metrics']
    final_equity = equity_df['equity'].iloc[-1]

    # Save run to history for benchmarking
    run_label = f"Cartera {len(portfolio_symbols)} act." if analysis_mode == "Cartera Multi-Activo" else symbol
    st.session_state["backtest_history"].append({
        "name": f"{run_label} {interval} ({datetime.datetime.now().strftime('%H:%M:%S')})",
        "equity_df": equity_df.copy(),
        "metrics": metrics
    })

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resumen de Cartera", "🕯️ Gráficos Avanzados",
        "🌡️ Rentabilidad Mensual", "📋 Operaciones",
    ])

    # TAB 1 — RESUMEN
    with tab1:
        st.markdown('<div class="section-label">MÉTRICAS CONSOLIDADAS DE LA CARTERA</div>', unsafe_allow_html=True)
        render_kpis(metrics, float(initial_capital), final_equity)

        # ── Strategy save ──
        st.markdown('<div class="section-label">💾 GUARDAR ESTRATEGIA / CARTERA</div>', unsafe_allow_html=True)
        save_col1, save_col2 = st.columns([3, 1])
        save_col1.text_input(
            "Nombre de la Estrategia / Cartera",
            placeholder="Ej: CARTERA_TopCrypto_FullQuant",
            label_visibility="collapsed",
            key="strategy_name_input_key"
        )
        custom_port_str = st.session_state.get("cfg_custom_portfolio_symbols", "")
        save_col2.button(
            "💾 Guardar",
            use_container_width=True,
            on_click=_on_save_clicked,
            args=(analysis_mode, symbol, portfolio_symbols, custom_port_str, interval, start_date, end_date, context_len, horizon_len, step_size, stop_loss_pct, exit_mode, take_profit_pct, trailing_sl_pct, break_even, break_even_trigger_pct, threshold_pct, initial_capital, overlapping, max_positions, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier)
        )
        if "save_status" in st.session_state:
            msg_type, msg_text = st.session_state["save_status"]
            if msg_type == "warning":
                st.warning(msg_text)
            else:
                st.success(msg_text)
            del st.session_state["save_status"]

        st.markdown('<div class="section-label">CURVA DE EQUIDAD BENCHMARKING</div>', unsafe_allow_html=True)
        fig_equity = build_equity_chart(results, df_dict, context_len, float(initial_capital))
        st.plotly_chart(fig_equity, use_container_width=True)

    # TAB 2 — GRÁFICOS AVANZADOS
    with tab2:
        if analysis_mode == "Cartera Multi-Activo":
            col_adv_sym, _ = st.columns([2, 2])
            selected_adv_sym = col_adv_sym.selectbox("🔍 Seleccionar Activo de la Cartera para inspeccionar", portfolio_symbols)
            adv_df = df_dict[selected_adv_sym]
            adv_trades = results['asset_results'][selected_adv_sym]['trades']
            adv_equity = results['asset_results'][selected_adv_sym]['equity_df']
        else:
            adv_df = df_dict[symbol]
            adv_trades = trades_df
            adv_equity = equity_df

        st.markdown('<div class="section-label">VELAS JAPONESAS + SEÑALES + DRAWDOWN + VOLUMEN</div>', unsafe_allow_html=True)
        fig_adv = build_advanced_chart(adv_df, adv_trades, adv_equity)
        st.plotly_chart(fig_adv, use_container_width=True)
        st.markdown("**Leyenda de señales:**")
        leg_cols = st.columns(5)
        leg_cols[0].markdown("🟢 **▲ ENTER LONG**")
        leg_cols[1].markdown("🔴 **▼ ENTER SHORT**")
        leg_cols[2].markdown("🟠 **✕ STOP LOSS**")
        leg_cols[3].markdown("🟡 **◆ BREAK-EVEN**")
        leg_cols[4].markdown("🔵 **⭐ TRAILING SL**")

    # TAB 3 — HEATMAP
    with tab3:
        st.markdown('<div class="section-label">RETORNOS MENSUALES DE LA CARTERA</div>', unsafe_allow_html=True)
        fig_heat = build_monthly_heatmap(equity_df)
        if fig_heat is not None:
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Verde = meses rentables · Rojo = meses con pérdidas.")
        else:
            st.info("No hay suficientes datos para generar el heatmap mensual.")

    # TAB 4 — OPERACIONES
    with tab4:
        st.markdown('<div class="section-label">REGISTRO COMPLETO DE OPERACIONES & EXPORTACIÓN</div>', unsafe_allow_html=True)
        if not trades_df.empty:
            styled_df = style_trades(trades_df)
            st.dataframe(styled_df, use_container_width=True, height=400)

            # CSV Download Button
            csv_data = trades_df.to_csv(index=False).encode('utf-8')
            file_prefix = f"CARTERA_{len(portfolio_symbols)}activos" if analysis_mode == "Cartera Multi-Activo" else symbol
            st.download_button(
                label="📥 Descargar Registro de Operaciones (CSV)",
                data=csv_data,
                file_name=f"{file_prefix}_{interval}_trades_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            entry_types = {'ENTER_LONG', 'ENTER_SHORT'}
            sl_types    = {'CLOSE_LONG_SL', 'CLOSE_SHORT_SL'}
            be_types    = {'CLOSE_LONG_BE', 'CLOSE_SHORT_BE'}
            tsl_types   = {'CLOSE_LONG_TSL', 'CLOSE_SHORT_TSL'}
            tp_types    = {'CLOSE_LONG_TP', 'CLOSE_SHORT_TP'}
            close_types = {'CLOSE_LONG', 'CLOSE_SHORT', 'CLOSE_LONG_END', 'CLOSE_SHORT_END'}
            n_sl      = len(trades_df[trades_df['type'].isin(sl_types)])
            n_be      = len(trades_df[trades_df['type'].isin(be_types)])
            n_tsl     = len(trades_df[trades_df['type'].isin(tsl_types)])
            n_tp      = len(trades_df[trades_df['type'].isin(tp_types)])
            n_close   = len(trades_df[trades_df['type'].isin(close_types)])
            n_entries = len(trades_df[trades_df['type'].isin(entry_types)])

            n_high_conf = len(trades_df[trades_df.get('confidence', '') == 'High']) if 'confidence' in trades_df.columns else 0
            n_filt      = metrics.get('filtered_trades', 0)

            if n_entries > 0 or n_filt > 0:
                st.markdown(f"""
                <div class="trade-stats">
                  <div><span class="stat-label">ENTRADAS</span><br>
                       <span class="stat-value white">{n_entries}</span></div>
                  <div><span class="stat-label">ALTA CONFIANZA</span><br>
                       <span class="stat-value cyan">{n_high_conf}</span></div>
                  <div><span class="stat-label">FILTRADAS (INCERTIDUMBRE)</span><br>
                       <span class="stat-value purple">{n_filt}</span></div>
                  <div><span class="stat-label">CIERRES NORMALES</span><br>
                       <span class="stat-value green">{n_close}</span></div>
                  <div><span class="stat-label">TAKE PROFIT</span><br>
                       <span class="stat-value green">{n_tp}</span></div>
                  <div><span class="stat-label">BREAK-EVEN HIT</span><br>
                       <span class="stat-value gold">{n_be}</span></div>
                  <div><span class="stat-label">TRAILING SL</span><br>
                       <span class="stat-value green">{n_tsl}</span></div>
                  <div><span class="stat-label">STOP-LOSS HIT</span><br>
                       <span class="stat-value red">{n_sl}</span></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No se ejecutó ninguna operación en el periodo analizado.")

        st.markdown("---")
        with st.expander("📖 Estrategia Aplicada — Descripción Completa"):
            mode_label = f"**Simultáneo** (máx. {max_positions} posiciones)" if overlapping else "**Secuencial** (1 posición a la vez)"
            exit_label = f"**Trailing Stop Loss** ({trailing_sl_pct*100:.1f}% distancia)" if trailing_sl else f"**Take Profit Fijo** ({take_profit_pct*100:.1f}%)"
            be_label   = f"**Activo** (Gatillo {break_even_trigger_pct*100:.1f}%)" if break_even else "**Inactivo**"
            ds_label   = f"**Activo** ({confidence_multiplier}x en Alta Confianza)" if dynamic_sizing else "**Inactivo**"
            uf_label   = f"**Activo** (Máx. Incertidumbre {max_uncertainty_pct*100:.1f}%)" if uncertainty_filter else "**Inactivo**"
            asl_label  = f"**Activo** ({volatility_multiplier}k·σ adaptativo)" if adaptive_sl else "**Inactivo**"
            activos_txt = ", ".join(portfolio_symbols) if analysis_mode == "Cartera Multi-Activo" else symbol
            st.markdown(f"""
### Modelo: TimesFM 2.5 (Google DeepMind)
Un modelo fundacional de series temporales univariantes entrenado por Google. Opera en modo **zero-shot**
(sin reentrenamiento) y genera predicciones probabilísticas (cuantiles q10–q90).

### Lógica de la Estrategia / Cartera

| Parámetro | Valor |
|-----------|-------|
| Modo Análisis | `{analysis_mode}` |
| Activo(s) | `{activos_txt}` |
| Intervalo | `{interval}` |
| Context Length | `{context_len}` velas |
| Horizon Length | `{horizon_len}` velas |
| Paso de Evaluación | `{step_size}` velas |
| Stop-Loss Inicial | `{stop_loss_pct*100:.1f}%` |
| Stop-Loss Adaptativo | {asl_label} |
| Modo Salida | {exit_label} |
| Break-Even Protection | {be_label} |
| Dynamic Confidence Sizing | {ds_label} |
| Uncertainty Filter | {uf_label} |
| Umbral entrada | `{threshold_pct*100:.1f}%` |
| Capital Global | `${initial_capital:,}` USD |
| Modo de Ejecución | {mode_label} |
""")

else:
    # EMPTY STATE
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🎯</div>
        <div class="title">Configura los parámetros y ejecuta el backtest</div>
        <div class="desc">
            Ajusta el activo o los pares de la cartera, el intervalo y los parámetros de riesgo en el panel lateral.<br>
            Pulsa <strong class="accent">🚀 Ejecutar Backtest</strong> cuando estés listo.
        </div>
    </div>
    """, unsafe_allow_html=True)
