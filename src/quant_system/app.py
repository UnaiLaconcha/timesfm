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

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D1321;
    border-right: 1px solid #1E2A45;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7C8CF8;
}

/* Header banner */
.quant-header {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
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
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.5px;
}
.quant-header p {
    color: #8DA4C4;
    margin: 6px 0 0;
    font-size: 0.9rem;
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
    margin: 20px 0;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1E2A45;
    border-radius: 12px;
    padding: 18px 20px;
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
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 1.7rem;
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
    margin: 18px 0 8px;
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
  <p>Motor de inversión algorítmica · Google TimesFM 2.5 × Binance · Estrategia Long/Short</p>
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
        "symbol":                 "cfg_symbol",
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

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

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

    st.markdown("### 📡 Activo & Datos")
    symbol = st.text_input("Par de Trading", value="BTCUSDT", key="cfg_symbol")

    _interval_options = ["1h", "4h", "1d"]
    interval = st.selectbox("Intervalo Temporal", _interval_options, key="cfg_interval")

    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Fecha Inicio", value=datetime.date(2026, 1, 1), key="cfg_start_date")
    end_date = col_d2.date_input("Fecha Fin", value=datetime.date.today(), key="cfg_end_date")

    st.markdown("### 🧠 Modelo TimesFM")
    context_len = st.slider("Context Length (velas)", 128, 1024, 512, 128,
                             help="Número de velas históricas que recibe el modelo como contexto.",
                             key="cfg_context_len")
    horizon_len = st.slider("Horizon Length (velas)", 1, 96, 24, 1,
                             help="Cuántas velas en el futuro predice el modelo para tomar la decisión.",
                             key="cfg_horizon_len")
    step_size = st.slider("Paso de Evaluación (velas)", 1, 24, 6, 1,
                           help="Frecuencia (en velas) con la que se toman decisiones. Un valor de 1 evalúa cada hora/vela (lento), mientras que 6 evalúa cada 6 velas.",
                           key="cfg_step_size")

    st.markdown("### 🛡️ Gestión de Riesgo & Salida")
    stop_loss_pct = st.number_input("Stop Loss Inicial (%)", min_value=0.5, max_value=15.0, value=2.0, step=0.5, key="cfg_stop_loss_pct") / 100.0
    
    adaptive_sl = st.checkbox(
        "Stop Loss Adaptativo (Volatilidad)",
        value=False,
        key="cfg_adaptive_sl",
        help="Ajusta dinámicamente el porcentaje de Stop Loss inicial según la desviación estándar de la volatilidad del mercado."
    )
    if adaptive_sl:
        volatility_multiplier = st.slider(
            "Multiplicador Volatilidad (k·σ)",
            1.0, 4.0, 2.0, 0.5,
            key="cfg_volatility_multiplier",
            help="Factor k aplicado a la desviación estándar de la volatilidad del contexto."
        )
    else:
        volatility_multiplier = 2.0

    exit_mode = st.radio(
        "Modo de Salida",
        ["Take Profit Fijo", "Trailing Stop Loss (Dinámico)"],
        key="cfg_exit_mode",
        help="Selecciona si deseas cerrar posiciones por objetivo fijo o acompañar la tendencia con Stop Loss dinámico."
    )

    if exit_mode == "Trailing Stop Loss (Dinámico)":
        trailing_sl = True
        trailing_sl_pct = st.number_input("Distancia Trailing SL (%)", min_value=0.5, max_value=15.0, value=2.0, step=0.5, key="cfg_trailing_sl_pct", help="Distancia porcentual respecto al precio pico/valle a la que persigue el Stop Loss.") / 100.0
        take_profit_pct = 0.0
    else:
        trailing_sl = False
        trailing_sl_pct = 0.02
        take_profit_pct = st.number_input("Take Profit (%)", min_value=0.0, max_value=30.0, value=4.0, step=0.5, key="cfg_take_profit_pct") / 100.0

    break_even = st.checkbox(
        "Activar Break-Even (Mover SL a Entrada)",
        value=False,
        key="cfg_break_even",
        help="Mueve el Stop Loss al precio de entrada una vez alcanzado un beneficio determinado para garantizar cero pérdidas."
    )
    if break_even:
        break_even_trigger_pct = st.number_input(
            "Gatillo Break-Even (%)",
            min_value=0.1, max_value=15.0, value=1.5, step=0.1,
            key="cfg_break_even_trigger_pct",
            help="Porcentaje de beneficio necesario para mover el Stop Loss a precio de entrada."
        ) / 100.0
    else:
        break_even_trigger_pct = 0.015

    threshold_pct = st.number_input("Umbral de Entrada (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="cfg_threshold_pct") / 100.0
    initial_capital = st.number_input("Capital Inicial (USD)", min_value=100, max_value=100000, value=1000, step=100, key="cfg_initial_capital")

    st.markdown("### 💎 Dimensionamiento de Posición")
    dynamic_sizing = st.checkbox(
        "Dimensionamiento por Confianza",
        value=False,
        key="cfg_dynamic_sizing",
        help="Escala el tamaño de la posición cuando el cuantil pesimista de TimesFM respalda la dirección."
    )
    if dynamic_sizing:
        confidence_multiplier = st.slider(
            "Multiplicador Alta Confianza",
            1.1, 3.0, 1.5, 0.1,
            key="cfg_confidence_multiplier",
            help="Factor de capital asignado cuando el modelo muestra máxima confianza en los cuantiles."
        )
    else:
        confidence_multiplier = 1.5

    st.markdown("### 🔬 Filtros de Calidad de Señal")
    uncertainty_filter = st.checkbox(
        "Filtro de Incertidumbre",
        value=False,
        key="cfg_uncertainty_filter",
        help="Descarta operaciones cuando la dispersión de cuantiles q90-q10 sea demasiado alta (mercado muy caótico)."
    )
    if uncertainty_filter:
        max_uncertainty_pct = st.slider(
            "Máx. Incertidumbre Permitida (%)",
            1.0, 15.0, 5.0, 0.5,
            key="cfg_max_uncertainty_pct",
            help="Amplitud máxima permitida entre q90 y q10 respecto al precio."
        ) / 100.0
    else:
        max_uncertainty_pct = 0.05

    st.markdown("### ⚡ Modo de Ejecución")
    overlapping = st.checkbox("Operaciones Simultáneas", value=True, key="cfg_overlapping",
                              help="Si está activo, se pueden abrir múltiples posiciones a la vez. Si no, solo una operación a la vez (modo clásico).")
    if overlapping:
        max_positions = st.slider("Máx. Posiciones Simultáneas", 2, 10, 5, 1, key="cfg_max_positions",
                                  help="Número máximo de operaciones que pueden estar abiertas simultáneamente. El capital se reparte entre los slots disponibles.")
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
        <div class="kpi-label">Operaciones</div>
        <div class="kpi-value white">{ntrades}</div>
        <div class="kpi-sub">Señales ejecutadas por el bot</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def build_equity_chart(equity_df, df, ctx_len, init_cap):
    start_price = df['close'].iloc[ctx_len]
    bnh_equity = init_cap * (df['close'].iloc[ctx_len:] / start_price)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df['equity'],
        name="Estrategia TimesFM",
        line=dict(color="#00FFA3", width=2),
        fill='tozeroy',
        fillcolor='rgba(0,255,163,0.06)',
    ))
    fig.add_trace(go.Scatter(
        x=bnh_equity.index, y=bnh_equity.values,
        name="Buy & Hold",
        line=dict(color="#7C8CF8", width=1.5, dash='dash'),
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text="Equity Curve vs. Buy & Hold", font=dict(size=14, color="#E8EAF6")),
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
        title=dict(text="Retorno Mensual de la Estrategia (%)", font=dict(size=14)),
        height=max(200, 80 * len(y) + 80),
    )
    return fig


def style_trades(df):
    """Format trades for display without pandas Styler (avoids jinja2 dependency)."""
    display_df = df.copy()
    if 'price' in display_df.columns:
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
    if 'capital' in display_df.columns:
        display_df['capital'] = display_df['capital'].apply(lambda x: f"${x:,.2f}")
    return display_df


def _on_save_clicked(sym, intv, s_date, e_date, c_len, h_len, s_size, sl_pct, ex_mode, tp_pct, tsl_pct, be, be_trig, th_pct, init_cap, overl, max_pos, dyn_size, conf_mult, uncert_filt, max_uncert, adapt_sl, vol_mult):
    name = st.session_state.get("strategy_name_input_key", "").strip()
    if not name:
        st.session_state["save_status"] = ("warning", "⚠️ Escribe un nombre para la estrategia antes de guardar.")
    else:
        clean_name = name.replace(" ", "_")
        params_to_save = {
            "symbol": sym,
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
        st.session_state["save_status"] = ("success", f"✅ Estrategia **{clean_name}** guardada correctamente.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    # LOAD DATA
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
            df = loader.fetch_historical_data(symbol, interval, start_str, end_str)
        except Exception as e:
            st.error(f"Error al descargar datos de Binance: {e}")
            st.stop()

    n_candles = len(df)
    st.markdown('<div class="section-label">✅ DATOS CARGADOS</div>', unsafe_allow_html=True)
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Velas descargadas", n_candles)
    col_info2.metric("Desde", str(df.index[0].date()))
    col_info3.metric("Hasta", str(df.index[-1].date()))

    if n_candles < context_len + horizon_len:
        st.error(f"⚠️ Se necesitan al menos **{context_len + horizon_len}** velas. Tienes {n_candles}.")
        st.stop()

    # LOAD MODEL
    with st.spinner("🧠 Cargando modelo TimesFM 2.5 (primera carga ~30s)..."):
        try:
            predictor = TimesFMPredictor(context_len=context_len, horizon_len=horizon_len)
        except Exception as e:
            st.error(f"Error al cargar el modelo TimesFM: {e}")
            st.stop()

    # RUN BACKTEST
    with st.spinner("⚙️ Ejecutando simulación de backtesting..."):
        try:
            engine = BacktestEngine(initial_capital=float(initial_capital), fee_rate=0.0004)
            results = engine.run_backtest(
                df, predictor, horizon_len, stop_loss_pct, threshold_pct,
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

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resumen", "🕯️ Gráficos Avanzados",
        "🌡️ Rentabilidad Mensual", "📋 Operaciones",
    ])

    # TAB 1 — RESUMEN
    with tab1:
        st.markdown('<div class="section-label">KPIs PRINCIPALES</div>', unsafe_allow_html=True)
        render_kpis(metrics, float(initial_capital), final_equity)

        # ── Strategy save ──
        st.markdown('<div class="section-label">💾 GUARDAR ESTRATEGIA</div>', unsafe_allow_html=True)
        save_col1, save_col2 = st.columns([3, 1])
        save_col1.text_input(
            "Nombre de la Estrategia",
            placeholder="Ej: BTC_1h_aggresiva_v2",
            label_visibility="collapsed",
            key="strategy_name_input_key"
        )
        save_col2.button(
            "💾 Guardar",
            use_container_width=True,
            on_click=_on_save_clicked,
            args=(symbol, interval, start_date, end_date, context_len, horizon_len, step_size, stop_loss_pct, exit_mode, take_profit_pct, trailing_sl_pct, break_even, break_even_trigger_pct, threshold_pct, initial_capital, overlapping, max_positions, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier)
        )
        if "save_status" in st.session_state:
            msg_type, msg_text = st.session_state["save_status"]
            if msg_type == "warning":
                st.warning(msg_text)
            else:
                st.success(msg_text)
            del st.session_state["save_status"]

        st.markdown('<div class="section-label">CURVA DE EQUIDAD</div>', unsafe_allow_html=True)
        fig_equity = build_equity_chart(equity_df, df, context_len, float(initial_capital))
        st.plotly_chart(fig_equity, use_container_width=True)
        st.markdown(
            "<div style='background-color: #1E293B; padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #00FFA3;'>"
            "💡 <b>¿Qué significa la Curva de Equidad?</b><br>"
            "<span style='color:#A0AEC0; font-size: 0.9em;'>"
            "Este gráfico representa la evolución de tu capital a lo largo del tiempo. "
            "<ul>"
            "<li>La <b>línea verde continua (Estrategia TimesFM)</b> muestra cómo crece o disminuye tu dinero aplicando automáticamente las operaciones predictivas del bot.</li>"
            "<li>La <b>línea azul punteada (Buy & Hold)</b> simula qué hubiera pasado si simplemente compraras el activo el primer día y lo mantuvieras sin hacer nada.</li>"
            "</ul>"
            "Es la visualización más importante para saber si el algoritmo realmente está superando al mercado o si el riesgo asumido no compensa el rendimiento."
            "</span></div>", unsafe_allow_html=True
        )

    # TAB 2 — GRÁFICOS AVANZADOS
    with tab2:
        st.markdown('<div class="section-label">VELAS JAPONESAS + SEÑALES + DRAWDOWN + VOLUMEN</div>',
                    unsafe_allow_html=True)
        fig_adv = build_advanced_chart(df, trades_df, equity_df)
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
        st.markdown('<div class="section-label">RETORNOS MENSUALES DE LA ESTRATEGIA</div>',
                    unsafe_allow_html=True)
        fig_heat = build_monthly_heatmap(equity_df)
        if fig_heat is not None:
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Verde = meses rentables · Rojo = meses con pérdidas.")
        else:
            st.info("No hay suficientes datos para generar el heatmap mensual.")

    # TAB 4 — OPERACIONES
    with tab4:
        st.markdown('<div class="section-label">REGISTRO COMPLETO DE OPERACIONES</div>',
                    unsafe_allow_html=True)
        if not trades_df.empty:
            st.dataframe(style_trades(trades_df), use_container_width=True, height=400)

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
            st.markdown(f"""
### Modelo: TimesFM 2.5 (Google DeepMind)
Un modelo fundacional de series temporales univariantes entrenado por Google. Opera en modo **zero-shot**
(sin reentrenamiento) y genera predicciones probabilísticas (cuantiles q10–q90).

### Lógica de la Estrategia

| Parámetro | Valor |
|-----------|-------|
| Activo | `{symbol}` |
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
| Capital inicial | `${initial_capital:,}` USD |
| Modo de Ejecución | {mode_label} |
""")

else:
    # EMPTY STATE
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🎯</div>
        <div class="title">Configura los parámetros y ejecuta el backtest</div>
        <div class="desc">
            Ajusta el activo, el intervalo y los parámetros de riesgo en el panel lateral.<br>
            Pulsa <strong class="accent">🚀 Ejecutar Backtest</strong> cuando estés listo.
        </div>
    </div>
    """, unsafe_allow_html=True)
