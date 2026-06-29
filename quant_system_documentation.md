# Documentación Exhaustiva y Manual Técnico del Sistema Cuantitativo: TimesFM + Binance

Este documento constituye la especificación técnica completa, la formulación matemática rigurosa y el manual operativo del sistema cuantitativo de backtesting basado en **Google TimesFM 2.5** e integrado con **Binance**. 

Está diseñado para proporcionar a cualquier desarrollador, analista cuantitativo o **modelo de Inteligencia Artificial** una comprensión absoluta y detallada de la arquitectura de código, la fundamentación matemática de cada indicador y estrategia, el funcionamiento de los hiperparámetros y la ejecución del sistema.

---

## 1. Arquitectura del Sistema y Módulos de Código

El sistema está construido de forma modular bajo el directorio `src/quant_system/`. Cada componente tiene una responsabilidad claramente delimitada dentro de la canalización (*pipeline*) cuantitativa:

```
src/quant_system/
├── data_loader.py       ← Módulo de adquisición y formateo de datos históricos (Binance API)
├── model_inference.py   ← Módulo de inferencia del modelo fundacional Google TimesFM 2.5
├── backtester.py        ← Motor institucional de simulación Walk-Forward y gestión de riesgo
├── app.py               ← Interfaz web interactiva (Streamlit), gráficos Plotly y gestión de estado
└── saved_strategies/    ← Repositorio de configuraciones y carteras serializadas en JSON
```

### 1.1. Módulo de Datos (`data_loader.py`)
* **Clase Principal**: `BinanceLoader`
* **Funcionalidad**: Interactúa con los puntos de enlace públicos de la API de Binance (mediante `python-binance`) sin requerir llaves API (autenticación pública).
* **Marcos Temporales Soportados**: Admite los 15 intervalos estándar de Binance:
  $$\text{Intervalos} \in \{1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M\}$$
* **Alineación Histórica de Contexto**: Para evitar sesgos de inicio en las predicciones, el método `fetch_historical_data` calcula dinámicamente un margen de retroceso temporal (*lookback timedelta*) en función del parámetro `context_len`. De este modo, al alcanzar la fecha exacta de inicio del backtest (`start_date`), el modelo ya dispone de la ventana completa de contexto requerida para inferir de forma inmediata.
* **Estructura de Salida**: Devuelve un `pandas.DataFrame` indexado por marca temporal UTC con columnas normalizadas: `['open', 'high', 'low', 'close', 'volume']`.

### 1.2. Módulo de Inferencia (`model_inference.py`)
* **Clase Principal**: `TimesFMPredictor`
* **Arquitectura del Modelo**: Encapsula el modelo fundacional univariante de series temporales de Google **TimesFM 2.5** (200 millones de parámetros, checkpoint PyTorch `google/timesfm-2.5-200m-pytorch`).
* **Inferencia Probabilística por Cuantiles**: Configurado con la cabeza de cuantiles continuos (`use_continuous_quantile_head=True`), el modelo genera en cada paso tanto el pronóstico puntual (mediana $q_{50}$) como una distribución probabilística de 10 cuantiles ($q_{10}, q_{20}, \dots, q_{90}$).
* **Aceleración por Lotes (`predict_batch`)**: Recibe una lista de matrices unidimensionales de contexto de tamaño $[B, N_{context}]$ (donde $B$ es el tamaño del lote) y devuelve matrices de pronóstico punto $[B, N_{horizon}]$ y cuantiles $[B, N_{horizon}, 10]$, optimizando de forma masiva el uso de la CPU/GPU.

### 1.3. Motor de Backtesting y Riesgo (`backtester.py`)
* **Clase Principal**: `BacktestEngine`
* **Modos de Inversión**: Soporta tanto la evaluación sobre un **Activo Único** (`run_backtest`) como la gestión de una **Cartera Multi-Activo** (`run_portfolio_backtest`).
* **Simulación Walk-Forward**: Itera cronológicamente sobre el historial. Para garantizar alta eficiencia computacional sin perder precisión, permite ajustar el paso de evaluación (`step_size`).
* **Inferencia en Mini-Lotes**: Procesa las ventanas de contexto en bloques de 64 muestras (`batch_size = 64`), reduciendo el tiempo de cálculo hasta en un 95%.
* **Estructura de Comisiones y Fricción**: Resta en cada operación de entrada y salida una comisión tipo *Taker Fee* de Binance ($0.04\%$) más el porcentaje de deslizamiento (*Slippage*) configurable.
* **Sistema de Callbacks**: Incorpora el argumento `progress_callback` para notificar el porcentaje de avance y el estado de la simulación en tiempo real a la interfaz de usuario.

### 1.4. Capa de Presentación e Interfaz (`app.py`)
* **Framework**: `Streamlit` combinado con `Plotly Graph Objects` para renderizado interactivo de gráficos financieros y simulaciones de Monte Carlo.
* **Persistencia de Estado**: Implementa una gestión estricta con `st.session_state` para almacenar los resultados del último backtest activo. Esto evita que los gráficos se reinicien al interactuar con selectores secundarios o cambiar entre pestañas del dashboard.
* **Diseño e Iconografía Multiplataforma**: Utiliza CSS personalizado con una pila de fuentes en cascada que incluye fallbacks explícitos (`Noto Color Emoji`, `Apple Color Emoji`, `Segoe UI Emoji`). Para prevenir fallos de renderizado (glifos vacíos `[x]`) en sistemas operativos Linux o servidores sin fuentes de color instaladas, la interfaz emplea texto limpio, símbolos universales y etiquetas CSS pulidas.

---

## 2. Guía de Instalación, Entorno y Ejecución

### 2.1. Gestión del Entorno Virtual con `uv`
El proyecto utiliza un entorno virtual localizado en `.venv/` gestionado mediante el paquete de alto rendimiento `uv`. Todas las dependencias cuantitativas están definidas como un grupo opcional en `pyproject.toml`.

**Instalación paso a paso desde la terminal:**

```bash
# 1. Clonar o posicionarse en el directorio del proyecto
cd /ruta/al/proyecto/times-fm

# 2. Instalar el paquete en modo editable junto con las dependencias del grupo [quant]
uv pip install -e ".[quant]" --extra-index-url https://download.pytorch.org/whl/cpu
```

> **Regla de Entorno:** Se debe utilizar siempre `uv pip` o `uv run` para garantizar que los paquetes se instalen e interpreten exclusivamente dentro del entorno `.venv/` aislado del proyecto.

### 2.2. Comando de Lanzamiento del Dashboard
Para iniciar la aplicación web en modo de desarrollo o producción:

```bash
uv run streamlit run src/quant_system/app.py
```

El servidor local se desplegará por defecto en el puerto 8501: `http://localhost:8501`.

---

## 3. Especificación Matemática Rigurosa y Lógica de Inversión

El motor de backtesting evalúa las condiciones del mercado en cada paso $t$ del bucle Walk-Forward. A continuación se detallan las ecuaciones y algoritmos exactos que rigen el comportamiento del sistema.

### 3.1. Señal de Predicción y Evaluación por Trayectoria AUC
En cualquier instante de evaluación $t$, disponiendo de una ventana de contexto de precios de cierre $\{P_{t-N_{ctx}+1}, \dots, P_t\}$, el modelo TimesFM predice una serie futura sobre el horizonte $h \in \{1, \dots, N_{hor}\}$, denotada como $\hat{P}_{t+h}$.

El **Movimiento Porcentual Esperado** ($\Delta_{exp}$) admite dos modos de cálculo (`signal_mode`):

1. **Modo "Punto Final"**: Evalúa el cambio en el último punto del horizonte proyectado ($h = N_{hor}$):
   $$\Delta_{exp} = \frac{\hat{P}_{t+N_{hor}} - P_t}{P_t}$$

2. **Modo "Trayectoria AUC" (Area Under Curve)**: Calcula el movimiento relativo promediando la integral discreta de la trayectoria predicha a lo largo de todo el horizonte:
   $$\Delta_{exp} = \frac{1}{N_{hor}} \sum_{h=1}^{N_{hor}} \frac{\hat{P}_{t+h} - P_t}{P_t}$$

Sea $\theta = \text{threshold\_pct}$ el umbral porcentual mínimo exigido. La generación de señales según el parámetro de dirección de operaciones (`trade_direction`) se rige por:

* **Modo "Long & Short"**:
  $$\text{Señal}_t = \begin{cases} \text{LONG}, & \text{si } \Delta_{exp} > \theta \\ \text{SHORT}, & \text{si } \Delta_{exp} < -\theta \\ \text{NEUTRAL}, & \text{en otro caso} \end{cases}$$

* **Modo "Solo Long"**:
  $$\text{Señal}_t = \begin{cases} \text{LONG}, & \text{si } \Delta_{exp} > \theta \\ \text{NEUTRAL}, & \text{en otro caso} \end{cases}$$

* **Modo "Solo Short"**:
  $$\text{Señal}_t = \begin{cases} \text{SHORT}, & \text{si } \Delta_{exp} < -\theta \\ \text{NEUTRAL}, & \text{en otro caso} \end{cases}$$

---

### 3.2. Asignación de Cartera por Paridad de Riesgo (*Risk Parity*)
En el modo cartera multi-activo (`portfolio_allocation_mode`), el capital inicial $C_{total}$ se distribuye según una de dos filosofías:

1. **Asignación Equitativa ($1/N$)**: Cada activo recibe $C_i = C_{total} / N$.
2. **Paridad de Riesgo (*Risk Parity*)**: El capital se asigna inversamente proporcional a la volatilidad histórica de cada activo ($\sigma_i$). Sea $\sigma_i$ la desviación estándar de los retornos del activo $i$:
   $$w_i = \frac{1/\sigma_i}{\sum_{k=1}^N 1/\sigma_k}, \quad C_i = C_{total} \cdot w_i$$
   De este modo, los activos menos volátiles reciben más capital y los más volátiles reciben menos capital, equilibrando el riesgo de la cartera.

---

### 3.3. Modelo de Deslizamiento (*Slippage*) y Tasas de Financiación (*Funding Rates*)
* **Fricción Total por Operación**: En cada apertura o cierre de posición, se descuenta la suma de la comisión base ($fee = 0.04\%$) y el deslizamiento por impacto de mercado ($slippage = S_{pct}$):
  $$\text{Fricción Total} = fee + S_{pct}$$
* **Tasas de Financiación Periódica (*Funding Rates*)**: Mientras una posición permanezca abierta, en cada paso $t$ del bucle se aplica una deducción porcentual $F_{pct}$ sobre el capital invertido y disponible:
  $$C(\tau) = C(\tau-1) \cdot (1 - F_{pct})$$

---

### 3.4. Formulación Matemática de los Mecanismos de Riesgo

#### A. Stop Loss ($SL$) y Take Profit ($TP$) Fijos
Si una posición se abre al precio $P_{entry}$ en el instante $t_0$:
* **Para posición Long**:
  $$SL_{price} = P_{entry} \cdot (1 - \text{stop\_loss\_pct})$$
  $$TP_{price} = P_{entry} \cdot (1 + \text{take\_profit\_pct})$$
* **Para posición Short**:
  $$SL_{price} = P_{entry} \cdot (1 + \text{stop\_loss\_pct})$$
  $$TP_{price} = P_{entry} \cdot (1 - \text{take\_profit\_pct})$$

#### B. Stop Loss Adaptativo por Volatilidad (Volatility Adaptive SL)
Calcula la volatilidad histórica local a partir de los retornos en la ventana de contexto:

$$r_\tau = \frac{P_\tau - P_{\tau-1}}{P_{\tau-1}}, \quad \forall \tau \in [t-N_{ctx}+1, t], \quad \sigma_{local} = \sqrt{\frac{1}{N_{ctx}-1} \sum_{\tau} (r_\tau - \bar{r})^2}$$

$$\text{SL}_{effective} = \max\left(\text{stop\_loss\_pct}, \, k \cdot \sigma_{local}\right)$$

#### C. Protección Break-Even ($BE$)
Al alcanzar una ganancia de desarrollo $\eta = \text{break\_even\_trigger\_pct}$, el Stop Loss se traslada al precio de entrada $P_{entry}$.

#### D. Trailing Stop Loss Dinámico ($TSL$)
Rastrae el precio máximo/mínimo alcanzado ($P_{peak} / P_{trough}$) y actualiza dinámicamente el nivel de parada.

#### E. Dimensionamiento por Confianza Cuantílica (Quantile Sizing)
Si en Long $q_{10}(t+N_{hor}) > P_t$ o en Short $q_{90}(t+N_{hor}) < P_t$, el capital asignado se multiplica por $c_{mult}$ (ej. $1.5\text{x}$).

#### F. Filtro de Incertidumbre Probabilística (Uncertainty Filter)
Si la dispersión relativa intercuantílica $U_t = \frac{q_{90} - q_{10}}{P_t} > U_{max}$, la señal de entrada se descarta.

---

## 4. Definición Detallada e Indicadores del Dashboard y Monte Carlo

### 4.1. Indicadores Estadísticos Principales
* **Retorno Total ($R_{total}$)**: $\frac{E_{final} - E_{initial}}{E_{initial}}$
* **Máximo Drawdown ($MDD$)**: Caída porcentual máxima desde el pico histórico acumulado.
* **Ratio de Sharpe ($SR$)**: Exceso de retorno ajustado por volatilidad total anualizada ($\sqrt{365}$).
* **Ratio de Sortino ($SoR$)**: Retorno ajustado por la volatilidad a la baja ($\sigma_{down}$).
* **Win Rate ($WR$)** y **Profit Factor ($PF$)**: Proporción de aciertos y ratio ganancia/pérdida bruta.
* **Expectancia por Operación ($E_{trade}$)**: Ganancia esperada por trade en dólares.

### 4.2. Simulaciones de Monte Carlo y Estrés ($VaR$ y $CVaR$)
La pestaña de análisis de estrés ejecuta 1,000 permutaciones aleatorias (*bootstrapping*) sobre los retornos de las operaciones ejecutadas para generar la distribución probabilística de la equidad proyectada:
* **Value at Risk ($VaR_{95\%}$ y $VaR_{99\%}$)**: Pérdida porcentual máxima esperada con un 95% y 99% de nivel de confianza.
* **Conditional VaR ($CVaR_{95\%}$)**: Pérdida promedio esperada en el 5% de los peores escenarios simulados.
* **Probabilidad de Ruina**: Porcentaje de trayectorias simuladas donde el capital final sufre una caída superior al 50%.

---

## 5. Estructura de Archivos JSON de Configuración (`saved_strategies/`)

Esquema de serialización JSON actualizado:

```json
{
  "analysis_mode": "Cartera Multi-Activo",
  "quote_asset": "USDC",
  "trade_direction": "Solo Long",
  "portfolio_allocation_mode": "Paridad de Riesgo (Risk Parity)",
  "signal_mode": "Trayectoria AUC",
  "symbol": "BTCUSDC",
  "portfolio_symbols": [
    "BTCUSDC",
    "ETHUSDC",
    "SOLUSDC"
  ],
  "custom_portfolio_symbols": "",
  "interval": "1d",
  "start_date": "2026-01-01",
  "end_date": "2026-06-29",
  "context_len": 512,
  "horizon_len": 24,
  "step_size": 1,
  "initial_capital": 1000.0,
  "stop_loss_pct": 3.0,
  "slippage_pct": 0.05,
  "funding_rate_pct": 0.01,
  "exit_mode": "Take Profit Fijo",
  "take_profit_pct": 6.0,
  "trailing_sl_pct": 2.0,
  "break_even": true,
  "break_even_trigger_pct": 2.0,
  "threshold_pct": 1.5,
  "overlapping": false,
  "max_positions": 1,
  "dynamic_sizing": false,
  "confidence_multiplier": 1.5,
  "uncertainty_filter": true,
  "max_uncertainty_pct": 6.0,
  "adaptive_sl": false,
  "volatility_multiplier": 2.0
}
```

### Tabla de Mapeo de Claves JSON
| Clave JSON | Tipo | Descripción y Correspondencia en el Código |
|---|---|---|
| `portfolio_allocation_mode` | `string` | `"Equitativa (1/N)"` o `"Paridad de Riesgo (Risk Parity)"`. |
| `signal_mode` | `string` | `"Punto Final"` o `"Trayectoria AUC"`. |
| `slippage_pct` | `float` | Deslizamiento por impacto de mercado ($0-100$). |
| `funding_rate_pct` | `float` | Tasa de financiación periódica ($0-100$). |
| `analysis_mode` | `string` | `"Activo Único"` o `"Cartera Multi-Activo"`. |
| `quote_asset` | `string` | Mercado base: `"USDT"` o `"USDC"`. |
| `trade_direction` | `string` | `"Long & Short"`, `"Solo Long"`, o `"Solo Short"`. |

---

## 6. Guía de Desarrollo e Iteración para Futuros Desarrolladores e Inteligencias Artificiales

1. **Inferencia Batch**: Extender métricas cuantitativas dentro de `_precompute_forecasts` en `backtester.py`.
2. **Persistencia de Estado**: Registrar cualquier widget nuevo en `key_map` de `_apply_strategy_to_session_state` y en `_on_save_clicked`.
3. **Determinismo Histórico**: Preservar `"Long & Short"`, `slippage_pct=0.0` y `funding_rate_pct=0.0` como valores predeterminados para garantizar reproducibilidad exacta de ejecuciones pasadas.
