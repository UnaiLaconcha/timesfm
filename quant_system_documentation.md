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
* **Estructura de Comisiones**: Resta en cada operación de entrada y salida una comisión tipo *Taker Fee* de Binance configurada por defecto en $0.04\%$ ($0.0004$).
* **Sistema de Callbacks**: Incorpora el argumento `progress_callback` para notificar el porcentaje de avance y el estado de la simulación en tiempo real a la interfaz de usuario.

### 1.4. Capa de Presentación e Interfaz (`app.py`)
* **Framework**: `Streamlit` combinado con `Plotly Graph Objects` para renderizado interactivo de gráficos financieros.
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

### 3.1. Señal de Predicción y Movimiento Esperado
En cualquier instante de evaluación $t$, disponiendo de una ventana de contexto de precios de cierre $\{P_{t-N_{ctx}+1}, \dots, P_t\}$, el modelo TimesFM predice una serie futura sobre el horizonte $h \in \{1, \dots, N_{hor}\}$, denotada como $\hat{P}_{t+h}$.

El **Movimiento Porcentual Esperado** ($\Delta_{exp}$) se calcula considerando el último valor del horizonte proyectado ($h = N_{hor}$):

$$\Delta_{exp} = \frac{\hat{P}_{t+N_{hor}} - P_t}{P_t}$$

Sea $\theta = \text{threshold\_pct}$ el umbral porcentual mínimo exigido (ej. $1.0\% = 0.01$). La generación de señales según el parámetro de dirección de operaciones (`trade_direction`) se rige por:

* **Modo "Long & Short"**:
  $$\text{Señal}_t = \begin{cases} \text{LONG}, & \text{si } \Delta_{exp} > \theta \\ \text{SHORT}, & \text{si } \Delta_{exp} < -\theta \\ \text{NEUTRAL}, & \text{en otro caso} \end{cases}$$

* **Modo "Solo Long"**:
  $$\text{Señal}_t = \begin{cases} \text{LONG}, & \text{si } \Delta_{exp} > \theta \\ \text{NEUTRAL}, & \text{en otro caso} \end{cases}$$

* **Modo "Solo Short"**:
  $$\text{Señal}_t = \begin{cases} \text{SHORT}, & \text{si } \Delta_{exp} < -\theta \\ \text{NEUTRAL}, & \text{en otro caso} \end{cases}$$

---

### 3.2. Formulación Matemática de los Mecanismos de Riesgo

#### A. Stop Loss ($SL$) y Take Profit ($TP$) Fijos
Si una posición se abre al precio $P_{entry}$ en el instante $t_0$:
* **Para posición Long**:
  $$SL_{price} = P_{entry} \cdot (1 - \text{stop\_loss\_pct})$$
  $$TP_{price} = P_{entry} \cdot (1 + \text{take\_profit\_pct})$$
* **Para posición Short**:
  $$SL_{price} = P_{entry} \cdot (1 + \text{stop\_loss\_pct})$$
  $$TP_{price} = P_{entry} \cdot (1 - \text{take\_profit\_pct})$$

#### B. Stop Loss Adaptativo por Volatilidad (Volatility Adaptive SL)
En lugar de un porcentaje rígido, calcula la volatilidad histórica local a partir de los retornos logarítmicos o porcentuales de las velas de contexto:

$$r_\tau = \frac{P_\tau - P_{\tau-1}}{P_{\tau-1}}, \quad \forall \tau \in [t-N_{ctx}+1, t]$$

$$\sigma_{local} = \sqrt{\frac{1}{N_{ctx}-1} \sum_{\tau} (r_\tau - \bar{r})^2}$$

El porcentaje de Stop Loss adaptativo efectivo ($\text{SL}_{effective}$) se determina multiplicando $\sigma_{local}$ por el parámetro $k = \text{volatility\_multiplier}$ y aplicando un piso de seguridad con el Stop Loss base:

$$\text{SL}_{effective} = \max\left(\text{stop\_loss\_pct}, \, k \cdot \sigma_{local}\right)$$

#### C. Protección Break-Even ($BE$)
Permite asegurar una posición en riesgo cero trasladando el Stop Loss al precio de entrada $P_{entry}$ una vez alcanzado un cierto umbral de ganancia en desarrollo $\eta = \text{break\_even\_trigger\_pct}$:
* **Long**: Si en cualquier instante $\tau > t_0$ el precio máximo alcanza $P_\tau \ge P_{entry} \cdot (1 + \eta)$, se activa la bandera $BE_{active} = \text{True}$ y se fija $SL_{price} = P_{entry}$.
* **Short**: Si el precio mínimo alcanza $P_\tau \le P_{entry} \cdot (1 - \eta)$, se activa $BE_{active} = \text{True}$ y se fija $SL_{price} = P_{entry}$.

#### D. Trailing Stop Loss Dinámico ($TSL$)
Mantiene un nivel de Stop Loss dinámico que acompaña al precio a medida que la operación acumula ganancias, garantizando la captura de tendencias:
* **Long**: Registra el precio máximo alcanzado desde la entrada $P_{peak}(\tau) = \max_{s \in [t_0, \tau]} P_s$. El precio de parada dinámico evoluciona como:
  $$SL_{dynamic}(\tau) = \max\left(SL_{base}, \, P_{peak}(\tau) \cdot (1 - \text{trailing\_sl\_pct})\right)$$
* **Short**: Registra el precio mínimo alcanzado $P_{trough}(\tau) = \min_{s \in [t_0, \tau]} P_s$. El precio de parada dinámico evoluciona como:
  $$SL_{dynamic}(\tau) = \min\left(SL_{base}, \, P_{trough}(\tau) \cdot (1 + \text{trailing\_sl\_pct})\right)$$

#### E. Dimensionamiento por Confianza Cuantílica (Quantile Sizing)
Ajusta el tamaño del capital de la posición en función de la distribución probabilística proyectada por los cuantiles $q_{10}$ y $q_{90}$ de TimesFM:
* **Entrada Long de Alta Confianza**: Si el cuantil pesimista $q_{10}$ al final del horizonte está por encima del precio actual:
  $$q_{10}(t+N_{hor}) > P_t \implies \text{Multiplicador de Capital} = c_{mult} \quad (\text{confidence\_multiplier})$$
* **Entrada Short de Alta Confianza**: Si el cuantil optimista $q_{90}$ al final del horizonte está por debajo del precio actual:
  $$q_{90}(t+N_{hor}) < P_t \implies \text{Multiplicador de Capital} = c_{mult}$$
En caso contrario, la posición opera con el tamaño estándar ($1.0\text{x}$).

#### F. Filtro de Incertidumbre Probabilística (Uncertainty Filter)
Evalúa el ancho de la banda de incertidumbre del modelo al final del horizonte de predicción. Se define la dispersión relativa de incertidumbre $U_t$ como:

$$U_t = \frac{q_{90}(t+N_{hor}) - q_{10}(t+N_{hor})}{P_t}$$

Si $U_t > U_{max}$ (donde $U_{max} = \text{max\_uncertainty\_pct}$), el sistema desestima la señal de entrada y permanece en liquidez, evitando operar en situaciones de alta inconsistencia del modelo.

#### G. Gestión de Operaciones Simultáneas (Overlapping Trades)
En el modo de ejecuciones simultáneas (`overlapping = True`), el motor permite mantener hasta $M_{pos} = \text{max\_positions}$ operaciones abiertas al mismo tiempo.
* **Asignación de Capital**: Sea $C_{avail}(\tau)$ el capital líquido no invertido en el instante $\tau$, y $N_{active}(\tau)$ el número de posiciones abiertas en ese momento. Al generarse una nueva señal, el capital asignado a la nueva posición $j$ es:
  $$C_{allocated, j} = \frac{C_{avail}(\tau)}{M_{pos} - N_{active}(\tau)}$$
* **Valoración Global (Mark-to-Market)**: La equidad total del portafolio $E(\tau)$ en cualquier vela $\tau$ es la suma del capital disponible más la valoración no realizada de cada posición $j$:
  $$E(\tau) = C_{avail}(\tau) + \sum_{j=1}^{N_{active}(\tau)} MTM_j(\tau)$$
  donde $MTM_j(\tau) = C_{allocated, j} \cdot \left(1 \pm \frac{P_\tau - P_{entry, j}}{P_{entry, j}}\right)$.

---

## 4. Definición Detallada y Fórmulas Matemáticas de los Indicadores del Dashboard

El dashboard de evaluación cuantitativa presenta una batería de indicadores de rendimiento y métricas de riesgo de grado institucional. A continuación se desglosa la definición matemática exacta de cada uno.

### 4.1. Retorno Total ($R_{total}$)
Mide la ganancia o pérdida porcentual neta acumulada por la cuenta desde el inicio de la simulación hasta la última vela.

$$R_{total} = \frac{E_{final} - E_{initial}}{E_{initial}}$$

*Donde $E_{initial}$ es el capital inicial (ej. $\$1000$ USD) y $E_{final}$ es el capital resultante final tras cerrar todas las posiciones.*

### 4.2. P&L Neta en USD ($PnL_{usd}$)
Representa la variación absoluta del capital expresada en dólares estadounidenses.

$$PnL_{usd} = E_{final} - E_{initial}$$

### 4.3. Máximo Drawdown ($MDD$)
Mide la máxima caída porcentual experimentada por la curva de equidad desde un pico histórico (*peak*) hasta un valle posterior (*trough*). Es el indicador fundamental del riesgo de la estrategia.

Dado el historial de equidad $E(t)$ para $t \in [0, T]$, se define primero el pico acumulado $H(t)$:

$$H(t) = \max_{\tau \le t} E(\tau)$$

El Drawdown porcentual en cualquier punto $t$ es:

$$DD(t) = \frac{E(t) - H(t)}{H(t)}$$

El Máximo Drawdown ($MDD$) es la caída más severa observada en todo el periodo:

$$MDD = \min_{t \in [0, T]} DD(t)$$

### 4.4. Ratio de Sharpe ($SR$)
Mide el retorno excedente de la estrategia por unidad de riesgo o volatilidad total. Asumiendo una tasa libre de riesgo $R_f = 0$, se calcula a partir de los retornos diarios de la curva de equidad $r_d$:

$$r_d = \frac{E_d - E_{d-1}}{E_{d-1}}$$

$$\bar{r}_d = \frac{1}{N_d} \sum_{d=1}^{N_d} r_d, \quad \sigma_d = \sqrt{\frac{1}{N_d - 1} \sum_{d=1}^{N_d} (r_d - \bar{r}_d)^2}$$

El Ratio de Sharpe anualizado (considerando 365 días de negociación en criptomonedas) es:

$$SR = \frac{\bar{r}_d}{\sigma_d} \cdot \sqrt{365}$$

*Interpretación:* $SR > 1.0$ indica un buen ajuste riesgo-beneficio; $SR > 2.0$ representa un rendimiento institucional excelente.

### 4.5. Ratio de Sortino ($SoR$)
A diferencia del Ratio de Sharpe, el Ratio de Sortino evalúa el retorno ajustado únicamente por la **volatilidad a la baja** (*downside risk*), ignorando las fluctuaciones positivas.

Se define la desviación estándar a la baja ($\sigma_{down}$):

$$\sigma_{down} = \sqrt{\frac{1}{N_d} \sum_{d=1}^{N_d} \left(\min(0, r_d)\right)^2}$$

El Ratio de Sortino anualizado se formula como:

$$SoR = \frac{\bar{r}_d}{\sigma_{down}} \cdot \sqrt{365}$$

### 4.6. Tasa de Acierto / Win Rate ($WR$)
Porcentaje de operaciones cerradas con beneficio neto positivo respecto al total de operaciones ejecutadas.

$$WR = \frac{N_{ganadoras}}{N_{total\_trades}}$$

*Donde $N_{ganadoras}$ es la cantidad de trades con $PnL_{usd} > 0$.*

### 4.7. Factor de Beneficio / Profit Factor ($PF$)
Relación entre la ganancia bruta acumulada por las operaciones ganadoras y la pérdida bruta acumulada por las operaciones perdedoras.

$$PF = \frac{\sum_{j \in \text{ganadoras}} PnL_j}{\sum_{k \in \text{perdedoras}} |PnL_k|}$$

*Interpretación:* $PF > 1.0$ indica una estrategia rentable. $PF \ge 1.5$ refleja una sólida ventaja cuantitativa.

### 4.8. Expectancia por Operación ($E_{trade}$)
Monto promedio en dólares que la estrategia espera ganar (o perder) en cada operación ejecutada.

$$E_{trade} = \left(WR \cdot \bar{W}\right) - \left((1 - WR) \cdot |\bar{L}|\right)$$

*Donde $\bar{W}$ es la ganancia promedio de las operaciones ganadoras en USD, y $|\bar{L}|$ es la pérdida promedio de las operaciones perdedoras en USD.*

### 4.9. Desglose de Operaciones y Filtros ($N_{entries}, N_{high}, N_{filt}$)
* **Total Operaciones ($N_{trades}$)**: Número total de entradas ejecutadas por el bot.
* **Alta Confianza ($N_{high}$)**: Cantidad de entradas donde la predicción cuantílica activó el multiplicador de tamaño por alta certidumbre.
* **Filtradas por Incertidumbre ($N_{filt}$)**: Número de oportunidades de entrada que fueron descartadas automáticamente por el motor debido a que la dispersión cuantílica superó el límite `max_uncertainty_pct`.

---

## 5. Estructura de Archivos JSON de Configuración (`saved_strategies/`)

Las estrategias y carteras se guardan y cargan mediante archivos JSON estructurados. A continuación se muestra el esquema completo de parámetros con la descripción y tipo de dato de cada clave:

```json
{
  "analysis_mode": "Cartera Multi-Activo",
  "quote_asset": "USDC",
  "trade_direction": "Solo Long",
  "symbol": "BTCUSDC",
  "portfolio_symbols": [
    "BTCUSDC",
    "ETHUSDC",
    "SOLUSDC",
    "BNBUSDC"
  ],
  "custom_portfolio_symbols": "",
  "interval": "1d",
  "start_date": "2026-01-01",
  "end_date": "2026-06-28",
  "context_len": 512,
  "horizon_len": 24,
  "step_size": 1,
  "initial_capital": 1000.0,
  "stop_loss_pct": 3.0,
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
| `analysis_mode` | `string` | `"Activo Único"` o `"Cartera Multi-Activo"`. |
| `quote_asset` | `string` | Mercado base: `"USDT"` o `"USDC"`. |
| `trade_direction` | `string` | `"Long & Short"`, `"Solo Long"`, o `"Solo Short"`. |
| `symbol` | `string` | Par principal seleccionado para activo único (ej. `"BTCUSDC"`). |
| `portfolio_symbols` | `array[str]` | Lista de pares incluidos en el backtest multi-activo. |
| `custom_portfolio_symbols`| `string` | Cadena de texto con pares adicionales ingresados manualmente. |
| `interval` | `string` | Marco temporal de velas Binance (`"1m"` a `"1M"`). |
| `start_date` / `end_date` | `string` | Fechas límites en formato ISO (`"YYYY-MM-DD"`). |
| `context_len` | `integer` | Número de velas de contexto para TimesFM (ej. `512`). |
| `horizon_len` | `integer` | Velas proyectadas hacia el futuro por el modelo (ej. `24`). |
| `step_size` | `integer` | Paso del bucle walk-forward (ej. `1` o `6`). |
| `initial_capital` | `float` | Capital inicial total en USD para la simulación. |
| `stop_loss_pct` | `float` | Porcentaje de Stop Loss inicial (expresado en escala $0-100$ en JSON). |
| `exit_mode` | `string` | `"Take Profit Fijo"` o `"Trailing Stop Loss (Dinámico)"`. |
| `take_profit_pct` | `float` | Porcentaje de Take Profit fijo ($0-100$). |
| `trailing_sl_pct` | `float` | Distancia porcentual del Trailing Stop Loss ($0-100$). |
| `break_even` | `boolean` | `true` para activar protección a precio de entrada. |
| `break_even_trigger_pct` | `float` | Porcentaje de ganancia para activar Break-Even ($0-100$). |
| `threshold_pct` | `float` | Umbral porcentual mínimo para validar una señal de entrada. |
| `overlapping` | `boolean` | `true` para permitir ejecuciones simultáneas de posiciones. |
| `max_positions` | `integer` | Límite máximo de posiciones abiertas simultáneamente ($2-10$). |
| `dynamic_sizing` | `boolean` | `true` para activar Quantile Sizing basado en confianza cuantílica. |
| `confidence_multiplier` | `float` | Factor multiplicador de capital en operaciones de alta confianza ($1.1-3.0$). |
| `uncertainty_filter` | `boolean` | `true` para filtrar entradas con dispersión cuantílica alta. |
| `max_uncertainty_pct` | `float` | Límite máximo de dispersión cuantílica tolerada ($1.0-15.0$). |
| `adaptive_sl` | `boolean` | `true` para ajustar el Stop Loss según la volatilidad local ($\sigma$). |
| `volatility_multiplier` | `float` | Multiplicador $k$ aplicado a la volatilidad ($\sigma$). |

---

## 6. Guía de Desarrollo e Iteración para Futuros Desarrolladores e Inteligencias Artificiales

Si eres un desarrollador o un modelo de Inteligencia Artificial trabajando en este repositorio, considera las siguientes pautas técnicas clave para modificar o expandir el código sin introducir regresiones:

### 6.1. Extensión de Métricas y Cálculos en Inferencia por Lotes
Cualquier nueva métrica de filtrado o análisis predictivo derivado del modelo debe integrarse dentro de la función `_precompute_forecasts` en `backtester.py`. De este modo se aprovecha el empaquetado por lotes `predict_batch` de `model_inference.py`, previniendo cuellos de botella en la ejecución del bucle.

### 6.2. Persistencia de Parámetros de Sesión
Si se añade un nuevo control o parámetro en la barra lateral de `app.py`:
1. Debe incluir una clave de estado `key="cfg_nombre_parametro"`.
2. Debe registrarse en la función `_apply_strategy_to_session_state` para que se actualice al cargar archivos JSON guardados.
3. Debe incluirse en la lista de argumentos que recibe `_on_save_clicked` para serializarse correctamente en el disco.

### 6.3. Mantener Determinismo y Compatibilidad Histórica
La opción predeterminada de cualquier nuevo selector de dirección o filtrado debe conservar la ejecución simétrica original (`trade_direction = "Long & Short"`). Esto asegura que los backtests de referencia continúen arrojando resultados deterministas idénticos a los históricos.
