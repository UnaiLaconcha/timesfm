# Documentación del Sistema Cuantitativo: TimesFM + Binance

Esta documentación explica la arquitectura, lógica y comandos de ejecución del sistema de backtesting implementado. Está redactada para que cualquier desarrollador o **futuro modelo de IA** comprenda rápidamente el estado actual del código y sepa cómo iterar sobre él.

---

## 1. Arquitectura del Proyecto

El sistema se ha diseñado de forma modular dentro del directorio `src/quant_system/`.

* **`data_loader.py`**: 
  * Descarga datos históricos de velas (Klines) de la API pública de Binance usando `python-binance`.
  * **No requiere API keys.**
  * Devuelve un `pandas.DataFrame` estructurado con columnas estándar (`open`, `high`, `low`, `close`, `volume`) e índice temporal.
* **`model_inference.py`**: 
  * Encapsula el modelo fundacional de Google: `TimesFM 2.5`.
  * Define la clase `TimesFMPredictor`, inicializando los hiperparámetros obligatorios (`context_len` y `horizon_len`).
  * Utiliza `torch` con alta precisión en la multiplicación de matrices.
* **`backtester.py`**:
  * Contiene la clase `BacktestEngine`, el motor del sistema.
  * Implementa un *Walk-Forward Backtest* (simulación paso a paso a través del tiempo).
  * Por razones de rendimiento, avanza en bloques (ej. `step_size = 6`) en lugar de vela por vela para no saturar las inferencias del modelo.
  * **Lógica implementada:** Long/Short, Stop-Loss fijo y deducción de *Taker Fees* de Binance (`0.04%`).
* **`app.py`**:
  * Interfaz de usuario construida con `Streamlit`.
  * Une los 3 módulos anteriores. Contiene los controles de configuración, invoca las descargas/predicciones y visualiza la curva de capital (`Equity Curve`) utilizando `Plotly`.

---

## 2. Comandos de Ejecución y Configuración

Para levantar el sistema desde cero (o si cambias de máquina), debes ejecutar los siguientes comandos en tu terminal.

### 2.1. Entorno y Dependencias

El proyecto ya tiene su entorno virtual en `.venv/` **creado y gestionado por `uv`** (ver `pyvenv.cfg`). Las dependencias del sistema cuantitativo están declaradas como extras en `pyproject.toml` bajo `[project.optional-dependencies]` con el nombre `quant`.

**Para instalar todas las dependencias del sistema cuantitativo con `uv`:**

```bash
# Instalar el extra [quant] (incluye timesfm, torch, streamlit, plotly, pandas, python-binance)
uv pip install -e ".[quant]" --extra-index-url https://download.pytorch.org/whl/cpu
```

> **¿Por qué `uv pip install` y no `pip install`?**
> `uv` respeta el entorno `.venv` del proyecto y es 10–100x más rápido que pip.
> Jamás usar `pip install` a nivel de sistema. Todo va dentro del `.venv` del proyecto.

### 2.2. Ejecutar el Dashboard

Para lanzar la interfaz visual interactiva, el servidor de Streamlit debe ejecutarse **con el entorno virtual activado**:

```bash
# Opción 1: Con uv run (recomendado - no necesita activar el venv manualmente)
uv run streamlit run src/quant_system/app.py

# Opción 2: Activando el venv manualmente
source .venv/bin/activate
streamlit run src/quant_system/app.py
```

> El dashboard estará disponible en: `http://localhost:8501`

---

## 3. Lógica Interna de Inversión (Detallada)

El sistema de backtesting simula la ejecución paso a paso (Walk-Forward) a lo largo del historial de precios descargado de Binance. Aquí se explica la lógica y el papel de cada parámetro:

### 3.1. Papel de los Parámetros del Panel Lateral
* **`Fecha Inicio` / `Fecha Fin`**: Rango de tiempo para el cual se desea evaluar y ejecutar la estrategia de inversión. Por defecto, inicia el 1 de enero de 2026 y termina en la fecha actual (o la que se configure).
  * *Nota de alineación:* El cargador de datos calcula de manera automática una ventana adicional de retroceso (`timedelta` correspondiente a `context_len` velas hacia atrás de la fecha de inicio). De esta forma, el primer día de simulación de inversión (el 1 de enero de 2026 por defecto) el modelo ya cuenta con el historial completo de contexto requerido y comienza a predecir de forma inmediata.
* **`Context Length` (`context_len`)**: La cantidad de velas del pasado inmediato que el modelo de TimesFM utiliza como entrada. Para pronósticos robustos, se recomiendan ventanas amplias (ej. 512 velas).
* **`Horizon Length` (`horizon_len`)**: El número exacto de velas hacia el futuro que predice el modelo. Por ejemplo, con un valor de 24 en velas de 4h, el modelo proyecta el precio a 96 horas vista. La decisión de inversión evalúa el movimiento esperado en el último punto de esta predicción (`horizon_len - 1`).
* **`Paso de Evaluación` (`step_size`)**: Frecuencia con la que el backtester avanza el bucle y ejecuta una inferencia. Si se configura en `1`, se evalúa la señal en cada una de las velas históricas (ej. cada 1 hora). Si se configura en `6`, el bucle avanza de 6 en 6 velas (evaluando cada 6 horas), lo que reduce considerablemente el tiempo del cálculo a cambio de menor resolución.
* **`Umbral de Entrada (%)` (`threshold_pct`)**: Movimiento porcentual mínimo esperado para abrir una posición. Si el modelo proyecta que el precio subirá más del umbral, se abre un **Long** (comprado). Si proyecta que caerá más allá del umbral negativo, se abre un **Short** (vendido).
* **`Stop Loss (%)` (`stop_loss_pct`)**: Límite máximo de pérdidas. Si el precio se mueve en contra de la posición por este porcentaje o más desde el punto de entrada, la operación se cierra inmediatamente.
* **`Take Profit (%)` (`take_profit_pct`)**: Objetivo de ganancias. Si el precio se mueve a favor de la posición por este porcentaje o más desde el punto de entrada, la posición se liquida para asegurar las ganancias. Si se configura en `0%`, queda deshabilitado.

### 3.2. Ciclo de Vida de una Operación (Modo Secuencial)
1. **Evaluación de Señales:** En cada paso del bucle (definido por `step_size`):
   * Se extraen las últimas `context_len` velas hasta el índice actual.
   * Se invoca el modelo `TimesFM` para generar la predicción sobre las próximas `horizon_len` velas.
   * Se calcula el movimiento porcentual esperado: `expected_move = (Precio_Predicho_Fin_Horizonte - Precio_Actual) / Precio_Actual`.
2. **Entrada al Mercado (Si no hay posición activa):**
   * Si `expected_move > threshold_pct` → Se abre **Long** al precio actual.
   * Si `expected_move < -threshold_pct` → Se abre **Short** al precio actual.
   * Cada apertura descuenta un `0.04%` de comisión de Binance.
3. **Monitoreo y Salida del Mercado (Si hay posición activa):**
   * Mientras la posición esté abierta, **el modelo no vuelve a realizar predicciones**. El bucle simplemente avanza y monitorea el precio.
   * **Stop-Loss (SL):** Si el precio cruza el umbral de pérdidas máximo establecido (`stop_loss_pct`), la operación se cierra inmediatamente.
   * **Take Profit (TP):** Si el precio alcanza el objetivo de ganancias establecido (`take_profit_pct`), la operación se liquida para asegurar las ganancias.
   * **Cierre por Fin de Historial:** Si la simulación termina y aún hay una posición abierta, se fuerza el cierre en la última vela disponible.
   * Al cerrar, se calcula el capital resultante final y se descuenta la comisión de salida (`0.04%`).

### 3.3. Modo de Operaciones Simultáneas (Overlapping Trades)
Cuando el checkbox **"Operaciones Simultáneas"** está activado en el dashboard (`overlapping=True`), el motor permite mantener hasta `max_positions` operaciones abiertas simultáneamente (configurable de 2 a 10).
* **Modelo de Asignación de Capital:** Para evitar inflar o distorsionar los resultados, el sistema utiliza un pool de capital no invertido (`available_capital`). Cuando el modelo genera una nueva señal y hay slots disponibles (`len(active_positions) < max_positions`), a la nueva posición se le asigna una porción equitativa del capital disponible (`available_capital / slots_restantes`).
* **Monitoreo Independiente:** En cada paso temporal, el sistema comprueba individualmente los niveles de Stop-Loss y Take-Profit de **todas** las posiciones activas. Cada posición se liquida de forma independiente cuando alcanza sus objetivos.
* **Composición de Equidad:** La curva de equidad en cada paso se calcula sumando el capital no invertido más la valoración mark-to-market actual de todas las operaciones abiertas.

### 3.4. Guardado y Carga de Estrategias (Config Management)
El sistema incluye un gestor de configuraciones en formato JSON guardados en la carpeta `src/quant_system/saved_strategies/`.
* **Guardar Estrategia:** En la pestaña "📊 Resumen", el usuario puede escribir un nombre y pulsar "💾 Guardar". Esto serializa todos los hiperparámetros actuales (activo, intervalo, fechas, context, horizon, step size, riesgo y modo de ejecución) en un archivo `.json`.
* **Cargar Estrategia:** En la parte superior del panel lateral izquierdo, el desplegable "Cargar Estrategia Guardada" muestra todos los archivos guardados. Al seleccionar uno, el sistema inyecta los parámetros en `st.session_state` y fuerza un refresco visual (`st.rerun()`), actualizando instantáneamente todos los sliders e inputs del dashboard.

### 3.5. Modo Trailing Stop Loss (Stop Loss Dinámico)
Como alternativa al Take Profit fijo, el sistema incluye la opción de **Trailing Stop Loss (Dinámico)** para maximizar las ganancias durante tendencias prolongadas a favor de la posición.
* **Lógica en Posición Long:** El sistema registra de manera continua el precio máximo alcanzado (`peak_price`) desde la apertura. El nivel de Stop Loss se actualiza dinámicamente según la fórmula: `dynamic_sl = max(entry_price * (1 - stop_loss_pct), peak_price * (1 - trailing_sl_pct))`. Si el precio se da la vuelta y cruza este umbral dinámico, la posición se liquida registrándose con el tipo `CLOSE_LONG_TSL`.
* **Lógica en Posición Short:** El sistema registra de manera continua el precio mínimo alcanzado (`lowest_price`). El nivel de Stop Loss dinámico se actualiza a: `dynamic_sl = min(entry_price * (1 + stop_loss_pct), lowest_price * (1 + trailing_sl_pct))`. Si el precio repunta y toca el nivel dinámico, la posición se liquida con el tipo `CLOSE_SHORT_TSL`.
* **Integración:** Funciona tanto en ejecuciones de posición única como en operaciones simultáneas (donde cada posición rastrea su propio Trailing SL independiente).

### 3.6. Modo Break-Even Stop Loss (Protección a Entrada)
El sistema permite activar la función **Break-Even**, la cual protege la cuenta trasladando el Stop Loss al precio exacto de entrada en cuanto la operación alcanza un beneficio porcentual determinado (`break_even_trigger_pct`).
* **Activación y Ejecución:** En cada vela, si una posición alcanza el porcentaje de ganancia especificado (ej. +1.5%), el Stop Loss de esa operación se actualiza automáticamente a `entry_price`. Si el mercado se da la vuelta y retrocede, la operación se liquida en el punto de entrada registrándose con el tipo `CLOSE_LONG_BE` o `CLOSE_SHORT_BE`, garantizando que la operación termine con cero pérdidas netas.
* **Combinabilidad:** Puede utilizarse de forma independiente con Take Profit Fijo o combinarse dinámicamente con Trailing Stop Loss.

### 3.7. Dimensionamiento Dinámico por Confianza (Quantile Sizing)
El sistema incluye la función **Dynamic Confidence Sizing**, la cual ajusta dinámicamente el tamaño del capital de cada posición basándose en los cuantiles probabilísticos de TimesFM 2.5 ($q_{10}$ a $q_{90}$).
* **Evaluación Cuantitativa:** Al evaluar una entrada en Long, se analiza el cuantil pesimista $q_{10}$. Si $q_{10} > \text{precio\_actual}$ (incluso en el escenario pesimista el modelo predice beneficio), se clasifica como una operación de `Alta Confianza` y se multiplica el capital asignado por el factor `confidence_multiplier` (ej. 1.5x o 2.0x). En Short, se evalúa si $q_{90} < \text{precio\_actual}$.
* **Registro:** Las operaciones ejecutadas bajo esta condición se registran con la etiqueta de confianza `High` en el libro de órdenes.

### 3.8. Modo Filtro de Incertidumbre (Uncertainty Filter)
El sistema integra el **Uncertainty Filter** para descartar señales de trading en periodos de volatilidad o ruido de mercado excesivo.
* **Métrica de Incertidumbre:** Calcula la dispersión relativa de los cuantiles al final del horizonte: $\text{uncertainty\_spread} = (q_{90} - q_{10}) / \text{precio\_actual}$.
* **Filtrado:** Si la dispersión supera el umbral máximo tolerado `max_uncertainty_pct` (ej. 5.0%), el sistema descarta la señal de entrada y no abre la posición, protegiendo la cuenta contra falsas rupturas. El número de entradas filtradas se contabiliza en la pestaña Operaciones.





---

## 4. 🤖 Notas para futuros modelos de IA (Contexto para Mejoras)

Si eres un modelo de Inteligencia Artificial leyendo esto para mejorar el sistema, aquí tienes los puntos críticos sobre los que debes iterar:

> [!NOTE]
> **Contexto Técnico:** El modelo `TimesFM` base es *univariante*, por lo tanto el predictor (`model_inference.py`) actualmente solo recibe el array unidimensional de cierres (`close`).

### Áreas de Mejora Sugeridas:

1. **Punto ciego del Step Size:** 
   En `backtester.py`, la variable `step_size = 6` (avanzar de 6 en 6 velas) se implementó para agilizar el tiempo de backtest en la prueba de concepto. **Tarea futura:** Mejorar la eficiencia del loop (o implementar inferencia por lotes/batch forecasting en TimesFM) para poder iterar con `step_size = 1` y tener un Stop-Loss 100% preciso vela a vela.
2. **Integración de Covariables (XReg):**
   TimesFM 2.5 permite variables exógenas (covariables). **Tarea futura:** Modificar `data_loader.py` para extraer el `volume` y covariables de calendario (hora, día de la semana), y actualizar `model_inference.py` para usar `model.forecast_with_covariates()`.
3. **Gestión de Riesgo Avanzada (Uso de Cuantiles):**
   Actualmente el motor usa `point_fc[-1]` (el pronóstico mediano) para decidir la dirección. **Tarea futura:** Modificar `backtester.py` para que lea `quant_fc` (por ejemplo el cuantil 10 y el cuantil 90) y no opere si la banda de incertidumbre/riesgo es muy ancha.
4. **Position Sizing Dinámico:**
   Actualmente el bot apuesta su porción asignada en cada operación. **Tarea futura:** Añadir una fracción de Kelly o un modelo de riesgo fijo (ej. arriesgar solo el 2% del capital en caso de tocar Stop-Loss).
5. **Comisiones de Funding (Futuros):**
   Si la estrategia es de Futuros Perpetuos, **Tarea futura:** añadir las *Funding Rates* periódicas que se cobran/pagan por mantener posiciones Short/Long prolongadas.

---

## 5. Dashboard y Visualización (`app.py`)

El dashboard actual actúa como el centro de control del sistema de backtesting. Está diseñado para ser simple pero altamente reactivo a los parámetros de entrada.

**Implementación Actual:**
* **Tecnología:** `Streamlit` para el renderizado web y reactividad, `Plotly Graph Objects` para los gráficos.
* **Sidebar (Panel Lateral):** Contiene el gestor de estrategias guardadas, selectores de activo, intervalo, fechas, parámetros del modelo (`context_len`, `horizon_len`, `step_size`), gestión de riesgo (Stop-Loss, Take-Profit, Umbral) y el modo de ejecución simultánea con slider de posiciones máximas.
* **Panel Principal:**
  * **Pestaña 📊 Resumen:** KPIs principales, guardado de estrategias en JSON y la gráfico interactivo de Curva de Equidad vs Buy & Hold.
  * **Pestaña 🕯️ Gráficos Avanzados:** Gráfico profesional de velas OHLC con marcadores de compra/venta/SL/TP, gráfico de Drawdown subacuático y barras de Volumen.
  * **Pestaña 🌡️ Rentabilidad Mensual:** Heatmap interactivo con matriz de retornos porcentuales por mes y año.
  * **Pestaña 📋 Operaciones:** Tabla contable detallada (Trade Log) con contadores de entradas, cierres normales, TP y SL hit, e informe descriptivo de la estrategia.

