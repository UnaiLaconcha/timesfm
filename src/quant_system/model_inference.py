import numpy as np
import torch
import timesfm
import warnings


def _get_safe_device() -> str:
    """Detects available device safely. Falls back to CPU on CUDA errors (common in WSL2)."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # Force CUDA context initialization to catch WSL2 errors early
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except Exception as e:
        warnings.warn(f"[TimesFM] CUDA disponible pero falló la inicialización ({e}). Usando CPU.", stacklevel=2)
        return "cpu"


class TimesFMPredictor:
    def __init__(self, context_len: int = 512, horizon_len: int = 24):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.model = None
        self.device = _get_safe_device()
        self._load_model()

    def _load_model(self):
        torch.set_float32_matmul_precision("high")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_len,
                max_horizon=self.horizon_len,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                infer_is_positive=True
            )
        )

    def _forecast_safe(self, horizon: int, inputs: list):
        """Runs forecast with automatic CPU fallback on CUDA errors."""
        try:
            return self.model.forecast(horizon=horizon, inputs=inputs)
        except RuntimeError as e:
            err = str(e).lower()
            if "cuda" in err and self.device == "cuda":
                warnings.warn(f"[TimesFM] Error CUDA durante inferencia, cambiando a CPU: {e}", stacklevel=3)
                self.device = "cpu"
                return self.model.forecast(horizon=horizon, inputs=inputs)
            raise

    def predict(self, input_array: np.ndarray):
        """
        Input should be a 1D numpy array.
        Returns point_forecast and quantile_forecast.
        """
        point_forecast, quantile_forecast = self._forecast_safe(
            horizon=self.horizon_len,
            inputs=[input_array]
        )
        return point_forecast[0], quantile_forecast[0]

    def predict_batch(self, inputs_list: list):
        """
        Input should be a list of 1D numpy arrays.
        Returns point_forecasts (batch, horizon) and quantile_forecasts (batch, horizon, quantiles).
        """
        if not inputs_list:
            return np.array([]), np.array([])
        point_forecasts, quantile_forecasts = self._forecast_safe(
            horizon=self.horizon_len,
            inputs=inputs_list
        )
        return point_forecasts, quantile_forecasts
