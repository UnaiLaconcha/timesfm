import numpy as np
import torch
import timesfm

class TimesFMPredictor:
    def __init__(self, context_len: int = 512, horizon_len: int = 24):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.model = None
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
        
    def predict(self, input_array: np.ndarray):
        """
        Input should be a 1D numpy array.
        Returns point_forecast and quantile_forecast.
        """
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=self.horizon_len,
            inputs=[input_array]
        )
        return point_forecast[0], quantile_forecast[0]
