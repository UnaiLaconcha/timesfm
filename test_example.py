import torch
import numpy as np
import timesfm

print("Checking PyTorch availability...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

torch.set_float32_matmul_precision("high")

print("Loading TimesFM 2.5 model (from_pretrained)...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

print("Compiling model configuration...")
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

print("Running dummy forecast...")
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],  # Two dummy inputs
)

print("Verification Succeeded!")
print(f"Point forecast shape: {point_forecast.shape} (Expected: (2, 12))")
print(f"Quantile forecast shape: {quantile_forecast.shape} (Expected: (2, 12, 10))")
