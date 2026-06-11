Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Python = "C:\Users\yanbo\.conda\envs\ntt-gpu-cu128\python.exe"

& $Python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

& $Python .\src\ntt\tcn\train.py --model-config configs/tcn_gated/medium.json --training-config configs/tcn_gated/training_formal_medium.json

& $Python .\src\ntt\tcn\infer.py --checkpoint outputs/tcn_gated/medium/checkpoints/best.pt --input data/aligned/aligned_dry.wav --output outputs/tcn_gated/medium/prediction.wav --device auto

& $Python .\src\ntt\evaluation\evaluate_test_split.py --metadata data/chunks/metadata.json --tcn-checkpoint outputs/tcn_gated/medium/checkpoints/best.pt --model-name GatedTCN-Medium --out reports/test_metrics_tcn_medium.json --device auto

& $Python .\src\ntt\tcn\benchmark.py --checkpoint outputs/tcn_gated/medium/checkpoints/best.pt --input data/aligned/aligned_dry.wav --device auto

& $Python .\src\ntt\evaluation\compare_models.py `
  --target data/aligned/aligned_wet.wav `
  --a1-pred outputs/a1_baseline/prediction.wav `
  --a2-lite-pred outputs/a2_baseline/a2_lite_prediction.wav `
  --a2-full-pred outputs/a2_baseline/a2_full_prediction.wav `
  --tcn-small-pred outputs/tcn_gated/small/prediction.wav `
  --tcn-small-checkpoint outputs/tcn_gated/small/checkpoints/best.pt `
  --tcn-medium-pred outputs/tcn_gated/medium/prediction.wav `
  --tcn-medium-checkpoint outputs/tcn_gated/medium/checkpoints/best.pt `
  --tcn-large-pred outputs/tcn_gated/large/prediction.wav `
  --tcn-large-checkpoint outputs/tcn_gated/large/checkpoints/best.pt
