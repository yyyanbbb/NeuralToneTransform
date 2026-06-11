Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$GlobalGpuPython = "C:\Users\yanbo\.conda\envs\ntt-gpu-cu128\python.exe"

& $GlobalGpuPython -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
