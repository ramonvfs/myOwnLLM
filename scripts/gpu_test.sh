#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --partition=gpu-4-h200        # troque para gpu-4-a100 / gpu-4-h200 para comparar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:02:00
#SBATCH --output=/home/rvfdsouza/%x-%j.out

nvidia-smi
cd ~/myownllm
uv run python - <<'EOF'
import time, torch
print(torch.__version__, torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
print("bf16 suportado:", torch.cuda.is_bf16_supported())
n = 8192
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        continue
    a = torch.randn(n, n, device="cuda", dtype=dtype); b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3): a @ b
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): a @ b
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 20
    print(f"{dtype}: {2 * n**3 / dt / 1e12:.1f} TFLOP/s")
EOF
