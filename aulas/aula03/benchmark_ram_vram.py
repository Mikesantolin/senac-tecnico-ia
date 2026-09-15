# Aula 3 - Benchmark RAM vs VRAM (PyTorch)
import torch
import time

N = 10_000_000
a_cpu = torch.randn(N)
b_cpu = torch.randn(N)

start = time.time()
for _ in range(100):
    c_cpu = a_cpu + b_cpu
cpu_time = (time.time() - start) / 100
print(f"CPU (RAM): {cpu_time*1000:.3f} ms")

if torch.cuda.is_available():
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        c_gpu = a_gpu + b_gpu
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 100
    print(f"GPU (VRAM): {gpu_time*1000:.3f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")

    # Validação de corretude
    assert torch.allclose(c_cpu, c_gpu.cpu(), atol=1e-5), "Resultados CPU/GPU divergem!"
    print("✓ Resultados conferem: RAM == VRAM")
else:
    print("GPU não disponível.")
