# Aula 7 - Benchmark: Transformada de Fourier (CPU vs GPU)
import numpy as np
import time

N = 2**22 
sinal_cpu = np.random.randn(N).astype(np.float32)

print("Processando na CPU...")
inicio = time.time()
fft_cpu = np.fft.fft(sinal_cpu)
tempo_cpu = time.time() - inicio
print(f"-> Tempo CPU: {tempo_cpu * 1000:.2f} ms\n")

try:
    import cupy as cp
    print("Processando na GPU (CUDA)...")
    sinal_gpu = cp.array(sinal_cpu)
    
    _ = cp.fft.fft(sinal_gpu) # Warm-up
    cp.cuda.Stream.null.synchronize()

    inicio = time.time()
    fft_gpu = cp.fft.fft(sinal_gpu)
    cp.cuda.Stream.null.synchronize()
    
    tempo_gpu = time.time() - inicio
    print(f"-> Tempo GPU: {tempo_gpu * 1000:.2f} ms")
    print(f"\n🚀 Speedup: A GPU foi {tempo_cpu/tempo_gpu:.1f}x mais rápida!")
except ImportError:
    print("CuPy não instalado. Instale usando: !pip install cupy-cuda12x")
