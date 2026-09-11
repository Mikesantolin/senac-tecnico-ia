# Aula 7 - Benchmark FFT CPU vs GPU (CuPy)
import numpy as np
import time

N = 2**22
sinal_cpu = np.random.randn(N).astype(np.float32)

inicio = time.time()
_ = np.fft.fft(sinal_cpu)
tempo_cpu = time.time() - inicio
print(f"CPU FFT: {tempo_cpu * 1000:.2f} ms")

try:
    import cupy as cp
    sinal_gpu = cp.array(sinal_cpu)
    cp.cuda.Stream.null.synchronize()

    inicio = time.time()
    _ = cp.fft.fft(sinal_gpu)
    cp.cuda.Stream.null.synchronize()
    tempo_gpu = time.time() - inicio
    print(f"GPU FFT: {tempo_gpu * 1000:.2f} ms")
    print(f"Speedup: {tempo_cpu/tempo_gpu:.1f}x")
except ImportError:
    print("CuPy não instalado.")
