# Aula 2 - Comparação Sequencial vs SIMD (NumPy)
import numpy as np
import time

N = 1_000_000
a = list(range(N))
b = list(range(N))

start = time.time()
c_seq = [a[i] + b[i] for i in range(N)]
seq_time = time.time() - start
print(f"Sequencial: {seq_time:.4f}s")

a_np = np.arange(N)
b_np = np.arange(N)
start = time.time()
c_np = a_np + b_np
np_time = time.time() - start
print(f"NumPy (SIMD): {np_time:.4f}s")
print(f"Speedup: {seq_time / np_time:.1f}x mais rápido!")
