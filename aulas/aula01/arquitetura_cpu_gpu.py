# Aula 1 - Arquitetura de Computadores: Sequencial vs Paralelo
# Demonstra por que a CPU sozinha não escala para operações matriciais de IA.
import numpy as np
import time

N = 200  # Matrizes N×N

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

print(f"Multiplicação de matrizes {N}x{N}")
print(f"Total de operações: {N**3 * 2:,} (multiply-add)\n")

# Abordagem sequencial: 3 loops aninhados (como Von Neumann processa, um cálculo por vez)
inicio = time.time()
C = np.zeros((N, N), dtype=np.float32)
for i in range(N):
    for j in range(N):
        soma = 0.0
        for k in range(N):
            soma += A[i, k] * B[k, j]
        C[i, j] = soma
tempo_seq = time.time() - inicio
print(f"Sequencial (3 loops): {tempo_seq:.2f}s")

# Abordagem vetorizada: NumPy delega para bibliotecas otimizadas (BLAS)
# que exploram paralelismo de hardware (múltiplos núcleos + instruções SIMD)
inicio = time.time()
C_np = A @ B
tempo_np = time.time() - inicio
print(f"Vetorizado (NumPy):   {tempo_np:.6f}s")

# Validação: ambos produzem o mesmo resultado
assert np.allclose(C, C_np, atol=1e-2), "Resultados divergem!"
print(f"\n✓ Resultados conferem — Speedup: {tempo_seq / tempo_np:.0f}x")
print(f"→ Em redes neurais, essas matrizes têm milhões de linhas. Sem paralelismo, é inviável.")
