# Aula 8 - Otimização de Memória: Global vs Compartilhada (Tiling)
import numpy as np
import time
from numba import cuda, float32

@cuda.jit
def matmul_global(A, B, C):
    row, col = cuda.grid(2)
    M, K = A.shape
    _, N = B.shape
    if row < M and col < N:
        soma = 0.0
        for k in range(K):
            soma += A[row, k] * B[k, col]
        C[row, col] = soma

TILE = 16
@cuda.jit
def matmul_shared(A, B, C):
    tile_A = cuda.shared.array((TILE, TILE), dtype=float32)
    tile_B = cuda.shared.array((TILE, TILE), dtype=float32)
    row, col = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    M, K = A.shape
    _, N = B.shape
    soma = float32(0.0)

    for t in range((K + TILE - 1) // TILE):
        kr = t * TILE + ty
        kc = t * TILE + tx
        tile_A[ty, tx] = A[row, kr] if (row < M and kr < K) else 0.0
        tile_B[ty, tx] = B[kc, col] if (kc < K and col < N) else 0.0
        
        cuda.syncthreads() 

        for k in range(TILE):
            soma += tile_A[ty, k] * tile_B[k, tx] 
            
        cuda.syncthreads()

    if row < M and col < N:
        C[row, col] = soma

N = 512
print(f"Gerando matrizes de teste ({N}x{N})...")
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
C_g = np.zeros((N, N), dtype=np.float32)
C_s = np.zeros((N, N), dtype=np.float32)

A_d, B_d = cuda.to_device(A), cuda.to_device(B)
C_g_d, C_s_d = cuda.to_device(C_g), cuda.to_device(C_s)
TPB = 16
BPG = (N + TPB - 1) // TPB

t0 = time.time()
matmul_global[(BPG, BPG), (TPB, TPB)](A_d, B_d, C_g_d)
cuda.synchronize()
t_global = time.time() - t0

t0 = time.time()
matmul_shared[(BPG, BPG), (TILE, TILE)](A_d, B_d, C_s_d)
cuda.synchronize()
t_shared = time.time() - t0

print("\n" + "="*45)
print(f"📊 RESULTADOS DE PERFORMANCE:")
print(f"-> Memória Global (VRAM): {t_global * 1000:.2f} ms")
print(f"-> Mem. Compartilhada (SRAM): {t_shared * 1000:.2f} ms")
print(f"🚀 Speedup com Tiling: {t_global / t_shared:.1f}x mais rápido!")
print("="*45)
