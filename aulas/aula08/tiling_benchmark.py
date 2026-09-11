# Aula 8 - Otimização de Memória: Global vs Tiling (Numba CUDA)
import numpy as np
import time
from numba import cuda, float32

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
        tile_B[ty, tx] = B[kc, col] if (kc < K and kc < N) else 0.0
        cuda.syncthreads()

        for k in range(TILE):
            soma += tile_A[ty, k] * tile_B[k, tx]
        cuda.syncthreads()

    if row < M and col < N:
        C[row, col] = soma

print("Script de Tiling pronto para execução em ambiente CUDA.")
