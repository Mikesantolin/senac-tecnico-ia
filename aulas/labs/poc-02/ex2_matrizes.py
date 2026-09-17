import numpy as np
from numba import cuda, float32

# Definindo o tamanho do bloco (tile) como constante para alocar memória
TAMANHO_BLOCO = 16

@cuda.jit
def mult_matriz_compartilhada(A, B, C):
    # Alocação de memória compartilhada (acesso ultrarrápido, restrito ao bloco atual)
    sA = cuda.shared.array(shape=(TAMANHO_BLOCO, TAMANHO_BLOCO), dtype=float32)
    sB = cuda.shared.array(shape=(TAMANHO_BLOCO, TAMANHO_BLOCO), dtype=float32)

    # Identificando as posições 2D (Grades e Blocos)
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    # Variável temporária mantida nos registradores da thread
    tmp = 0.0
    
    # Loop dividindo a operação em "ladrilhos" (tiles) para caber na memória compartilhada
    for i in range(bpg):
        # Cada thread carrega um elemento da memória global para a compartilhada
        sA[tx, ty] = A[x, ty + i * TAMANHO_BLOCO]
        sB[tx, ty] = B[tx + i * TAMANHO_BLOCO, y]
        
        # Sincronização: garante que todas as threads do bloco terminaram de carregar os dados
        cuda.syncthreads()

        # Computando o resultado parcial usando a memória rápida
        for j in range(TAMANHO_BLOCO):
            tmp += sA[tx, j] * sB[j, ty]
        
        # Sincronização: garante que o cálculo terminou antes do próximo loop sobrescrever sA e sB
        cuda.syncthreads()

    # Gravação do resultado final de volta na memória global
    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp

# 1. Preparação dos dados
N = 32 # Tamanho da matriz (32x32)
A = np.ones((N, N), dtype=np.float32) * 2.0  # Matriz preenchida com 2
B = np.ones((N, N), dtype=np.float32) * 3.0  # Matriz preenchida com 3
C = np.zeros((N, N), dtype=np.float32)

# 2. Configuração 2D para blocos e grades
threads_bloco = (TAMANHO_BLOCO, TAMANHO_BLOCO)
blocos_grade_x = int(np.ceil(A.shape[0] / threads_bloco[0]))
blocos_grade_y = int(np.ceil(B.shape[1] / threads_bloco[1]))
blocos_grade = (blocos_grade_x, blocos_grade_y)

# 3. Chamada do Kernel
mult_matriz_compartilhada[blocos_grade, threads_bloco](A, B, C)

print("--- Multiplicação de Matrizes com Memória Compartilhada ---")
print("Matriz C (Amostra 4x4 do resultado final):\n", C[:4, :4])