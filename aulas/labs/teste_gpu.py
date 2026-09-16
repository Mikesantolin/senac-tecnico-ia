import numpy as np
from numba import cuda

# Definição do Kernel (rodando na GPU)
@cuda.jit
def soma_vetores(a, b, c):
    # Mapeamento de threads, blocos e grades
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    
    # Cálculo do índice global para paralelismo
    i = tx + bx * bw
    if i < a.size:
        c[i] = a[i] + b[i]

# Alocação de Memória
N = 100000
a = np.ones(N)
b = np.ones(N)
c = np.zeros(N)

# Configuração da Grade (Grid) e Blocos (Blocks)
threads_por_bloco = 256
blocos_por_grade = (a.size + (threads_por_bloco - 1)) // threads_por_bloco

# Execução do Kernel
soma_vetores[blocos_por_grade, threads_por_bloco](a, b, c)
print("Soma concluída. Amostra do resultado (GPU):", c[:5])