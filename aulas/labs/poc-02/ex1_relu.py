import numpy as np
from numba import cuda

@cuda.jit
def aplicar_relu(vetor_entrada, vetor_saida):
    # Identificação da posição global da thread (1D)
    posicao = cuda.grid(1)
    
    # Prevenção de acesso fora dos limites da memória global
    if posicao < vetor_entrada.size:
        valor = vetor_entrada[posicao]
        
        # Lógica da função ReLU (comum em pipeline de IA)
        if valor > 0:
            vetor_saida[posicao] = valor
        else:
            vetor_saida[posicao] = 0.0

# 1. Preparação dos dados (arrays NumPy)
N = 20
np.random.seed(42) # Mantém os valores fixos para demonstração
dados_entrada = np.random.uniform(-10, 10, N).astype(np.float32)
dados_saida = np.zeros(N, dtype=np.float32)

# 2. Configuração de execução (threads e blocos)
threads_por_bloco = 32
blocos = (N + (threads_por_bloco - 1)) // threads_por_bloco

# 3. Chamada do kernel
aplicar_relu[blocos, threads_por_bloco](dados_entrada, dados_saida)

print("--- Função de Ativação ReLU (GPU) ---")
print("Entrada:", np.round(dados_entrada, 2))
print("Saída:  ", np.round(dados_saida, 2))