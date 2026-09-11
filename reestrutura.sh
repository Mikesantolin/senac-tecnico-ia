#!/bin/bash

echo "Criando a estrutura de pastas por aula para o repositório..."

# Criar pastas organizadas por aula
mkdir -p aulas/aula01 aulas/aula02 aulas/aula03 aulas/aula04 aulas/aula05 aulas/aula06 aulas/aula07 aulas/aula08 docs bash_scripts

# =========================================================
# 1. CRIAR README
# =========================================================
cat << 'EOF' > README.md
# Tecnologia e Infraestrutura para Inteligência Artificial

Repositório estruturado por aulas para o curso de IA focado em negócios, infraestrutura e arquitetura de computadores.
Contém scripts práticos, exemplos de código CUDA (Python) e scripts de administração Linux (Bash) utilizados nas Aulas 1 a 8.

## Estrutura do Repositório
- `/aulas/aula01` a `/aulas/aula08`: Códigos e materiais práticos separados por aula.
- `/docs`: Resumos consolidados e guias de estudo.
- `/bash_scripts`: Scripts para servidores Linux/WSL.
EOF

# =========================================================
# 2. CRIAR DOCUMENTAÇÃO NA PASTA DOCS
# =========================================================
cat << 'EOF' > docs/resumo_aulas_1_a_8.md
# Guia de Estudo e Resumo Consolidado: Aulas 1 a 8
## Fundamentos de Tecnologia e Infraestrutura para I.A.

---

### Bloco 1: Fundamentos

#### Aula 1: Introdução às Arquiteturas de Computadores e GPUs
* **Von Neumann vs. Harvard:** Von Neumann compartilha memória para dados e instruções (gargalo no barramento); Harvard separa as memórias para acesso simultâneo.
* **CPU vs. GPU:** CPU focada em tarefas sequenciais complexas (poucos núcleos); GPU focada em paralelismo massivo (milhares de núcleos para matrizes e IA).

#### Aula 2: Modelos de Processamento (SIMD, MIMD, RISC, CISC)
* **SIMD:** Uma instrução aplicada a múltiplos dados simultaneamente (GPUs, NumPy).
* **MIMD:** Múltiplas instruções em múltiplos dados (CPUs multi-core).
* **RISC vs. CISC:** RISC foca em instruções simples e fixas (baixo consumo, ARM); CISC foca em instruções complexas e variáveis (x86).

#### Aula 3: Estrutura de Memória em GPUs
* **Hierarquia:** Registradores (mais rápidos, por thread) -> Memória Compartilhada / SRAM (cache manual por bloco) -> Cache L1/L2 -> Memória Global / VRAM (alta capacidade, latência alta).
* **O Gargalo do Barramento:** Transferências CPU <-> GPU via PCIe são o gargalo principal e devem ser minimizadas.

#### Aula 4: Fundamentos de Processos e Threads
* **Processos vs. Threads:** Processos isolam memória (contornam o GIL do Python); Threads compartilham memória (leves, mas limitadas pelo GIL em tarefas CPU-bound).
* **Hierarquia CUDA:** Threads agrupadas em Warps (32 threads SIMD), Blocos (compartilham SRAM) e Grades (problema completo).

#### Aula 5: Protocolos de Redes e Interação com GPUs
* **IPv4 vs. IPv6:** Esgotamento do IPv4 impulsionou o IPv6 com endereçamento massivo.
* **TCP vs. UDP:** TCP garante entrega confiável (transferência de dados); UDP prioriza velocidade sem confirmação (telemetria).
* **SSH e Rsync:** Essenciais para controle remoto de servidores de GPU e sincronização de datasets.

#### Aula 6: Sistemas Operacionais Linux e GPU
* **Estrutura Virtual:** Uso de `/dev`, `/proc` e `/sys` para interagir com o kernel e estado das GPUs.
* **Automação:** Uso combinado de `cron` para tarefas agendadas e `systemd` para serviços contínuos, além de sessões persistentes com `tmux` e `screen`.

---

### Bloco 2: Programação e Otimização

#### Aula 7: Introdução ao Modelo CUDA
* **Kernels:** Funções executadas em paralelo por milhares de threads diretamente na VRAM.
* **Índices Globais:** Combinação de `blockIdx`, `blockDim` e `threadIdx` para mapear dados unicamente.
* **Sincronização:** `cuda.synchronize()` garante a conclusão dos cálculos na GPU antes de retornar os dados para a CPU.

#### Aula 8: Manipulação de Memória em CUDA (Tiling)
* **Regra 90/10:** 90% do tempo de processamento em I.A. é gasto em acessos à memória.
* **Tiling:** Técnica de carregar pedaços de dados da VRAM para a Memória Compartilhada rápida, permitindo reutilização e eliminando o gargalo de largura de banda.
* **Coalescing:** Acessos consecutivos à memória unificados em transações eficientes.
EOF

cat << 'EOF' > docs/questionario.md
# Questionário de Revisão (Aulas 1 a 7)
## Fundamentos de Tecnologia e Infraestrutura para I.A.

> **Instruções:** Utilize o resumo consolidado das aulas como guia de consulta para responder às questões abaixo, focando na aplicação prática e na visão de negócios.

---

### BLOCO 1: FUNDAMENTOS

#### Aula 1: Quem Faz o Quê? (CPU vs. GPU)
1. Qual é a principal diferença funcional, apresentada na analogia da aula, entre a CPU (o cérebro central) e a GPU (a placa de vídeo)?
2. Por que as CPUs tradicionais não conseguem atender sozinhas à demanda pesada de processamento exigida pelo treinamento de modelos modernos de Inteligência Artificial?
3. Na visão de infraestrutura de uma empresa, o que define a escolha entre utilizar o poder de uma CPU ou de uma GPU para uma determinada tarefa computacional?

#### Aula 2: Como Eles Trabalham? (Organização do Trabalho)
4. Explique como funciona o modelo SIMD (Sincronia Total) utilizado pelas GPUs, comparando-o com a dinâmica de uma equipe de trabalho.
5. Qual é a principal vantagem da arquitetura de processamento RISC (com instruções simples e padronizadas) em comparação à CISC na fabricação de dispositivos modernos?
6. Como a escolha entre arquiteturas voltadas para eficiência energética (como em dispositivos móveis) impacta o desenvolvimento de soluções em tecnologia?

#### Aula 3: A Logística e a Memória (Por que a IA fica lenta?)
7. O que é o "gargalo da rodovia" mencionado ao discutir a transferência de dados entre a memória RAM e a VRAM da placa de vídeo?
8. Descreva a diferença prática entre os registradores (na mão do operário) e a Memória Global/VRAM dentro da hierarquia de memória da GPU.
9. Por que a lentidão na busca de informações em camadas de memória distantes pode prejudicar a performance geral de um algoritmo de Inteligência Artificial?

#### Aula 4: Como a Máquina Divide o Trabalho (Processos vs. Threads)
10. Qual é a principal diferença de segurança e custo operacional entre abrir um novo "Processo" e alocar novas "Threads" em um sistema computacional?
11. O que é o GIL (Global Interpreter Lock) na linguagem Python e de que maneira ele restringe o aproveitamento de múltiplos núcleos em processamentos pesados por threads?
12. O que ocorre com o desempenho de uma GPU quando o fluxo de processamento paralelizável sofre com a chamada "Divergência" (excesso de desvios condicionais do tipo "se/senão")?

#### Aula 5: A Logística e a Comunicação da I.A. (Redes e Transferência)
13. Explique por que a escassez de endereços no padrão IPv4 gerou a necessidade urgente de migração para o IPv6 no ecossistema de grandes data centers.
14. Qual é o critério técnico e de negócio que define quando uma aplicação deve optar pelo protocolo de transporte TCP em detrimento do UDP?
15. Ao transferir um volume massivo de dados (como centenas de gigabytes) para um servidor em nuvem, por que a utilização da ferramenta `rsync` é financeiramente e operacionalmente superior ao uso de ferramentas tradicionais como o `scp`?

#### Aula 6: Sistemas Operacionais Linux e Gerenciamento de GPU
16. Qual é a função dos diretórios virtuais `/proc` e `/sys` no sistema operacional Linux para a administração de servidores de infraestrutura?
17. Por que ferramentas como `screen`, `tmux` ou `nohup` são consideradas essenciais para profissionais que gerenciam treinamentos longos de I.A. em servidores remotos via SSH?
18. Qual é a utilidade prática do agendador de tarefas `cron` combinado com o `systemd` na rotina de manutenção de um ambiente produtivo de GPUs?

---

### BLOCO 2: PROGRAMAÇÃO

#### Aula 7: Introdução ao Modelo CUDA
19. O que é um *kernel* dentro da plataforma de computação paralela CUDA desenvolvida pela NVIDIA?
20. Como a hierarquia CUDA organiza o problema computacional utilizando Grades (Grids), Blocos (Blocks) e Threads individuais controladas por índices?
21. Por que a etapa de sincronização (`cuda.synchronize()`) é obrigatória antes de transferir de volta os dados processados na VRAM para a memória da CPU?
EOF

# =========================================================
# 3. CRIAR SCRIPTS BASH NAS PASTAS CORRETAS
# =========================================================
cat << 'EOF' > bash_scripts/comandos_transferencia.sh
#!/bin/bash
# Exemplos práticos da Aula 5 (IP/Rede)

# Envio simples de arquivo
# scp relatorio.txt usuario_mint@172.31.15.50:/home/usuario_mint/

# Sincronização inteligente de diretório
# rsync -avzP meus_dados/ usuario_mint@172.31.15.50:/home/usuario_mint/destino_nuvem/
EOF

cat << 'EOF' > bash_scripts/gpu_status.sh
#!/bin/bash
# Script de monitoramento da Aula 6
echo -e "\n=== Monitor de Recursos de I.A. ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
else
    echo "Nenhuma GPU gerenciável encontrada no sistema."
fi
EOF
chmod +x bash_scripts/gpu_status.sh

# =========================================================
# 4. DISTRIBUIR SCRIPTS PYTHON POR PASTA DE AULA (/aulas/aulaXX)
# =========================================================

# --- Aula 2 ---
cat << 'EOF' > aulas/aula02/simd_numpy.py
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
EOF

# --- Aula 3 ---
cat << 'EOF' > aulas/aula03/benchmark_ram_vram.py
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
else:
    print("GPU não disponível.")
EOF

# --- Aula 4 ---
cat << 'EOF' > aulas/aula04/processos_threads.py
# Aula 4 - Threading vs Multiprocessing em Python (CPU-bound)
import multiprocessing
import threading
import time
import math

def calcular_primos(limite):
    primos = []
    for n in range(2, limite):
        if all(n % i != 0 for i in range(2, int(math.sqrt(n)) + 1)):
            primos.append(n)
    return len(primos)

LIMITE = 50_000
N_TAREFAS = 4

inicio = time.time()
with multiprocessing.Pool(processes=N_TAREFAS) as pool:
    pool.starmap(calcular_primos, [(LIMITE,)] * N_TAREFAS)
print(f"Multiprocessing: {time.time() - inicio:.2f}s (Paralelismo real)")
EOF

# --- Aula 7 ---
cat << 'EOF' > aulas/aula07/fft_benchmark.py
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
EOF

# --- Aula 8 ---
cat << 'EOF' > aulas/aula08/tiling_benchmark.py
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
EOF

cat << 'EOF' > aulas/aula08/stress_nvtop.py
# Aula 8 - Teste de estresse para monitoramento via nvtop
import cupy as cp
import time

print("Iniciando estresse da GPU para monitoramento...")
TAMANHO = 8192
A = cp.random.randn(TAMANHO, TAMANHO, dtype=cp.float32)
B = cp.random.randn(TAMANHO, TAMANHO, dtype=cp.float32)

try:
    while True:
        _ = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
except KeyboardInterrupt:
    print("Estresse encerrado.")
EOF

# =========================================================
# 5. ATUALIZAR GIT E ENVIAR PARA O GITHUB
# =========================================================
git add .
git commit -m "refactor: organiza scripts python por pastas de aulas de 1 a 8"
git push -u origin main

echo ""
echo "✅ Estrutura reorganizada com sucesso por pastas de aulas e enviada para o GitHub!"