#!/bin/bash

echo "Recriando a estrutura completa do curso na pasta atual..."

# Criar subpastas
mkdir -p docs bash_scripts colab_scripts

# =========================================================
# 1. CRIAR README
# =========================================================
cat << 'EOF' > README.md
# Tecnologia e Infraestrutura para Inteligência Artificial

Repositório estruturado para o curso de IA focado em negócios e infraestrutura.
Contém scripts práticos, exemplos de código CUDA (Python) e scripts de administração Linux (Bash) utilizados nas Aulas 1 a 8.

## Estrutura do Repositório
- `/docs`: Resumos consolidados e guias de estudo (Aulas 1 a 8) e gabaritos.
- `/bash_scripts`: Scripts para rodar em servidores Linux/WSL (Rsync, SCP, Monitoramento de GPU).
- `/colab_scripts`: Códigos Python (CuPy/Numba) prontos para uso no Google Colab, focados em experimentação e análise de negócio.
EOF

# =========================================================
# 2. CRIAR DOCUMENTAÇÃO (RESUMO E GABARITO)
# =========================================================
cat << 'EOF' > docs/resumo_aulas_1_a_8.txt
======================================================
GUIA DE ESTUDO E RESUMO CONSOLIDADO: AULAS 1 A 8
Fundamentos de Tecnologia e Infraestrutura para I.A.
======================================================

--- BLOCO 1: FUNDAMENTOS ---

## Aula 1: Quem Faz o Quê? (CPU vs. GPU)
* Função Principal: A CPU atua como o cérebro central (chef experiente para decisões complexas), enquanto a GPU funciona como a "fábrica" (galpão com milhares de operários executando cálculos matemáticos simultâneos).
* Critério de Escolha: Tarefas lógicas vão para a CPU; processamento matemático paralelo vai para a GPU.

## Aula 2: Como Eles Trabalham?
* SIMD (Sincronia Total): Modelo padrão da GPU onde um comando único comanda vários elementos executando a mesma instrução em dados diferentes.
* RISC vs. CISC: RISC foca em instruções simples (alta eficiência energética); CISC lida com comandos complexos.

## Aula 3: A Logística e a Memória
* O Gargalo da Rodovia: A lentidão na transferência de dados entre a RAM e a VRAM através do barramento PCIe.
* Impacto na Performance: A distância das camadas de memória (Registradores rápidos vs. VRAM lenta) determina se a GPU fica ociosa.

## Aula 4: Processos vs. Threads
* Processos criam filiais isoladas (seguras e caras), threads compartilham o mesmo espaço (rápidas, baratas, mas vulneráveis).
* O GIL do Python: Mecanismo que impede o verdadeiro paralelismo de múltiplas threads na CPU.
* Divergência na GPU: Desvios condicionais ("se/senão") forçam partes do grupo a esperar, reduzindo o desempenho.

## Aula 5: Redes e Transferência
* TCP vs. UDP: TCP prioriza a entrega segura (dados financeiros); UDP aposta na velocidade sem confirmação (streaming).
* Ferramentas de Transferência: O rsync otimiza grandes volumes permitindo retomada de quedas, superando o scp.

## Aula 6: Linux e Gerenciamento de GPU
* Sessões Persistentes: Ferramentas como screen e tmux protegem treinamentos de interrupções por quedas de conexão SSH.
* Automação: cron (agendamento) com systemd (serviços contínuos) garantem a estabilidade da infraestrutura.

--- BLOCO 2: PROGRAMAÇÃO E OTIMIZAÇÃO ---

## Aula 7: Introdução ao Modelo CUDA
* O Conceito de Kernel: Funções executadas em paralelo por milhares de threads diretamente na GPU.
* Hierarquia: Grade (Grid), Blocos (Blocks) e Threads individuais.
* Sincronização: cuda.synchronize() assegura que cálculos na VRAM terminaram antes de voltar à CPU.

## Aula 8: Manipulação de Memória em CUDA (Tiling)
* A Regra 90/10: 90% do tempo de execução de um algoritmo de I.A. é gasto em acessos à memória.
* Memória Compartilhada (Shared Memory): Cache manual ultra rápido compartilhado entre threads do mesmo bloco.
* Tiling: Técnica de carregar pedaços (tiles) da VRAM para a Memória Compartilhada para aliviar o tráfego de dados.
* Coalescing: Acessos consecutivos otimizam as transações da GPU. Acessos espalhados geram lentidão severa.
EOF

cat << 'EOF' > docs/questionario_gabarito.txt
======================================================
GABARITO: QUESTIONÁRIO DE REVISÃO (AULAS 1 a 7)
======================================================
1. CPU vs GPU: CPU foca em tarefas sequenciais complexas; GPU foca em cálculos paralelos massivos.
2. Gargalo da CPU: O treinamento exige matrizes gigantes que a CPU sequencial demora muito para processar.
3. Gargalo de Memória: A limitação de banda do barramento PCIe entre RAM e VRAM.
4. Processos vs Threads: Processos não compartilham memória (mais seguros); Threads compartilham (mais rápidas, menos seguras).
5. GIL do Python: Trava que impede a execução de código Python em múltiplos núcleos simultaneamente.
6. Divergência: Uso excessivo de IF/ELSE quebra a sincronia dos workers na GPU.
7. TCP vs UDP: TCP garante a entrega do pacote de dados; UDP envia o pacote priorizando tempo sem checar recebimento.
8. Rsync: Ideal para IA pois retoma transferências pesadas de onde pararam.
9. Tmux/Screen: Mantém o job rodando no servidor mesmo se a rede do seu notebook cair.
10. CUDA Kernel: Função escrita para rodar na GPU.
11. Tiling (Aula 8): Estratégia de negócio para poupar custo de nuvem usando cache compartilhado para reduzir tempo de GPU ativa.
EOF

# =========================================================
# 3. CRIAR SCRIPTS BASH (Aulas 5 e 6)
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
# 4. CRIAR SCRIPTS COLAB (Aulas 7 e 8)
# =========================================================
cat << 'EOF' > colab_scripts/aula07_fft_benchmark.py
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
EOF

cat << 'EOF' > colab_scripts/aula08_tiling_benchmark.py
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
EOF

cat << 'EOF' > colab_scripts/aula08_stress_nvtop.py
# Aula 8 - Teste de Estresse para monitoramento via NVTOP
import cupy as cp
import time

print("🔥 Iniciando o teste de estresse da GPU...")
TAMANHO = 16384 
A = cp.random.randn(TAMANHO, TAMANHO, dtype=cp.float32)
B = cp.random.randn(TAMANHO, TAMANHO, dtype=cp.float32)

ciclo = 0
try:
    while True:
        C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
        ciclo += 1
        if ciclo % 10 == 0:
            print(f"Lote {ciclo} concluído. GPU cravada em 100%...")
except KeyboardInterrupt:
    print("\n✅ Teste interrompido com segurança.")
    del A, B, C
    cp.get_default_memory_pool().free_all_blocks()
EOF

# =========================================================
# 5. CONFIGURAR GIT E ENVIAR PARA O GITHUB
# =========================================================
git init
git add .
git commit -m "feat: recriando estrutura completa do curso de infra para IA"
git branch -M main
git remote add origin https://github.com/jonasmaffei/senac-tecnico-ia.git
git push -u origin main

echo ""
echo "✅ Estrutura criada e enviada com sucesso para o GitHub!"