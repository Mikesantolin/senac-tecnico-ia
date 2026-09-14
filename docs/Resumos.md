# Guia de Estudo e Resumo Consolidado
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

#### Aula 9: Alternativas ao CUDA: OpenCL
*## 1. Contextualização e Suporte
* **Objetivo:** Viabilizar computação paralela multiplataforma e heterogênea.
* **Cenário de Negócio:** Atender clientes com datacenters baseados em GPUs AMD e Intel (incompatíveis com CUDA).
* **Suporte por Fabricante:**
  * **NVIDIA:** OpenCL 3.0 (suportado, mas com preferência por CUDA).
  * **AMD:** OpenCL 3.0 (suportado nativamente via ROCm).
  * **Intel:** OpenCL 3.0 (CPU, GPU integrada e Arc).
  * **Apple:** OpenCL 1.2 (*deprecated* / descontinuado em favor do Metal).
  * **Qualcomm:** OpenCL 2.0 (GPUs mobile / Snapdragon).

## 2. Arquitetura e Componentes do OpenCL
* **Platform:** Conjunto de drivers do fabricante do hardware.
* **Device:** Unidade física de processamento (CPU, GPU, FPGA, DSP).
* **Context:** Gerenciador que agrupa dispositivos, buffers de memória e filas.
* **Command Queue:** Fila de despacho de operações (kernels e cópias de dados).
* **Kernel OpenCL:** Função escrita em C99 compilada em tempo de execução (*JIT*).
* **Buffer:** Área de memória alocada explicitamente no dispositivo (`cl.Buffer`).

## 3. Mapeamento Lógico (CUDA vs. OpenCL)
| Conceito | CUDA (NVIDIA) | OpenCL (Khronos) |
| :--- | :--- | :--- |
| **Unidade de Execução** | `thread` | `work-item` |
| **Grupo de Execução** | `block` | `work-group` |
| **Conjunto Completo** | `grid` | `NDRange` |
| **ID Local** | `threadIdx.x` | `get_local_id(0)` |
| **ID do Grupo** | `blockIdx.x` | `get_group_id(0)` |
| **ID Global** | `cuda.grid(1)` | `get_global_id(0)` |
| **Tamanho do Grupo** | `blockDim.x` | `get_local_size(0)` |
| **Memória Compartilhada**| `cuda.shared.array()` | `__local float[]` |
| **Sincronização** | `cuda.syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` |
| **Memória Constante** | `__constant__` | `__constant` |

## 4. Trade-offs (Portabilidade vs. Ecossistema)
* **Vantagens:** 
  * Portabilidade multi-vendor (NVIDIA, AMD, Intel, CPUs, FPGAs).
  * Eliminação de *vendor lock-in*.
  * Fallback nativo para CPU.
* **Desvantagens:** 
  * Código mais verboso (~3× mais longo).
  * Compilação JIT gera latência na inicialização.
  * Ecossistema de alto nível para Deep Learning (PyTorch/TensorFlow/cuDNN) fortemente acoplado à NVIDIA.