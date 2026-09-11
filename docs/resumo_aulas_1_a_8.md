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
