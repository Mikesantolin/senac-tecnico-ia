# Trilha de Aprendizado (Timeline): Do Silício à Computação Heterogênea
## A Cadeia Causal da Engenharia de Infraestrutura para I.A.

> **Filosofia da Trilha:** Em Inteligência Artificial, código ineficiente não é apenas lento — ele queima orçamento de nuvem, desperdiça energia e trava a escala do negócio. Cada aula resolve o gargalo deixado pela anterior, formando um encadeamento lógico inevitável do silício ao ecossistema multi-vendor.

---

## O Fluxo Contínuo de Dependência Técnica

```text
[A1] Von Neumann/Harvard ──> [A2] SIMD/MIMD/RISC/CISC ──> [A3] Hierarquia de Memória (VRAM/PCIe)
                                                                      │
                                                                      ▼
[A10] AMD ROCm & HIP <── [A9] OpenCL Multi-Vendor <── [A8] Tiling & Coalescing <── [A7] Kernels CUDA ──> [A6] Linux Ops ──> [A5] Redes/Rsync ──> [A4] Processos/GIL
```

---

## Detalhamento Sequencial da Trilha

### Aula 1: Introdução às Arquiteturas de Computadores e GPUs
* **Conceito/Fundamento:** Von Neumann compartilha memória para dados e instruções (gargalo no barramento); Harvard separa as memórias para acesso simultâneo. CPU focada em tarefas sequenciais complexas (poucos núcleos); GPU focada em paralelismo massivo (milhares de núcleos para matrizes e IA).
* **O que se aprende:** O gargalo estrutural de Von Neumann vs. Harvard, e a divisão de papéis entre a CPU (general sequencial) e a GPU (exército massivo matricial).
* **O problema que fica em aberto:** Sabemos *que* a GPU processa matrizes em massa, mas *como* os dados e instruções se movem e se organizam dentro do chip?
* **Impacto de negócio:** Impede a contratação cega de instâncias caras de nuvem sem saber se o workload é matricial ou sequencial.

### Aula 2: Modelos de Processamento (SIMD, MIMD, RISC, CISC)
* **Conceito/Fundamento:** SIMD aplica uma instrução a múltiplos dados simultaneamente (GPUs, NumPy); MIMD executa múltiplas instruções em múltiplos dados (CPUs multi-core). RISC foca em instruções simples e fixas (baixo consumo, ARM); CISC foca em instruções complexas e variáveis (x86).
* **O que se aprende:** Taxonomia de Flynn (SIMD vetorizado vs. MIMD multi-core) e o balanço energético de instruções (RISC fixo/baixo consumo em ARM vs. CISC complexo em x86).
* **Conexão com a Aula 1:** Explica o *mecanismo de execução* da GPU (SIMD expandido em massa) e justifica por que dispositivos de borda (câmeras, edge) escolhem RISC.
* **O problema que fica em aberto:** O dado chega vetorizado, mas quanto tempo ele gasta para ser buscado na hierarquia de armazenamento?

### Aula 3: Estrutura de Memória em GPUs
* **Conceito/Fundamento:** A hierarquia vai de Registradores (mais rápidos, por thread) ➔ Memória Compartilhada / SRAM (cache manual por bloco) ➔ Cache L1/L2 ➔ Memória Global / VRAM (alta capacidade, latência alta). Transferências CPU ⟷ GPU via PCIe são o gargalo principal e devem ser minimizadas.
* **O que se aprende:** A pirâmide de latência (Registradores 1 ciclo -> SRAM Compartilhada 5 ciclos -> VRAM Global 500 ciclos) e o gargalo estrangulador do barramento PCIe.
* **Conexão com a Aula 2:** Descobrimos o verdadeiro vilão de performance: notória falta de largura de banda de entrega de dados pela rodovia PCIe/VRAM, e não falta de cálculo no SIMD.
* **O problema que fica em aberto:** A VRAM está ali, mas quem despacha os lotes de dados do sistema operacional para o barramento sem travar o processador?

### Aula 4: Fundamentos de Processos e Threads
* **Conceito/Fundamento:** Processos isolam memória (contornam o GIL do Python); Threads compartilham memória (leves, mas limitadas pelo GIL em tarefas CPU-bound). Hierarquia CUDA agrupa threads em Warps (32 threads SIMD), Blocos (compartilham SRAM) e Grades (problema completo).
* **O que se aprende:** Isolamento de memória via `multiprocessing` (contornando o GIL do Python no Data Loader) vs. levezas de threads limitadas a I/O.
* **Conexão com a Aula 3:** O gargalo de VRAM é alimentado por batches preparados em paralelo no host; se usarmos `threading` em CPU-bound, o GIL satura e a GPU fica ociosa esperando o dataset.
* **O problema que fica em aberto:** O dataset foi preparado pelo host, mas e quando ele está em um storage remoto de 200 GB fora da máquina local?

### Aula 5: Protocolos de Redes e Interação com GPUs
* **Conceito/Fundamento:** IPv4 lida com esgotamento de endereços impulsionando o IPv6 massivo; TCP garante entrega confiável (transferência de dados) enquanto UDP prioriza velocidade sem confirmação (telemetria). SSH e Rsync são essenciais para controle remoto de servidores de GPU e sincronização de datasets.
* **O que se aprende:** IPv4/IPv6, TCP (confiável/ordenado para datasets/modelos via `scp`/`rsync`) vs. UDP (rápido/sem confirmação para telemetria), e túneis SSH seguros.
* **Conexão com a Aula 4:** O data loader local da A4 não adianta se o cluster for distribuído; os dados precisam atravessar a malha de rede com resiliência a quedas (`rsync --partial`).
* **O problema que fica em aberto:** Os dados chegaram via rede ao nó remoto, mas o SO subjacente precisa sustentar o hardware e os drivers de vídeo por 12 horas seguidas.

### Aula 6: Sistemas Operacionais Linux e GPU
* **Conceito/Fundamento:** Uso de `/dev`, `/proc` e `/sys` para interagir com o kernel e estado das GPUs. Automação combinada de `cron` para tarefas agendadas e `systemd` para serviços contínuos, além de sessões persistentes com `tmux` e `screen`.
* **O que se aprende:** O kernel Linux como gestor de silício via `/dev/nvidia*`, `/proc`, `/sys`, resiliência contra queda SSH via `tmux`/`screen` e automação com `cron`/`systemd`.
* **Conexão com a Aula 5:** A rede entregou o arquivo, o SSH conectou, mas o job de treino longo precisa sobreviver a desconexões e auditar temperatura/recursos via terminal.
* **O problema que fica em aberto:** O ambiente está estável, seguro e monitorado, mas o código executado (Python puro/PyTorch padrão) ainda não espreme o máximo de eficiência do silício da VRAM.

### Aula 7: Introdução ao Modelo CUDA
* **Conceito/Fundamento:** Kernels são funções executadas em paralelo por milhares de threads diretamente na VRAM. Uso de índices globais (`blockIdx`, `blockDim`, `threadIdx`) para mapear dados unicamente, com `cuda.synchronize()` garantindo conclusão dos cálculos antes de retornar dados à CPU.
* **O que se aprende:** Escrita de *kernels* em Python/Numba, mapeamento de índice global (`blockIdx.x * blockDim.x + threadIdx.x`) e sincronização obrigatória (`cuda.synchronize()`).
* **Conexão com as Aulas 1-6:** Unimos o hardware da A1-3, a alimentação do host da A4-5 e o SO da A6 para rodar código customizado direto na VRAM da NVIDIA.
* **O problema que fica em aberto:** O kernel da A7 funciona, mas ele bate toda hora na VRAM Global lenta (A3), gerando desperdício massivo de ciclos de clock.

### Aula 8: Manipulação de Memória em CUDA (Tiling)
* **Conceito/Fundamento:** Regra 90/10 (90% do tempo de processamento em IA é gasto em acessos à memória). Tiling carrega pedaços da VRAM para a Memória Compartilhada rápida para reutilização. Coalescing une acessos consecutivos em transações eficientes.
* **O que se aprende:** A Regra 90/10, *Coalescing* (unir transações de memória global) e *Tiling* (carregar blocos de matrizes na SRAM compartilhada para reutilização local).
* **Conexão com a Aula 7:** Pegamos o kernel da A7 e o reescremos com estratégia de cache manual (SRAM da A3) para eliminar o gargalo de largura de banda da VRAM.
* **O problema que fica em aberto:** O código é hiper-otimizado para NVIDIA (CUDA/cuDNN), mas e quando o cliente corporativo exige execução em infraestrutura AMD/Intel?

### Aula 9: Alternativas ao CUDA: OpenCL
* **Conceito/Fundamento:** Viabilizar computação paralela multiplataforma e heterogênea para clientes com AMD/Intel. Suporte OpenCL 3.0 (NVIDIA/AMD/Intel), Apple deprecated 1.2 (Metal), Qualcomm 2.0 (Mobile). Arquitetura com *Platform*, *Device*, *Context*, *Command Queue*, kernel em C99 JIT e `cl.Buffer`. Mapeamento: thread/work-item, block/work-group, grid/NDRange, shared mem/`__local`, syncthreads/barrier, constant/`__constant`. Trade-offs: portabilidade multi-vendor sem vendor lock-in vs. código ~3x mais verboso, JIT cold start e ecossistema DL fraco.
* **O que se aprende:** Padrão aberto do Khronos Group, hierarquia *Platform -> Device -> Context -> Command Queue*, mapeamento de `work-items`/`work-groups` e trade-offs JIT vs AOT.
* **Conexão com a Trilha Inteira:** O aluno percebe que *não aprendeu CUDA isoladamente* — **aprendeu computação paralela de dados**. O OpenCL traduz a matriz mental de *tiling* (A8) e *kernels* (A7) para um padrão agnóstico multi-vendor.
* **O problema que fica em aberto:** O OpenCL exige reescrever o código em C99 manual. Como rodar modelos de IA prontos em PyTorch/TensorFlow no hardware AMD mantendo alta performance de nível industrial?

### Aula 10: Introdução ao ROCm e GPUs AMD
* **Conceito/Fundamento:** Configurar e executar aplicações de IA em GPUs AMD com o ecossistema ROCm, abstraindo a portabilidade via HIP e contêineres Docker para eliminar o *vendor lock-in*.
* **O que se aprende:** Arquitetura ROCm (`KFD`, `ROCr`, `HIP`), equivalência funcional (`rocBLAS`, `MIOpen`, `rocm-smi`), conversão via `hipify` e contêineres oficiais `rocm/pytorch`.
* **Conexão com a Aula 9:** O OpenCL mostrou a teoria multi-vendor; a Aula 10 entrega a solução industrial de alto nível, provando que pipelines PyTorch em CUDA rodam transparentemente via HIP em GPUs AMD (como a MI300X) sem alterar o código Python.

---

## Matriz de Domínio por Elo da Corrente

| Elo | Pergunta Crítica Respondida | Evidência Prática no Repositório |
| :--- | :--- | :--- |
| **A1-A3** | "Por que meu hardware afeta a conta de luz e nuvem?" | Diagnóstico de arquitetura e escolha de hardware edge/cloud. |
| **A4-A6** | "Como mexer no cluster remoto sem perder o job de 12h?" | Script de sync `rsync`, túnel SSH, monitoramento `nvtop`/`tmux`. |
| **A7-A8** | "Como espremer 100% da VRAM da placa de vídeo?" | Kernel otimizado com *tiling*, *coalescing* e medição de latência. |
| **A9** | "E se o cliente exigir rodar em cluster AMD/Intel?" | Tradução mental de paralelismo para padrão aberto agnóstico (OpenCL). |
| **A10** | "Como migrar um pipeline CUDA existente para GPUs AMD reduzindo custos?" | Benchmark PyTorch transparente via HIP e Docker ROCm (`rocm-smi`). |