# Guia de Estudo e Resumo Consolidado: Aulas 1 a 8
## Fundamentos de Tecnologia e Infraestrutura para I.A.

---

### Bloco 1: Fundamentos

#### Aula 1: Quem Faz o Quê? (CPU vs. GPU)
* **Função Principal:** A CPU atua como o cérebro central (chef experiente para decisões complexas), enquanto a GPU funciona como a "fábrica" (galpão com milhares de operários executando cálculos matemáticos simultâneos).
* **Critério de Escolha:** Tarefas lógicas vão para a CPU; processamento matemático paralelo vai para a GPU.

#### Aula 2: Como Eles Trabalham?
* **SIMD (Sincronia Total):** Modelo padrão da GPU onde um comando único comanda vários elementos executando a mesma instrução em dados diferentes.
* **RISC vs. CISC:** RISC foca em instruções simples (alta eficiência energética); CISC lida com comandos complexos.

#### Aula 3: A Logística e a Memória
* **O Gargalo da Rodovia:** A lentidão na transferência de dados entre a RAM e a VRAM através do barramento PCIe.
* **Impacto na Performance:** A distância das camadas de memória (Registradores rápidos vs. VRAM lenta) determina se a GPU fica ociosa.

#### Aula 4: Processos vs. Threads
* **Processos vs. Threads:** Processos criam filiais isoladas (seguras e caras), threads compartilham o mesmo espaço (rápidas, baratas, mas vulneráveis).
* **O GIL do Python:** Mecanismo que impede o verdadeiro paralelismo de múltiplas threads na CPU.
* **Divergência na GPU:** Desvios condicionais ("se/senão") forçam partes do grupo a esperar, reduzindo o desempenho.

#### Aula 5: Redes e Transferência
* **TCP vs. UDP:** TCP prioriza a entrega segura (dados financeiros); UDP aposta na velocidade sem confirmação (streaming).
* **Ferramentas de Transferência:** O `rsync` otimiza grandes volumes permitindo retomada de quedas, superando o `scp`.

#### Aula 6: Linux e Gerenciamento de GPU
* **Sessões Persistentes:** Ferramentas como `screen` e `tmux` protegem treinamentos de interrupções por quedas de conexão SSH.
* **Automação:** O `cron` (agendamento) combinado com o `systemd` (serviços contínuos) garante a estabilidade da infraestrutura.

---

### Bloco 2: Programação e Otimização

#### Aula 7: Introdução ao Modelo CUDA
* **O Conceito de Kernel:** Funções executadas em paralelo por milhares de threads diretamente na GPU.
* **Hierarquia:** Grade (`Grid`), Blocos (`Blocks`) e Threads individuais.
* **Sincronização:** `cuda.synchronize()` assegura que cálculos na VRAM terminaram antes de voltar à CPU.

#### Aula 8: Manipulação de Memória em CUDA (Tiling)
* **A Regra 90/10:** 90% do tempo de execução de um algoritmo de I.A. é gasto em acessos à memória.
* **Memória Compartilhada (Shared Memory):** Cache manual ultra rápido compartilhado entre threads do mesmo bloco.
* **Tiling:** Técnica de carregar pedaços (`tiles`) da VRAM para a Memória Compartilhada para aliviar o tráfego de dados.
* **Coalescing:** Acessos consecutivos otimizam as transações da GPU. Acessos espalhados geram lentidão severa.
