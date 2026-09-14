# Atividades Conceituais: Aula 9 - Alternativas ao CUDA: OpenCL

1. **O Preço da Flexibilidade (JIT)**: O OpenCL compila o código no primeiro uso (*Just-In-Time*). Em um serviço de áudio que precisa processar o primeiro lote de dados sem travar, qual é o impacto prático dessa compilação inicial?
2. **O Mito do "Roda em Tudo"**: A Apple depreciou o OpenCL em favor do Metal. Como essa movimentação de mercado desafia a premissa de que o OpenCL resolve 100% dos cenários multi-hardware?
3. **Visibilidade de Memória**: O OpenCL exige a criação explícita de buffers (`cl.Buffer`). Em termos de clareza de engenharia, por que forçar o desenvolvedor a declarar o tráfego de dados ajuda a mapear gargalos?
4. **O Dilema do Prazo (MVP)**: Se o cliente exige uma entrega de curto prazo (2 semanas) mas o hardware de destino é heterogêneo (AMD/Intel), o uso de OpenCL acelera ou retarda o *time-to-market* da startup? Justifique.