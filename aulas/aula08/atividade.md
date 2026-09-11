# Atividade de Pesquisa: Otimização de Memória e Custos em I.A.
## Aula 8: Manipulação de Memória em CUDA e Tiling

> **Objetivo:** Investigar como o gerenciamento de largura de banda e memória em hardware de alto desempenho afeta a eficiência energética, o tempo de processamento e a fatura de infraestrutura de grandes empresas de Inteligência Artificial.

---

### Parte 1: Investigação Técnica (Hardware e Arquitetura)

1. **O Impacto do Coalescing:**
   * Pesquise o que acontece a nível de hardware em uma GPU NVIDIA quando threads de um mesmo warp fazem acessos **coalescidos** à memória global versus acessos **desalinhados/espalhados**.
   * *Pergunta para entrega:* Quantas transações de barramento a memória precisa realizar no pior cenário e qual é a perda estimada de desempenho?

2. **Memória Compartilhada (SRAM) vs. Cache L1:**
   * A Memória Compartilhada do CUDA é controlada explicitamente pelo programador (cache manual), enquanto o Cache L1 é gerenciado de forma automática pelo hardware.
   * *Pergunta para entrega:* Em termos de engenharia de software, quais são as vantagens e desvantagens de ter que gerenciar manualmente o cache (Tiling) em vez de confiar 100% no cache automático do processador?

---

### Parte 2: Visão de Negócios e Infraestrutura em Nuvem

3. **O Custo do Minuto de GPU na Nuvem:**
   * Pesquise o custo médio de locação de instâncias de nuvem equipadas com GPUs de alta performance (como a NVIDIA A100 ou H100).
   * *Pergunta para entrega:* Se um algoritmo de treinamento de redes neurais otimizado com Tiling reduz o tempo de processamento de uma época de 10 horas para 2 horas, qual é a economia financeira estimada para uma empresa que roda esse treinamento diariamente?

4. **Eficiência Energética (Green AI):**
   * O consumo de energia de data centers de IA é um dos maiores desafios globais da atualidade.
   * *Pergunta para entrega:* Explique como a redução de acessos desnecessários à Memória Global (VRAM lenta) impacta diretamente o consumo de Watts (energia) de uma placa de vídeo durante uma carga de trabalho pesada.

---
> **Dica:** Lembre-se de que o melhor engenheiro de I.A. não é apenas quem domina a matemática dos modelos, mas quem compreende a física do hardware e sabe otimizar o uso da memória para garantir escala e viabilidade financeira.