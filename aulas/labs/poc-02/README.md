# Práticas de Programação CUDA para IA

Este diretório contém os exercícios práticos desenvolvidos para as aulas do curso técnico de Inteligência Artificial no SenacTech. O objetivo é demonstrar os fundamentos da arquitetura de GPUs, hierarquia de memória e o modelo de programação CUDA utilizando Python e a biblioteca Numba.

Como os laboratórios focam na lógica de paralelismo e nem sempre as máquinas possuem GPUs NVIDIA dedicadas disponíveis para todos, este laboratório utiliza o **Simulador CUDA** rodando em containers Docker no WSL.

## 🛠️ Pré-requisitos

Para executar estas práticas, os alunos precisam ter configurado em suas máquinas:
* **WSL 2** (Windows Subsystem for Linux) com uma distribuição Linux (ex: Ubuntu).
* **Docker** instalado e rodando no WSL.
* Conhecimentos básicos de terminal Linux (navegação de diretórios e execução de scripts).

## 🚀 Configuração do Ambiente (Container)

Para garantir que todos os alunos tenham as mesmas dependências (Python, NumPy e Numba) sem poluir o sistema operacional host, utilizaremos uma imagem Docker leve.

**1. Construir a imagem da aula:**
No terminal do WSL, navegue até a pasta contendo o `Dockerfile` e execute:
`docker build -t aula-cuda-simulador .`

**2. Iniciar o container com o Simulador CUDA:**
Para rodar os testes utilizando a CPU (simulando a arquitetura da GPU), inicie o container mapeando os arquivos locais e injetando a variável de ambiente `NUMBA_ENABLE_CUDASIM=1`:
`docker run -it --rm -e NUMBA_ENABLE_CUDASIM=1 -v $(pwd):/app aula-cuda-simulador bash`
*(Você estará dentro do terminal do container, no diretório `/app`, pronto para rodar os scripts).*

---

## 💻 Exercícios Práticos

### Exercício 1: Função de Ativação (ReLU) e Memória Global
**Arquivo:** `ex1_relu.py`

Este script demonstra o conceito fundamental de paralelismo em tarefas de Inteligência Artificial. Ele aplica uma função de ativação ReLU (Rectified Linear Unit) em um vetor de dados, simulando como uma rede neural processa informações em lote na **Memória Global** da GPU.
* **Conceitos abordados:** Arrays 1D, `cuda.grid(1)`, mapeamento de threads simples e prevenção de acesso fora dos limites (out-of-bounds).
* **Como executar:** `python3 ex1_relu.py`

### Exercício 2: Multiplicação de Matrizes e Memória Compartilhada
**Arquivo:** `ex2_matrizes.py`

A multiplicação de matrizes é a operação matemática central no treinamento de modelos de Deep Learning. Este exercício ilustra o conceito de **Hierarquia de Memória**, utilizando a **Memória Compartilhada** (Shared Memory) para otimizar o acesso aos dados dentro de um mesmo bloco de threads.
* **Conceitos abordados:** Grades e Blocos 2D (`cuda.grid(2)`), alocação de `cuda.shared.array`, transferência de dados da memória global para a compartilhada, e barreiras de sincronização (`cuda.syncthreads()`).
* **Como executar:** `python3 ex2_matrizes.py`

---
> **Aviso Importante sobre o Simulador:** O simulador executa as threads sequencialmente na CPU. Ele é excelente para validar a matemática e debugar o código, mas não reflete o ganho de velocidade de uma GPU real e não acusa erros de sincronização (condições de corrida).