# 🧠 Tecnologia e Infraestrutura para Inteligência Artificial

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Repositório estruturado por aulas para o curso técnico de IA do Senac, cobrindo desde arquitetura de computadores e hierarquia de memória até programação CUDA e deploy de LLMs locais.

Cada aula resolve o gargalo que a anterior deixou em aberto, formando uma cadeia causal completa:

```
Silício → Modelos de Execução → Memória → Processos → Redes → Linux → CUDA → Tiling → OpenCL/LLMs
```

---

## 📋 Pré-requisitos

| Requisito | Detalhes |
| :--- | :--- |
| **Python** | 3.10 ou superior |
| **NumPy** | Obrigatório (todas as aulas) |
| **PyTorch** | Aula 3 (benchmark RAM vs VRAM) |
| **CuPy** | Aulas 7 e 8 (FFT e estresse de GPU) |
| **Numba** | Aula 8 (kernels CUDA com Tiling) |
| **GPU NVIDIA** | Recomendada para Aulas 3, 7 e 8 (scripts têm fallback para CPU) |
| **Docker** | Aula 9 (Open WebUI — opcional) |

## 🚀 Instalação

```bash
# Clone o repositório
git clone https://github.com/jonasmaffei/senac-tecnico-ia.git
cd senac-tecnico-ia

# Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Instale as dependências
pip install -r requirements.txt
```

> **Nota sobre GPU:** O PyTorch com CUDA deve ser instalado separadamente via [pytorch.org/get-started](https://pytorch.org/get-started). O CuPy também requer versão compatível com seu CUDA (`pip install cupy-cuda12x`).

---

## 📚 Índice das Aulas

### Bloco 1 — Fundamentos de Hardware e Infraestrutura

| Aula | Tema | Scripts |
| :---: | :--- | :--- |
| **01** | Arquitetura de Computadores (Von Neumann, CPU vs GPU) | [`arquitetura_cpu_gpu.py`](aulas/aula01/arquitetura_cpu_gpu.py) |
| **02** | Modelos de Processamento (SIMD, RISC vs CISC) | [`simd_numpy.py`](aulas/aula02/simd_numpy.py) |
| **03** | Hierarquia de Memória (RAM vs VRAM, PCIe) | [`benchmark_ram_vram.py`](aulas/aula03/benchmark_ram_vram.py) |
| **04** | Processos e Threads (GIL, Multiprocessing) | [`processos_threads.py`](aulas/aula04/processos_threads.py) |
| **05** | Redes e Transferência de Dados (TCP/UDP, SSH, rsync) | [`demo_tcp_udp.py`](aulas/aula05/demo_tcp_udp.py), [`comandos_transferencia.sh`](aulas/aula05/comandos_transferencia.sh) |
| **06** | Linux e GPUs (/proc, /sys, tmux, cron) | [`monitoramento_linux.py`](aulas/aula06/monitoramento_linux.py), [`gpu_status.sh`](aulas/aula06/gpu_status.sh) |

### Bloco 2 — Programação e Otimização em GPU

| Aula | Tema | Scripts |
| :---: | :--- | :--- |
| **07** | Introdução ao CUDA (Kernels, CuPy FFT) | [`fft_benchmark.py`](aulas/aula07/fft_benchmark.py) |
| **08** | Tiling e Otimização de Memória (Shared Memory) | [`tiling_benchmark.py`](aulas/aula08/tiling_benchmark.py), [`stress_nvtop.py`](aulas/aula08/stress_nvtop.py) |
| **09** | Alternativas ao CUDA (OpenCL) + LLMs Locais | [Tutorial Ollama](aulas/aula09/hands-on-ollama.md), [Tutorial Open WebUI](aulas/aula09/hands-on-frontend-ollama.md) |

---

## 📖 Documentação

| Documento | Descrição |
| :--- | :--- |
| [`resumos.md`](docs/resumos.md) | Resumos teóricos consolidados de todas as 9 aulas |
| [`timeline-engenharia.md`](docs/timeline-engenharia.md) | Blueprint causal de engenharia: por que cada aula existe |
| [`questionario.md`](docs/questionario.md) | 21 questões dissertativas (avaliação) |
| [`questionario-gabarito.md`](docs/questionario-gabarito.md) | Gabarito sintetizado |
| [`materiais-complementares.md`](docs/materiais-complementares.md) | Curadoria de links, playlists e cursos externos |

---

## 🗂️ Estrutura do Repositório

```
senac-tecnico-ia/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── aulas/
│   ├── aula01/   → Arquitetura CPU vs GPU
│   ├── aula02/   → SIMD e vetorização (NumPy)
│   ├── aula03/   → Benchmark RAM vs VRAM (PyTorch)
│   ├── aula04/   → Multiprocessing vs Threading
│   ├── aula05/   → TCP/UDP + comandos de rede (scp, rsync)
│   ├── aula06/   → Monitoramento Linux + GPU status
│   ├── aula07/   → FFT CPU vs GPU (CuPy)
│   ├── aula08/   → Tiling CUDA + estresse de GPU
│   └── aula09/   → OpenCL + Tutoriais Ollama/Docker
└── docs/         → Resumos, questionários e materiais complementares
```

---

## 📄 Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).
