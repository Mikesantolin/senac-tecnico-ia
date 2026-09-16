# Aula 11: Aplicação de Modelos de IA em GPUs NVIDIA e AMD

## 📌 Resumo Teórico e Objetivos

Nesta aula de síntese prática do Bloco 2, unificamos os conhecimentos adquiridos sobre ecossistemas GPU (CUDA, OpenCL e ROCm) aplicando-os em um experimento real de treinamento e benchmarking com **ResNet-18** e **ResNet-50**, registrando métricas objetivas com **Weights & Biases (W&B)**.

### Objetivos da Aula
1. **Comparar Ecossistemas:** Analisar o comportamento de modelos reais (PyTorch) em ambientes NVIDIA (CUDA) e AMD (ROCm/HIP).
2. **Coleta de Métricas Objetivas:** Monitorar Throughput (imagens/s), VRAM alocada (MB), tempo por época e perda (Loss) via Weights & Biases.
3. **Benchmarking de Operações:** Avaliar o tempo de execução e GFLOPS para operações essenciais de IA (MatMul FP32/FP16, Conv2D, CrossEntropy e FFT).
4. **Mixed Precision (AMP FP16/BF16):** Demonstrar otimizações com `torch.cuda.amp` sem alteração de código entre CUDA e ROCm.
5. **Relatório Estratégico:** Capacitar a elaboração de trade-offs técnicos e operacionais para decisões de infraestrutura (TCO, setup, ecossistema e riscos).

---

## 🛠️ Métricas-Chave para Avaliação Comparativa

| Métrica | Unidade | Descrição / Importância |
| :--- | :--- | :--- |
| **Throughput** | `imagens/segundo` | Quantidade de amostras processadas por segundo no loop de treino/inferência. |
| **VRAM Alocada** | `MB` ou `GB` | Memória de vídeo consumida durante os passos de *forward* e *backward*. |
| **Tempo por Época** | `segundos` | Duração total para iterar por todo o dataset/batches programados. |
| **Consumo Energético** | `Watts` (W) | Consumo médio via `nvidia-smi` ou `rocm-smi`. |
| **Tempo de Setup** | `Horas` (h) | Complexidade de configuração do ambiente, drivers e dependências. |
| **Custo Relativo** | `US$/hora` | Valor de locação de instâncias em nuvem (AWS, GCP, Azure, Lambda Labs). |

---

## 💻 1. Configuração de Ambiente Unificado

O PyTorch utiliza a camada **HIP** em plataformas AMD para mapear chamadas da API `torch.cuda` nativamente. O código abaixo detecta automaticamente a plataforma ativa:

```python
# Configurar ambiente unificado — detecta CUDA ou ROCm automaticamente
import torch
import torchvision
import platform
import subprocess
import time

def detectar_ambiente():
    """Detecta e imprime informações do ambiente de execução."""
    info = {
        "sistema":  platform.system() + " " + platform.release(),
        "python":   platform.python_version(),
        "pytorch":  torch.__version__,
        "cuda_ok":  torch.cuda.is_available(),
    }
    if info["cuda_ok"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu_nome"]  = props.name
        info["vram_gb"]   = props.total_memory // (1024 ** 3)
        info["sm_count"]  = props.multi_processor_count
        info["backend"]   = "ROCm/HIP " + (torch.version.hip or "?") \
                             if hasattr(torch.version, "hip") and torch.version.hip \
                             else "CUDA " + (torch.version.cuda or "?")

    for k, v in info.items():
        print(f"  {k:<14}: {v}")
    return info

print("=" * 48)
print("  Ambiente de Treinamento")
print("=" * 48)
env = detectar_ambiente()
device = "cuda" if env["cuda_ok"] else "cpu"
print(f"\n→ Dispositivo ativo: {device.upper()}")
```

---

## 🏎️ 2. Benchmark de Operações Fundamentais de IA

Para avaliar o desempenho raw do hardware antes de aplicar modelos complexos, utiliza-se a medição direta de operações fundamentais:

```python
# Benchmark de Operações Fundamentais de IA
import torch, time, math

device = "cuda" if torch.cuda.is_available() else "cpu"

def medir(fn, warmup=3, repeticoes=20):
    """Mede tempo médio de uma função GPU com warm-up."""
    for _ in range(warmup):
        fn()
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeticoes):
        fn()
    if device == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / repeticoes * 1000  # ms

N = 2048
A = torch.randn(N, N, device=device, dtype=torch.float32)
B = torch.randn(N, N, device=device, dtype=torch.float32)

print(f"Dispositivo: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
print(f"{'Operação':<30} {'Tempo (ms)':>12} {'GFLOPS':>10}")
print("-" * 55)

# 1. Multiplicação de matrizes (FP32)
t = medir(lambda: torch.matmul(A, B))
gflops = 2 * N**3 / (t/1000) / 1e9
print(f"{'MatMul FP32 (2048×2048)':<30} {t:>11.2f}ms {gflops:>9.1f}")

# 2. Multiplicação de matrizes (FP16 / mixed precision)
Ah = A.half(); Bh = B.half()
t = medir(lambda: torch.matmul(Ah, Bh))
gflops = 2 * N**3 / (t/1000) / 1e9
print(f"{'MatMul FP16 (2048×2048)':<30} {t:>11.2f}ms {gflops:>9.1f}")

# 3. Convolução 2D
conv  = torch.nn.Conv2d(256, 256, 3, padding=1).to(device)
x_img = torch.randn(16, 256, 56, 56, device=device)
t = medir(lambda: conv(x_img))
print(f"{'Conv2D 256ch 56×56 (bs=16)':<30} {t:>11.2f}ms {'N/A':>9}")

# 4. Softmax + CrossEntropy
logits = torch.randn(512, 1000, device=device)
labels = torch.randint(0, 1000, (512,), device=device)
ce = torch.nn.CrossEntropyLoss()
t = medir(lambda: ce(logits, labels))
print(f"{'CrossEntropy 512×1000':<30} {t:>11.2f}ms {'N/A':>9}")

# 5. FFT
sinal = torch.randn(32, 2**16, device=device)
t = medir(lambda: torch.fft.fft(sinal))
print(f"{'FFT batch=32, N=65536':<30} {t:>11.2f}ms {'N/A':>9}")
```

### Resultados Ilustrativos de Referência

| Operação | NVIDIA T4 (CUDA) | AMD MI300X (ROCm) | Speedup AMD |
| :--- | :---: | :---: | :---: |
| **MatMul FP32 (2048×2048)** | 18.4 ms | 4.1 ms | **4.5×** |
| **MatMul FP16 (2048×2048)** | 9.1 ms | 1.8 ms | **5.1×** |
| **Conv2D 256ch 56×56** | 3.2 ms | 1.1 ms | **2.9×** |
| **CrossEntropy 512×1000** | 0.4 ms | 0.2 ms | **2.0×** |
| **FFT batch=32, N=65536** | 1.8 ms | 0.6 ms | **3.0×** |

---

## 📈 3. Treinamento Otimizado com Mixed Precision (AMP) e W&B

O trecho a seguir integra o monitoramento ao vivo via Weights & Biases e a otimização de memória por `torch.cuda.amp`:

```python
import wandb
import torch, torchvision.models as models
import torch.nn as nn, time

# Inicialização do rastreamento
wandb.init(
    project = "aula11-cuda-vs-rocm",
    name    = f"resnet18-{'rocm' if hasattr(torch.version, 'hip') and torch.version.hip else 'cuda'}",
    config  = {
        "arquitetura" : "ResNet-18",
        "batch_size"  : 64,
        "lr"          : 1e-3,
        "epochs"      : 5,
        "dispositivo" : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "backend"     : "ROCm" if (hasattr(torch.version,"hip") and torch.version.hip) else "CUDA",
    }
)

device = "cuda" if torch.cuda.is_available() else "cpu"
modelo = models.resnet18(weights=None).to(device)
criterio   = nn.CrossEntropyLoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
scaler     = torch.cuda.amp.GradScaler()

for epoch in range(1, 6):
    modelo.train()
    perdas, throughputs = [], []
    
    for _ in range(30):
        imgs   = torch.randn(64, 3, 224, 224, device=device)
        labels = torch.randint(0, 10, (64,), device=device)

        t0 = time.time()
        otimizador.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            out  = modelo(imgs)
            loss = criterio(out, labels)

        scaler.scale(loss).backward()
        scaler.step(otimizador)
        scaler.update()

        if device == "cuda": torch.cuda.synchronize()
        t_batch = time.time() - t0
        
        perdas.append(loss.item())
        throughputs.append(64 / t_batch)

    vram = torch.cuda.memory_allocated() // (1024**2) if device == "cuda" else 0
    loss_med = sum(perdas) / len(perdas)
    tp_med   = sum(throughputs) / len(throughputs)

    wandb.log({
        "epoch"                 : epoch,
        "train/loss"            : loss_med,
        "perf/throughput_imgs_s": tp_med,
        "mem/vram_mb"           : vram
    })

wandb.finish()
```

---

## ⚖️ 4. Matriz Comparativa Estratégica: CUDA vs. ROCm

| Critério | NVIDIA CUDA | AMD ROCm |
| :--- | :--- | :--- |
| **Maturidade do Ecossistema** | ⭐⭐⭐⭐⭐ Muito alta (padrão da indústria) | ⭐⭐⭐⭐ Alta em crescimento acelerado |
| **Suporte PyTorch / JAX** | Nativo e otimizado via cuDNN/cuBLAS | Nativo via camada de compatibilidade HIP |
| **Suporte TensorFlow** | Nativo | Suporte experimental / via containers |
| **Licenciamento** | Proprietário / Fechado | Open-Source (código-fonte aberto) |
| **Custo de Hardware / TCO** | Custo mais elevado por TFLOPS | ~30% mais barato (melhor custo/benefício) |
| **Capacidade de VRAM** | Até 80 GB (H100) / 144 GB (H200) | Até 192 GB (Instinct MI300X) |
| **Complexidade de Setup** | Instalação simples de drivers no host | Recomendado uso de contêineres Docker |
| **Suporte da Comunidade** | Vasta documentação e fóruns ativos | Comunidade em expansão, mantida por AMD/Meta |

---

## 📝 Atividades Conceituais e Discussão

1. **Interpretando Desempenho Isolado:** O benchmark indicou que a GPU AMD MI300X teve um throughput significativamente maior que a NVIDIA Tesla T4. Essa comparação é totalmente justa do ponto de vista arquitetural ou existem diferenças de geração/categoria entre os chips? Quais outros fatores operacionais devem ser avaliados?
2. **Mitigação de Lock-in em Camadas Baixas:** Se o pipeline de I.A. da empresa utiliza chamadas diretas às extensões proprietárias de CUDA C (em vez das abstrações puras do PyTorch), qual é o impacto e o risco na migração para ROCm? De que forma a ferramenta `hipify` auxilia nesse processo?
3. **Análise de Custos Ocultos de Migração:** Uma equipe composta por engenheiros treinados exclusivamente em CUDA precisa migrar para ROCm. Quais custos indiretos (treinamento, refatoração de CI/CD, tempo de homologação) devem ser contrabalançados com o menor preço do hardware AMD?
4. **Impacto Prático de Precision Formats:** Na escolha entre FP32, FP16 e BF16 para modelos de visão computacional e LLMs, qual formato apresenta melhor equilíbrio entre consumo de VRAM e estabilidade numérica sem exigir escala de gradiente (`GradScaler`)?

---

## 📌 Tarefa de Casa (Opção Técnica)

Elabore um relatório técnico formatado para apresentação ao CTO contendo:
1. **Execução de Treinamento:** Treinar **ResNet-50** nos dois ambientes (CUDA e ROCm) com 5 épocas, `batch_size=64` e W&B ativo.
2. **Comparativo Tabular:** Registrar Loss por época, Throughput FP32 vs. Throughput FP16, VRAM máxima consumida.
3. **Estimativa Financeira:** Calcular o custo total para 100 épocas comparando instâncias AWS `g5.xlarge` (NVIDIA A10G) vs. opções baseadas em AMD.
4. **Recomendação Final:** Uma síntese executiva recomendando a plataforma ideal considerando os 3 anos de ciclo de vida do projeto.
