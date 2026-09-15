# Hands-on / Prática: Introdução ao ROCm e GPUs AMD via PyTorch & Docker
## Aula 10: Ecossistema ROCm, HIP e Portabilidade CUDA

> **Público-Alvo:** Estudantes e profissionais de diversas áreas (Engenharia, Gestão, Negócios, TI) buscando compreender a portabilidade de IA sem depender de um único fabricante de hardware (*Vendor Lock-in*).

---

### 1. Contextualização da Prática

Nesta atividade, você simulará a avaliação técnica da startup fictícia que adquiriu um servidor com GPUs **AMD Instinct MI300X** / **Radeon RX 7900 XTX**.

O objetivo é verificar se o código PyTorch escrito originalmente para ecossistemas NVIDIA (CUDA) roda de maneira transparente e sem modificações em GPUs AMD através da camada **HIP (ROCm)** e contêineres **Docker**.

---

### 2. Script de Diagnóstico e Benchmark Portável (`rocm_pytorch_benchmark.py`)

Crie o arquivo `rocm_pytorch_benchmark.py` com o conteúdo abaixo. Ele detecta automaticamente se o ambiente está executando sob **CUDA nativo (NVIDIA)** ou **ROCm / HIP (AMD)**, além de medir a taxa de processamento (throughput) do modelo **ResNet-18**.

```python
"""
Script de Benchmark e Diagnóstico de Portabilidade (CUDA <-> ROCm)
Aula 10 - Ecossistema ROCm e GPUs AMD
"""

import time
import platform
import torch
import torch.nn as nn
import torchvision

def verificar_ambiente():
    print("=" * 60)
    print("  DIAGNÓSTICO DO AMBIENTE DE DEEP LEARNING")
    print("=" * 60)
    print(f"Sistema Operacional : {platform.system()} {platform.release()}")
    print(f"Versão do Python    : {platform.python_version()}")
    print(f"Versão do PyTorch   : {torch.__version__}")

    gpu_disponivel = torch.cuda.is_available()
    print(f"GPU Disponível?     : {'SIM' if gpu_disponivel else 'NÃO (Executando em CPU)'}")

    if gpu_disponivel:
        gpu_nome = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Nome do Dispositivo : {gpu_nome}")
        print(f"VRAM Total          : {vram_gb:.2f} GB")

        # Verifica se o PyTorch está compilado com ROCm/HIP ou CUDA nativo
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"Backend Detectado   : ROCm / HIP (AMD) - Versão {torch.version.hip}")
        else:
            print(f"Backend Detectado   : CUDA Nativo (NVIDIA) - Versão {torch.version.cuda}")
    else:
        print("Aviso: Nenhuma aceleração por GPU foi detectada.")
    print("=" * 60 + "\n")

def executar_benchmark_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Benchmark de Treinamento ResNet-18 no dispositivo: {device.type.upper()}...")

    # Configurações do Lote Sintético (Imagens simuladas)
    N_BATCHES = 50
    BATCH_SIZE = 64
    INPUT_SHAPE = (BATCH_SIZE, 3, 224, 224)  # Formato padrão ImageNet

    # Carrega modelo ResNet-18 e transfere para a GPU/CPU
    modelo = torchvision.models.resnet18(weights=None).to(device)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.SGD(modelo.parameters(), lr=0.01, momentum=0.9)

    modelo.train()
    tempos_batch = []
    total_amostras = 0

    # Aquecimento (Warm-up) da GPU
    if device.type == "cuda":
        d_dummy = torch.randn(*INPUT_SHAPE, device=device)
        _ = modelo(d_dummy)
        torch.cuda.synchronize()

    for i in range(N_BATCHES):
        # Gera dados sintéticos em memória para focar no teste de processamento da GPU
        imgs = torch.randn(*INPUT_SHAPE, device=device)
        labels = torch.randint(0, 1000, (BATCH_SIZE,), device=device)

        t0 = time.time()
        otimizador.zero_grad()
        saida = modelo(imgs)
        perda = criterio(saida, labels)
        perda.backward()
        otimizador.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        
        t_batch = time.time() - t0
        tempos_batch.append(t_batch)
        total_amostras += BATCH_SIZE

        if (i + 1) % 10 == 0:
            throughput_parcial = BATCH_SIZE / t_batch
            print(f"Batch [{i+1:02d}/{N_BATCHES}] | Perda: {perda.item():.4f} | Throughput: {throughput_parcial:.1f} imgs/s")

    t_medio = sum(tempos_batch) / len(tempos_batch)
    throughput_medio = BATCH_SIZE / t_medio

    print("\n" + "-" * 50)
    print("  RESULTADO DO BENCHMARK")
    print("-" * 50)
    print(f"Tempo médio por batch  : {t_medio*1000:.2f} ms")
    print(f"Throughput médio       : {throughput_medio:.1f} imagens/segundo")
    print(f"Total de amostras      : {total_amostras:,}")
    print("-" * 50)

if __name__ == "__main__":
    verificar_ambiente()
    executar_benchmark_resnet()
```

---

### 3. Como Executar via Docker (Ambiente AMD ROCm)

Para rodar em um servidor com GPUs AMD sem necessidade de instalar drivers complexos no host:

1. **Baixar a Imagem Oficial PyTorch + ROCm:**
   ```bash
   docker pull rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
   ```

2. **Iniciar o Container com Acesso ao Hardware AMD:**
   ```bash
   docker run -it --rm \
     --device=/dev/kfd \
     --device=/dev/dri \
     --group-add=video \
     --group-add=render \
     --ipc=host \
     --shm-size 8G \
     -v $(pwd):/workspace \
     rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0 \
     python3 /workspace/rocm_pytorch_benchmark.py
   ```

---

### 4. Monitoramento da GPU durante a Execução

Em um segundo terminal no host Linux com GPU AMD, utilize os comandos do **ROCm-SMI** para auditar o comportamento da placa:

```bash
# Monitorar utilização, VRAM e temperatura continuamente
watch -n 1 rocm-smi --showuse --showmeminfo vram --showtemp --showpower
```
