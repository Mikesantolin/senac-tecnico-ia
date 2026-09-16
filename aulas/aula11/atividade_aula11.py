# atividade_aula11.py
"""
Aula 11: Treinamento e Benchmark Comparativo CUDA vs ROCm
Aplicação de ResNet-18/50, Mixed Precision (AMP) e Coleta de Métricas
"""

import torch
import torch.nn as nn
import platform
import time
import json

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

def criar_modelo(n_classes=10):
    """Cria um modelo ResNet-18 ou fallback equivalente usando PyTorch puro."""
    if HAS_TORCHVISION:
        modelo = models.resnet18(weights=None)
        modelo.fc = nn.Linear(512, n_classes)
        return modelo
    else:
        # Fallback simples estilo CNN para ambientes sem torchvision instalado
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )

def detectar_ambiente():
    """Detecta a GPU e a pilha de execução (CUDA ou ROCm/HIP)."""
    info = {
        "sistema": platform.system() + " " + platform.release(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_ok": torch.cuda.is_available(),
    }
    if info["cuda_ok"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu_nome"] = props.name
        info["vram_gb"] = props.total_memory // (1024 ** 3)
        info["sm_count"] = props.multi_processor_count
        info["backend"] = (
            "ROCm/HIP " + (torch.version.hip or "?")
            if hasattr(torch.version, "hip") and torch.version.hip
            else "CUDA " + (torch.version.cuda or "?")
        )

    print("=" * 50)
    print("  Informações do Ambiente de Computação")
    print("=" * 50)
    for k, v in info.items():
        print(f"  {k:<14}: {v}")
    print("=" * 50)
    return info

def benchmark_operacoes(device):
    """Executa benchmarks de operações fundamentais de IA."""
    print("\n--- 1. Benchmark de Operações Fundamentais ---")
    def medir(fn, warmup=3, repeticoes=20):
        for _ in range(warmup):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeticoes):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / repeticoes * 1000

    N = 2048
    A = torch.randn(N, N, device=device, dtype=torch.float32)
    B = torch.randn(N, N, device=device, dtype=torch.float32)

    t_fp32 = medir(lambda: torch.matmul(A, B))
    gflops_fp32 = 2 * N**3 / (t_fp32 / 1000) / 1e9

    Ah, Bh = A.half(), B.half()
    t_fp16 = medir(lambda: torch.matmul(Ah, Bh))
    gflops_fp16 = 2 * N**3 / (t_fp16 / 1000) / 1e9

    conv = nn.Conv2d(256, 256, 3, padding=1).to(device)
    x_img = torch.randn(16, 256, 56, 56, device=device)
    t_conv = medir(lambda: conv(x_img))

    print(f"MatMul FP32 (2048x2048): {t_fp32:.2f} ms ({gflops_fp32:.1f} GFLOPS)")
    print(f"MatMul FP16 (2048x2048): {t_fp16:.2f} ms ({gflops_fp16:.1f} GFLOPS)")
    print(f"Conv2D 256ch 56x56     : {t_conv:.2f} ms")

def treinar_resnet(device, epochs=3, batch_size=64, use_amp=False):
    """Treina modelo e coleta métricas de loss, throughput e VRAM."""
    tipo_precision = "AMP FP16" if use_amp else "FP32 (Padrão)"
    print(f"\n--- 2. Treinamento ({tipo_precision}) ---")
    
    modelo = criar_modelo(n_classes=10).to(device)

    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.SGD(modelo.parameters(), lr=1e-3, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    historico = {"loss": [], "throughput": [], "vram_mb": []}

    for epoch in range(1, epochs + 1):
        modelo.train()
        perdas, throughputs = [], []
        
        for _ in range(20):  # 20 batches por época
            imgs = torch.randn(batch_size, 3, 224, 224, device=device)
            labels = torch.randint(0, 10, (batch_size,), device=device)

            t0 = time.time()
            otimizador.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = modelo(imgs)
                    loss = criterio(out, labels)
                scaler.scale(loss).backward()
                scaler.step(otimizador)
                scaler.update()
            else:
                out = modelo(imgs)
                loss = criterio(out, labels)
                loss.backward()
                otimizador.step()

            if device == "cuda":
                torch.cuda.synchronize()
            t_batch = time.time() - t0

            perdas.append(loss.item())
            throughputs.append(batch_size / t_batch)

        loss_med = sum(perdas) / len(perdas)
        tp_med = sum(throughputs) / len(throughputs)
        vram = torch.cuda.memory_allocated() // (1024**2) if device == "cuda" else 0

        historico["loss"].append(round(loss_med, 4))
        historico["throughput"].append(round(tp_med, 1))
        historico["vram_mb"].append(vram)

        print(
            f"Época {epoch}/{epochs} | Loss: {loss_med:.4f} | "
            f"Throughput: {tp_med:.1f} imgs/s | VRAM: {vram} MB"
        )

    return historico

if __name__ == "__main__":
    env = detectar_ambiente()
    device = "cuda" if env["cuda_ok"] else "cpu"
    print(f"\nDispositivo selecionado: {device.upper()}")

    benchmark_operacoes(device)

    # Comparativo FP32 vs AMP FP16
    hist_fp32 = treinar_resnet(device, epochs=3, use_amp=False)
    hist_amp = treinar_resnet(device, epochs=3, use_amp=True)

    print("\n--- Resumo de Desempenho ---")
    print(f"FP32 Throughput Médio : {sum(hist_fp32['throughput'])/len(hist_fp32['throughput']):.1f} imgs/s")
    print(f"AMP  Throughput Médio : {sum(hist_amp['throughput'])/len(hist_amp['throughput']):.1f} imgs/s")
