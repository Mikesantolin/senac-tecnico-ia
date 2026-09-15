# Aula 6 - Linux: Pseudo-Arquivos e Monitoramento de Sistema
# Demonstra como o Linux expõe informações de hardware via /proc e /sys.
import os
import platform
import subprocess
import shutil

print("=== Monitoramento de Sistema para I.A. ===\n")

sistema = platform.system()

if sistema == "Linux":
    # /proc/cpuinfo — informações do processador
    print("[/proc/cpuinfo]")
    with open('/proc/cpuinfo') as f:
        for line in f:
            if 'model name' in line:
                print(f"  CPU: {line.split(':')[1].strip()}")
                break

    # /proc/meminfo — memória do sistema
    print("\n[/proc/meminfo]")
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith(('MemTotal', 'MemAvailable')):
                chave, valor = line.split(':')
                mb = int(valor.strip().split()[0]) / 1024
                print(f"  {chave}: {mb:,.0f} MB")

    # /proc/uptime — tempo ligado
    print("\n[/proc/uptime]")
    with open('/proc/uptime') as f:
        segundos = float(f.read().split()[0])
        horas = segundos / 3600
        print(f"  Uptime: {horas:.1f} horas")

    # GPU NVIDIA via nvidia-smi
    print("\n[nvidia-smi]")
    if shutil.which('nvidia-smi'):
        resultado = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,temperature.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if resultado.returncode == 0:
            print(f"  GPU: {resultado.stdout.strip()}")
    else:
        print("  nvidia-smi não encontrado")

    print("\n→ Ferramentas como nvtop, htop e gpustat leem desses mesmos pseudo-arquivos")
    print("→ Use tmux/screen para manter sessões de monitoramento ativas remotamente")

else:
    print(f"Sistema detectado: {sistema}")
    print("Este script foi projetado para Linux (ambiente de servidores de IA).\n")
    print("Em Linux, o hardware é exposto como pseudo-arquivos (sem ocupar disco):")
    print("  /proc/cpuinfo  → modelo e frequência do processador")
    print("  /proc/meminfo  → uso de memória RAM em tempo real")
    print("  /proc/uptime   → tempo de atividade do servidor")
    print("  /sys/class/drm → dispositivos gráficos (GPUs)")
    print("  /dev/nvidia*   → dispositivos NVIDIA para CUDA")
    print("\n→ Essas interfaces permitem monitoramento sem instalar ferramentas extras")
    print("→ Scripts como gpu_status.sh automatizam essa leitura para operações de IA")
