import time
import torch
import torch_directml

print("=" * 60)
print("TESTE DE GPU - PyTorch + DirectML")
print("=" * 60)

device = torch_directml.device()

print("Dispositivo DirectML:")
print(device)

# Matriz grande
N = 4096

print()
print(f"Criando matrizes {N} x {N}...")

a = torch.randn(N, N, device=device)
b = torch.randn(N, N, device=device)

print("Matrizes criadas.")
print("Iniciando multiplicação de matrizes...")
print()

inicio = time.time()

for i in range(100):
    c = torch.matmul(a, b)

    if i % 10 == 0:
        print(f"Iteração {i}/100")

fim = time.time()

print()
print("=" * 60)
print("TESTE FINALIZADO")
print("=" * 60)
print(f"Tempo: {fim - inicio:.2f} segundos")
print()
print("Se o DirectML estiver utilizando a GPU,")
print("deverá haver atividade da GPU no Windows.")