# Aula 8 - Teste de estresse para monitoramento via nvtop
import cupy as cp
import time

print("Iniciando estresse da GPU para monitoramento...")
TAMANHO = 8192
A = cp.random.randn(TAMANHO, TAMANHO, dtype=cp.float32)
B = cp.random.randn(TAMANHO, TAMANHO, dtype=cp.float32)

try:
    while True:
        _ = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
except KeyboardInterrupt:
    print("Estresse encerrado.")
