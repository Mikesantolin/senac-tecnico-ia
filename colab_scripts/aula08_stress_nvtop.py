# Aula 8 - Teste de Estresse para monitoramento via NVTOP
import cupy as cp
import time

print("🔥 Iniciando o teste de estresse da GPU...")
TAMANHO = 16384 
A = cp.random.randn(TAMANHO, TAMANHO, dtype=cp.float32)
B = cp.random.randn(TAMANHO, TAMANHO, dtype=cp.float32)

ciclo = 0
try:
    while True:
        C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
        ciclo += 1
        if ciclo % 10 == 0:
            print(f"Lote {ciclo} concluído. GPU cravada em 100%...")
except KeyboardInterrupt:
    print("\n✅ Teste interrompido com segurança.")
    del A, B, C
    cp.get_default_memory_pool().free_all_blocks()
