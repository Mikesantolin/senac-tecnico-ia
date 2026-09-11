# Aula 4 - Threading vs Multiprocessing em Python (CPU-bound)
import multiprocessing
import threading
import time
import math

def calcular_primos(limite):
    primos = []
    for n in range(2, limite):
        if all(n % i != 0 for i in range(2, int(math.sqrt(n)) + 1)):
            primos.append(n)
    return len(primos)

LIMITE = 50_000
N_TAREFAS = 4

inicio = time.time()
with multiprocessing.Pool(processes=N_TAREFAS) as pool:
    pool.starmap(calcular_primos, [(LIMITE,)] * N_TAREFAS)
print(f"Multiprocessing: {time.time() - inicio:.2f}s (Paralelismo real)")
