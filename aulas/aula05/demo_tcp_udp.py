# Aula 5 - TCP vs UDP: Confiabilidade vs Velocidade
# Demonstra a diferença fundamental entre os protocolos de transporte.
import socket
import threading
import time

PORTA_TCP = 65432
PORTA_UDP = 65433
MENSAGENS = 200

# --- TCP: Conexão confiável com confirmação de entrega ---
def servidor_tcp():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', PORTA_TCP))
        s.listen()
        conn, _ = s.accept()
        with conn:
            recebidas = 0
            while True:
                dados = conn.recv(1024)
                if not dados:
                    break
                recebidas += 1
            conn.sendall(str(recebidas).encode())

def cliente_tcp():
    time.sleep(0.1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', PORTA_TCP))
        inicio = time.time()
        for i in range(MENSAGENS):
            s.sendall(f"msg-{i}".encode())
        s.shutdown(socket.SHUT_WR)
        recebidas = int(s.recv(1024).decode())
        tempo = time.time() - inicio
    print(f"  TCP: {recebidas}/{MENSAGENS} entregues em {tempo*1000:.1f}ms (garantia de entrega)")

# --- UDP: Disparo rápido sem confirmação ---
def servidor_udp(resultado):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('127.0.0.1', PORTA_UDP))
        s.settimeout(1.0)
        recebidas = 0
        try:
            while True:
                s.recvfrom(1024)
                recebidas += 1
        except socket.timeout:
            pass
    resultado.append(recebidas)

def cliente_udp():
    time.sleep(0.1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        inicio = time.time()
        for i in range(MENSAGENS):
            s.sendto(f"msg-{i}".encode(), ('127.0.0.1', PORTA_UDP))
        tempo = time.time() - inicio
    print(f"  UDP: {MENSAGENS} disparadas em {tempo*1000:.1f}ms")

print("=== Comparação TCP vs UDP ===\n")

print("[TCP - Transmission Control Protocol]")
t1 = threading.Thread(target=servidor_tcp)
t2 = threading.Thread(target=cliente_tcp)
t1.start(); t2.start()
t1.join(); t2.join()

print("\n[UDP - User Datagram Protocol]")
resultado_udp = []
t3 = threading.Thread(target=servidor_udp, args=(resultado_udp,))
t4 = threading.Thread(target=cliente_udp)
t3.start(); t4.start()
t3.join(); t4.join()
print(f"  UDP: {resultado_udp[0]}/{MENSAGENS} recebidas pelo servidor")

print("\n→ TCP: ideal para transferir modelos e datasets (integridade garantida)")
print("→ UDP: ideal para telemetria e métricas em tempo real (baixa latência)")
