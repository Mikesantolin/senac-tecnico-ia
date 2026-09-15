FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        libblas3 \
        libomp5 \
        liblapack3 \
        && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch-directml

COPY teste_gpu.py /app/teste_gpu.py

WORKDIR /app

CMD ["python3", "teste_gpu.py"]