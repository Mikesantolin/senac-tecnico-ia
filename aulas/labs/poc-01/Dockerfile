FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install numba numpy
WORKDIR /app
COPY . /app
CMD ["bash"]