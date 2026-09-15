Construa
docker build -t aula-gpu .
Executa
docker run --rm \
  --device=/dev/dxg \
  aula-gpu