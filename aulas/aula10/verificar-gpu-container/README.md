# Criando nosso primeiro container de teste de GPU

#### Crie uma pasta aonde será armazenado nosso repositório da aula

mkdir repos && cd repos

#### Clone o repositório da aula
git clone https://github.com/jonasmaffei/senac-tecnico-ia

###### Obs depois do primeiro clone, sempre que precisar atualizar o repositório na pasta rode: git pull

#### Navegue até a pasta aonde está a prática da aula de hoje
cd aulas/aula10/verificar-gpu-container

###### Construa o container
docker build . -t verificar-gpu-container
###### Executa
docker run --rm --device=/dev/dxg verificar-gpu-container