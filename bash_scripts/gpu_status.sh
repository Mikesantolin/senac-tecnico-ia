#!/bin/bash
# Script de monitoramento da Aula 6
echo -e "\n=== Monitor de Recursos de I.A. ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
else
    echo "Nenhuma GPU gerenciável encontrada no sistema."
fi
