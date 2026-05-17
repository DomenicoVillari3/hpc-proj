#!/bin/bash

# Definizione dei colori per l'output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Definizione della lista di IP e Nodi
declare -A hosts=(
    ["192.168.128.210"]="mpi1"
    ["192.168.128.211"]="mpi2"
    ["192.168.128.212"]="mpi3"
    ["192.168.128.213"]="mpi4"
    ["192.168.128.214"]="mpi5"
    ["192.168.128.215"]="mpi6"
    ["192.168.128.220"]="mpi7"
    ["192.168.128.221"]="mpi8"
    ["192.168.128.222"]="mpi9"
    ["192.168.128.223"]="mpi10"
    ["192.168.128.224"]="mpi11"
    ["192.168.128.225"]="mpi12"
)

echo "=== Controllo stato nodi MPI ==="

# Iteriamo sugli IP nell'ordine corretto
for ip in "192.168.128.210" "192.168.128.211" "192.168.128.212" "192.168.128.213" \
          "192.168.128.214" "192.168.128.215" "192.168.128.220" "192.168.128.221" \
          "192.168.128.222" "192.168.128.223" "192.168.128.224" "192.168.128.225"; do
    
    nome=${hosts[$ip]}
    
    # -c 1: invia solo 1 pacchetto ping
    # -W 1: aspetta al massimo 1 secondo per la risposta
    # > /dev/null 2>&1: nasconde l'output standard del ping
    ping -c 1 -W 1 "$ip" > /dev/null 2>&1
    
    # Controlliamo l'exit code del ping ($? è 0 se ha successo)
    if [ $? -eq 0 ]; then
        echo -e "[${GREEN}  UP  ${NC}] $nome ($ip)"
    else
        echo -e "[${RED} DOWN ${NC}] $nome ($ip)"
    fi
done

echo "================================"