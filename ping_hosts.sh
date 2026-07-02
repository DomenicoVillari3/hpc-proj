#!/bin/bash

# Definizione dei colori e stili per l'output
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Funzione per stampare gli header delle sezioni in modo pulito
print_header() {
    echo -e "\n${CYAN}${BOLD}====================================================${NC}"
    echo -e "${CYAN}${BOLD} $1 ${NC}"
    echo -e "${CYAN}${BOLD}====================================================${NC}"
}

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
    ["192.168.128.237"]="mpi13"
)

print_header "Controllo stato nodi MPI (Ping)"

# Iteriamo sugli IP nell'ordine corretto 
for ip in "192.168.128.210" "192.168.128.211" "192.168.128.212" "192.168.128.213" \
          "192.168.128.214" "192.168.128.215" "192.168.128.220" "192.168.128.221" \
          "192.168.128.222" "192.168.128.223" "192.168.128.224" "192.168.128.225" "192.168.128.237"; do
    
    nome=${hosts[$ip]}
    
    ping -c 1 -W 1 "$ip" > /dev/null 2>&1
    
    # L'uso di printf garantisce un allineamento perfetto anche tra nomi lunghi e corti
    if [ $? -eq 0 ]; then
        printf "[${GREEN}  UP  ${NC}] ${YELLOW}%-7s${NC} (%s)\n" "$nome" "$ip"
    else
        printf "[${RED} DOWN ${NC}] ${YELLOW}%-7s${NC} (%s)\n" "$nome" "$ip"
    fi
done

print_header "Controllo permessi (/home/mpiuser) diversi da 777"

for node in mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi12; do
    printf "${YELLOW}%-7s${NC} ➔  " "$node"
    ssh mpiuser@$node "stat -c '%a %n' /home/mpiuser" 2>&1
done

print_header "Controllo Hostname (Test SSH)"

for node in mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12 mpi13; do    
    printf "${YELLOW}%-7s${NC} ➔  " "$node"
    ssh -o ConnectTimeout=3 -o BatchMode=yes mpiuser@$node hostname 2>&1
done

print_header "Controllo Mount NFS "

for node in  mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12 mpi13; do
    printf "${YELLOW}%-7s${NC} ➔  " "$node"
    
    # Catturiamo l'output per evitare spazi vuoti disordinati se grep non trova nulla
    output=$(ssh mpiuser@$node "mount | grep test" 2>&1)
    if [ -z "$output" ]; then
        echo -e "${RED}[Nessun mount trovato]${NC}"
    else
        echo "$output"
    fi
done

print_header "Controllo Python (mpi4py)"

for node in mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12 mpi13; do
    printf "${YELLOW}%-7s${NC} ➔  " "$node"
    ssh mpiuser@$node "/home/mpiuser/test/.venv/bin/python3 -c 'from mpi4py import MPI; print(\"OK\")'" 2>&1
done

for node in mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12 mpi13; do
    printf "${YELLOW}%-7s${NC} ➔  " "$node"
  ssh mpiuser@$node "which perf"
done
echo -e "\n${CYAN}${BOLD}=== Esecuzione terminata ===${NC}\n"