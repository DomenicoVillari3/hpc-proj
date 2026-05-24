# HPC
## primo setup 
1. ```sudo apt-get update && sudo apt-get dist-upgrade -y && sudo snap refresh```

2. Configurazione /etc/hosts su tutti i nodi :
    ```
    192.168.128.210  mpi1
    192.168.128.211  mpi2
    192.168.128.212  mpi3
    192.168.128.213  mpi4
    192.168.128.214  mpi5
    192.168.128.215  mpi6
    192.168.128.220  mpi7
    192.168.128.221  mpi8
    192.168.128.222  mpi9
    192.168.128.223  mpi10
    192.168.128.224  mpi11
    192.168.128.225  mpi12

    ```

3. download opencl e driver Nvidia

    ```
    sudo apt-get install -y \ openmpi-bin \ libopenmpi-dev \ gcc \ build-essential

        mpicc --version       # deve mostrare gcc wrapper
        mpirun --version      # deve mostrare Open MPI x.y.z ``

        # Driver NVIDIA (se non già installati):
        sudo apt-get install -y nvidia-driver-535   # o la versione disponibile

        # Runtime OpenCL:
        sudo apt-get install -y \
            ocl-icd-opencl-dev \
            opencl-headers \
            clinfo


        clinfo | grep "Device Name"    # deve mostrare la GPU su ogni nodo
        nvidia-smi                     # deve mostrare la scheda
        ```




---



## Setup definitivo 
Ecco la Fase 1 aggiornata con tutto quello che è emerso durante il setup reale:

### FASE 1 — Setup Cluster
#### 1.1 — /etc/hosts (tutti gli 8 nodi)
```sudo nano /etc/hosts
Aggiungi su ogni nodo:
192.168.128.210  mpi1
192.168.128.211  mpi2
192.168.128.212  mpi3
192.168.128.213  mpi4
192.168.128.214  mpi5
192.168.128.215  mpi6
192.168.128.220  mpi7
192.168.128.221  mpi8
192.168.128.222  mpi9
192.168.128.223  mpi10
192.168.128.224  mpi11
192.168.128.225  mpi12
192.168.128.237  mpi13
```

Verifica da mpi1:

```for i in {2..13}; do ping -c1 mpi$i | grep "1 received" && echo "mpi$i è vivo"; done```

#### 1.2 — Utente mpiuser (tutti gli 8 nodi)
```
sudo adduser mpiuser
Stessa username e password su ogni nodo.
```

#### 1.3 — NOPASSWD sudo per amministratore (tutti gli 8 nodi)
Necessario per eseguire comandi remoti non interattivi via SSH:
sudo visudo
Aggiungi in fondo:
```
amministratore ALL=(ALL) NOPASSWD: ALL
```

#### 1.4 — SSH passwordless mpiuser (dal master)
```
su - mpiuser
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa

# Pre-accetta le chiavi host:
ssh-keyscan -H mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12>> ~/.ssh/known_hosts

# Copia chiave su tutti i nodi:
for node in mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12; do
    ssh-copy-id mpiuser@$node
done

oppure 

# Copia chiave su tutti i nodi:
for node in mpi13; do
    ssh-copy-id -i ~/.ssh/id_rsa.pub mpiuser@$node
done

eval $(ssh-agent)
ssh-add ~/.ssh/id_rsa
Se un nodo rifiuta la password, abilitare temporaneamente PasswordAuthentication yes in /etc/ssh/sshd_config e riavviare con sudo systemctl restart ssh.
Verifica:
for node in mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12; do
    echo -n "$node: "
    ssh -o BatchMode=yes -o ConnectTimeout=5 mpiuser@$node hostname 2>&1
done
```

Tutti devono rispondere senza password.

### 1.5 — NFS
#### Script master (~/setup_master.sh su mpi1, eseguito come amministratore):
```
#!/bin/bash

EXPORT_DIR="/home/mpiuser/test"

SUBNET="192.168.128.0/24"

echo "[INFO] Installing NFS server..."

sudo apt update -y

sudo apt install -y nfs-kernel-server

echo "[INFO] Creating export directory: $EXPORT_DIR"

sudo mkdir -p $EXPORT_DIR

sudo chown -R mpiuser:mpiuser $EXPORT_DIR

sudo chmod 755 $EXPORT_DIR

echo "[INFO] Configuring /etc/exports..."

sudo sed -i '\|'$EXPORT_DIR'|d' /etc/exports

echo "$EXPORT_DIR $SUBNET(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports

echo "[INFO] Applying exports..."

sudo exportfs -ra

echo "[INFO] Restarting NFS service..."

sudo systemctl restart nfs-kernel-server

echo "[SUCCESS] NFS Master ready!"

sudo exportfs -v

```

#### SETUP WORKER (setup_worker.sh )
```
#!/bin/bash

MASTER_IP="192.168.128.210"

EXPORT_DIR="/home/mpiuser/test"

MOUNT_POINT="/home/mpiuser/test"

echo "[INFO] Installing NFS client..."

sudo apt update -y

sudo apt install -y nfs-common

echo "[INFO] Creating mount point $MOUNT_POINT"

sudo mkdir -p $MOUNT_POINT

sudo chown -R mpiuser:mpiuser $MOUNT_POINT

echo "[INFO] Mounting NFS share..."

sudo mount $MASTER_IP:$EXPORT_DIR $MOUNT_POINT

if mountpoint -q $MOUNT_POINT; then

    echo "[SUCCESS] Mounted at $MOUNT_POINT"

else

    echo "[ERROR] Failed to mount $MOUNT_POINT"

    exit 1

fi
```

#### Verifica:
```
echo "NFS OK" | sudo tee /home/mpiuser/cloud/test.txt
for node in mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12; do
    echo -n "$node: "
    ssh amministratore@$node "cat /home/mpiuser/test/test.txt"
done
```




##### Output
```
for node in mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12; do
    echo -n "$node: "
    ssh mpiuser@$node "cat /home/mpiuser/test/test.txt"
done
mpiuser@mpi2's password: 
NFS OK
mpiuser@mpi3's password: 
NFS OK
mpiuser@mpi4's password: 
NFS OK
mpiuser@mpi5's password: 
NFS OK
mpiuser@mpi6's password: 
Permission denied, please try again.
mpiuser@mpi6's password: 
NFS OK
mpiuser@mpi7's password: 
NFS OK
mpiuser@mpi8's password: 
NFS OK
```


#### 1.6 — OpenMPI + Python (tutti gli 8 nodi)
```
sudo apt-get install -y openmpi-bin libopenmpi-dev gcc build-essential python3-pip
pip3 install mpi4py numpy pyopencl --break-system-packages
Verifica:
mpirun --version
python3 -c "from mpi4py import MPI; print('mpi4py OK')"
python3 -c "import pyopencl; print('pyopencl OK')"
```
##### Output 
```
for node in mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8; do
    echo -n "$node: "
    ssh mpiuser@$node "python3 -c 'from mpi4py import MPI; print(\"OK\")' 2>&1"
done
mpiuser@mpi1's password: 
Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

OK
mpi2: Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

OK
mpi3: Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

OK
mpi4: Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

OK
mpi5: Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

OK
mpi6: Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

OK
mpi7: Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

OK
mpi8: Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

Authorization required, but no authorization protocol specified

OK
```

## 1.7 — GPU NVIDIA + OpenCL (tutti gli 8 nodi)
```

sudo apt-get install -y ocl-icd-opencl-dev opencl-headers clinfo
Se nvidia-smi dà errore driver/library mismatch: sudo reboot.
```

Verifica:
```
for node in mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12; do
    echo -n "$node: "
    ssh amministratore@$node "nvidia-smi --query-gpu=name --format=csv,noheader 2>&1"
done
for node in mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 mpi7 mpi8 mpi9 mpi10 mpi11 mpi12; do
    echo -n "$node: "
    ssh amministratore@$node "nvidia-smi --query-gpu=name --format=csv,noheader 2>&1"
done

amministratore@mpi1's password: 
NVIDIA T1000
amministratore@mpi2's password: 
NVIDIA T1000
amministratore@mpi3's password: 
NVIDIA T1000
amministratore@mpi4's password: 
NVIDIA T1000
amministratore@mpi5's password: 
NVIDIA T1000
amministratore@mpi6's password: 
NVIDIA T1000
amministratore@mpi7's password: 
NVIDIA T1000
amministratore@mpi8's password: 
NVIDIA T1000
```


#### 1.8 — Hostfile MPI
/home/mpiuser/test/hostfile:
```
mpi1 slots=1
mpi2 slots=1
mpi3 slots=1
mpi4 slots=1
mpi5 slots=1
mpi6 slots=1
mpi7 slots=1
mpi8 slots=1
mpi9 slots=1
mpi10 slots=1
mpi11 slots=1
mpi12 slots=1
```


#### 1.9 — Test end-to-end mpi4py
 /home/mpiuser/test/hello_mpi.py
```
from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()
print(f"rank {rank}/{size} | host {host}")
mpirun --hostfile /home/mpiuser/test/hostfile -np 12 \
    python3 /home/mpiuser/test/hello_mpi.py
```
Output atteso: 8 righe, una per nodo.

Stato attuale:
* [x] /etc/hosts configurato
* [x] mpiuser creato
* [x] NFS attivo (cloud montato sui worker)
* [x] OpenMPI installato
* [x ] SSH passwordless mpiuser completo su tutti i nodi
* [ x] nvidia-smi funzionante su tutti i nodi
* [ x] mpi4py + pyopencl installati su tutti i nodi
* [ x] Test end-to-end mpi4py






