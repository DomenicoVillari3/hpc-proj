# HPC Project — Distributed SGD for Logistic Regression
**Studenti:** Domenico Villari (574194), Francesco Maria Russo (574193)  
**Corso:** High Performance Computing — Prof. Salvatore Distefano  
**Università degli Studi di Messina**

---

## Architettura del Progetto

Il progetto implementa un sistema di **Stochastic Gradient Descent (SGD) distribuito** per la regressione logistica su cluster HPC reale, articolato in 4 livelli di parallelismo progressivi.

### Stack Tecnologico
- **Linguaggio:** Python 3
- **MPI:** `mpi4py`
- **Task Parallelism:** `concurrent.futures.ThreadPoolExecutor`
- **GPU:** `pyopencl`
- **Calcolo numerico:** `numpy`
- **Dataset:** EPSILON (D=2000, N=500k), SUSY (D=18, N=5M)

### Cluster
- 8 nodi (1 master + 7 worker), subnet `192.168.128.0/24`
- GPU: NVIDIA T1000 su ogni nodo
- Filesystem condiviso: NFS su `/home/mpiuser/test`

---

## Struttura Directory

```
/home/mpiuser/test/
├── data/
│   ├── SUSY.csv
│   ├── epsilon_normalized
│   ├── epsilon_normalized.t
│   ├── susy_X_train.bin
│   ├── susy_y_train.bin
│   ├── susy_X_test.bin
│   ├── susy_y_test.bin
│   ├── susy_meta.json
│   ├── epsilon_X_train.bin
│   ├── epsilon_y_train.bin
│   ├── epsilon_X_test.bin
│   ├── epsilon_y_test.bin
│   └── epsilon_meta.json
├── src/
│   ├── preprocess_susy.py
│   ├── preprocess_epsilon.py
│   ├── sgd_sequential.py
│   ├── sgd_master_worker.py
│   ├── sgd_data_parallel.py
│   ├── sgd_task_parallel.py      # TODO
│   └── sgd_opencl.py             # TODO
├── kernels/
│   └── gradient.cl               # TODO
├── results/
│   ├── sequential.csv
│   ├── master_worker.csv
│   └── data_parallel_np*.csv
├── hostfile
└── .env
```

---

## Configurazione — `.env`

```dotenv
DATASET_DIR=/home/mpiuser/test/data/
DATASET_NAME=epsilon_normalized
RESULTS_DIR=/home/mpiuser/test/results/
SEED=42
EPOCHS=50
LEARNING_RATE=0.1
BATCH_SIZE=256
```

---

## Preprocessing

### `preprocess_susy.py`
Legge `SUSY.csv` (formato CSV, label in colonna 0, 18 feature).  
- Carica con `pandas.read_csv`
- Split 80/20 train/test
- Salva `susy_X_train.bin`, `susy_y_train.bin`, `susy_X_test.bin`, `susy_y_test.bin` in formato `float32` binario
- Salva `susy_meta.json` con `N_train`, `N_test`, `D`

```bash
python3 src/preprocess_susy.py
```

### `preprocess_epsilon.py`
Legge `epsilon_normalized` e `epsilon_normalized.t` (formato LibSVM, label ±1, D=2000).  
- Carica con `sklearn.datasets.load_svmlight_file`
- Converte label da {-1,+1} a {0,1}
- Train e test già separati dal dataset originale
- Salva binari float32 e `epsilon_meta.json`

```bash
python3 src/preprocess_epsilon.py
```

---

## Level 0 — Sequential Baseline

**File:** `sgd_sequential.py`

### Descrizione
Implementazione single-process di mini-batch SGD per regressione logistica. Serve come baseline per il calcolo dello speedup nei livelli successivi.

### Algoritmo
1. Carica dataset binario con `numpy.fromfile`
2. Per ogni epoca:
   - Shuffle del dataset con seed riproducibile
   - Per ogni mini-batch: calcola gradiente, aggiorna `w`
   - Learning rate decrescente per epoca: `lr = lr0 / (1 + epoch * 0.1)`
3. Valuta loss e accuracy ogni 10 epoche

### Funzioni chiave
```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def compute_loss(X, y, w):
    p = sigmoid(X @ w)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def compute_accuracy(X, y, w):
    preds = (sigmoid(X @ w) >= 0.5).astype(np.float32)
    return np.mean(preds == y)
```

### Output
Salva `results/sequential.csv` con colonne: `epoch, loss, accuracy, lr, epoch_time_s`

### Risultati (EPSILON, lr=0.1, batch=256)
| Metrica | Valore |
|---------|--------|
| Tempo per epoca | ~4.3s |
| Speedup baseline | 1.0x |

### Esecuzione
```bash
python3 src/sgd_sequential.py
```

---

## Level 1 — Master-Worker

**File:** `sgd_master_worker.py`

### Descrizione
Parallelismo Master-Worker con comunicazione punto-punto esplicita (`comm.Send` / `comm.Recv`). Il rank 0 agisce da master, i rank 1…P-1 da worker.

### Architettura
```
Master (rank 0)                    Worker (rank w)
─────────────────                  ───────────────
Carica dataset                     Riceve D dal master
Broadcast D                  →     
Per ogni step:                     Loop:
  Invia w a ogni worker      →       Riceve w
  Invia mini-batch a worker  →       Riceve mini-batch
  Attende gradiente da worker ←      Calcola gradiente
  Aggrega gradienti                  Invia gradiente
  Aggiorna w
```

### Pattern di comunicazione
- `comm.bcast(D, root=0)` — broadcast dimensione feature
- `comm.Send(w, dest=wid, tag=TAG_W)` — invio pesi ai worker
- `comm.send(batch_size, dest=wid, tag=TAG_DATA)` — invio dimensione batch
- `comm.Send(X_batch, dest=wid, tag=TAG_DATA+1)` — invio dati
- `comm.Recv(grad, source=wid, tag=TAG_GRAD+1)` — ricezione gradiente

### Limitazioni
Il master serializza la raccolta dei gradienti (O(P) comunicazioni sequenziali per step), diventando collo di bottiglia con molti worker. Questo motiva il passaggio al Level 2.

### Output
Salva `results/master_worker.csv`

### Risultati (EPSILON, 8 rank)
| Metrica | Valore |
|---------|--------|
| Tempo per epoca | ~30s |
| Note | Più lento del sequenziale per overhead comunicazione |

### Esecuzione
```bash
mpirun --hostfile /home/mpiuser/test/hostfile -np 8 \
    /home/mpiuser/test/.venv/bin/python3 src/sgd_master_worker.py
```

---

## Level 2 — MPI Data Parallelism (Sync SGD)

**File:** `sgd_data_parallel.py`

### Descrizione
Parallelismo dei dati con `MPI_Allreduce` collettivo. Ogni rank carica e processa il proprio shard del dataset. Non esiste un master dedicato — tutti i rank sono simmetrici.

### Architettura
```
Rank 0          Rank 1          ...     Rank P-1
──────          ──────                  ────────
Shard 0         Shard 1                 Shard P-1
↓               ↓                       ↓
grad_0          grad_1                  grad_{P-1}
        ↘       ↓       ↙
            Allreduce (SUM)
            grad_avg = sum / P
        ↗       ↑       ↖
w -= lr * grad_avg  (su tutti i rank)
```

### Partizionamento
```python
shard_size = N_train // size
start_idx  = rank * shard_size
end_idx    = start_idx + shard_size if rank < size - 1 else N_train
```

### Comunicazione
```python
comm.Allreduce(grad_local, grad_buf, op=MPI.SUM)
w -= lr * (grad_buf / size)
```

Una sola operazione collettiva per mini-batch, O(log P) nella rete.

### Invarianti di correttezza
- `w` è read-only durante il calcolo del gradiente
- Tutti i rank inizializzano `w` con lo stesso seed
- Dopo Allreduce tutti i rank hanno lo stesso `w` aggiornato (zero staleness)
- Seed di shuffle diversi per rank e per epoca per evitare correlazioni

### Output
Salva `results/data_parallel_np{size}.csv`

### Risultati (EPSILON, lr=0.01, batch=256)
| Rank | Tempo/epoca | Speedup | Efficienza |
|------|-------------|---------|------------|
| 1 (seq) | 4.3s | 1.0x | 100% |
| 4 | 1.07s | 4.0x | 100% |
| 7 | 0.62s | 6.9x | 98.6% |

### Esecuzione
```bash
# Strong scaling test:
for np in 2 4 8; do
    mpirun --hostfile /home/mpiuser/test/hostfile -np $np \
        /home/mpiuser/test/.venv/bin/python3 src/data_parallelism.py
done
```

---

## Prossimi Step

### Level 3 — Task Parallelism (`sgd_task_parallel.py`)
- `concurrent.futures.ThreadPoolExecutor` con 3 thread per rank
- `MPI_Iallreduce` non-bloccante
- Overlap: Task B (loss su validation) e Task C (statistiche) eseguiti mentre Allreduce è in corso
- Beneficio atteso: riduzione tempo per epoca del 10-20%

### Level 4 — OpenCL GPU (`sgd_opencl.py`)
- Kernel `gradient.cl`: ogni work-item calcola contributo su una feature
- Test su EPSILON (D=2000): GPU efficiente
- Test su SUSY (D=18): GPU inefficiente (caso contrasto)
- Overlap `MPI_Iallreduce` con operazioni GPU residue

---

## Note sul Report

**Giustificazione sistema distribuito:**  
Su EPSILON (D=2000, N=400k) il baseline sequenziale impiega ~4.3s/epoca. Con 7 worker MPI il tempo scende a ~0.62s/epoca (speedup 6.9x, efficienza 98.6%). La scalabilità quasi-lineare dimostra che il carico computazionale (calcolo del gradiente) domina sull'overhead di comunicazione Allreduce.

**SUSY come caso contrasto:**  
Con D=18 il calcolo del gradiente è trascurabile rispetto alla latenza di rete. Il Master-Worker peggiora le prestazioni del sequenziale. La GPU Level 4 su SUSY mostrerà analogamente inefficienza per insufficienza di work-item (18 < dimensione warp = 32).