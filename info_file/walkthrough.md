—

# TODO 
Ecco la roadmap completa aggiornata in Python:

ROADMAP COMPLETA — SGD Distribuito su Cluster HPC (Python)

## FASE 1 — Setup Cluster
* [x] /etc/hosts configurato su tutti gli 8 nodi
* [x] Utente mpiuser creato su tutti i nodi
* [x] NFS mount attivo su tutti gli 8 nodi (/home/mpiuser/test)
* [x] OpenMPI installato su tutti i nodi
* [ x] SSH passwordless mpiuser funzionante su tutti i nodi
* [ x] nvidia-smi funzionante su tutti gli 8 nodi
* [x ] clinfo mostra GPU su tutti i nodi
* [ x] Installazione dipendenze Python su tutti i nodi:
pip install mpi4py numpy pyopencl
* [ ] Hostfile definitivo in /home/mpiuser/test/hostfile:
mpi1 slots=1
mpi2 slots=1
mpi3 slots=1
mpi4 slots=1
mpi5 slots=1
mpi6 slots=1
mpi7 slots=1
mpi8 slots=1
* [ ] Test end-to-end mpi4py:
mpirun --hostfile hostfile -np 8 python3 hello_mpi.py

## FASE 2 — Struttura Directory Progetto
```
/home/mpiuser/test/
├── data/
│   ├── epsilon_train.bin
│   ├── epsilon_test.bin
│   ├── susy_train.bin
│   └── susy_test.bin
├── src/
│   ├── preprocess.py
│   ├── sgd_sequential.py
│   ├── sgd_master_worker.py
│   ├── sgd_mpi.py
│   ├── sgd_mpi_tasks.py
│   └── sgd_opencl.py
├── kernels/
│   └── gradient.cl
├── results/
│   └── *.csv
├── plots/
│   └── *.png
├── hostfile
└── run_all.sh
```
## FASE 3 — Preprocessing Dataset
* [ ] Download EPSILON da LIBSVM repository (~12GB, D=2000, N=500k) (SKIPPATO)
* [x ] Download SUSY (~2.4GB, D=18, N=5M)
* [ x] preprocess.py — parsing LibSVM → numpy array → flat binary float32
* [ ]x Salva X_train.bin, y_train.bin, X_test.bin, y_test.bin per ogni dataset
* [ x] Salva metadata: N, D, N_test in file .json
* [ x] Verifica lettura da ogni nodo via NFS (VA TUTTO OK)

## FASE 4 — Level 0: Sequential Baseline
File: sgd_sequential.py
* [ ] Lettura dataset binario con numpy.fromfile
* [ ] Mini-batch SGD con sigmoid e cross-entropy loss
* [ ] Learning rate decrescente: lr = lr0 / (1 + t)
* [ ] Calcolo accuracy su test set ogni 10 epoche
* [ ] Salva tempi e loss in results/sequential.csv
* [ ] Run su EPSILON e SUSY — verifica convergenza

## FASE 5 — Level 1: Master-Worker
File: sgd_master_worker.py — mpi4py
* [ ] Rank 0 (master): carica dataset, divide in chunk, distribuisce mini-batch ai worker con comm.send
* [ ] Rank 1…P (worker): riceve mini-batch, calcola gradiente locale, invia al master con comm.send
* [ ] Master aggrega gradienti, aggiorna w, broadcast nuovo w
* [ ] Nessun MPI_Allreduce — comunicazione punto-punto esplicita
* [ ] Test con 2, 4, 8 rank — misura tempi
* [ ] Confronto speedup vs Level 0

## FASE 6 — Level 2: MPI Data Parallelism (Sync SGD)
File: sgd_mpi.py — mpi4py
* [ ] Ogni rank carica la propria partizione del dataset (shard)
* [ ] Ogni rank calcola gradiente sul proprio shard con numpy
* [ ] comm.Allreduce per aggregare gradienti (divide per P dopo)
* [ ] Aggiornamento sincrono di w — zero staleness garantito
* [ ] Strong scaling: 1, 2, 4, 8 rank — salva tempi in results/mpi.csv
* [ ] Verifica che loss converge identicamente a Level 0

## FASE 7 — Level 3: Task Parallelism
File: sgd_mpi_tasks.py — mpi4py + concurrent.futures
* [ ] ThreadPoolExecutor con 3 worker thread per rank
* [ ] Task A (critico): calcolo gradiente su mini-batch — w read-only durante il task
* [ ] Task B (overlap): calcolo loss su validation set — eseguito in parallelo con Allreduce
* [ ] Task C (overlap): aggiornamento statistiche/logging — eseguito in parallelo con Allreduce
* [ ] comm.Iallreduce non-bloccante — overlap con Task B e C
* [ ] Sequenza per step:
    1. Submit Task A → attendi risultato (gradiente)
    2. Lancia Iallreduce (non-bloccante)
    3. Submit Task B e Task C in parallelo
    4. request.Wait() — attendi Allreduce
    5. Aggiorna w
* [ ] Grafo dipendenze task da disegnare (TikZ/draw.io) per il report
* [ ] Misura beneficio overlap: tempo Task B+C vs tempo Allreduce
* [ ] Salva tempi in results/tasks.csv

## FASE 8 — Level 4: OpenCL GPU
File: sgd_opencl.py — mpi4py + pyopencl
Kernel: kernels/gradient.cl
* [ ] Kernel OpenCL: ogni work-item calcola contributo su una feature d
* [ ] Bounds guard: if (gid >= D) return;
* [ ] Setup pyopencl: platform, device, context, queue
* [ ] Trasferimento X_shard, y_shard, w → GPU buffer
* [ ] Enqueue kernel, leggi gradiente risultante
* [ ] comm.Iallreduce non-bloccante overlap con operazioni GPU residue
* [ ] Test su EPSILON (D=2000) — GPU efficiente, molti work-item
* [ ] Test su SUSY (D=18) — GPU inefficiente, pochi work-item (caso contrasto)
* [ ] Salva tempi in results/opencl.csv

## FASE 9 — Performance Evaluation
* [ ] Strong scaling MPI: 1→8 rank, EPSILON e SUSY — speedup e efficiency
* [ ] Task parallelism benefit: Level 2 vs Level 3 — overlap misurabile in secondi
* [ ] GPU vs CPU: Level 3 vs Level 4 su D=18 e D=2000
* [ ] Temporal breakdown per step: Task A / Allreduce / Task B+C / update w
* [ ] Script plots/generate_plots.py con matplotlib:
    * Speedup curve (teorico Amdahl vs misurato)
    * Efficiency curve
    * Temporal breakdown stacked bar
    * GPU vs CPU bar chart per D=18 e D=2000
* [ ] Tabelle CSV riassuntive

FASE 10 — Report LaTeX e Presentazione
* [ ] Sezioni: Introduzione, Design PCAM, Implementazione (4 livelli), Risultati, Conclusioni
* [ ] Diagramma PCAM (draw.io o TikZ)
* [ ] Grafo dipendenze task Level 3 (TikZ)
* [ ] Analisi Amdahl — stima speedup teorico vs misurato
* [ ] Analisi GPU efficiency: perché SUSY è inefficiente (D=18 < warp size)
* [ ] Slide presentazione (max 15 slide)

Prossimo step immediato: completare Fase 1 (SSH passwordless + nvidia-smi), poi inizio Fase 3 con preprocess.py. Vuoi che generi il codice per la Fase 3?
