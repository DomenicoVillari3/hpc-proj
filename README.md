# HPC Project — SGD distribuito per regressione logistica

Progetto di **High Performance Computing** per confrontare l'addestramento di un modello di regressione logistica binaria con mini-batch Stochastic Gradient Descent (SGD) su CPU, cluster MPI e GPU OpenCL.

Il repository implementa una baseline sequenziale, comunicazione master-worker, parallelismo dei dati, sovrapposizione tra comunicazione e task CPU e calcolo del gradiente su GPU. Include inoltre una coda di job MPI per eseguire configurazioni di iperparametri in gruppi di processi.

**Autori:** Domenico Villari e Francesco Maria Russo  
**Corso:** High Performance Computing — Prof. Salvatore Distefano  
**Università:** Università degli Studi di Messina

## Implementazioni

| Modalità | Sorgente | Funzionamento |
| --- | --- | --- |
| L0 — Sequenziale | `src/sgd_sequential.py` | Calcolo del gradiente e aggiornamento dei pesi in un solo processo con NumPy. |
| L1 — Master-worker | `src/master-worker.py` | Il rank 0 distribuisce pesi e mini-batch, riceve i gradienti e aggiorna il modello tramite comunicazioni punto-punto. |
| L2 — Data parallelism | `src/data_parallelism.py` | Ogni rank elabora una partizione del training set; `MPI.Allreduce` somma i gradienti e tutti i rank aggiornano i pesi. |
| L3 — Task parallelism | `src/task_parallelism.py` | `ThreadPoolExecutor` e `MPI.Iallreduce` sovrappongono la comunicazione con valutazione locale e preparazione del batch successivo. |
| L4 — OpenCL | `src/sgd_opencl.py` | Il gradiente viene calcolato su GPU; valutazione e preparazione dei batch restano su CPU durante `MPI.Iallreduce`. |
| Ricerca di iperparametri | `src/job_queue.py` | Un master assegna configurazioni CPU/GPU a gruppi di tre rank, ciascuno con il proprio comunicatore MPI. |

Il learning rate diminuisce secondo `lr(epoch) = lr0 / (1 + 0.1 * epoch)`, con epoche indicizzate da zero. Il modello usa la sigmoide, la cross-entropy binaria e una soglia di classificazione di 0.5. I pesi sono aggiornati in `float64`; dati e operazioni dei kernel GPU usano `float32`.

## Struttura del repository

| Percorso | Contenuto |
| --- | --- |
| `src/` | Preprocessing, implementazioni SGD e funzioni comuni in `utils.py`. |
| `kernels/gradient.cl` | Kernel OpenCL `forward_kernel` e `gradient_kernel`. |
| `results/` | CSV degli esperimenti, ricerca di iperparametri e log del monitoraggio GPU. |
| `log_files/` | Log di esecuzioni OpenCL già presenti. |
| `info_file/` | Note di setup, sviluppo e roadmap del progetto. |
| `requirements.txt` | Dipendenze Python con versioni fissate. |
| `hostfile` | Elenco dei nodi MPI: `mpi1`–`mpi13`, con uno slot per nodo. |
| `hello_mpi.py` | Verifica essenziale dell'avvio dei processi MPI. |
| `run_experiments.sh` | Esecuzione delle configurazioni attive per data, task e GPU parallelism. |
| `run_opencl_experiments_with_monitor.sh` | Esperimenti OpenCL con monitoraggio remoto tramite `nvidia-smi`. |
| `show_log_gpu_usage.sh` | Riepilogo dell'utilizzo GPU dai log. |
| `ping_hosts.sh` | Diagnostica della raggiungibilità dei nodi. |

La directory `data/` e il file `.env` devono essere creati localmente. I dataset non sono inclusi nel repository. Le note in `info_file/` descrivono anche fasi precedenti: per nomi dei file e comportamento attuale, fare riferimento ai sorgenti e alle istruzioni seguenti.

## Requisiti e installazione

- Python 3 compatibile con le versioni in `requirements.txt`.
- Open MPI e relative librerie di sviluppo per le modalità distribuite.
- Runtime OpenCL/loader di sistema; PyOpenCL è importato da `utils.py` anche nelle modalità CPU.
- Una GPU con runtime OpenCL per L4 e per i job GPU. Il codice preferisce la piattaforma NVIDIA e seleziona la prima GPU disponibile sulla piattaforma scelta.
- Per un cluster: nomi dei nodi risolvibili, accesso SSH senza richiesta interattiva di password e progetto, dati e interprete disponibili allo stesso percorso sui nodi, ad esempio tramite NFS.

Su Ubuntu/Debian, un esempio di installazione delle dipendenze di sistema è:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-venv build-essential \
    openmpi-bin libopenmpi-dev ocl-icd-opencl-dev opencl-headers clinfo
```

Installare anche il driver e il runtime OpenCL del produttore della GPU: il solo loader OpenCL non fornisce un dispositivo di calcolo.

```bash
git clone https://github.com/DomenicoVillari3/hpc-proj.git
cd hpc-proj
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data results
```

I programmi Python non richiedono una compilazione separata. Il programma OpenCL viene compilato sul dispositivo durante l'avvio del training GPU.

## Configurazione

Creare `.env` nella radice del progetto. Sostituire `/percorso/assoluto/hpc-proj` con il percorso reale, identico su tutti i nodi del cluster.

```dotenv
DATASET_DIR=/percorso/assoluto/hpc-proj/data
DATASET_NAME=epsilon_normalized
DATASET_FILE=epsilon_normalized
DATASET_FILE_TEST=epsilon_normalized.t
RESULTS_DIR=/percorso/assoluto/hpc-proj/results
KERNEL_DIR=/percorso/assoluto/hpc-proj/kernels
SEED=42
EPOCHS=50
LEARNING_RATE=0.1
BATCH_SIZE=4096
```

| Variabile | Utilizzo |
| --- | --- |
| `DATASET_DIR` | Directory dei file originali e dei binari preprocessati. |
| `DATASET_NAME` | Prefisso dei binari e dei metadati caricati durante il training. |
| `DATASET_FILE` | File originale di training EPSILON oppure CSV SUSY. |
| `DATASET_FILE_TEST` | File originale di test EPSILON; non utilizzato dal preprocessing SUSY. |
| `RESULTS_DIR` | Directory di destinazione dei CSV. |
| `KERNEL_DIR` | Directory che contiene `gradient.cl`. |
| `SEED` | Seed per inizializzazione e campionamento. |
| `EPOCHS` | Numero di epoche. |
| `LEARNING_RATE` | Learning rate iniziale del training ordinario. |
| `BATCH_SIZE` | Dimensione del mini-batch locale, per worker/rank nelle modalità distribuite. |

La configurazione è letta tramite `python-dotenv`; non sono previsti argomenti CLI per gli iperparametri. I valori sopra sono un esempio di avvio, non una configurazione ottimale per tutti gli esperimenti.

## Dataset e preprocessing

### EPSILON

Inserire in `data/` i file decompressi `epsilon_normalized` e `epsilon_normalized.t`, in formato LibSVM. Il preprocessing usa 2000 feature, mantiene la separazione train/test originale e converte le label da `{-1, +1}` a `{0, 1}`.

```bash
.venv/bin/python3 src/preprocess.py
```

Per il dataset completo, le note del progetto riportano 400.000 campioni di training e 100.000 di test. Le sole matrici dense `float32` richiedono circa 4 GB complessivi; parsing e copie temporanee aumentano il fabbisogno di RAM.

### SUSY

Inserire il file decompresso `SUSY.csv` in `data/` e impostare:

```dotenv
DATASET_NAME=susy
DATASET_FILE=SUSY.csv
```

```bash
.venv/bin/python3 src/preprocess_susy.py
```

Il CSV contiene la label nella prima colonna e 18 feature nelle successive. Il codice usa il primo 80% delle righe per il training e il restante 20% per il test, senza shuffle preliminare. Usare `DATASET_NAME=susy`: i nomi dei binari SUSY sono fissi nel sorgente.

### Formato generato

Entrambi i preprocessing producono quattro binari `float32` e un file JSON, dove `<dataset>` è `epsilon_normalized` oppure `susy`:

- `<dataset>_X_train.bin` e `<dataset>_y_train.bin`
- `<dataset>_X_test.bin` e `<dataset>_y_test.bin`
- `<dataset>_meta.json`, con le chiavi `N_train`, `N_test` e `D`

## Esecuzione

Eseguire i comandi dalla radice del progetto, dopo aver completato configurazione e preprocessing.

### Verifica MPI e OpenCL

Un test MPI locale non richiede il `hostfile` del cluster:

```bash
mpirun -np 2 .venv/bin/python3 hello_mpi.py
```

Per il cluster, adattare `hostfile` ai nodi disponibili e verificare l'avvio remoto:

```bash
mpirun --hostfile hostfile -np 2 "$PWD/.venv/bin/python3" "$PWD/hello_mpi.py"
clinfo
```

Su nodi NVIDIA, `nvidia-smi` permette anche di verificare driver e GPU. Il numero di rank deve rispettare gli slot disponibili nel `hostfile`.

### Training

```bash
# L0 — Baseline sequenziale
.venv/bin/python3 src/sgd_sequential.py

# L1 — Un master e tre worker; richiede almeno due rank
mpirun --hostfile hostfile -np 4 "$PWD/.venv/bin/python3" "$PWD/src/master-worker.py"

# L2 — SGD sincrono con Allreduce
mpirun --hostfile hostfile -np 4 "$PWD/.venv/bin/python3" "$PWD/src/data_parallelism.py"

# L3 — Task CPU e comunicazione non bloccante
mpirun --hostfile hostfile -np 4 "$PWD/.venv/bin/python3" "$PWD/src/task_parallelism.py"

# L4 — Gradiente GPU e comunicazione non bloccante
mpirun --hostfile hostfile -np 4 "$PWD/.venv/bin/python3" "$PWD/src/sgd_opencl.py"
```

Per un'esecuzione locale delle modalità MPI, omettere `--hostfile hostfile` e scegliere un numero di processi compatibile con la macchina.

### Coda di job e ricerca di iperparametri

`src/job_queue.py` riserva il rank 0 al coordinamento e crea gruppi da tre rank. Usare almeno quattro processi; con 13 processi si ottengono quattro gruppi. Le configurazioni sono definite nella lista `CONFIGS` del sorgente:

| Modalità | Learning rate iniziale | Batch locale |
| --- | --- | --- |
| CPU | 0.16 | 4096 |
| CPU | 0.64 | 16384 |
| GPU | 0.16 | 4096 |
| GPU | 0.64 | 16384 |

Questo script cerca `.env` nel percorso fisso `/home/mpiuser/test/src/.env`. Per eseguirlo da un'altra posizione, esportare la configurazione della radice e inoltrarla ai rank:

```bash
set -a
source .env
set +a
mpirun --hostfile hostfile -np 13 \
    -x DATASET_DIR -x DATASET_NAME -x RESULTS_DIR -x KERNEL_DIR -x EPOCHS -x SEED \
    "$PWD/.venv/bin/python3" "$PWD/src/job_queue.py"
```

La coda usa learning rate e batch di `CONFIGS`, indipendentemente da `LEARNING_RATE` e `BATCH_SIZE` del file `.env`. Produce un CSV per job e un riepilogo ordinato per accuracy sul test set. Per una ricerca con test set indipendente, selezionare gli iperparametri usando la validation e riservare il test alla valutazione finale.

### Script degli esperimenti

Gli script shell sono configurati per `/home/mpiuser/test`: adattare `BASE`, interprete e nodi prima di usarli su un'altra installazione.

```bash
bash run_experiments.sh
bash run_opencl_experiments_with_monitor.sh
bash show_log_gpu_usage.sh
```

- `run_experiments.sh` esegue attualmente L2, L3 e L4 con **3 rank**; la baseline sequenziale è commentata. Il comando finale `ls` cerca CSV direttamente in `results/`, mentre molte implementazioni li scrivono in sottodirectory: può terminare con errore anche dopo training riusciti.
- `run_opencl_experiments_with_monitor.sh` esegue L4 con **1, 2, 4, 8 e 13 rank**, raccogliendo log applicativi e campioni di utilizzo/memoria GPU via SSH ogni secondo.
- `show_log_gpu_usage.sh` riporta utilizzo medio e massimo per nodo; il caso a un rank usa il tag `opencl_baseline` nello script di raccolta, mentre lo script di riepilogo cerca `opencl`, quindi può mostrare `no data` per quel caso.

## Risultati e metriche

I programmi registrano le metriche alla prima epoca e poi ogni dieci epoche. I percorsi seguenti riflettono i nomi usati attualmente dal codice, relativi a `RESULTS_DIR`:

| Implementazione | CSV generato |
| --- | --- |
| Sequenziale | `sequential/sequential_<dataset>.csv` |
| Master-worker | `master_worker.csv` |
| Data parallelism | `data_parallelism/data_parallel__second_test_<dataset>_np<P>.csv` |
| Task parallelism | `task_parallelism/tasks_second_test_<dataset>_np<P>.csv` |
| OpenCL | `opencl/opencl_second_test_<dataset>_np<P>.csv` |
| Coda di job | `job_<dataset>_<cpu|gpu>_lr<lr>_b<batch>_g<gruppo>.csv` e `job_summary_<dataset>.csv` |

`<P>` indica il numero di rank. I risultati già versionati includono anche esecuzioni precedenti con nomi e colonne differenti; la coda scrive direttamente in `RESULTS_DIR`, quindi impostarlo a una sottodirectory come `results/hyperparams` per raccogliere lì i nuovi job.

Le metriche principali sono `loss`, `accuracy` ed `epoch_time_s`. Le implementazioni distribuite aggiungono, secondo la modalità, validation locale, tempo di Allreduce, shuffle, task CPU, calcolo GPU e attesa residua.

Nella baseline sequenziale e nel master-worker, `loss` è calcolata sul training set; nelle modalità L2–L4 è calcolata sul test set. La validation di L2–L4 usa il 5% iniziale della partizione locale; le metriche registrate riguardano il rank 0. `gpu_time_s` comprende l'esecuzione della funzione GPU, inclusi allocazioni, trasferimenti e sincronizzazione. `overlap_s` è una stima costruita dai timer del codice, non una misura diretta del tempo di comunicazione nascosto.

Per valutare lo scaling si possono usare `speedup(P) = T(1) / T(P)` ed `efficienza(P) = speedup(P) / P`. Confrontare configurazioni dichiarate, hardware, costo della valutazione e qualità del modello: aumentando i rank a batch locale fisso cambia anche il batch globale. Inoltre L3/L4 campionano mini-batch casuali a ogni step, mentre L0/L2 percorrono dati rimescolati; le epoche non rappresentano esattamente lo stesso lavoro.

I CSV vengono sovrascritti quando si ripete la stessa configurazione. Conservare separatamente risultati e `.env` di ciascuna prova. Il codice salva metriche, ma non esporta i pesi finali del modello.

## Vincoli operativi

- **Dimensione dei batch GPU:** usare un `BATCH_SIZE` multiplo di **256**. La griglia di `forward_kernel` è arrotondata a 256 work-item e il kernel attuale non controlla il limite sul numero di campioni; un batch non multiplo può causare accessi fuori dai buffer.
- **Dimensione delle partizioni:** nelle modalità L3/L4 e nella coda, il batch deve entrare nella partizione locale dopo il 5% di validation. Ogni rank dello stesso comunicatore deve eseguire lo stesso numero di step collettivi; verificare questo vincolo quando si cambiano dataset, batch o numero di rank.
- **RAM per rank:** L2–L4 e i job leggono l'intero training set prima di copiarne la partizione locale. Aumentare i rank non elimina quindi il costo della copia completa in memoria per processo.
- **GPU per nodo:** ogni rank GPU seleziona la prima GPU disponibile; il `hostfile` fornito usa uno slot per nodo. Più rank sullo stesso nodo condividono il dispositivo.

## Documentazione e licenza

Le note aggiuntive sono disponibili in [setup iniziale](info_file/setup_iniziale.md), [primo sviluppo](info_file/primo_sviluppo.md), [preprocessing](info_file/preprocessing_e_seq.md) e [walkthrough](info_file/walkthrough.md).

Al momento il repository non contiene un file `LICENSE`.
