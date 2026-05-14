import numpy as np
import json
import os
from sklearn.datasets import load_svmlight_file
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATASET_DIR")
DATASET_NAME = os.getenv("DATASET_NAME")
TRAIN_FILE = os.path.join(DATA_DIR, os.getenv("DATASET_FILE"))
TEST_FILE  = os.path.join(DATA_DIR, os.getenv("DATASET_FILE_TEST"))
OUTPUT   = DATA_DIR


print("[INFO] Caricamento epsilon train...")
X_train, y_train = load_svmlight_file(TRAIN_FILE, dtype=np.float32, n_features=2000)
X_train = X_train.toarray()

print("[INFO] Caricamento epsilon test...")
X_test, y_test = load_svmlight_file(TEST_FILE, dtype=np.float32, n_features=2000)
X_test = X_test.toarray()

# Label: -1/+1 → 0/1
y_train = ((y_train + 1) / 2).astype(np.float32)
y_test  = ((y_test  + 1) / 2).astype(np.float32)

N_train, D = X_train.shape
N_test     = X_test.shape[0]

print(f"[INFO] N_train={N_train} | N_test={N_test} | D={D}")

# Salva binario float32
X_train.tofile(os.path.join(OUTPUT, f"{DATASET_NAME}_X_train.bin"))
y_train.tofile(os.path.join(OUTPUT, f"{DATASET_NAME}_y_train.bin"))
X_test.tofile(os.path.join(OUTPUT,  f"{DATASET_NAME}_X_test.bin"))
y_test.tofile(os.path.join(OUTPUT,  f"{DATASET_NAME}_y_test.bin"))

meta = {"N_train": int(N_train), "N_test": int(N_test), "D": int(D)}
with open(os.path.join(OUTPUT, f"{DATASET_NAME}_meta.json"), "w") as f:
    json.dump(meta, f)

print(f"[DONE] Train: {N_train} | Test: {N_test} | D={D}")
print(f"[DONE] File salvati in {OUTPUT}")