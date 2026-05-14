import numpy as np
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

INPUT  = os.path.join(os.getenv("DATASET_DIR"), os.getenv("DATASET_FILE"))
OUTPUT = os.getenv("DATASET_DIR")
DATASET_NAME = os.getenv("DATASET_NAME")

print("[INFO] Caricamento SUSY...")
df = pd.read_csv(INPUT, header=None, dtype=np.float32)

N, cols = df.shape
D = cols - 1  # 18 feature
print(f"[INFO] N={N}, D={D}")

data = df.values

# Split 80/20
split = int(N * 0.8)
X_train = data[:split, 1:]
y_train = data[:split, 0]
X_test  = data[split:, 1:]
y_test  = data[split:, 0]

# Salva binario float32
X_train.tofile(os.path.join(OUTPUT, "susy_X_train.bin"))
y_train.tofile(os.path.join(OUTPUT, "susy_y_train.bin"))
X_test.tofile(os.path.join(OUTPUT,  "susy_X_test.bin"))
y_test.tofile(os.path.join(OUTPUT,  "susy_y_test.bin"))

# Metadata
meta = {"N_train": int(split), "N_test": int(N - split), "D": D}
with open(os.path.join(OUTPUT, f"{DATASET_NAME}_meta.json"), "w") as f:
    json.dump(meta, f)

print(f"[DONE] Train: {split} samples | Test: {N-split} samples | D={D}")
print(f"[DONE] File salvati in {OUTPUT}")