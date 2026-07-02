#!/bin/bash
# run_experiments.sh
# I parametri (batch, lr, epochs, seed) vengono dal .env

set -e

BASE=/home/mpiuser/test
PYTHON=$BASE/.venv/bin/python3
HOSTFILE=$BASE/hostfile
SRC=$BASE/src
RESULTS=$BASE/results

echo "=============================================="
echo "  HPC Project — Full Experiment Suite"
echo "  $(date)"
echo "=============================================="

run_exp() {
    local label=$1
    local np=$2
    local script=$3

    echo ""
    echo "----------------------------------------------"
    echo "  START: $label | np=$np | dataset=$dataset"
    echo "  $(date +%H:%M:%S)"
    echo "----------------------------------------------"

    mpirun --hostfile $HOSTFILE -np $np \
        $PYTHON $SRC/$script

    echo "  DONE: $label | $(date +%H:%M:%S)"
}

# ──────────────────────────────────────────────
# L0 — Sequential Baseline
# ──────────────────────────────────────────────
#run_exp "L0 Sequential — EPSILON" 1 sgd_sequential.py

# ──────────────────────────────────────────────
# L2 — MPI Data Parallelism (Strong Scaling)
# ──────────────────────────────────────────────
# for np in 2 4 8 13; do
run_exp "L2 Data Parallel np=3 — EPSILON" 3 data_parallelism.py
# done

# ──────────────────────────────────────────────
# L3 — Task Parallelism
# ──────────────────────────────────────────────

for np in 3; do
    run_exp "L3 Task Parallel np=$np — EPSILON" $np task_parallelism.py
done

# ──────────────────────────────────────────────
# L4 — OpenCL GPU
# ──────────────────────────────────────────────
for np in 3; do
    run_exp "L4 OpenCL np=$np — EPSILON" $np sgd_opencl.py
done

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS DONE — $(date)"
echo "=============================================="
echo ""
echo "Results in $RESULTS:"
ls -lh $RESULTS/*.csv