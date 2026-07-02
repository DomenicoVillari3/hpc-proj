#!/bin/bash
set -e

BASE=/home/mpiuser/test
PYTHON=$BASE/.venv/bin/python3
HOSTFILE=$BASE/hostfile
SRC=$BASE/src
RESULTS=$BASE/results
GPU_LOGS=$RESULTS/gpu_monitoring
mkdir -p $GPU_LOGS

NODES=(mpi1 mpi2 mpi3 mpi4 mpi5 mpi6 \
       mpi7 mpi8 mpi9 mpi10 mpi11 mpi12 mpi13)

start_gpu_monitor() {
    local np=$1
    local tag=$2
    for i in $(seq 0 $((np-1))); do
        NODE=${NODES[$i]}
        ssh mpiuser@$NODE \
            "nvidia-smi \
            --query-gpu=timestamp,utilization.gpu,memory.used \
            --format=csv -l 1 \
            > $GPU_LOGS/gpu_${NODE}_${tag}.log 2>&1 &
            echo \$! > /tmp/nvidia_smi_pid" &
    done
    sleep 2
}

stop_gpu_monitor() {
    local np=$1
    for i in $(seq 0 $((np-1))); do
        NODE=${NODES[$i]}
        ssh mpiuser@$NODE \
            "kill \$(cat /tmp/nvidia_smi_pid) 2>/dev/null; \
             rm -f /tmp/nvidia_smi_pid" &
    done
    wait
}

run_exp() {
    local label=$1
    local np=$2
    local script=$3

    echo "[$(date +%H:%M:%S)] $label np=$np starting..."

    start_gpu_monitor $np "${label}_np${np}"

    mpirun --hostfile $HOSTFILE -np $np \
        $PYTHON $SRC/$script \
        2>&1 | tee $GPU_LOGS/${label}_np${np}.log

    stop_gpu_monitor $np

    echo "[$(date +%H:%M:%S)] $label np=$np done."
}

echo "[$(date +%H:%M:%S)] experiment suite started."

run_exp "opencl_baseline" 1 sgd_opencl.py
run_exp "opencl" 2 sgd_opencl.py
run_exp "opencl" 4 sgd_opencl.py
run_exp "opencl" 8 sgd_opencl.py
run_exp "opencl" 13 sgd_opencl.py

echo "[$(date +%H:%M:%S)] all done. results in $GPU_LOGS"