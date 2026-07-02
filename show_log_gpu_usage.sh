#!/bin/bash
for np in 1 2 4 8 13; do
    echo "=== np=$np ==="
    for node in $(seq -f "mpi%g" 1 $np); do
        echo -n "$node: "
        grep -v "timestamp" \
            /home/mpiuser/test/results/gpu_monitoring/gpu_${node}_opencl_np${np}.log 2>/dev/null | \
        awk -F',' '{
            gsub(/ /, "", $2);
            gsub(/%/, "", $2);
            if($2+0 > max) max=$2+0;
            sum+=$2+0; n++
        } END {
            if(n>0) printf "avg=%.1f%% max=%d%%\n", sum/n, max;
            else print "no data"
        }'
    done
    echo ""
done