#!/bin/bash

mkdir -p ./results

export k_val=5
export decay_val=0.9
N_ITER=10

for i in `seq 1 5`; do
    python pall_svm_ssk_aprx.py | tee  "./results/aprx_k${k_val}_dcy_${decay_val}_iter${i}.log"
done
