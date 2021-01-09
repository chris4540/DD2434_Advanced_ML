#!/bin/bash

mkdir -p ./results

export k_val=5
export decay_val=0.9

N_ITER=5

for i in `seq 1 ${N_ITER}`; do
    python para_svm_ssk.py | tee  "./results/exact_k${k_val}_dcy_${decay_val}_iter${i}.log"
done
