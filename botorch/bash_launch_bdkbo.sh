#!/bin/bash

# start from 0, inclusive
for r in {1..1}
do
    fname="nohup${r}.out"
    nohup python run_bdkbo.py -i $r -k Lin -a EI -o nano -p ./test/results/paper_2022/botorch/bnn_test_ei/ >$fname &
done
