#!/bin/bash -

# npf file
NPF_SCRIPT_1="custom_math.npf"
NPF_SCRIPT_2="iperf-advanced.npf"
# Path for logs
LOG_FILE="npf_results.log"
# Path for results
NPF_RESULT_PATH="results"

python3 ../npf_regress.py --test $NPF_SCRIPT_1
# python3 ../npf_regress.py --test $NPF_SCRIPT_2

# Run npf (we don't generate plots here)
# python3 ../npf_regress.py --test $NPF_SCRIPT --result-path $NPF_RESULT_PATH --no-graph --single-output result.csv &> $LOG_FILE

