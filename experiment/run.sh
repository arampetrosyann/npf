#!/bin/bash -

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
	PYTHON_BIN="python3"
fi

# npf file
NPF_SCRIPT_1="$SCRIPT_DIR/custom_math.npf"
# NPF_SCRIPT_2="$SCRIPT_DIR/iperf-advanced.npf"
# Path for logs
LOG_FILE="npf_results.log"
# Path for results
NPF_RESULT_PATH="results"

"$PYTHON_BIN" "$SCRIPT_DIR/../npf_regress.py" --test "$NPF_SCRIPT_1"
# python3 ../npf_regress.py --test $NPF_SCRIPT_2

# Run npf (we don't generate plots here)
# python3 ../npf_regress.py --test $NPF_SCRIPT --result-path $NPF_RESULT_PATH --no-graph --single-output result.csv &> $LOG_FILE

