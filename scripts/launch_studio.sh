#!/bin/sh
# Use the existing CPU environment; no package/environment mutation.
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/python_src${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n seisloc python -m wavedispx.desktop.studio "$@"
