#!/bin/sh
# No installation or environment mutation: use the existing project and seisloc.
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/python_src${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n seisloc python -m wavedispx.classroom.app "$@"
