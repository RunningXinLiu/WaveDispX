"""Frozen desktop entry and actual-solver self-test used by clean CI machines."""
from pathlib import Path
import sys
from wavedispx.classroom.runtime import prepare_runtime
prepare_runtime()

if '--self-test' in sys.argv:
    from wavedispx.classroom.bundle_check import main
    raise SystemExit(main(Path(sys.argv[sys.argv.index('--self-test')+1])))
from wavedispx.classroom.app import main
raise SystemExit(main())
