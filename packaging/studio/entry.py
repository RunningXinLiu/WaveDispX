"""Studio-only distribution entry; self-test runs the actual packaged UI."""
from pathlib import Path
import sys
from wavedispx.desktop.runtime import prepare_runtime
prepare_runtime()
if '--self-test' in sys.argv:
    from wavedispx.desktop.bundle_check import main
    raise SystemExit(main(Path(sys.argv[sys.argv.index('--self-test')+1])))
from wavedispx.desktop.studio import main
raise SystemExit(main())
