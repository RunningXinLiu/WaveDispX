import wavedispx
from wavedispx.cli import _parser
from wavedispx.workflow import WaveDispXWorkflow


def test_wavedispx_is_the_primary_public_brand():
    assert wavedispx.__version__ == "0.2.0"
    assert wavedispx.WaveDispXWorkflow is WaveDispXWorkflow
    assert _parser().prog == "wavedispx"
