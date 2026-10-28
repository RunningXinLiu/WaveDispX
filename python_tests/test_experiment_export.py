"""Saved CSV-based experiments must be reloadable outside their source folder."""
from pathlib import Path
from copy import deepcopy
from wavedispx.app_service import load_template_mapping
from wavedispx.config import config_from_mapping, load_config
from wavedispx.workflow import WaveDispXWorkflow
from wavedispx.experiment import ExperimentResult, ExperimentRequest, save_experiment

ROOT=Path(__file__).resolve().parents[1]


def test_saved_csv_configuration_is_self_contained(tmp_path):
    path=ROOT/'examples/custom_geometry/passive_switching_array.toml'
    mapping=load_template_mapping(path)
    mapping['run']['output_directory']=str(tmp_path/'results')
    original=deepcopy(mapping)
    wf=WaveDispXWorkflow(config_from_mapping(mapping,base_directory=path.parent))
    directory=save_experiment(ExperimentResult(ExperimentRequest(product='energy')),wf,mapping)
    restored=load_config(directory/'input.toml')
    assert mapping==original
    for new,old in ((restored.receivers.coordinates_file,wf.config.receivers.coordinates_file),
                    (restored.ambient_sources.coordinates_file,wf.config.ambient_sources.coordinates_file),
                    (restored.ambient_sources.activity_file,wf.config.ambient_sources.activity_file)):
        assert new.is_relative_to(directory)
        assert new.read_bytes()==old.read_bytes()
