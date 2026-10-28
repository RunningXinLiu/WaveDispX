"""Bundle-specific path changes preserve scientific and user-supplied inputs."""
from copy import deepcopy
from pathlib import Path
import sys
from wavedispx.desktop import runtime


def test_source_mapping_unchanged(tmp_path):
    m={'run':{'output_directory':'../output_python/demo'},'model':{'layers':[{'vs_mps':220.}]}}
    assert runtime.runtime_mapping(m,tmp_path)==m


def test_frozen_builtin_output_only(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'frozen',True,raising=False)
    monkeypatch.setattr(sys,'_MEIPASS',str(tmp_path),raising=False)
    m={'run':{'output_directory':'../output_python/demo','workers':2},'model':{'layers':[{'vs_mps':220.}]}}
    before=deepcopy(m)
    result=runtime.runtime_mapping(m,tmp_path/'examples')
    assert m==before
    assert result['model']==m['model']
    assert Path(result['run']['output_directory']).is_absolute()
    assert Path(result['run']['output_directory']).parts[-3:]==('WaveDispX','results','demo')
    assert result['run']['workers']==2
    assert runtime.runtime_mapping(m,tmp_path/'user_config')==m
    m['run']['output_directory']=str(tmp_path/'explicit_user_output')
    assert runtime.runtime_mapping(m,tmp_path/'examples')==m


def test_prepared_bundle_requires_native_files(tmp_path,monkeypatch):
    import pytest
    monkeypatch.setattr(sys,'frozen',True,raising=False)
    monkeypatch.setattr(sys,'_MEIPASS',str(tmp_path),raising=False)
    with pytest.raises(FileNotFoundError,match='发行包缺少必要文件'):
        runtime.prepare_runtime()
