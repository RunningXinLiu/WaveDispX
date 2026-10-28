"""Generate small CPU source baselines on the server, retaining exact inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from wavedispx.desktop.bundle_cases import CASES, case_mapping, arrays
from wavedispx.config import config_from_mapping
from wavedispx.workflow import WaveDispXWorkflow
from wavedispx.experiment import ExperimentRequest, run_experiment, save_experiment

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
reference={}
report={'platform':platform.platform(),'cases':{},'tolerances':{'raw_and_energy_relative_L2':1e-5,'theory_atol_mps':.002,'theory_rtol':2e-6}}
for name in CASES:
    start=time.monotonic()
    mapping,base=case_mapping(root,name,args.output/name)
    wf=WaveDispXWorkflow(config_from_mapping(mapping,base_directory=base))
    result=run_experiment(wf,ExperimentRequest.from_mapping(mapping))
    saved=save_experiment(result,wf,mapping)
    reference.update({name+'__'+k:v for k,v in arrays(wf,result).items()})
    report['cases'][name]={'input':mapping,'base_directory':str(base),'saved':str(saved),'elapsed_s':time.monotonic()-start,'statuses':result.statuses}
    print(name,report['cases'][name]['elapsed_s'],flush=True)
np.savez_compressed(args.output/'baseline.npz',**reference)
report['baseline_sha256']=hashlib.sha256((args.output/'baseline.npz').read_bytes()).hexdigest()
report['sources_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in ('python_src/wavedispx','cpp') for p in (root/folder).rglob('*')
    if p.is_file() and p.suffix in ('.py','.cpp','.so')}
(args.output/'provenance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
