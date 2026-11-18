"""Finish audits and reports automatically after the independently running jobs."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import hashlib
from .dataset import HERE
from experiments.reference_sno_gmsfem.dataset import _write_json


def main():
    repo=HERE.parents[1];results=HERE/'results'
    env=dict(os.environ,VECLIB_MAXIMUM_THREADS='2',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    def record(stage,**more):
        _write_json(results/'completion_status.json',dict(stage=stage,**more))
        print(stage,flush=True)
    def execute(args,log):
        with (results/log).open('w') as stream:
            subprocess.run(args,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def check_imports():
        manifest=json.loads((results/'additional_source_manifest.json').read_text())
        for path,digest in manifest['source_sha256'].items():
            if hashlib.sha256((repo/path).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Supplemental frozen implementation changed: {path}')
    try:
        record('waiting_for_15_neural_runs')
        while True:
            path=results/'training_summary.json'
            if path.exists() and len(json.loads(path.read_text())['runs'])==15:break
            processes=subprocess.check_output(['ps','-axo','command'],text=True)
            if '-m experiments.highk_svd_gmsfem.run train' not in processes:
                raise RuntimeError('Trainer stopped before all 15 runs completed; inspect training.log')
            time.sleep(10)
        record('verifying_neural_checkpoints')
        check_imports()
        execute([sys.executable,'-m','experiments.highk_svd_gmsfem.run','verify'],'checkpoint_replay.log')
        record('waiting_for_gmsfem')
        while True:
            path=results/'parallel_status.json'
            value=json.loads(path.read_text()) if path.exists() else {}
            if value.get('status')=='complete':break
            if value.get('status')=='failed':raise RuntimeError('GMsFEM worker failed; inspect worker logs')
            time.sleep(10)
        record('auditing_native_fem_labels')
        check_imports()
        execute(['bash','scripts/docker.sh','compose','exec','-T','-e','OMP_NUM_THREADS=2',
                 '-e','OPENBLAS_NUM_THREADS=2','dolfinx-complex','python3','-u','-m',
                 'experiments.highk_svd_gmsfem.run','audit-labels'],'fem_label_replay.log')
        record('building_verified_report')
        execute([sys.executable,'-m','experiments.highk_svd_gmsfem.run','report'],'report_generation.log')
        check_imports()
        record('complete')
    except Exception as exc:
        record('failed',error=str(exc))
        raise


if __name__=='__main__':main()
