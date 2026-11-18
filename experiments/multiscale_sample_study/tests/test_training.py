"""A complete tiny training run, exact resume and incompatible-cache rejection."""
import json

import numpy as np
import pytest
import torch

from neural_operator.data import Dataset
from experiments.multiscale_sample_study import training
from experiments.multiscale_sample_study.sno import SNOAdapter


def setup_case(root):
    rng=np.random.default_rng(1)
    data=Dataset(A=np.ones((6,4)),n=np.ones((6,4)),
        u_scat=2+rng.normal(size=(6,4))+1j*rng.normal(size=(6,4)),
        theta=np.zeros(6),k0=np.linspace(.1,4,6),medium_id=np.zeros(6,dtype=int),
        split=np.array([1,2,0,0,0,0]),coordinates=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]),
        triangles=np.array([[0,1,2],[1,3,2]]),frequency_id=np.arange(6),split_strategy='frequency')
    directory=root/'cache/n4/sno'
    directory.mkdir(parents=True)
    SNOAdapter(data,np.arange(2,6),rank=2,width=2,depth=1).save_preprocessing(directory)
    protocol=dict(device='cpu',sno_device='cpu',epochs=3,batch_size=2,depth=1,sno_width=2,
                  learning_rate=.001,weight_decay=.0001,patience_checks=15,validation_interval=1)
    return data,protocol


def test_full_training_replays_and_rejects_changed_specification(tmp_path):
    data,p=setup_case(tmp_path)
    result=training.train_one(tmp_path,data,p,4,'mscale_sno',11,np.array([1.]))
    directory=training.run_directory(tmp_path,4,'mscale_sno',11)
    assert result['epochs_completed']==3
    adapter=training.load_adapter(tmp_path,4,'mscale_sno',p)
    checkpoint=torch.load(directory/'model.pt',weights_only=False)
    adapter.model.load_state_dict(checkpoint['state_dict'])
    with np.load(directory/'predictions.npz') as saved:
        np.testing.assert_allclose(adapter.predict_fields(np.array([1])),saved['prediction'],rtol=0,atol=0)
    with pytest.raises(ValueError,match='specification differs'):
        training.train_one(tmp_path,data,dict(p,epochs=4),4,'mscale_sno',11,np.array([1.]))


def test_interrupt_after_committed_validation_resumes_exactly(tmp_path,monkeypatch):
    full=tmp_path/'full'; resumed=tmp_path/'resumed'
    data,p=setup_case(full)
    setup_case(resumed)
    training.train_one(full,data,p,4,'mscale_sno',11,np.array([1.]))
    write=training._write_json
    def interrupt(path,value):
        write(path,value)
        if path.name=='history.json' and len(value)==1:
            raise RuntimeError('test interruption')
    with monkeypatch.context() as patch:
        patch.setattr(training,'_write_json',interrupt)
        with pytest.raises(RuntimeError,match='interruption'):
            training.train_one(resumed,data,p,4,'mscale_sno',11,np.array([1.]))
    training.train_one(resumed,data,p,4,'mscale_sno',11,np.array([1.]))
    a=torch.load(training.run_directory(full,4,'mscale_sno',11)/'model.pt',weights_only=False)
    b=torch.load(training.run_directory(resumed,4,'mscale_sno',11)/'model.pt',weights_only=False)
    assert a['best_epoch']==b['best_epoch']
    for key in a['state_dict']:
        torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)
