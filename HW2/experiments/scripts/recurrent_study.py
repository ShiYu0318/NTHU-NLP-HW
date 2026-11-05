"""Shared model and immutable-driver adapters for the post-submission study."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import torch
from paths import CONTINUATION_RUN

SOURCE_DRIVER=CONTINUATION_RUN/'training_driver.ipynb'


def make_driver(cell_type, smoke_epochs=None):
    if cell_type not in {'GRU','LSTM'}:
        raise ValueError('Unsupported recurrent cell')
    driver=copy.deepcopy(json.loads(SOURCE_DRIVER.read_text()))
    model=''.join(driver['cells'][15]['source'])
    assert model.count('torch.nn.LSTM(')==2
    driver['cells'][15]['source']=model.replace('torch.nn.LSTM(',f'torch.nn.{cell_type}(').splitlines(True)
    setup=''.join(driver['cells'][17]['source'])
    setup=setup.replace('config = dict(embed_dim=embed_dim,','config = dict(cell_type=cell_type, embed_dim=embed_dim,',1)
    old="run_path = output_path / 'runs' / f'lstm_e{embed_dim}_h{hidden_dim}_seed2{order_suffix}'"
    assert old in setup
    setup=setup.replace(old,"run_path = output_path / 'state'")
    setup=setup.replace("state['config'].setdefault('answer_order', 'normal')", "state['config'].setdefault('answer_order', 'normal')\n    state['config'].setdefault('cell_type', 'LSTM')")
    driver['cells'][17]['source']=setup.splitlines(True)
    loop=''.join(driver['cells'][19]['source'])
    loop=loop.replace("(output_path / 'STOP').exists()", "((output_path / 'STOP').exists() or (output_path.parent / 'STOP').exists())")
    if smoke_epochs is not None:
        assert smoke_epochs>0
        loop=loop.replace('for epoch in __import__("itertools").count(start_epoch):',f'for epoch in range(start_epoch, start_epoch + {smoke_epochs}):')
    driver['cells'][19]['source']=loop.splitlines(True)
    driver['cells'][20]['source']=['# Research execution snapshot\n','Configuration and measured results are stored alongside this file; this is not the submitted homework report.\n']
    for c in driver['cells']:
        if c['cell_type']=='code':
            compile(''.join(c['source']),'<recurrent-study>','exec')
            c['outputs']=[];c['execution_count']=None
    return driver


def make_model(driver, config):
    mapping=config['char_to_id']
    ns=dict(torch=torch,char_to_id=mapping,id_to_char={i:c for c,i in mapping.items()},answer_order='normal')
    for node in ast.parse(''.join(driver['cells'][7]['source'])).body:
        if isinstance(node,ast.FunctionDef) and node.name=='reorder_answer_digits':
            exec(compile(ast.Module(body=[node],type_ignores=[]),'answer-order','exec'),ns)
    exec(''.join(driver['cells'][15]['source']),ns)
    return ns['CharRNN'](config['vocab_size'],config['embed_dim'],config['hidden_dim'])


def convergence_status(records):
    source=''.join(json.loads(SOURCE_DRIVER.read_text())['cells'][17]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='convergence_status')
    ns={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'convergence','exec'),ns)
    return ns['convergence_status'](records)


def study_hw2(study):
    study=Path(study).resolve()
    if study.parent.name!='runs' or study.parent.parent.name!='experiments':
        raise ValueError('A study must live under HW2/experiments/runs/')
    return study.parents[2]


def verify_study(study):
    hw=study_hw2(study)
    manifest=json.loads((Path(study)/'frozen_submission.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((hw/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen homework artifact changed: {name}')
