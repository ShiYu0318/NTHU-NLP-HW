"""Prepare a Chinese Draft PR only for a verified, completed research study."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from paths import WORKSPACE
from recurrent_study import verify_study
from train_recurrent import atomic_json

REPOSITORY='ShiYu0318/NTHU-NLP-HW'
WORK_BRANCH='work/hw2-gru-comparison'


def publication_plan(root):
    root=Path(root).resolve();verify_study(root)
    if (root/'STOP').exists():return dict(publishable=False,reason='Study is paused')
    comparison=json.loads((root/'comparison.json').read_text())
    if not comparison.get('useful_finding'):return dict(publishable=False,reason='No selected finding')
    for role in ['gru','lstm']:
        p=root/role;verified=p/'verification.json'
        if not verified.exists() or not json.loads(verified.read_text()).get('converged'):
            return dict(publishable=False,reason='Both models must converge and be independently verified')
        v=json.loads(verified.read_text());m=json.loads((p/'checkpoint_metadata.json').read_text())
        assert hashlib.sha256((p/'checkpoint.pt').read_bytes()).hexdigest()==v['checkpoint_sha256']==m['checkpoint_sha256']
        assert v['evaluation']['total']==v['diagnostic']['total']==10000
    names=['comparison.json','report.md','learning_curves.png','study.json','frozen_submission.json']
    for role in ['gru','lstm']:
        names += [role+'/'+name for name in ['experiment.json','verification.json','checkpoint_metadata.json',
                  'state/history.json','source_notebook.ipynb','evaluation_predictions.csv','diagnostic_predictions.csv']]
    for name in names:
        assert (root/name).is_file(),name
        assert (root/name).stat().st_size<50*1024**2,name
    body=(root/'report.md').read_text()
    if body.startswith('# '):body='\n'.join(body.splitlines()[1:]).lstrip()
    assert [x for x in body.splitlines() if x.startswith('## ')]==['## 摘要','## 實作方法','## 實驗結果']
    return dict(publishable=True,files=names,body=body,branch='feat/hw2-gru-lstm-'+root.name.removeprefix('gru_lstm_'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-dir',required=True,type=Path)
    parser.add_argument('--publish',action='store_true',help='Explicitly commit selected results, push a feature branch and create a Draft PR')
    args=parser.parse_args();root=args.study_dir.resolve();plan=publication_plan(root)
    if not plan['publishable'] or not args.publish:
        print(json.dumps({k:v for k,v in plan.items() if k!='body'},indent=2));return
    def run(*command):
        p=subprocess.run(command,cwd=WORKSPACE,capture_output=True,text=True)
        if p.returncode:raise RuntimeError(p.stderr or p.stdout)
        return p.stdout.strip()
    def git(*args):return run('git',*args)
    assert git('branch','--show-current')==WORK_BRANCH
    assert not git('status','--porcelain'), 'Unexpected worktree changes; refusing to commit them.'
    assert json.loads(run('gh','repo','view','--json','nameWithOwner'))['nameWithOwner']==REPOSITORY
    subprocess.run([sys.executable,'-m','unittest','discover','-s','HW2/experiments/tests','-v'],cwd=WORKSPACE,check=True)
    destination=WORKSPACE/'HW2/experiments/studies'/root.name
    for name in plan['files']:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    git('add','--',str(destination));git('diff','--cached','--check')
    if git('diff','--cached','--name-only'):
        git('commit','-m','docs(hw2): report converged GRU and LSTM comparison')
    verify_study(root);git('fetch','origin')
    for message in git('log','--format=%B','origin/main..HEAD').splitlines():
        assert message.isascii(),'Commit messages must be English'
    head=git('rev-parse','HEAD');branch=plan['branch']
    existing=subprocess.run(['git','rev-parse','--verify','refs/heads/'+branch],cwd=WORKSPACE,capture_output=True,text=True)
    if existing.returncode==0:assert existing.stdout.strip()==head,'Existing feature branch differs; refusing to overwrite.'
    else:git('branch',branch,head)
    gh=shutil.which('gh')
    if gh is None:raise FileNotFoundError('GitHub CLI is required to publish the authorized draft.')
    git('-c','credential.helper=','-c','credential.helper=!'+shlex.quote(gh)+' auth git-credential','push','-u','origin',branch)
    found=json.loads(run('gh','pr','list','--repo',REPOSITORY,'--head',branch,'--state','all','--json','url,state,isDraft'))
    if found:
        url=found[0]['url']
    else:
        body_file=root/'pr_body.md';body_file.write_text(plan['body'])
        url=run('gh','pr','create','--repo',REPOSITORY,'--base','main','--head',branch,'--draft',
                '--title','feat(hw2): 比較 GRU 與 LSTM 的收斂表現與參數效率','--body-file',str(body_file))
    atomic_json(root/'publication.json',dict(url=url,branch=branch,head_sha=head,draft=True,merged=False))
    print('Draft PR:',url,flush=True)

if __name__=='__main__':main()
