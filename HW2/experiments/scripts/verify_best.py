"""Archive and independently verify the current best; coding assistance: OpenAI Codex."""
import ast,hashlib,io,json,time
from pathlib import Path
import pandas as pd
import torch

from paths import CONTINUATION_RUN, DATA, verify_submission
root=CONTINUATION_RUN
metadata_bytes=(root/'checkpoint_metadata.json').read_bytes()
metadata=json.loads(metadata_bytes)
checkpoint=(root/'checkpoint.pt').read_bytes()
assert (root/'checkpoint_metadata.json').read_bytes()==metadata_bytes,'Selection changed; retry.'
assert hashlib.sha256(checkpoint).hexdigest()==metadata['checkpoint_sha256']
dest=root/'best_archive'/f"epoch_{metadata['epoch']}"
dest.mkdir(parents=True,exist_ok=True)
for name,content in [('checkpoint.pt',checkpoint),('checkpoint_metadata.json',metadata_bytes)]:
 target=dest/name
 if target.exists():assert target.read_bytes()==content
 else:target.write_bytes(content)
config=metadata['config'];n=json.loads((root/'training_driver.ipynb').read_text())
ns=dict(torch=torch,answer_order=config.get('answer_order','normal'),char_to_id=config['char_to_id'],id_to_char={i:c for c,i in config['char_to_id'].items()})
for node in ast.parse(''.join(n['cells'][7]['source'])).body:
 if isinstance(node,ast.FunctionDef) and node.name=='reorder_answer_digits':
  exec(compile(ast.Module(body=[node],type_ignores=[]),'answer_order','exec'),ns)
exec(''.join(n['cells'][15]['source']),ns)
torch.set_num_threads(2)
model=ns['CharRNN'](config['vocab_size'],config['embed_dim'],config['hidden_dim']).eval()
model.load_state_dict(torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=True))
frame=pd.read_csv(DATA/'arithmetic_eval.csv',dtype=str)
matched=0;start=time.monotonic()
with torch.inference_mode():
 for index,row in enumerate(frame.itertuples(index=False),1):
  prediction=''.join(model.generator(row.src,max_len=len(row.src)+16))[len(row.src):]
  matched+=int(prediction==row.tgt)
  if index%2000==0:print(f'Independent saved-checkpoint verification: {index}/{len(frame)}',flush=True)
result=dict(epoch=metadata['epoch'],correct=matched,total=len(frame),exact_match=matched/len(frame),
 checkpoint_sha256=metadata['checkpoint_sha256'],seconds=time.monotonic()-start,full_answer_generation=True)
assert result['exact_match']==metadata['exact_match'],result
verify_submission(root/'protected_submission_sha256.json')
result['submission_unchanged']=True
(dest/'independent_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
