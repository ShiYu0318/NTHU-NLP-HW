import copy
import io
import json
from pathlib import Path
import sys
import unittest

import torch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))

class RecurrentStudyTests(unittest.TestCase):
    def helper(self):
        self.assertTrue((SCRIPTS/'recurrent_study.py').exists(),'Recurrent study helper is missing')
        import recurrent_study
        return recurrent_study

    def model(self,kind):
        h=self.helper()
        config=json.loads((SCRIPTS.parent/'runs/continuation_20261104_230700/checkpoint_metadata.json').read_text())['config']
        config=dict(config,embed_dim=8,hidden_dim=12)
        return h.make_model(h.make_driver(kind),config),config

    def test_models_change_only_recurrent_cell_type(self):
        gru,_=self.model('GRU');lstm,_=self.model('LSTM')
        self.assertEqual(sum(isinstance(m,torch.nn.GRU) for m in gru.modules()),2)
        self.assertEqual(sum(isinstance(m,torch.nn.LSTM) for m in lstm.modules()),2)
        self.assertEqual(gru.embedding.weight.shape,lstm.embedding.weight.shape)
        self.assertEqual(gru.linear[-1].weight.shape,lstm.linear[-1].weight.shape)
        self.assertLess(sum(p.numel() for p in gru.parameters()),sum(p.numel() for p in lstm.parameters()))

    def test_cached_gru_decoder_matches_uncached_generation(self):
        model,c=self.model('GRU');model.eval();mapping=c['char_to_id'];reverse={i:t for t,i in mapping.items()}
        prompt='1+2+3=';tokens=[mapping[x] for x in prompt];limit=len(tokens)+5
        with torch.no_grad():
            while len(tokens)<limit:
                x=torch.tensor([tokens]);x=model.embedding(x)
                x,_=model.rnn_layer1(x);x,_=model.rnn_layer2(x)
                token=int(model.linear(x[:,-1,:]).argmax(-1).item())
                if token==mapping['<eos>']:break
                tokens.append(token)
        expected=[reverse[i] for i in tokens]
        self.assertEqual(model.generator(prompt,max_len=limit),expected)

    def test_adam_resume_preserves_next_parameter_update(self):
        model,c=self.model('GRU');model2=copy.deepcopy(model)
        def step(m,opt):
            opt.zero_grad(set_to_none=True)
            y=m(torch.tensor([[2,3,4]]),torch.tensor([3]))
            torch.nn.functional.cross_entropy(y.reshape(-1,18),torch.tensor([0,5,1]),ignore_index=0).backward()
            torch.nn.utils.clip_grad_value_(m.parameters(),1)
            opt.step()
        opt=torch.optim.Adam(model.parameters(),lr=.001);step(model,opt);step(model,opt)
        opt2=torch.optim.Adam(model2.parameters(),lr=.001);step(model2,opt2)
        stream=io.BytesIO();torch.save(dict(model=model2.state_dict(),optimizer=opt2.state_dict()),stream);stream.seek(0)
        state=torch.load(stream,map_location='cpu',weights_only=True)
        model3,_=self.model('GRU');model3.load_state_dict(state['model'])
        opt3=torch.optim.Adam(model3.parameters(),lr=.001);opt3.load_state_dict(state['optimizer']);step(model3,opt3)
        for a,b in zip(model.parameters(),model3.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_convergence_requires_plateau_and_low_learning_rate(self):
        f=self.helper().convergence_status
        def row(score,lr=1e-5,complete=True):return dict(exact_match=score,learning_rate=lr,completed_epoch=complete,trained_batches_this_invocation=1)
        self.assertFalse(f([row(.99)]*12)['converged'])
        self.assertTrue(f([row(.99)]*13)['converged'])
        self.assertFalse(f([row(.99,.001)]*30)['converged'])
        self.assertFalse(f([row(.99)]*12+[row(.9902)])['converged'])
        self.assertFalse(f([row(.99,complete=False)]*30)['converged'])

if __name__=='__main__':unittest.main()
