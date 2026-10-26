"""Reproduce the single controlled V3 CNN experiment (fresh training, seed 42).

Run from the repository root with its training dependencies installed. Writes
rank_classifier_v3/ and never overwrites published models. Source-round grouping
is documented in VALIDATION.md. The reserved round is used only for final reporting.
"""
import sys,random,re,json,csv
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import train_rank_classifier_v2 as old
from rank_model import RankCNN,preprocess_crop,RANKS
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)
out=ROOT/'rank_classifier_v3';out.mkdir(exist_ok=True)
rows=[]
# Exclude clipped 10 examples identified in the raw-crop audit.
for s in old.load_all_samples():
 if s['path'].name.startswith(('candidate_0022_','candidate_0026_')):continue
 f=int(re.search(r'(\d{6})_\d+\.png$',s['path'].name).group(1))
 rows.append(dict(key=s['path'].name,crop=cv2.imread(str(s['path'])),rank=s['rank'],frame=f,group='round2' if 90<=f<=160 else ('round1' if f<80 else 'round3' if 180<=f<270 else s['group'])))
fixture=json.loads((ROOT/'tests/fixtures/card_tracking.json').read_text())
with np.load(ROOT/'tests/fixtures/card_tracking_crops.npz', allow_pickle=False) as crops:
 for r in fixture['records']:
  if not r['identity']:continue
  rank=r['identity'].split('_')[-1]
  if rank not in RANKS:continue
  f=int(r['frame'][-10:-4]);group=r['identity'].split('_')[0]
  rows.append(dict(key=r['key'],crop=crops[r['key']],rank=rank,frame=f,group={'r1':'round1','r2':'round2','r3':'round3'}[group]))
# Entire reviewed second round stays out of both optimization and checkpoint selection.
test=[r for r in rows if r['group']=='round2'];pool=[r for r in rows if r['group']!='round2']
groups=sorted(set(r['group'] for r in pool));random.Random(42).shuffle(groups)
vg=set(groups[:max(1,round(.2*len(groups)))]);train=[r for r in pool if r['group'] not in vg];val=[r for r in pool if r['group'] in vg]
print('split',len(train),len(val),len(test),'valgroups',sorted(vg),flush=True)
print('train ranks',dict(Counter(r['rank'] for r in train)),flush=True)
class Data(Dataset):
 def __init__(self,rows,augment):self.rows=rows;self.augment=augment
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];tensor=preprocess_crop(r['crop'])[0]
  if self.augment:
   gray=((1-tensor[0].numpy())*255).astype('uint8');gray=old.augment_gray(gray);tensor=torch.from_numpy((255-gray.astype('float32'))/255).unsqueeze(0)
  return tensor,RANKS.index(r['rank'])
model=RankCNN();optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
counts=Counter(r['rank'] for r in train);weights=torch.tensor([1/max(counts[r],1)**.5 for r in RANKS]);lossfn=torch.nn.CrossEntropyLoss(weight=weights,label_smoothing=.05)
loader=DataLoader(Data(train,True),batch_size=16,shuffle=True)
def evaluate(rows,net):
 net.eval();images=torch.cat([preprocess_crop(r['crop']) for r in rows]);
 with torch.no_grad():probs=net(images).softmax(1)
 pred=probs.argmax(1).tolist();return sum(RANKS[p]==r['rank'] for p,r in zip(pred,rows)),pred,probs
best=-1
for epoch in range(60):
 model.train()
 for x,y in loader:
  optim.zero_grad();loss=lossfn(model(x),y);loss.backward();optim.step()
 good,_,_=evaluate(val,model)
 if good>best:
  best=good;torch.save(dict(model_state_dict=model.state_dict(),ranks=RANKS,image_size=64,epoch=epoch+1,preprocessing='cnn_v2',training='fresh seed 42, 60 epochs, global source groups; reviewed round2 excluded'),out/'candidate.pt')
 if epoch%10==0:print('epoch',epoch+1,'val',good,len(val),flush=True)
model.load_state_dict(torch.load(out/'candidate.pt',weights_only=True)['model_state_dict'])
for name,path in [('baseline',ROOT/'models/rank_classifier_v2.pt'),('candidate',out/'candidate.pt')]:
 net=RankCNN();net.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['model_state_dict']);good,pred,probs=evaluate(test,net)
 print(name,'reserved round',good,len(test),flush=True)
 results=[dict(key=r['key'],actual=r['rank'],predicted=RANKS[p],confidence=float(prob.max())) for r,p,prob in zip(test,pred,probs)]
 (out/(name+'_reserved.json')).write_text(json.dumps(results,indent=2))
(out/'split.json').write_text(json.dumps({k:[{a:b for a,b in r.items() if a!='crop'} for r in v] for k,v in [('train',train),('validation',val),('test',test)]},indent=2))
