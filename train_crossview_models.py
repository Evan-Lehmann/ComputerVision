"""Train development-only suit and compressed-rank readers from audited CSV crops.

The primary V5 weights are never modified. This needs local private development
crops; do not supply acceptance recordings. Saves counts/groups and checkpoints.
"""
import argparse,csv,hashlib,json,random
from pathlib import Path
from collections import Counter
import cv2,numpy as np,torch
from torch.utils.data import DataLoader,Dataset
from card_suits import SuitCNN,SUITS,suit_tensor
from rank_model import RankCNN,RANKS,preprocess_crop
from train_rank_classifier_v4 import load_data,evaluate
from train_rank_classifier_v5 import load_reviewed


def load_audited(folder):
    rows=[];folder=Path(folder).resolve();labels={}
    with (folder/'labels.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            kind=row.get('kind');label=row.get('label');group=row.get('card_group','').strip()
            if kind not in ['rank','suit'] or label not in (RANKS if kind=='rank' else SUITS) or not group:
                raise ValueError('Choose an audited rank/suit and a physical-card group.')
            key=(kind,group)
            if key in labels and labels[key]!=label:raise ValueError('Conflicting labels for a physical card.')
            labels[key]=label
            path=(folder/row['crop']).resolve()
            if not path.is_relative_to(folder):raise ValueError('Crop path leaves audited folder.')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Checksum mismatch.')
            crop=cv2.imread(str(path))
            if crop is None:raise ValueError('Unreadable audited crop.')
            rows.append(dict(row,crop=crop))
    if not rows:raise ValueError('No audited crops found.')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audited',type=Path,required=True);parser.add_argument('--rank-development',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--suit-only',action='store_true');parser.add_argument('--suit-selection-groups',nargs='+',default=['r1:7','r2:5','r2:7','r1:6']);args=parser.parse_args()
    random.seed(73);np.random.seed(73);torch.manual_seed(73);torch.set_num_threads(2)
    rows=load_audited(args.audited)
    args.output.mkdir(parents=True,exist_ok=False)
    suit_rows=[r for r in rows if r['kind']=='suit' and r['label'] in SUITS and suit_tensor(r['crop']) is not None]
    selection_groups=set(args.suit_selection_groups)
    train=[r for r in suit_rows if r['card_group'] not in selection_groups];val=[r for r in suit_rows if r['card_group'] in selection_groups]
    assert set(r['label'] for r in val)==set(SUITS)
    optimization=train+[r for r in train if ':mini' in r['card_group'] or '_slot' in r['card_group']]*4
    class Suits(Dataset):
        def __len__(self):return len(optimization)
        def __getitem__(self,i):
            r=optimization[i];tensor=suit_tensor(r['crop'])[0][0]
            # Small changes preserve icon geometry and colour metadata.
            a=tensor[0].numpy();angle=random.uniform(-4,4);m=cv2.getRotationMatrix2D((16,16),angle,random.uniform(.9,1.1));a=cv2.warpAffine(a,m,(32,32))
            return torch.from_numpy(a).unsqueeze(0),SUITS.index(r['label'])
    model=SuitCNN();opt=torch.optim.AdamW(model.parameters(),lr=.001);loader=DataLoader(Suits(),batch_size=24,shuffle=True);best=(-1,-float('inf'));epoch_selected=0
    for epoch in range(80):
        model.train()
        for x,y in loader:
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y,label_smoothing=.03);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            x=torch.cat([suit_tensor(r['crop'])[0] for r in val]);y=torch.tensor([SUITS.index(r['label']) for r in val]);p=model(x);correct=int((p.argmax(1)==y).sum());vl=float(torch.nn.functional.cross_entropy(p,y))
        if (correct,-vl)>best:
            best=(correct,-vl);epoch_selected=epoch+1;torch.save(dict(model_state_dict=model.state_dict(),suits=SUITS),args.output/'suit_classifier_v1.pt')
    print('suit',len(train),len(val),best,flush=True)
    if args.suit_only:
        report=dict(suit_seed=73,suit_training_crops=len(train),suit_optimization_crops=len(optimization),new_round_mini_suit_training_weight=5,older_mini_suit_training_weight=5,suit_selection_crops=len(val),suit_selection_groups=sorted(selection_groups),suit_selection_correct=best[0],suit_selected_epoch=epoch_selected)
        report['suit_classifier_sha256']=hashlib.sha256((args.output/'suit_classifier_v1.pt').read_bytes()).hexdigest()
        (args.output/'suit_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        return
    random.seed(74);np.random.seed(74);torch.manual_seed(74)
    base=load_data();extra=load_reviewed(args.rank_development,{'new_r1','new_r2','new_r3'})
    mini=[dict(key='mini_'+str(i),rank=r['label'],group=r['card_group'],crop=r['crop']) for i,r in enumerate(rows) if r['kind']=='rank' and r['label'] in RANKS]
    training=base['train']+extra+mini*8
    class Ranks(Dataset):
        def __len__(self):return len(training)
        def __getitem__(self,i):
            r=training[i];crop=r['crop']
            if not r['key'].startswith('mini_') and random.random()<.6:
                crop=cv2.resize(crop,(max(6,round(crop.shape[1]*random.uniform(.4,.7))),max(7,round(crop.shape[0]*random.uniform(.3,.6)))),interpolation=cv2.INTER_AREA)
            tensor=preprocess_crop(crop)[0]
            return tensor,RANKS.index(r['rank'])
    rank=RankCNN();rank.load_state_dict(torch.load(Path(__file__).parent/'models/rank_classifier_v5.pt',map_location='cpu',weights_only=True)['model_state_dict']);opt=torch.optim.AdamW(rank.parameters(),lr=.0003);loader=DataLoader(Ranks(),batch_size=20,shuffle=True);best_rank=(-1,-float('inf'));selected=0
    for epoch in range(40):
        rank.train()
        for x,y in loader:
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(rank(x),y,label_smoothing=.03);loss.backward();opt.step()
        correct,_=evaluate(rank,base['validation'])
        with torch.no_grad():
            x=torch.cat([preprocess_crop(r['crop']) for r in base['validation']]);y=torch.tensor([RANKS.index(r['rank']) for r in base['validation']]);vl=float(torch.nn.functional.cross_entropy(rank(x),y))
        if (correct,-vl)>best_rank:
            best_rank=(correct,-vl);selected=epoch+1;torch.save(dict(model_state_dict=rank.state_dict(),ranks=RANKS,image_size=64),args.output/'rank_classifier_mini_v1.pt')
        if epoch%10==0:print('mini epoch',epoch+1,correct,flush=True)
    rank.load_state_dict(torch.load(args.output/'rank_classifier_mini_v1.pt',map_location='cpu',weights_only=True)['model_state_dict']);correct,_=evaluate(rank,mini)
    report=dict(suit_seed=73,mini_seed=74,suit_training_crops=len(train),suit_optimization_crops=len(optimization),new_round_mini_suit_training_weight=5,older_mini_suit_training_weight=5,suit_selection_crops=len(val),suit_selection_groups=sorted(selection_groups),suit_selection_correct=best[0],suit_selected_epoch=epoch_selected,mini_real_crops=len(mini),mini_unrepresented_rank_labels=sorted(set(RANKS)-{r['rank'] for r in mini}),mini_rank_counts=dict(Counter(r['rank'] for r in mini)),mini_physical_slot_groups=len(set(r['group'] for r in mini)),mini_selected_epoch=selected,mini_selection_correct=best_rank[0],mini_training_diagnostic_correct=correct,limitations=['r1-r5 are now development data. No independent acceptance set is used here.','Real miniature crops do not cover every rank; missing labels rely on historical and simulated downsampling examples.','Repeated views/printed corners are correlated; diagnostics are not general accuracy.'])
    report['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.pt')};(args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
