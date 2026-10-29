"""Train a rank candidate from explicitly audited development crops.

Original V4 training/selection groups stay fixed. New recording groups are
training-only and must be listed explicitly; reserved recordings must not be
provided. Outputs go to the requested directory; app weights are not replaced.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from rank_model import RankCNN,RANKS
from train_rank_classifier_v4 import load_data,RankData,evaluate


def load_reviewed(folder, allowed_groups):
    folder=Path(folder).resolve();rows=[];seen=set();cards={}
    with (folder/'labels.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            group=row['recording_group'];card=row['card_group'];rank=row['rank']
            if group not in allowed_groups or rank not in RANKS:
                raise ValueError('Unapproved recording group or invalid rank.')
            if row.get('annotation_source') not in ['visual_frame_audit','user_confirmed_review']:
                raise ValueError('Explicitly audited labels are required.')
            if card in cards and cards[card]!=(group,rank):
                raise ValueError('One physical card has conflicting labels/groups.')
            cards[card]=(group,rank)
            path=(folder/row['crop']).resolve()
            if not path.is_relative_to(folder):
                raise ValueError('Crop path leaves the labelled folder.')
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest!=row['sha256']:
                raise ValueError('Crop checksum mismatch.')
            if (card,digest) in seen:
                continue
            seen.add((card,digest));crop=cv2.imread(str(path))
            if crop is None:
                raise ValueError('Unreadable crop.')
            rows.append(dict(key=str(len(rows)),rank=rank,group=group,card_group=card,crop=crop))
    if not rows or set(r['group'] for r in rows)!=set(allowed_groups):
        raise ValueError('Every allowed development group must have labelled crops.')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed',type=Path,required=True)
    parser.add_argument('--development-groups',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)
    data=load_data();extra=load_reviewed(args.reviewed,set(args.development_groups))
    if set(r['group'] for r in extra)&set(r['group'] for r in data['validation']+data['test']):
        raise ValueError('New training groups overlap historical evaluation groups.')
    training=data['train']+extra;args.output.mkdir(parents=True,exist_ok=False)
    counts=Counter(r['rank'] for r in training)
    model=RankCNN();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor([1/max(counts[r],1)**.5 for r in RANKS]),label_smoothing=.05)
    loader=DataLoader(RankData(training),batch_size=16,shuffle=True)
    best=(-1,float('-inf'));history=[]
    print('training',len(training),'new',len(extra),'selection',len(data['validation']),flush=True)
    for epoch in range(60):
        model.train()
        for images,labels in loader:
            opt.zero_grad();loss=loss_fn(model(images),labels);loss.backward();opt.step()
        correct,predictions=evaluate(model,data['validation'])
        # Tie-break on held selection-set log loss, never on new recordings.
        from rank_model import preprocess_crop
        with torch.no_grad():
            logits=model(torch.cat([preprocess_crop(r['crop']) for r in data['validation']]))
            selection_loss=float(torch.nn.functional.cross_entropy(logits,torch.tensor([RANKS.index(r['rank']) for r in data['validation']])))
        score=(correct,-selection_loss)
        if score>best:
            best=score
            torch.save(dict(model_state_dict=model.state_dict(),ranks=RANKS,image_size=64,epoch=epoch+1,preprocessing='cnn_v2',training='Fresh seed 42; fixed V4 training + audited development groups; reserved recordings excluded'),args.output/'candidate.pt')
        history.append(dict(epoch=epoch+1,selection_correct=correct,selection_log_loss=selection_loss))
        if epoch%10==0:print('epoch',epoch+1,'selection',correct,'loss',round(selection_loss,4),flush=True)
    checkpoint=torch.load(args.output/'candidate.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model_state_dict'])
    report=dict(training_crops=len(training),historical_training_crops=len(data['train']),new_training_crops=len(extra),new_physical_cards=len(set(r['card_group'] for r in extra)),development_groups=sorted(args.development_groups),rank_counts=dict(counts),selected_epoch=checkpoint['epoch'],selection_rank_counts=dict(Counter(r['rank'] for r in data['validation'])),history=history,limitations=['Development/training and correlated crop observations are not independent accuracy measurements.','Historical selection set is small and lacks six ranks. Reserved recordings were not loaded.'])
    report['diagnostics']={}
    for part,rows in [('historical_test',data['test']),('oct26',data['oct26']),('new_training',extra)]:
        correct,predictions=evaluate(model,rows);report['diagnostics'][part]=dict(correct=correct,total=len(rows));print(part,correct,len(rows),flush=True)
    report['sha256']=hashlib.sha256((args.output/'candidate.pt').read_bytes()).hexdigest()
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
