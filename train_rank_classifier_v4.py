"""Controlled follow-up: V3's fixed split plus 15 reviewed Oct-26 crops.

Same CNN, preprocessing, seed and 60-epoch budget. Published models are never
modified by this script. Oct-26 is development/training data after this change.
"""
import json
import random
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rank_model import RankCNN, RANKS, preprocess_crop
from train_rank_classifier_v2 import augment_gray

ROOT = Path(__file__).resolve().parent


def load_data():
    split = json.loads((ROOT/'validation/v3_training_split.json').read_text())
    fixture_dir = ROOT/'tests/fixtures'
    with np.load(fixture_dir/'card_tracking_crops.npz', allow_pickle=False) as file:
        historical = {key: file[key] for key in file.files}
    with np.load(fixture_dir/'oct26_round_crops.npz', allow_pickle=False) as file:
        latest = {key: file[key] for key in file.files}
    def read(row):
        key = row['key']
        if key in historical:
            crop = historical[key]
        else:
            paths = [ROOT/'rank_value_dataset'/key, ROOT/'rank_value_candidates'/key]
            path = next((p for p in paths if p.is_file()), None)
            crop = cv2.imread(str(path)) if path else None
        if crop is None:
            raise ValueError('Missing training crop: ' + key)
        return dict(row, crop=crop)
    data = {part: [read(row) for row in rows] for part,rows in split.items()}
    fixture = json.loads((fixture_dir/'oct26_round.json').read_text())
    selected = set(fixture['training_keys'])
    data['oct26'] = [dict(key=row['key'],rank=row['rank'],frame=row['sample_index'],
                         group='oct26_round',crop=latest[row['key']]) for row in fixture['records']]
    data['train'].extend(row for row in data['oct26'] if row['key'] in selected)
    assert all(row['group'] != 'round2' for row in data['train'] + data['validation'])
    return data


class RankData(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        tensor = preprocess_crop(row['crop'])[0]
        gray = ((1 - tensor[0].numpy()) * 255).astype(np.uint8)
        gray = augment_gray(gray)
        tensor = torch.from_numpy((255 - gray.astype(np.float32)) / 255).unsqueeze(0)
        return tensor, RANKS.index(row['rank'])


def evaluate(model, rows):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start+32]
            tensors = torch.cat([preprocess_crop(row['crop']) for row in batch])
            probabilities = model(tensors).softmax(1)
            for row, p in zip(batch, probabilities):
                predicted = RANKS[int(p.argmax())]
                output.append(dict(key=row['key'],actual=row['rank'],predicted=predicted,
                                   confidence=float(p.max())))
    return sum(row['actual'] == row['predicted'] for row in output), output


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    data = load_data()
    out = ROOT/'rank_classifier_v4'
    out.mkdir(exist_ok=True)
    model = RankCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    counts = Counter(row['rank'] for row in data['train'])
    weights = torch.tensor([1 / max(counts[rank],1)**.5 for rank in RANKS])
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)
    loader = DataLoader(RankData(data['train']), batch_size=16, shuffle=True)
    best = -1
    print('split', {key: len(rows) for key,rows in data.items()}, flush=True)
    for epoch in range(60):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
        correct, _ = evaluate(model, data['validation'])
        if correct > best:
            best = correct
            torch.save(dict(model_state_dict=model.state_dict(),ranks=RANKS,image_size=64,
                epoch=epoch+1,preprocessing='cnn_v2',training='Fresh seed 42; V3 fixed split + 15 Oct26 training crops; round2 excluded'),out/'candidate.pt')
        if epoch % 10 == 0:
            print('epoch', epoch+1, 'validation', correct, len(data['validation']), flush=True)
    report = {}
    for name,path in [('v3',ROOT/'models/rank_classifier_v3.pt'),('v4',out/'candidate.pt')]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        report[name] = {}
        for part in ['test','oct26']:
            correct, predictions = evaluate(model, data[part])
            print(name, part, correct, len(predictions), flush=True)
            report[name][part] = dict(correct=correct,total=len(predictions),predictions=predictions)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest = {part: [{k:v for k,v in row.items() if k!='crop'} for row in rows]
                for part,rows in data.items()}
    (out/'split.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
