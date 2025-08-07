import os
import argparse
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# -----------------------------------------------------------------------------
#  Local imports – make sure GraphSeg.py, GraphLoss.py and your dataset module
#  are on PYTHONPATH or in the same folder as this file.
# -----------------------------------------------------------------------------
from models.GraphSeg import create_model
from models.GraphLoss import create_loss  # returns GraphLoss (Chamfer + smoothness)
from dataset import VesselPatchDataset  # adapt the import if the filename differs

# -----------------------------------------------------------------------------
#  Simple 3‑level 3D encoder → feature pyramid that GraphSeg expects
#    f1 : [B, 128, 8 ,  8 ,  8 ]
#    f2 : [B,  64, 16, 16, 16]
#    f3 : [B,  32, 32, 32, 32]
# -----------------------------------------------------------------------------
class Encoder3D(nn.Module):
    """Minimal 3-level encoder that supplies feature volumes required by GraphSeg.
    Input patch is assumed to be 64×64×64 voxels (1 channel)."""

    def __init__(self, in_ch: int = 1):
        super().__init__()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_ch, 32)     # 64³ → 64³
        self.pool1 = nn.MaxPool3d(2)          # 64³ → 32³
        self.enc2 = conv_block(32, 64)        # 32³ → 32³
        self.pool2 = nn.MaxPool3d(2)          # 32³ → 16³
        self.enc3 = conv_block(64, 128)       # 16³ → 16³
        self.pool3 = nn.MaxPool3d(2)          # 16³ → 8³ (unused)

    def forward(self, x):
        f_low  = self.enc1(x)
        x      = self.pool1(f_low)
        f_mid  = self.enc2(x)
        x      = self.pool2(f_mid)
        f_high = self.enc3(x)
        _      = self.pool3(f_high)           # keeps RF
        return [f_high, f_mid, f_low]         # coarse→fine


class CoronaryGeoNet(nn.Module):
    """Encoder3D + GraphSeg = full vascular mesh generator"""

    def __init__(self):
        super().__init__()
        self.encoder   = Encoder3D()
        self.graph_seg = create_model()

    def forward(self, x):
        feats = self.encoder(x)
        verts_list, faces_list = self.graph_seg(feats)
        return verts_list, faces_list

# -----------------------------------------------------------------------------
#  Seed util & checkpoint helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state: dict, ckpt_dir: Path, is_best: bool = False):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / 'latest.pth')
    if is_best:
        torch.save(state, ckpt_dir / 'best.pth')

# -----------------------------------------------------------------------------
#  Variable-length collate_fn  ——  vols stacked, points kept as list
# -----------------------------------------------------------------------------

def collate_varlen(batch):
    vols, pts = zip(*batch)                                # tuples length B
    vols = torch.from_numpy(np.stack(vols))                # [B,1,64,64,64]
    pts  = [torch.from_numpy(p) for p in pts]              # list[Tensor(N_i,3)]
    return vols, pts

# -----------------------------------------------------------------------------
#  Train / Val loops  ——  iterate over each sample inside the batch
# -----------------------------------------------------------------------------

def train_one_epoch(model, criterion, loader, optimizer, device):
    model.train()
    running = 0.0

    for vols, gt_list in tqdm(loader, desc='Train', leave=False):
        vols = vols.to(device, dtype=torch.float32)
        pred_verts_list, pred_faces_list = model(vols)     # lists over scales

        # accumulate loss over samples (variable-length GT)
        loss = 0.0
        for b, gt_verts in enumerate(gt_list):
            gt_verts = gt_verts.to(device, dtype=torch.float32)
            pv = [v[b:b+1] for v in pred_verts_list]
            pf = [f[b:b+1] for f in pred_faces_list]
            loss += criterion((None, pv, pf), (None, gt_verts))
        loss = loss / vols.size(0)                         # mean over batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * vols.size(0)

    return running / len(loader.dataset)


def validate(model, criterion, loader, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for vols, gt_list in tqdm(loader, desc='Val  ', leave=False):
            vols = vols.to(device, dtype=torch.float32)
            pred_verts_list, pred_faces_list = model(vols)
            loss = 0.0
            for b, gt_verts in enumerate(gt_list):
                gt_verts = gt_verts.to(device, dtype=torch.float32)
                pv = [v[b:b+1] for v in pred_verts_list]
                pf = [f[b:b+1] for f in pred_faces_list]
                loss += criterion((None, pv, pf), (None, gt_verts))
            loss = loss / vols.size(0)
            running += loss.item() * vols.size(0)
    return running / len(loader.dataset)

# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.get('seed', 42))

    train_ds = VesselPatchDataset(cfg['json_path'], cfg['train_indexes'], isTrain=True)
    val_ds   = VesselPatchDataset(cfg['json_path'], cfg['val_indexes'],   isTrain=False)
    

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True,
                              drop_last=False, collate_fn=collate_varlen)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'], pin_memory=True,
                              collate_fn=collate_varlen)
    print("data is loaded!!")

    model      = CoronaryGeoNet().to(device)
    print("model is loaded!!")
    
    criterion  = create_loss()
    optimizer  = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["epochs"])

    best_val   = float('inf')
    ckpt_dir   = Path(cfg['output_dir'])

    for epoch in range(1, cfg['epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")
        tr_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)
        val_loss = validate(model, criterion, val_loader, device)
        scheduler.step()

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val': best_val,
        }, ckpt_dir, is_best)

        print(f"  Train {tr_loss:.4f} | Val {val_loss:.4f} | Best {best_val:.4f}")

# -----------------------------------------------------------------------------
#  CLI wrapper
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    default_cfg = {
        'json_path': 'data.json',
        'train_indexes': [1,2,3,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
        'val_indexes':   [1,2,3],
        'batch_size': 4,
        'num_workers': 4,
        'lr': 3e-5,
        'epochs': 30,
        'seed': 42,
        'output_dir': './checkpoints/' + datetime.now().strftime('%Y%m%d-%H%M%S'),
    }

    if args.config:
        with open(args.config) as fp:
            user_cfg = json.load(fp)
        default_cfg.update(user_cfg)

    main(default_cfg)
