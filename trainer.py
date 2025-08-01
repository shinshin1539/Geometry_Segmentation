import os
import argparse
import json
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
    """Minimal 3‑level encoder that supplies the feature volumes required by
    GraphSeg. Input volumes are expected to be 64×64×64 voxels. If your patch
    size is different, adjust the `pool` strides accordingly or swap in a more
    powerful backbone (e.g. 3D UNet, nnUNet encoder, etc.)."""

    def __init__(self, in_ch=1):
        super().__init__()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_ch, 32)   # (64³) → (64³)
        self.pool1 = nn.MaxPool3d(2)        # (64³) → (32³)
        self.enc2 = conv_block(32, 64)      # (32³) → (32³)
        self.pool2 = nn.MaxPool3d(2)        # (32³) → (16³)
        self.enc3 = conv_block(64, 128)     # (16³) → (16³)
        self.pool3 = nn.MaxPool3d(2)        # (16³) → (8³)

    def forward(self, x):
        f_low = self.enc1(x)
        x = self.pool1(f_low)
        f_mid = self.enc2(x)
        x = self.pool2(f_mid)
        f_high = self.enc3(x)
        x = self.pool3(f_high)  # final pooling just to keep receptive‑field, not used

        # return pyramid coarse→fine, matching GraphSeg order
        #  [128×8³, 64×16³, 32×32³]
        return [f_high, f_mid, f_low]


class CoronaryGeoNet(nn.Module):
    """End‑to‑end model = 3D encoder + GraphSeg."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D()
        self.graph_seg = create_model()  # default dims: (coords=3, hidden=192)

    def forward(self, x):
        feats = self.encoder(x)            # list of 3 feature maps
        verts_list, faces_list = self.graph_seg(feats)
        return verts_list, faces_list


# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, checkpoint_dir: Path, is_best: bool = False):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = checkpoint_dir / 'latest.pth'
    torch.save(state, filename)
    if is_best:
        best_name = checkpoint_dir / 'best.pth'
        torch.save(state, best_name)


# -----------------------------------------------------------------------------
#  Training & validation loops
# -----------------------------------------------------------------------------

def train_one_epoch(model, criterion, loader, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for volume, gt_verts in tqdm(loader, desc='Train', leave=False):
        # Adjust this unpacking according to your Dataset.__getitem__
        volume = volume.to(device, dtype=torch.float32)        # [B, 1, 64,64,64]
        gt_verts = gt_verts.to(device, dtype=torch.float32)    # [B, N, 3]

        optimizer.zero_grad()
        pred_verts, pred_faces = model(volume)                # lists length = 3
        loss = criterion((None, pred_verts, pred_faces), (None, gt_verts))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * volume.size(0)

    return epoch_loss / len(loader.dataset)


def validate(model, criterion, loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for volume, gt_verts in tqdm(loader, desc='Val  ', leave=False):
            volume = volume.to(device, dtype=torch.float32)
            gt_verts = gt_verts.to(device, dtype=torch.float32)

            pred_verts, pred_faces = model(volume)
            loss = criterion((None, pred_verts, pred_faces), (None, gt_verts))
            val_loss += loss.item() * volume.size(0)

    return val_loss / len(loader.dataset)


# -----------------------------------------------------------------------------
#  Main entry
# -----------------------------------------------------------------------------

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.get('seed', 42))

    # ---------------------------------------------------------------------
    #  Dataset & loaders – customise VesselPatchDataset as needed
    # ---------------------------------------------------------------------
    train_ds = VesselPatchDataset(cfg['json_path'], cfg['train_indexes'], isTrain=True)
    val_ds = VesselPatchDataset(cfg['json_path'], cfg['val_indexes'], isTrain=False)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True)

    # ---------------------------------------------------------------------
    #  Model, loss, optimiser, scheduler
    # ---------------------------------------------------------------------
    model = CoronaryGeoNet().to(device)
    criterion = create_loss()

    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg['t0'], T_mult=1)

    # ---------------------------------------------------------------------
    #  Training loop
    # ---------------------------------------------------------------------
    best_val = float('inf')
    checkpoint_dir = Path(cfg['output_dir'])

    for epoch in range(1, cfg['epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)
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
        }, checkpoint_dir, is_best)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GraphSeg based coronary model')
    parser.add_argument('--config', type=str, required=False, default=None,
                        help='Path to a JSON config. If omitted, default params are used.')
    args = parser.parse_args()

    # Default hyper‑parameters – override via JSON file for experiments
    default_cfg = {
        'json_path': 'data/dataset.json',   # dataset split description
        'train_indexes': list(range(0, 160)),
        'val_indexes':   list(range(160, 200)),
        'batch_size': 4,
        'num_workers': 4,
        'lr': 1e-4,
        'epochs': 200,
        't0': 20,                 # Cosine scheduler period
        'seed': 42,
        'output_dir': './checkpoints/' + datetime.now().strftime('%Y%m%d-%H%M%S'),
    }

    if args.config is not None:
        with open(args.config, 'r') as fp:
            user_cfg = json.load(fp)
        default_cfg.update(user_cfg)

    main(default_cfg)
