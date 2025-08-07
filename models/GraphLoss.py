# ============================================================================
# GraphLoss.py  (variable‑length GT compatible)
# ============================================================================
from pytorch3d.loss import (
    chamfer_distance,
    mesh_laplacian_smoothing,
    mesh_edge_loss,
    mesh_normal_consistency,
)
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import torch
import torch.nn as nn


class MeshLoss(nn.Module):
    def __init__(
        self,
        *,
        weight_chamfer=1.0,
        num_samples=15000,
        weight_edge=0.5,
        weight_norm=0.02,
        weight_lapa=0.02,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.wc = weight_chamfer
        self.we = weight_edge
        self.wn = weight_norm
        self.wl = weight_lapa

    def forward(self, input, target):
        verts, faces = input                # Tensor(Np,3), Tensor(Fp,3)
        gt_verts = target                   # Tensor(Nt,3)
        mesh = Meshes(verts=[verts], faces=[faces])
        pts_pred = sample_points_from_meshes(mesh, num_samples=self.num_samples)

        cham = chamfer_distance(pts_pred, gt_verts[None])[0] * self.wc
        lapa = mesh_laplacian_smoothing(mesh) * self.wl
        edge = mesh_edge_loss(mesh) * self.we
        norm = mesh_normal_consistency(mesh) * self.wn
        return cham, lapa, edge, norm


class GraphLoss(nn.Module):
    """Aggregates MeshLoss over multi‑scale vertex predictions."""

    def __init__(self, **kwargs):
        super().__init__()
        self.base = MeshLoss(**kwargs)

    def forward(self, inputs, target):
        _, p_verts, p_faces = inputs  # lists over scales, each (1,Np,3)
        _, gt_verts = target          # Tensor(Nt,3)

        cham = lapa = edge = norm = 0.0
        for v, f in zip(p_verts, p_faces):
            c, l, e, n = self.base([v.squeeze(0), f.squeeze(0)], gt_verts)
            cham += c; lapa += l; edge += e; norm += n
        return cham + lapa + edge + norm


def create_loss(**kwargs):
    """Back‑compat wrapper so older scripts can `from GraphLoss import create_loss`."""
    return GraphLoss(**kwargs)

