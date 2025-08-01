# GraphLoss.py
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
        num_samples=5000,
        weight_edge=1.0,
        weight_norm=0.1,
        weight_lapa=0.1,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.wc = weight_chamfer
        self.we = weight_edge
        self.wn = weight_norm
        self.wl = weight_lapa

    def forward(self, input, target):
        """
        input  : [verts_b, faces_b]  (Tensor(Np,3), Tensor(Fp,3))
        target : gt verts            (Tensor(Nt,3))
        """
        verts, faces = input
        gt_verts = target

        mesh = Meshes(verts=[verts], faces=[faces])  # list 化が必須
        pts_pred = sample_points_from_meshes(mesh, num_samples=self.num_samples)

        chamfer = chamfer_distance(pts_pred, gt_verts[None])[0] * self.wc
        lapa    = mesh_laplacian_smoothing(mesh) * self.wl
        edge    = mesh_edge_loss(mesh)           * self.we
        norm    = mesh_normal_consistency(mesh)  * self.wn
        return chamfer, lapa, edge, norm


class GraphLoss(nn.Module):
    """
    - p_verts / p_faces : 各スケールとも shape = (1,Np,3) / (1,Fp,3)
    - gt_verts          : (Nt,3)  – 1 サンプル分のみ
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = MeshLoss(**kwargs)

    def forward(self, inputs, target):
        _, p_verts_list, p_faces_list = inputs
        _, gt_verts = target

        chamfer = lapa = edge = norm = 0.0
        for v, f in zip(p_verts_list, p_faces_list):
            # squeeze(0) : バッチ次元を落とす → (Np,3)
            c, l, e, n = self.criterion([v.squeeze(0), f.squeeze(0)], gt_verts)
            chamfer += c; lapa += l; edge += e; norm += n

        return chamfer + lapa + edge + norm

