
import os
import random
from typing import Sequence, Tuple, List

import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

from utils import get_json, get_csv  # project‑specific helpers


class VesselPatchDataset(Dataset):
    """Dataset that returns a (C=1,D,H,W) patch and normalised mesh points."""

    # voxel dimensions of the resampled cube in (D,Z)‑first ndarray order
    patch_size: Tuple[int, int, int] = (64, 64, 64)  # (Z, Y, X)

    # physical edge length of the crop cube in millimetres (isotropic)
    actual_patch_size: Tuple[float, float, float] = (32.0, 32.0, 32.0)  # mm

    # target isotropic voxel spacing after resample (X, Y, Z)
    target_spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # mm

    # ------------------------------------------------------------------
    # init / bookkeeping
    # ------------------------------------------------------------------
    def __init__(self, json_path: str, indexes: Sequence[int], isTrain: bool = True):
        super().__init__()
        cfg = get_json(json_path)

        self.isTrain = isTrain
        self.images: List[str] = []
        self.centerlines: List[str] = []
        self.meshes: List[str] = []

        for idx in indexes:
            for item in cfg[idx]:
                self.images.append(os.path.join(cfg['dir'], item['image']))
                self.centerlines.append(os.path.join(cfg['dir'], item['centerline']))
                self.meshes.append(os.path.join(cfg['dir'], item['mesh']))

    def __len__(self) -> int:
        return len(self.images)

    # ------------------------------------------------------------------
    # data‑augmentation helpers (operate on ndarray (Z,Y,X) & pts (Z,Y,X))
    # ------------------------------------------------------------------
    def _random_flip(self, vol: np.ndarray, pts: np.ndarray, axis_prob: float = 0.5):
        for ax in range(3):  # 0‑Z, 1‑Y, 2‑X
            if random.random() < axis_prob:
                vol = np.flip(vol, axis=ax).copy()
                pts[:, ax] = 1.0 - pts[:, ax]
        return vol, pts

    def _random_rotate90(self, vol: np.ndarray, pts: np.ndarray, axis_prob: float = 0.5):
        for ax in range(3):
            if random.random() < axis_prob:
                nxt = (ax + 1) % 3
                vol = np.rot90(vol, axes=(ax, nxt)).copy()
                pts[:, [ax, nxt]] = pts[:, [nxt, ax]]
        return vol, pts

    # ------------------------------------------------------------------
    # resampling & cropping utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _resample_patch_ndarray(
        arr: np.ndarray,
        cur_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float],
        is_label: bool,
        cur_origin: Tuple[float, float, float],
        direction=None,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Resample `arr` (Z,Y,X) to `target_spacing` using SimpleITK."""
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.SetSpacing(cur_spacing)  # (X, Y, Z)
        img.SetOrigin(cur_origin)
        if direction is not None:
            img.SetDirection(direction)

        size_xyz = np.array(img.GetSize())
        spacing_xyz = np.array(img.GetSpacing())
        target_xyz = np.array(target_spacing)
        new_size = np.round(size_xyz * spacing_xyz / target_xyz).astype(int).tolist()

        res = sitk.ResampleImageFilter()
        res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        res.SetOutputSpacing(target_xyz.tolist())
        res.SetSize(new_size)
        res.SetOutputOrigin(img.GetOrigin())
        res.SetOutputDirection(img.GetDirection())

        img_res = res.Execute(img)
        return sitk.GetArrayFromImage(img_res), img_res.GetSpacing()

    def _random_crop(self, itk_img: sitk.Image, centerline_vox: np.ndarray):
        """
        * 画像の端をはみ出さないよう index をクリップ
        * 中心線が小枝でパッチが小さい時も OK
        """
        # ------- 1) ランダム中心を voxel で取得 -------
        cz, cy, cx = map(int, random.choice(centerline_vox))

        # ------- 2) 必要なパッチ寸法 (voxel) -------
        sx, sy, sz = itk_img.GetSpacing()        # (x,y,z)
        pw = int(round(self.actual_patch_size[2] / sx))
        ph = int(round(self.actual_patch_size[1] / sy))
        pd = int(round(self.actual_patch_size[0] / sz))

        # 画像サイズ (voxel)
        W, H, D = itk_img.GetWidth(), itk_img.GetHeight(), itk_img.GetDepth()

        # ------- 3) index をクリップ -------
        x0 = np.clip(cx - pw // 2, 0, max(W - pw, 0))
        y0 = np.clip(cy - ph // 2, 0, max(H - ph, 0))
        z0 = np.clip(cz - pd // 2, 0, max(D - pd, 0))

        roi = sitk.RegionOfInterest(
            itk_img,
            size=[pw, ph, pd],
            index=[int(x0), int(y0), int(z0)],
        )
        crop_arr   = sitk.GetArrayFromImage(roi)          # (Z,Y,X)
        origin_mm  = itk_img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0)))

        # ------- 4) リサンプリング -------
        res_crop, _ = self._resample_patch_ndarray(
            crop_arr,
            cur_spacing=(sx, sy, sz),
            target_spacing=self.target_spacing,
            is_label=True,
            cur_origin=origin_mm,
        )
        return res_crop, origin_mm

    def _crop_resample_normalise_pts_mm(self, verts_mm: np.ndarray, origin_mm):
        """Map mesh verts (mm, order Z,X,Y) into [0,1] cube coordinates (Z,Y,X)."""
        ox, oy, oz = origin_mm  # careful: origin_mm is (x,y,z)
        wx, wy, wz = self.actual_patch_size  # physical cube extents

        inside = (
            (verts_mm[:, 0] >= ox) & (verts_mm[:, 0] < ox + wx) &  # Z range
            (verts_mm[:, 1] >= oy) & (verts_mm[:, 1] < oy + wy) &  # X range
            (verts_mm[:, 2] >= oz) & (verts_mm[:, 2] < oz + wz)    # Y range
        )
        pts = verts_mm[inside]
        if pts.size == 0:
            print("Warning: no mesh points inside patch, returning empty array.")
            return np.zeros((0, 3), np.float32)

        local_mm = pts - np.array([oz, ox, oy], dtype=np.float32)
        pts_vox = local_mm / np.array(self.target_spacing, dtype=np.float32)

        pd, ph, pw = self.patch_size  # (Z,Y,X)
        pts_norm = pts_vox / np.array([pd, ph, pw], dtype=np.float32)
        return pts_norm.astype(np.float32)

    # ------------------------------------------------------------------
    # main fetch
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        itk_img = sitk.ReadImage(self.images[idx])
        centerline_vox = np.loadtxt(self.centerlines[idx], dtype=np.float32)
        verts_mm = np.loadtxt(self.meshes[idx], dtype=np.float32)

        patch, origin_mm = self._random_crop(itk_img, centerline_vox)
        points = self._crop_resample_normalise_pts_mm(verts_mm, origin_mm)

        if self.isTrain:
            patch, points = self._random_flip(patch, points)
            patch, points = self._random_rotate90(patch, points)

        patch = patch[np.newaxis, ...].astype(np.float32)  # add channel dim (1,D,H,W)
        return patch, points

train_ds = VesselPatchDataset("data.json", indexes=["1", "2"], isTrain=False)