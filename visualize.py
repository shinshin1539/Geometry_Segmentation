# Visual check of mesh.xyz and centerline.txt
#
# ⚙️ 使い方
# 1. `case_dir` を “mesh.xyz” と “centerline.txt” が入っている症例フォルダに書き換えてください。
#    └ 例: Path("./data/CoronaryArtery/case_1")
# 2. 実行すると 3D プロットが表示されます。
#    - 灰点: メッシュ (ランダム抽出 5k 点)
#    - 赤線: 中心線
#
# 前提:
# * mesh.xyz … space 区切り (x y z) [mm]   ← LPS で保存済み
# * centerline.txt … space 区切り (z y x) voxel index
# * lab.nii.gz … 同フォルダに存在 (中心線を world 座標へ変換するため)

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- ユーザー設定 ----------------
case_dir = Path("./data/CoronaryArtery/case_2")   # ←必要に応じ変更
# ---------------------------------------------

mesh_path = case_dir / "mesh.xyz"
cl_path = case_dir / "centerline.xyz"
nii_path = case_dir / "seg.nii.gz"

# --- load mesh ---
mesh = np.loadtxt(mesh_path, dtype=np.float32)
if mesh.shape[0] > 5000:                       # down‑sample for speed
    idx = np.random.choice(mesh.shape[0], 5000, replace=False)
    mesh = mesh[idx]

# --- load centerline (voxel) & convert to world‑mm(LPS) ---
cl_vox = np.loadtxt(cl_path, dtype=np.int32)   # (z,y,x)
img = nib.load(str(nii_path))
aff = img.affine                               # voxel→RAS
# voxel (i,j,k) は nibabel では (x,y,z) = (col,row,slice)
# zyx → ijk に並べ替え
ijk = cl_vox[:, ::-1]                          # (x,y,z) with original order z,y,x
cl_ras = (aff @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
cl_ras[:, :2] *= -1                            # RAS → LPS

# --- plot ---
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mesh[:,0], mesh[:,1], mesh[:,2], s=1, alpha=0.3)
ax.scatter(cl_ras[:,0], cl_ras[:,1], cl_ras[:,2], linewidth=2)
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
ax.set_title(f"Mesh & Centerline : {case_dir.name}")
plt.show()
