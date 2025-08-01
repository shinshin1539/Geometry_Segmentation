from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes

def label_to_xyz_lps(label_path: Path,
                     level: float = 0.5,
                     step: int = 1) -> Path:
    """
    lab.nii(.gz) → mesh.xyz  (ITK/LPS 座標) で保存
    """
    label_path = Path(label_path)   
    img = nib.load(str(label_path))
    vol = img.get_fdata().astype(np.uint8)
    verts, _, _, _ = marching_cubes(vol, level=level, step_size=step)

    # voxel → RAS world
    ras = (img.affine @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]

    # ---- RAS → LPS 変換 ----
    lps = ras.copy()
    lps[:, 0] *= -1      # R→L
    lps[:, 1] *= -1      # A→P
    # Z はそのまま

    out = label_path.with_name("mesh.xyz")
    np.savetxt(out, lps, fmt="%.4f")
    return out

dir = './data/CoronaryArtery/'

for i in range(1, 3):
    path = dir + f"case_{i}/lab.nii.gz"
    print(path)
    label_to_xyz_lps(path, level=0.5, step=1)
    print(f"Converted {path} to mesh.xyz")