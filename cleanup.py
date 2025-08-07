import nibabel as nib
# from scipy.ndimage import label
import numpy as np
import itk
from typing import Optional
import networkx as nx
from pathlib import Path
from collections import deque

# def remove_small_components(seg: np.ndarray, min_size: int = 1000) -> np.ndarray:
#     """
#     小さな3D連結成分を除去する

#     Parameters
#     ----------
#     seg : np.ndarray
#         二値の3Dセグメンテーションマスク（0 or 1）
#     min_size : int
#         このボクセル数未満の領域を削除

#     Returns
#     -------
#     cleaned : np.ndarray
#         ノイズを除去した2値マスク
#     """
#     labeled, num = label(seg)
#     counts = np.bincount(labeled.ravel())

#     remove = np.isin(labeled, np.where(counts < min_size)[0])
#     seg[remove] = 0
#     return seg

def skeletonize_sitk(
        in_mask_path: str,
        threshold: Optional[float] = None,
        pre_radius: int = 2,
):
    """
    Parameters
    ----------
    in_mask_path : str
        入力ボリューム（.nii.gz, .mha など）。連続値でも OK。
    out_skel_path : str
        出力スケルトンボリューム（True=1, False=0）。
    """
    
    img = itk.imread(in_mask_path, itk.F)
    
    if threshold is None:
        lower = 1e-6
    
    else:
        lower = threshold
        
    mask = itk.binary_threshold_image_filter(img, lower_threshold=lower, upper_threshold=float('inf'), inside_value=1, outside_value=0 ).astype(itk.UC)
        
    se = itk.FlatStructuringElement[3].Ball(pre_radius)
    mask = itk.binary_dilate_image_filter(mask, kernel=se,
                                          foreground_value=1, background_value=0)
    mask = itk.binary_morphological_closing_image_filter(mask, kernel=se,
                                                         foreground_value=1,
                                                         safe_border=True)
    
    #----------------スケルトン化-------------#
    thinner = itk.BinaryThinningImageFilter3D.New(Input=mask)
    thinner.Update()
    skel = thinner.GetOutput()
    return skel

def skel_to_graph(itk_skel):
    
    skel = itk.array_from_image(itk_skel).astype(bool)
    coords = np.stack(np.nonzero(skel), axis=1)
    
    G = nx.Graph()
    
    for idx, (z,y,x) in enumerate(coords):
        G.add_node(idx, pos=(z,y,x))
        
    offsets = np.array([[dz,dy,dx] for dz in [-1,0,1] for dy in [-1,0,1] for dx in [-1,0,1] if not (dz == 0 and dy == 0 and dx == 0)], dtype = int)
    voxel_index = {tuple(c): i for i , c in enumerate(coords)}
    for i, p in enumerate(coords):
        neigh = p + offsets
        for q in neigh:
            key = tuple(q)
            if skel[key]:
                j = voxel_index[key]
                if i < j:
                    G.add_edge(i, j)
                    
    return G, coords

def reconstruct_volume_from_seeds(
    orig_arr: np.ndarray,
    seed_coords: np.ndarray,
    connectivity: int = 26
) -> np.ndarray:
    """
    orig_arr: bool array (Z,Y,X) の元マスク
    seed_coords: (M,3) のスケルトン座標 [(z,y,x),...]
    connectivity: 6 or 26
    戻り値: uint8 array  1 が到達可能な領域
    """
    shape = orig_arr.shape
    visited = np.zeros(shape, dtype=bool)
    q = deque()
    # 初期種（シード）を登録
    for z, y, x in seed_coords:
        visited[z, y, x] = True
        q.append((int(z), int(y), int(x)))

    # 近傍オフセット生成
    if connectivity == 6:
        offs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    else:
        offs = [(dz,dy,dx)
                for dz in (-1,0,1)
                for dy in (-1,0,1)
                for dx in (-1,0,1)
                if not (dz==dy==dx==0)]

    # BFS：マスク内のボクセルだけを訪問
    while q:
        z,y,x = q.popleft()
        for dz,dy,dx in offs:
            nz, ny, nx = z+dz, y+dy, x+dx
            if (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]
                and orig_arr[nz,ny,nx]
                and not visited[nz,ny,nx]):
                visited[nz,ny,nx] = True
                q.append((nz,ny,nx))

    return visited.astype(np.uint8)

def cleanup_coronary_segmentation(nii_path,out_path, pre_radius=2, connectivity=26):
    """
    Coronary artery segmentationのノイズ除去と保存
    """
    
    skel_img = skeletonize_sitk(nii_path,None, pre_radius=pre_radius)
    
    G, coords = skel_to_graph(skel_img)
    
    # --- 2) 主要２成分のノード集合を取得 ---
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps:
        raise RuntimeError("スケルトン化後に連結成分が見つかりません。")
    selected = set(comps[0])
    if len(comps) > 1:
        selected |= set(comps[1])
    seed_coords = coords[list(selected)]  # shape=(M,3)

    # --- 3) 元マスクを読み込み & Numpy 化 ---
    orig_img = itk.imread(nii_path, itk.UC)
    orig_arr = itk.array_from_image(orig_img).astype(bool)

    # --- 4) BFS で再構成 ---
    recon_arr = reconstruct_volume_from_seeds(orig_arr, seed_coords, connectivity=connectivity)

    # --- 5) ITK 画像に戻してメタデータコピー & 保存 ---
    recon_img = itk.GetImageFromArray(recon_arr)
    recon_img.SetSpacing(orig_img.GetSpacing())
    recon_img.SetOrigin(orig_img.GetOrigin())
    recon_img.SetDirection(orig_img.GetDirection())
    itk.imwrite(recon_img, out_path)
    print(f"Saved cleaned segmentation to {out_path}")
    
# cleanup_coronary_segmentation("data/CoronaryArtery/case_2/seg.nii.gz", "data/CoronaryArtery/case_2/seg_cleaned.nii.gz", pre_radius=2)
    


base_dir = "data/CoronaryArtery/"
for i in range(1, 11):
    dir = base_dir + f"case_{i}/"
    nii_path = dir + f"coronary_{i}.nii.gz"
    out_path = dir + "seg.nii.gz"
    cleanup_coronary_segmentation(nii_path, out_path)