#Conda環境の itk3Dでうごく

from typing import Optional
import itk
import numpy as np
import networkx as nx
from pathlib import Path

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
    return skel #itk image

    
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

def save_centerline(coords_zyx, out_path):
    np.savetxt(out_path, coords_zyx, fmt="%d")
    
    
base_dir = "data/CoronaryArtery/"
for i in range(1,3):
    dir = base_dir + f"case_{i}/"
    nii_path = dir + f"seg.nii.gz"
    out_path = dir + "centerline.xyz"
    
    _, coords = skel_to_graph(skeletonize_sitk(nii_path, threshold=0.5, pre_radius=2))
    save_centerline(coords, out_path)
    print(f"Saved centerline to {out_path}")
    