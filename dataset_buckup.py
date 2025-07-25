import os
import random, math
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as dataset
from utils import get_json, get_medical_image, save_csv, get_csv, norm_zero_one

class VesselPatchDataset(dataset):
    patch_size = (64, 64, 64) #boxel
    actual_patch_size = (32,32,32) #mm
    target_spacing = (0.5, 0.5, 0.5) #mm
    
    """
    "image":  nifty from nnUnet
    "mesh":   imageの表面のサンプリング点群(XYZ)RAS mm
    "centerline":  imageの中心線の点群(XYZ) ボクセル
    """
    def __init__(self, json_path, indexes,isTrain=True):
        super().__init__()
        config = get_json(json_path)
        
        self.isTrain = isTrain
        self.images = []
        self.centerlines = []
        self.meshes = []
        
        for idx in indexes:
            items = config[idx]
            
            for item in items:
                image_path = item['image']
                centerline_path = item.get('centerline')
                mesh_path = item['mesh']
                self.images.append(os.path.join(config["dir"], image_path))
                self.centerlines.append(os.path.join(config["dir"], centerline_path))
                self.meshes.append(os.path.join(config["dir"], mesh_path))
                
    def __len__(self):
        return len(self.images)
    
    def _random_flip(self, img, points, axis_prob=0.5):
        for ax in range(3):
            if random.random() < axis_prob:
                img = np.flip(img, axis=ax).copy()
                points[:, ax] = 1 - points[:, ax]          
        return img, points
    
    def _random_rotate90(self, img, points, axis_prob=0.5):
        for ax in range(3):
            if random.random() < axis_prob:
                img = np.rot90(img, axes=(ax, (ax + 1) % 3)).copy()
                points[:, [ax, (ax + 1) % 3]] = points[:, [(ax + 1) % 3, ax]]
        return img, points
    
    def _resample_patch_ndarray(self, arr, cur_spacing, target_spacing=(0.5, 0.5, 0.5),
                           is_label=True, cur_origin=(0.,0.,0.), direction=None):
            
        img_sitk = sitk.GetImageFromArray(arr.astype(np.uint8))
        img_sitk.SetSpacing(cur_spacing)
        img_sitk.SetOrigin(cur_origin)
        if direction is not None:
            img_sitk.SetDirection(direction)

        # ----- 新しいサイズを mm で合わせて計算 -----
        cur_size = np.array(img_sitk.GetSize())      # (X,Y,Z)
        cur_spacing_xyz = np.array(img_sitk.GetSpacing())
        tgt_spacing_xyz = np.array(target_spacing[::-1])

        new_size = np.round(cur_size * cur_spacing_xyz / tgt_spacing_xyz).astype(int).tolist()

        # ----- Resample -----
        res = sitk.ResampleImageFilter()
        res.SetOutputSpacing(tgt_spacing_xyz.tolist())
        res.SetSize(new_size)
        res.SetOutputOrigin(img_sitk.GetOrigin())
        res.SetOutputDirection(img_sitk.GetDirection())
        res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

        img_resamp = res.Execute(img_sitk)
        arr_resamp = sitk.GetArrayFromImage(img_resamp)      # (Z,Y,X) back

        return arr_resamp, img_resamp.GetSpacing() 
    
    def _randam_crop(self, img, centerline):
        # ランダムにパッチを切り出す
        #サイズは物理座法基準で切り出して、その後にボクセルサイズに変換する
        center= random.choice(centerline)
        cx, cy, cz = map(int, (center[2], center[1], center[0]))
        
        sx, sy, sz = img.GetSpacing()
        pw = int(np.round(self.actual_patch_size[0] / sx))
        ph = int(np.round(self.actual_patch_size[1] / sy))
        pd = int(np.round(self.actual_patch_size[2] / sz))
        
        x0 = cx - pw // 2
        y0 = cy - ph // 2
        z0 = cz - pd // 2

        region = sitk.RegionOfInterest(img,size=[pw, ph, pd],index=[x0, y0, z0])
        crop_img = sitk.GetArrayFromImage(region) 
        
        origin_mm = img.TransformIndexToPhysicalPoint((x0, y0, z0))
        
        crop_resamp, new_spacing = self._resample_patch_ndarray(
        crop_img,
        cur_spacing=(sx, sy, sz),
        target_spacing=self.target_spacing,
        is_label=True,
        cur_origin=img.TransformIndexToPhysicalPoint((x0,y0,z0)))
        
        return crop_resamp, origin_mm
    
    
    def _crop_resample_normalize_pts_mm(self,
                                        verts_mm,          # (N,3) Z,X,Y [mm]
                                        origin_mm,         # (ox,oy,oz) patch 原点 mm
                                        ):
        """
        画像パッチと同じ ROI & リサンプリング倍率で点群を [0,1] に正規化
        """
        ox,oy,oz = origin_mm
        wx,wy,wz = self.actual_patch_size 
        # ---- ① ROI 内に入る点だけ抽出 -----------------------------
        inside = (
            (verts_mm[:,0] >= ox) & (verts_mm[:,0] < ox+wx) &
            (verts_mm[:,1] >= oy) & (verts_mm[:,1] < oy+wy) &
            (verts_mm[:,2] >= oz) & (verts_mm[:,2] < oz+wz)
        )
        pts = verts_mm[inside]
        if len(pts)==0:
            return np.zeros((1,3), np.float32)

        # ---- ② パッチ原点からの mm オフセット ----------------------
        pts_local_mm = pts - np.array([ox,oy,oz], np.float32)

        # ---- ③ target_spacing で voxel 数へ ------------------------
        pts_vox = pts_local_mm / np.array(self.target_spacing, np.float32)  # (M,3) X,Y,Z voxel

        # ---- ④ [0,1] 正規化  --------------------------------------
        # 画像パッチは self.patch_size = (64,64,64) voxel
        pw,ph,pd = self.patch_size   # 注意：X,Y,Z ⇔ W,H,D
        pts_norm = pts_vox / np.array([pw,ph,pd], np.float32)

        # (Z,Y,X) 順にしたい場合はこの時点で列を並べ替える
        return pts_norm[:, [2,1,0]].astype(np.float32)   # (M,3) (Z,Y,X)
            
            
    def __getitem__(self, idx):
        #一枚の画像からbatchサイズ枚のパッチを取得する
        #トレーニング時にはデータ拡張を行う
        image = get_medical_image(self.images[idx])
        centerlines =  np.array(get_csv(self.centerlines[idx]), np.float32) 
        
        verts = np.asarray(list(map(convert, get_csv(self.meshes[idx], delimiter=' '))))
        def convert(x):
            return [float(i) for i in x]
        
        
        patch, origin_mm= self._randam_crop(image, centerlines) 
        points = self._crop_resample_normalize_pts_mm(verts, origin_mm, )
        
        
        if self.isTrain:
            # データ拡張を施す
            patch = self._random_flip(patch,points)
            patch = self._random_rotate90(patch, points)
            
        return np.array(patch), np.array(points)