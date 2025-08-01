from dataset import VesselPatchDataset

train_ds = VesselPatchDataset("data.json", indexes=["1", "2"], isTrain=False)

patch, pts = train_ds[0]

print(f"Patch shape: {patch.shape}")
print(f"Points shape: {pts.shape}")