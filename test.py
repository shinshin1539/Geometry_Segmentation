from dataset import VesselPatchDataset

train_ds = VesselPatchDataset("data.json", indexes=["1", "2"], isTrain=False)

train_ds.visualize(0)