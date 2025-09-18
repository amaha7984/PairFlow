import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GenericI2IDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.sat_dir = os.path.join(root_dir, "trainA")
        self.map_dir = os.path.join(root_dir, "trainB")
        self.transform = transform

        self.sat_files = sorted(os.listdir(self.sat_dir))
        self.map_files = sorted(os.listdir(self.map_dir))
        assert len(self.sat_files) == len(self.map_files), "Mismatch in A/B images."

    def __len__(self):
        return len(self.sat_files)

    def __getitem__(self, idx):
        sat_path = os.path.join(self.sat_dir, self.sat_files[idx])
        map_path = os.path.join(self.map_dir, self.map_files[idx])

        sat_img = Image.open(sat_path).convert("RGB")
        map_img = Image.open(map_path).convert("RGB")

        if self.transform:
            sat_img = self.transform(sat_img)
            map_img = self.transform(map_img)

        return {"sat": sat_img, "map": map_img}
