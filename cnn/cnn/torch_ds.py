import glob

from torch.utils.data import Dataset
import PIL


class CarBikeDS(Dataset):
    def __init__(self, car_dir, bike_dir,
                 transform=None, target_transform=None):

        self.imgs = []
        self.transform = transform
        self.target_transform = target_transform

        for img_name in glob.glob(f'{car_dir.strip("/")}/*'):
            self.imgs.append((img_name, 1, ))

        for img_name in glob.glob(f'{bike_dir.strip("/")}/*'):
            self.imgs.append((img_name, 0, ))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        image = PIL.Image.open(path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
