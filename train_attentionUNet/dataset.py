#prepare the dataset
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dataset, mask_dataset, transforms):
        self.imageDataset = image_dataset
        self.maskDataset = mask_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.imageDataset)

    def __getitem__(self, idx):
        image = self.imageDataset[idx]
        mask = self.maskDataset[idx]
        image = self.transforms(image)
        mask = self.transforms(mask)
        return (image, mask)