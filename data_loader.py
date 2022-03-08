#prepare the dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms as trns
from torch.utils.data import DataLoader 
from dataset import SegmentationDataset
from sklearn.model_selection import train_test_split
import torch

class CustomLoader():
    def __init__(self, images_folder, masks_folder, SIZE=256, BATCH_SIZE = 4):
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.images = os.listdir(images_folder)
        self.masks = os.listdir(masks_folder)
        self.size = SIZE
        self.batch = BATCH_SIZE

    def iter_images(self):
        image_dataset = []
        for i, image_name in enumerate(self.images):
            image = Image.open(os.path.join(self.images_folder, image_name))
            image = image.resize((self.size, self.size))
            image_dataset.append(image)
        return image_dataset

    def iter_masks(self):
        mask_dataset = []
        for i, image_name in enumerate(self.masks):
            image = Image.open(os.path.join(self.masks_folder, image_name))
            image = image.resize((self.size, self.size))
            mask_dataset.append(image)
        return mask_dataset

    def get_items(self, image_dataset, mask_dataset):
        transforms = trns.Compose([trns.ToTensor()])
        X_train, X_rest, y_train, y_rest = train_test_split(image_dataset,mask_dataset, test_size = 0.20, random_state = 0)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size = 0.5, random_state = 0)
        train = SegmentationDataset(image_dataset= X_train, mask_dataset=y_train, transforms=transforms)
        val = SegmentationDataset(image_dataset=X_val, mask_dataset=y_val, transforms=transforms)
        test = SegmentationDataset(image_dataset=X_test, mask_dataset=y_test, transforms=transforms)
        print(f"[INFO] found {len(train)} examples in the training set...")
        print(f"[INFO] found {len(val)} examples in the test set...")
        print(f"[INFO] found {len(test)} examples in the training set...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        PIN_MEMORY = True if device == "cuda" else False
        trainLoader = DataLoader(train, shuffle=True, batch_size=self.batch, pin_memory=PIN_MEMORY)
        valLoader = DataLoader(val, shuffle=False, batch_size=self.batch, pin_memory=PIN_MEMORY)
        testLoader = DataLoader(test)
        return trainLoader, valLoader, testLoader












