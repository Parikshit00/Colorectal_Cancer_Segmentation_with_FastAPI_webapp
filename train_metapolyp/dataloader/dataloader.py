import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

def default_augment_seg(input_image, input_mask):
    input_image = transforms.ColorJitter(brightness=0.1, contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=0.01)(input_image)
        
    # Flipping randomly horizontally or vertically
    if torch.rand(1) > 0.5:
        input_image = transforms.functional.hflip(input_image)
        input_mask = transforms.functional.hflip(input_mask)
    if torch.rand(1) > 0.5:
        input_image = transforms.functional.vflip(input_image)
        input_mask = transforms.functional.vflip(input_mask)

    return input_image, input_mask

def BatchAdvAugmentSeg(imagesT, masksT):
    
    images, masks = default_augment_seg(imagesT, masksT)
    
    return images, masks

def build_decoder(with_labels=True, target_size=(256, 256), ext='png', segment=False, ext2='jpg'):
    def decode(path):
        img = Image.open(path).convert("RGB")
        img = img.resize(target_size)
        return transforms.ToTensor()(img)
    
    def decode_mask(path, gray=True):
        img = Image.open(path).convert("RGB")
        if gray:
            img = transforms.functional.rgb_to_grayscale(img)
        img = img.resize(target_size)
        img = transforms.ToTensor()(img)/ 255.0
        return img
    
    def decode_with_labels(path, label):
        return decode(path), label
    
    def decode_with_segments(path, path2, gray=True):
        return decode(path), decode_mask(path2, gray)
    
    if segment:
        return decode_with_segments
    
    return decode_with_labels if with_labels else decode
'''
class Decoder:
    def __init__(self, with_labels=True, target_size=(256, 256), ext='png', segment=False, ext2='png'):
        self.with_labels = with_labels
        self.target_size = target_size
        self.ext = ext
        self.segment = segment
        self.ext2 = ext2

    def decode(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize(self.target_size)
        return transforms.ToTensor()(img)

    def decode_mask(self, path, gray=True):
        img = Image.open(path).convert("RGB")
        if gray:
            img = transforms.functional.rgb_to_grayscale(img)
        img = img.resize(self.target_size)
        return transforms.ToTensor()(img)

    def decode_with_labels(self, path, label):
        return self.decode(path), label

    def decode_with_segments(self, path, path2, gray=True):
        return self.decode(path), self.decode_mask(path2, gray)

    def __call__(self):
        if self.segment:
            return self.decode_with_segments
        else:
            return self.decode_with_labels if self.with_labels else self.decode

'''

class CustomDataset(Dataset):
    def __init__(self, paths, labels, decode_fn=None, augment_fn=None, augment=True, cache_dir=""):
        self.paths = paths
        self.labels = labels
        self.decode_fn = decode_fn
        self.augment_fn = augment_fn
        self.augment = augment
        self.cache_dir = cache_dir

        if self.cache_dir != "" and os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        if self.decode_fn is None:
            self.decode_fn = build_decoder(self.labels is not None)
        if self.augment_fn is None:
            self.augment_fn = build_augmenter(self.labels is not None)
        label_path = self.labels[index]
        img, label = self.decode_fn(img_path, label_path)
        if self.augment:
            img, label = self.augment_fn(img, label)
        return img, label


def build_dataset(paths, labels, bsize=32, cache=True, decode_fn=None, augment_fn=None, augment=True, augmentAdv=False, augmentAdvSeg=False, repeat=True, shuffle=True, cache_dir=""):
    dataset = CustomDataset(paths, labels, decode_fn=decode_fn, augment_fn=augment_fn, augment=augment, cache_dir=cache_dir)
    dataloader = DataLoader(dataset, batch_size=bsize, shuffle=shuffle)

    return dataloader

def build_augmenter(with_labels=True):
    def augment(img):
        img = transforms.functional.to_pil_image(img)
        img = transforms.RandomVerticalFlip()(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.ColorJitter(brightness=0.1, contrast=(0.9, 1.1), saturation=(0.9,1.1), hue=0.02)(img)
        img = transforms.functional.to_tensor(img)
        return img
    
    def augment_with_labels(img, label):
        return augment(img), label
    
    return augment_with_labels if with_labels else augment
