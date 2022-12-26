import os
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple 
# 나중에 자료형 힌트 연습해보기, torchvision/datasets/folder.py 참조. (ImageFolder 원본 타고들어가면 나옴) 

from config import *


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    return classes, class_to_idx, idx_to_class


def make_dataset(src_root_path: str, dst_root_path: str) -> Tuple[List[str], Dict[str, int]]:
    images_path = []
    src_root_path = os.path.expanduser(src_root_path)
    dst_root_path = os.path.expanduser(dst_root_path)
    num_data_per_spot = {}
    for spot in sorted(os.listdir(src_root_path)):
        count = 0
        src_spot_path = os.path.join(src_root_path, spot)
        dst_spot_path = os.path.join(dst_root_path, spot)
        if not os.path.isdir(src_spot_path):
            continue
        for _root, _, fnames in sorted(os.walk(src_spot_path)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    src_path = os.path.join(_root, fname)
                    src_dst = (src_path, dst_spot_path)
                    images_path.append(src_dst)
                    count += 1
        num_data_per_spot[spot] = count
    return images_path, num_data_per_spot


def pil_loader(path: str):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageFolder(Dataset):
    def __init__(self, src_root_path: str, dst_root_path: str, transform=None, loader=pil_loader):
        classes, class_to_idx, idx_to_class = find_classes(src_root_path)
        images_path, num_data_per_spot = make_dataset(src_root_path, dst_root_path)
        if len(images_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + src_root_path + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.src_root_path = src_root_path
        self.dst_root_path = dst_root_path
        self.images_path = images_path
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.loader = loader
        self.num_data_per_spot = num_data_per_spot

    def __getitem__(self, index):
        src_path, dst_spot_path = self.images_path[index]
        img = self.loader(src_path)
        if self.transform is not None:
            img = self.transform(img)
        item = {
            "image": img,
            "src_path": src_path,
            "dst_spot_path": dst_spot_path
        }
        return item

    def __len__(self):
        return len(self.images_path)


class getData(object):
    def __init__(self, src_root_path: str, dst_root_path: str, transform, batch_size: int, num_workers: int):
        self.src_root_path = src_root_path
        self.dst_root_path = dst_root_path
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_dataset(self):
        return ImageFolder(
                    src_root_path=self.src_root_path, 
                    dst_root_path=self.dst_root_path, 
                    transform=self.transform, 
                    )
    
    def get_dataloader(self, shuffle: bool=False):
        return DataLoader(  
                    dataset=self.get_dataset(),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=shuffle
                    )