import os
import glob
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

from config import *

"""
※ os.walk 설명

    file_path = 'C:\\test'
    for file in os.walk(file_path):
        print(file)
    
    위 코드를 실행시키면 아래와 같은 출력이 발생한다.
    
    ('C:\\test', ['directory1'], ['file1.txt', 'file2.txt', 'file3.txt'])
    ('C:\\test\\directory1', [], ['file4.txt'])
    
    os.walk는 한 이터레이션마다 (root, [root 내의 디렉토리들], [root 내의 파일들])의 형태를 반환하고 이는 재귀적(recursive)으로 이루어진다.

"""


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename: str) -> bool:
    """
    any() : 하나라도 True인게 있으면 True 반환
    all() : 모두 True여야 True 반환
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir: str):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    return classes, class_to_idx, idx_to_class


def make_dataset(dir: str, class_to_idx: dict, max_data_per_class: dict, shuflle_dataset: bool=True):
    images = []
    dir = os.path.expanduser(dir) # 경로를 사용자의 home으로 변경하여 반환함. 실패하면 입력받은 경로 그대로 다시 반환함
    cls_num_dict = {}
    for cls in sorted(os.listdir(dir)):
        break_flag = False
        count = 0
        d = os.path.join(dir, cls)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            if shuflle_dataset:
                np.random.shuffle(fnames)
            else:
                fnames = sorted(fnames)
            for fname in fnames:
                if is_image_file(fname):
                    if cls in max_data_per_class:
                        if count >= max_data_per_class[cls]:
                            break_flag = True 
                            break
                    path = os.path.join(root, fname) # os.walk를 사용함으로써 한 label폴더에 이미지들이 각각 다른 디렉토리 계층에 있어도 인식할 수 있음
                    item = (path, class_to_idx[cls])
                    images.append(item)
                    count += 1
            if break_flag:
                break
        cls_num_dict[cls] = count
    return images, cls_num_dict


def pil_loader(path: str):
    # ResourceWarning을 피하기 위해 Image.open as img를 open as f로 감싼다고 함   
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageFolder(Dataset):
    """루트 디렉토리의 하위 디렉토리들을 각각 하나의 클래스로 취급하여 데이터셋 형성

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): 루트 디렉토리 경로
        transform (callable, optional): PIL 이미지를 받아서 transform된 이미지를 반환하는 함수. ex ``transforms.RandomCrop``
        target_transform (callable, optional): target에 대한 transform을 수행하는 함수 (일단 지금은 안 씀)
        loader (callable, optional): 이미지를 불러올 때 사용할 함수

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root: str, max_data_per_class: dict, transform=None, target_transform=None, shuflle_dataset: bool=True, loader=pil_loader):
        classes, class_to_idx, idx_to_class = find_classes(root)
        imgs, cls_num_dict= make_dataset(root, class_to_idx, max_data_per_class, shuflle_dataset)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.cls_num_dict = cls_num_dict

    def __getitem__(self, index):
        """
        Args:
            index: 데이터의 인덱스

        Returns:
            tuple (image, target): (이미지, 정수로 인코딩된 타겟)
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class getData(object):
    def __init__(self, path: str, transform, batch_size: int, num_workers: int, max_data_per_class: dict, shuflle_dataset: bool):
        self.path = path
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_data_per_class = max_data_per_class
        self.shuflle_dataset = shuflle_dataset

    def get_dataset(self):
        return ImageFolder(
                    root=self.path, 
                    max_data_per_class=self.max_data_per_class, 
                    transform=self.transform, 
                    shuflle_dataset=self.shuflle_dataset)
    
    def get_dataloader(self, shuffle: bool=True):
        return DataLoader(  
                    dataset=self.get_dataset(),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=shuffle)


# class Custom_dataset(Dataset):
#     def __init__(self, root_path, mode='train'):
#         cls_list = os.listdir(root_path)
#         for cls in cls_list:
#             cls_path = glob.glob(os.path.join(cls, "*"))
#             Image.open 
#         self.mode=mode

#     def __len__(self):
#         return len(self.root_path)

#     def __getitem__(self, idx):
#         img = self.root_path[idx]
#         if self.mode=='train':
#             img = TRAIN_TRANSFORM(image=img)

#         if self.mode=='valid':
#             img = VALID_TRANSFORM(image=img)

#         label = self.labels[idx]
#         return img, label