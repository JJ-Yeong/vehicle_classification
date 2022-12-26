import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

SRC_ROOT_PATH = "이미지파일_분류전"
DST_ROOT_PATH = "이미지파일_분류됨"
DO_PATH_LIST = ["경북"] # ["경기", "충북"]

NUM_CLASS = 12

# 순서 변경 금지!!! (현재 사용하는 모델이 이 순서대로 label encoding해서 학습됨. 추후에 다시 학습시킬 떄 weight파일에 label encoding 정보도 넣어서 불러올 수 있게 해야할듯)
NAME = ["bus", "car", "truck-4W-FT", "truck-4W-ST", "truck-5W-FT", "truck-5W-ST", "truck-6W-ST", "truck-m-a", "truck-m-b", "truck-m-c", "truck-s-a", "truck-s-b"]

MODEL_NAME = "tf_efficientnet_b3"
WEIGHT = "tf_efficientnet_b3_img300_bs32_lr0.004_SGD_best_f1_0.747_epoch7.pt"
RESIZE = 300
BATCH_SIZE = 32
NUM_WORKERS = 0

TRANSFORM = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor()
    ])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")