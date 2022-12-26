import torch
from torchvision.transforms import transforms

ROOT_PATH = "data_12_15/check"
TRAIN_PATH = "train"
VALID_PATH = "test"
PROJECT_SAVE_PATH = "runs"

TRAIN_MAX_DATA_PER_CLASS = {
    # "bus": 5,
    "car": 50000,
    # "truck-4W-FT": 5,
    # "truck-4W-ST": 5,
    # "truck-5W-FT": 5,
    # "truck-5W-ST": 5,
    # "truck-6W-ST": 5,
    # "truck-m-a" : 5,
    # "truck-m-b" : 5,
    # "truck-m-c" : 5,
    # "truck-s-a" : 5,
    # "truck-s-b" : 5
}
VALID_MAX_DATA_PER_CLASS = {
    # "bus": 5,
    "car": 50000,
    # "truck-4W-FT": 5,
    # "truck-4W-ST": 5,
    # "truck-5W-FT": 5,
    # "truck-5W-ST": 5,
    # "truck-6W-ST": 5,
    # "truck-m-a" : 5,
    # "truck-m-b" : 5,
    # "truck-m-c" : 5,
    # "truck-s-a" : 5,
    # "truck-s-b" : 5
}

SHUFFLE_DATASET = True 
# dataset형성단계에서 shuffle을 수행할 건지 선택. train, valid 모두 적용되며 dataloader의 shuffle보다 먼저 적용됨.
# Down Sampling 수행 시 웬만하면 True로 주는 것이 좋음.

NUM_CLASS = 12
NAME = ["bus", "car", "truck-4W-FT", "truck-4W-ST", "truck-5W-FT", "truck-5W-ST", "truck-6W-ST", "truck-m-a", "truck-m-b", "truck-m-c", "truck-s-a", "truck-s-b"]
IMG_FORMAT = [".jpg", ".jpeg", ".png"]

MODEL_NAME = "tf_efficientnet_b3" # "efficientnet_b7" 
OPTIMIZER = "SGD" # "SGD", "Adam", RAdam"
LR = 0.004
MOMENTUM = 0.937

EPOCHS = 100
RESIZE = 300 # 224
BATCH_SIZE = 32
NUM_WORKERS = 0
F1_PATIENCE = 50

TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor()
    ])

VALID_TRANSFORM = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor()
    ])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")