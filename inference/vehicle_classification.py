import os
from tqdm import tqdm
import torch

from vehicle_classification_util.dataloader import getData
from vehicle_classification_util.model import Network
from vehicle_classification_util.utils import make_dst_path, move_file
from vehicle_classification_config import *


def inference():
    for do in DO_PATH_LIST:
        src_root_path = os.path.join(SRC_ROOT_PATH, do)
        dst_root_path = os.path.join(DST_ROOT_PATH, do)
        make_dst_path(do, src_root_path, dst_root_path)

        data = getData(src_root_path, dst_root_path, TRANSFORM, BATCH_SIZE, NUM_WORKERS)
        loader = data.get_dataloader(shuffle=False)

        model = Network(mode="test", model=MODEL_NAME, num_class=NUM_CLASS).to(DEVICE)
        model_path = os.path.join("vehicle_classification_weight", WEIGHT)
        weight = torch.load(model_path)
        model.load_state_dict(weight["model"]) #FIXME 주의!! 2022-10-06이후의 가중치 파일은 .pt -> .pt.tar 확장자로 바뀌고 key값도 "model" -> "model_state_dict"로 변경되므로 코드 일부 수정 필요 

        with torch.no_grad():
            model.eval()
            description = f"[{do}] 분류 시작..."
            for batch in tqdm(loader, desc=description):
                imgs = batch["image"].clone().detach().float().cuda()
                pred_raw = model(imgs) # pred : 2차원 텐서 (batch_size가 32일 경우, pred.shape=(32, 12))
                pred_max = pred_raw.argmax(1).detach().cpu().numpy().tolist() # pred_max : 1차원 텐서 (batch_size가 32일 경우, pred_max.shape=(32))
                src_path_list = batch["src_path"]
                dst_spot_path_list = batch["dst_spot_path"]
                move_file(pred_max, src_path_list, dst_spot_path_list)


if __name__ == "__main__":
    inference()


                

            
