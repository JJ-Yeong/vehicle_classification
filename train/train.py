import os
import time
import argparse
from tqdm import tqdm
import pandas as pd
import pprint

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config import *
from preprocessing import move_invalid_file
from dataloader import getData
from utils import show_sample, get_swap_dict, score_function, model_save
from model import Network

"""
conda create -n opencv python=3.8.3 -y && 
conda activate opencv && 
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y && 
conda install -c anaconda numpy=1.22.3 -y && 
conda install -c anaconda pandas seaborn openpyxl ipython ipykernel -y && # pillow protobuf
conda install -c conda-forge debugpy matplotlib terminaltables tensorboard tqdm imgaug profilehooks pprintpp easydict filterpy -y && 
pip install torchsummary PyYAML onnxruntime-gpu opencv-python
"""

# 전체 데이터에 대한 (height, width)의 평균 : (108.62, 156.70)

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, help="학습을 이어할 때 사용할 parameter파일의 경로를 입력합니다. 입력이 없을 경우 initial parameter로 학습을 진행합니다.")
    parser.add_argument("--validity_check", default=False, action="store_true", help="유효하지 않은 데이터를 체크 후 이동 또는 삭제합니다.")
    parser.add_argument("--show_sample", default=False, action="store_true", help="첫 번째 배치에 해당하는 데이터들을 샘플로 보여줍니다.")
    parser.add_argument("--print_summary", default=False, action="store_true", help="모델의 아키텍쳐를 print합니다.")
    args = parser.parse_args()


    # 기본적인 유효성 검사 (처음 사용하는 데이터셋일 때만) 
    if args.validity_check:
        move_invalid_file(ROOT_PATH, TRAIN_PATH, VALID_PATH)


    # train 및 valid 파일 경로 생성
    train_path = os.path.join(ROOT_PATH, TRAIN_PATH)
    valid_path = os.path.join(ROOT_PATH, VALID_PATH)


    # Dataset 및 Dataloader 생성
    train_data = getData(train_path, TRAIN_TRANSFORM, BATCH_SIZE, NUM_WORKERS, TRAIN_MAX_DATA_PER_CLASS, SHUFFLE_DATASET)
    valid_data = getData(valid_path, VALID_TRANSFORM, BATCH_SIZE, NUM_WORKERS, VALID_MAX_DATA_PER_CLASS, SHUFFLE_DATASET)
    train_dataset = train_data.get_dataset()
    valid_dataset = valid_data.get_dataset()
    train_loader = train_data.get_dataloader(shuffle=True)
    valid_loader = valid_data.get_dataloader(shuffle=False)


    # 라벨 인코딩 및 디코딩을 위한 딕셔너리 생성
    encode_dict = train_dataset.class_to_idx
    decode_dict = train_dataset.idx_to_class
    # decode_dict = get_swap_dict(encode_dict)


    # 각 클래스 별로 할당된 데이터(이미지) 수를 얻기 위한 딕셔너리 생성 
    train_cls_num_dict = train_dataset.cls_num_dict
    valid_cls_num_dict = valid_dataset.cls_num_dict


    # 입력받은 이미지의 샘플을 BATCH_SIZE만큼 보여줌(plt.show)
    if args.show_sample:
        show_sample(train_loader, decode_dict)


    # 모델 생성
    model = Network(mode="train", model=MODEL_NAME, num_class=NUM_CLASS).to(DEVICE)
    if args.print_summary:
        if isinstance(RESIZE, int):
            summary(model, (3, RESIZE, RESIZE), device='cuda')
        else:
            print(f"RESIZE값이 '{RESIZE}'입니다!")


    # Optimizer
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(MOMENTUM, 0.999))
    elif OPTIMIZER == "RAdam": 
        optimizer = torch.optim.RAdam(model.parameters(), lr=LR, betas=(MOMENTUM, 0.999))
    else: 
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, nesterov=True)


    # Loss Function
    criterion = nn.CrossEntropyLoss()


    # Scaler
    scaler = torch.cuda.amp.GradScaler()


    # weight 저장할 경로 설정 및 생성
    project_root = os.path.join(PROJECT_SAVE_PATH, f"{MODEL_NAME}_img{RESIZE}_bs{BATCH_SIZE}_lr{LR}_{OPTIMIZER}")
    project_path = os.path.join(project_root, "exp01")
    if os.path.exists(project_path):
        li = sorted(os.listdir(project_root))
        new_project = f"exp{str(int(li[-1][-2:]) + 1).zfill(2)}"
        project_path = os.path.join(project_root, new_project)
    os.makedirs(os.path.join(project_path, "weight"), exist_ok=True)


    # TensorBoard writer활성화
    writer = SummaryWriter(project_path)


    print("\n=== Train 개수 ===")
    for cls in NAME:
        print(f"{cls} : {train_cls_num_dict[cls]}")
    print("\n=== Valid 개수 ===")
    for cls in NAME:
        print(f"{cls} : {valid_cls_num_dict[cls]}")

    MAX_LEN_CLASS_TEXT = len(max(NAME, key=len))
    # DataFrame에 Epoch당 평가지표를 기록
    cls_columns = ["Epoch"] + \
                [f"{decode_dict[cls]}_loss" for cls in range(NUM_CLASS)] + \
                [f"{decode_dict[cls]}_f1" for cls in range(NUM_CLASS)] + \
                [f"{decode_dict[cls]}_p" for cls in range(NUM_CLASS)] + \
                [f"{decode_dict[cls]}_r" for cls in range(NUM_CLASS)]
    cls_df = pd.DataFrame(columns=cls_columns, index=None)
    total_df = pd.DataFrame(columns=["Epoch", "total_loss", "accuracy", "f1", "best_f1", "p", "r"], index=None)
    patience = 0
    valid_f1_best = 0
    total_valid_loss_best = 99999


    # 체크포인트 불러오기
    # args.checkpoint_path = "C:/Users/JJY/Desktop/classification/runs/tf_efficientnet_b3_img300_bs32_lr0.004_SGD/exp02/weight/last.pt"
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("\n=== checkpoint 불러오기 성공! ===")


    print("\n=== 학습 시작!!! ===")
    for epoch in range(EPOCHS):
        ###### Epoch 시작 ######
        ###### Train 시작 ######
        start = time.time()
        train_loss = 0
        train_pred = []
        train_true = []

        model.train()
        for batch in tqdm(train_loader): # ascii : 문자 세 개가 있으면 각각의 문자는 순서대로 배경, 진행중, 진행완료를 표시함
            optimizer.zero_grad()
            x = batch[0].clone().detach().float().cuda()
            y = batch[1].clone().detach().long().cuda()

            with torch.cuda.amp.autocast():
                pred = model(x)
            loss = criterion(pred, y) # CrossEntropyLoss에 softmax가 포함되어 있으므로 밑에서 softmax를 안 쓴다

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            train_true += y.detach().cpu().numpy().tolist()
        
        total_train_loss = train_loss/len(train_loader)
        train_result_dict = score_function(train_true, train_pred, NAME)
        # total_train_precision, total_train_recall, total_train_f1, _ = score_function(train_true, train_pred)
        train_accuracy = train_result_dict["accuracy"]
        train_f1 = train_result_dict["macro avg"]["f1-score"]
        train_precision = train_result_dict["macro avg"]["precision"]
        train_recall = train_result_dict["macro avg"]["recall"]
        writer.add_scalar("train/loss", total_train_loss, epoch)
        writer.add_scalar("train/accuracy", train_accuracy, epoch)
        writer.add_scalar("train/f1score", train_f1, epoch)
        writer.add_scalar("train/precision", train_precision, epoch)
        writer.add_scalar("train/recall", train_recall, epoch)
        ###### Train 종료 ######

        ###### Valid 시작 ######
        valid_loss = 0
        valid_pred = []
        valid_true = []
        per_cls_loss = {}
        # per_cls_true = {}
        # per_cls_precision = {}
        # per_cls_recall = {}
        # per_cls_f1 = {}

        for cls in range(NUM_CLASS):
            per_cls_loss[cls] = 0
            # per_cls_true[cls] = []

        with torch.no_grad():
            model.eval()
            for batch in tqdm(valid_loader):
                x = batch[0].clone().detach().float().cuda()
                y = batch[1].clone().detach().long().cuda()

                pred = model(x)
                loss = criterion(pred, y)
                for p, cls in zip(pred, y):
                    cls_int = int(cls.detach().cpu().int())
                    per_cls_loss[cls_int] += criterion(p, cls).item()

                valid_loss += loss.item()
                valid_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                valid_true += y.detach().cpu().numpy().tolist()
                
        total_valid_loss = valid_loss/len(valid_loader)
        valid_result_dict = score_function(valid_true, valid_pred, NAME)
        # total_valid_precision, total_valid_recall, total_valid_f1, _ = score_function(valid_true, valid_pred)
        valid_accuracy = valid_result_dict["accuracy"]
        valid_f1 = valid_result_dict["macro avg"]["f1-score"]
        valid_precision = valid_result_dict["macro avg"]["precision"]
        valid_recall = valid_result_dict["macro avg"]["recall"]
        writer.add_scalar("valid/loss", total_valid_loss, epoch)
        writer.add_scalar("valid/accuracy", valid_accuracy, epoch)
        writer.add_scalar("valid/f1score", valid_f1, epoch)
        writer.add_scalar("valid/precision", valid_precision, epoch)
        writer.add_scalar("valid/recall", valid_recall, epoch)

        model_save_path = os.path.join(project_path, "weight", 'last.pt.tar')
        model_save(model, optimizer, model_save_path)
        if valid_f1 > valid_f1_best:
            model_save_path = os.path.join(project_path, "weight", f'best_f1_{valid_f1:.3f}_epoch{epoch}.pt.tar')
            model_save(model, optimizer, model_save_path)
            valid_f1_best = valid_f1
            patience = 0
        else:
            patience += 1

        if total_valid_loss < total_valid_loss_best:
            model_save_path = os.path.join(project_path, "weight", f'best_loss_{total_valid_loss:.3f}_epoch{epoch}.pt.tar')
            model_save(model, optimizer, model_save_path)
            total_valid_loss_best = total_valid_loss

        for cls_id in range(NUM_CLASS):
            decoded_cls = decode_dict[cls_id]
            cls_f1 = valid_result_dict[decoded_cls]["f1-score"]
            cls_precision = valid_result_dict[decoded_cls]["precision"]
            cls_recall = valid_result_dict[decoded_cls]["recall"]

        #     per_cls_true = []
        #     per_cls_pred = []
        #     for true in valid_true:
        #         if true==cls_id:
        #             per_cls_true.append(1)
        #         else:
        #             per_cls_true.append(0)
        #     for pred in valid_pred:
        #         if pred==cls_id:
        #             per_cls_pred.append(1)
        #         else:
        #             per_cls_pred.append(0)
        #     cls_precision, cls_recall, cls_f1, _ = score_function(per_cls_true, per_cls_pred)
        #     per_cls_precision[cls_id] = cls_precision
        #     per_cls_recall[cls_id] = cls_recall
        #     per_cls_f1[cls_id] = cls_f1

            cls_num = valid_cls_num_dict[decoded_cls]
            per_cls_loss[cls_id] /= cls_num
            writer.add_scalar(f"valid_per_class/{decoded_cls}_loss", per_cls_loss[cls_id], epoch)
            writer.add_scalar(f"valid_per_class/{decoded_cls}_f1score", cls_f1, epoch)
            writer.add_scalar(f"valid_per_class/{decoded_cls}_precision", cls_precision, epoch)
            writer.add_scalar(f"valid_per_class/{decoded_cls}_recall", cls_recall, epoch)
        

        TIME = (time.time() - start)/60
        ###### Valid 종료 ######
        

        print('\n')
        print(f'===================== Epoch : {epoch+1}/{EPOCHS}    time for this epoch : {TIME:.0f}m   left time : {TIME*(EPOCHS-epoch-1)//60:.0f}h{TIME*(EPOCHS-epoch-1)%60:.0f}m =====================')
        # pprint.pprint(train_result_dict)
        # pprint.pprint(valid_result_dict)
        print(f'TRAIN -> loss : {total_train_loss:.4f}     Accuracy : {train_accuracy:.4f}     F1 : {train_f1:.4f}')
        print(f'VALID -> loss : {total_valid_loss:.4f}     Accuracy : {valid_accuracy:.4f}     F1 : {valid_f1:.4f}     Best F1 : {valid_f1_best:.4f}     P : {valid_precision:.4f}     R : {valid_recall:.4f}')
        total_data = [epoch+1] + list(map(lambda x:round(x, 4), [total_valid_loss, valid_accuracy, valid_f1, valid_f1_best, valid_precision, valid_recall]))
        total_df.loc[epoch] = total_data
        cls_df.loc[epoch, "Epoch"] = epoch+1
        print('\n')
        for cls_id in range(NUM_CLASS):
            decoded_cls = decode_dict[cls_id]
            cls_f1 = valid_result_dict[decoded_cls]["f1-score"]
            cls_precision = valid_result_dict[decoded_cls]["precision"]
            cls_recall = valid_result_dict[decoded_cls]["recall"]
            cls_loss = per_cls_loss[cls_id]
            print(f"    [{decoded_cls:^{MAX_LEN_CLASS_TEXT}}] -> loss : {cls_loss:.2f}    F1 : {cls_f1:.2f}    P : {cls_precision:.2f}    R : {cls_recall:.2f}")
            cls_df.loc[epoch, f"{decoded_cls}_loss"] = round(cls_loss, 2)
            cls_df.loc[epoch, f"{decoded_cls}_f1"] = round(cls_f1, 2)
            cls_df.loc[epoch, f"{decoded_cls}_p"] = round(cls_precision, 2)
            cls_df.loc[epoch, f"{decoded_cls}_r"] = round(cls_recall, 2)

        print('\n\n')

        # Tensorboard Epoch 종료
        writer.flush()
        writer.close()
        if patience > F1_PATIENCE:
            break
        ###### Epoch 종료 ######
   
    ###### 전체 Epoch 종료 및 Metrics DataFrame 저장 ######
    total_df.to_excel(os.path.join(project_path, "total_result.xlsx"), index=False, encoding="cp949")
    cls_df.to_excel(os.path.join(project_path, "cls_result.xlsx"), index=False, encoding="cp949")

if __name__=="__main__":
    train()