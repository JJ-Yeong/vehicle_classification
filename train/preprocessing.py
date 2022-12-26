import os
import shutil
import cv2
import numpy as np

from tqdm import tqdm
from config import *

# 전체 데이터에 대한 (height, width)의 평균 : (108.62, 156.70)

def move_invalid_file(root_path: str, train_path: str, test_path: str) -> None:
    # height_list = []
    # width_list = []
    root = os.path.join(os.getcwd(), root_path)
    dst_path = os.path.join(root, "invalid_file")
    if not os.path.exists(dst_path): 
        os.mkdir(dst_path)

    for mode in [train_path, test_path]:
        sub_root = os.path.join(root, mode)
        cls_list = os.listdir(sub_root)
        if len(set(cls_list) ^ set(NAME)) > 0:
            raise Exception("폴더명과 config의 NAME이 일치하지 않습니다!")

        for i, cls in enumerate(cls_list, start=1):
            print(f"[{mode} {i}/{len(cls_list)}] {cls} 카운팅 시작!")
            file_list = os.listdir(os.path.join(sub_root, cls))

            for file in tqdm(file_list):
                stem, ext = os.path.splitext(file)
                file_path = os.path.join(sub_root, cls, file)
                img = cv2.imread(file_path) # img.shape = H x W x C
                if file == "desktop.ini":
                    os.remove(file)
                if (ext not in IMG_FORMAT) or (img is None):
                    print(f" {cls}: {file} is invalid!")
                    shutil.move(file_path, dst_path)
                    print(f"{cls}: {file} has been moved to {dst_path}!")
                    continue
                # height_list.append(img.shape[0])
                # width_list.append(img.shape[1])


# window_name = "hihello"
# cv2.imshow(window_name, img)
# cv2.moveWindow(window_name, 0, 0)
# key = cv2.waitKey(0)
# if key == 27:
#     cv2.destroyAllWindows()
# print(f"{np.mean(height_list):.2f}")
# print(f"{np.mean(width_list):.2f}")