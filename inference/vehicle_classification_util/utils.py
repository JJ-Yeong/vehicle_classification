import os
import shutil
from vehicle_classification_config import NAME


def make_dst_path(do: str, src_root_path: str, dst_root_path: str) -> None:
    print(f"[{do}] 디렉토리 생성 시작...")
    for spot_path in os.listdir(src_root_path):
        for cls in NAME:
            dst_path = os.path.join(dst_root_path, spot_path, cls)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                print(f"{dst_path}")
    print(f"[{do}] 디렉토리 생성 완료!")


def move_file(pred_max: list, src_path_list: list, dst_spot_path_list: str) -> None:
    for pred, src_path, dst_spot_path in zip(pred_max, src_path_list, dst_spot_path_list):
        dst_path = os.path.join(dst_spot_path, NAME[pred])
        shutil.copy(src_path, dst_path)