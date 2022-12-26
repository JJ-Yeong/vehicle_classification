import torch
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, ceil

from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report


"""
※ result_dict 구성 상세

{'accuracy': 0.075,
 'bus': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 10},
 'car': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 10},
 'macro avg': {'f1-score': 0.05789679456346123,
               'precision': 0.04773897866129324,
               'recall': 0.075,
               'support': 120},
 'truck-4W-FT': {'f1-score': 0.06060606060606061,
                 'precision': 0.043478260869565216,
                 'recall': 0.1,
                 'support': 10},
 'truck-4W-ST': {'f1-score': 0.0,
                 'precision': 0.0,
                 'recall': 0.0,
                 'support': 10},
 'truck-5W-FT': {'f1-score': 0.14814814814814817,
                 'precision': 0.11764705882352941,
                 'recall': 0.2,
                 'support': 10},
 'truck-5W-ST': {'f1-score': 0.0,
                 'precision': 0.0,
                 'recall': 0.0,
                 'support': 10},
 'truck-6W-ST': {'f1-score': 0.0,
                 'precision': 0.0,
                 'recall': 0.0,
                 'support': 10},
 'truck-m-a': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 10},
 'truck-m-b': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 10},
 'truck-m-c': {'f1-score': 0.16,
               'precision': 0.13333333333333333,
               'recall': 0.2,
               'support': 10},
 'truck-s-a': {'f1-score': 0.23076923076923075,
               'precision': 0.1875,
               'recall': 0.3,
               'support': 10},
 'truck-s-b': {'f1-score': 0.09523809523809525,
               'precision': 0.09090909090909091,
               'recall': 0.1,
               'support': 10},
 'weighted avg': {'f1-score': 0.05789679456346123,
                  'precision': 0.047738978661293244,
                  'recall': 0.075,
                  'support': 120}}
"""


def get_swap_dict(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def score_function(true, pred, target_name):
    # score = f1_score(true, pred, average="macro")
    # precision, recall, f_score, true_sum = precision_recall_fscore_support(true, pred, average="macro")
    # return precision, recall, f_score, true_sum
    result_dict = classification_report(true, pred, target_names=target_name, output_dict=True)
    return result_dict


def model_save(model, optimizer, path: str) -> None:
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path)


def show_sample(dataloader, decode_dict: dict) -> None:
    for batch_x, batch_y in dataloader:
        row_num = ceil(sqrt(len(batch_x)))
        plt.figure(figsize=(row_num*4, row_num*4))

        for i, (img, label) in enumerate(zip(batch_x, batch_y), start=1):
            plt.subplot(row_num, row_num, i)
            plt.axis("off")
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.title(f"{decode_dict[label.item()]}", fontdict={"size":8})
        
        plt.show()
        break
