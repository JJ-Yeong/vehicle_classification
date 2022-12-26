import torch.nn as nn
import timm


class Network(nn.Module):
    def __init__(self, mode :str="train", model :str="efficientnet_b7", num_class: int=12):
        super(Network, self).__init__()
        self.mode = mode
        if self.mode == 'train':
          self.model = timm.create_model(model, pretrained=True, num_classes=num_class, drop_path_rate = 0.2)
        if self.mode == 'test':
          self.model = timm.create_model(model, pretrained=True, num_classes=num_class, drop_path_rate = 0)
        
    def forward(self, x):
        x = self.model(x)
        return x