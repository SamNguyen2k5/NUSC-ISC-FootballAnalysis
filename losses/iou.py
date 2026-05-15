import torch.nn as nn
# from ultralytics.utils.loss import v8DetectionLoss

class IOULoss(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError()
        # self.fn = v8DetectionLoss()

    def forward(self, y_pred, y_true):
        raise NotImplementedError()
        # return self.fn(y_pred, y_true)