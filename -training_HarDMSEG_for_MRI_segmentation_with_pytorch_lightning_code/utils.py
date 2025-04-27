import torch
from torchmetrics import Metric

class IoU(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape

        for i in range(preds.shape[0]):
            pred_mask = torch.argmax(preds[i], dim=0).cpu()  
            true_mask = torch.argmax(target[i], dim=0).cpu()
            intersection = (pred_mask * true_mask).sum()
            union = (pred_mask + true_mask).sum()
            iou = intersection / union if union != 0 else 0
            self.total_iou += iou
            self.count += 1

    def compute(self):
        return self.total_iou.float() / self.count.float()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])










    