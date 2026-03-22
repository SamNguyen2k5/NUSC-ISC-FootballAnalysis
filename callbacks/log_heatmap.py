import torch
import numpy as np
import wandb
from pytorch_lightning.callbacks import Callback

class LogHeatmapCallback(Callback):
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

    def logging_images(self, mode, trainer, pl_module, outputs, batch, batch_idx):
        x, y = batch
        pl_module.eval()
        with torch.no_grad():
            y_pred = pl_module(x)
        
        images = []
        # Take up to num_samples from the batch
        n = min(x.shape[0], self.num_samples)
        
        for i in range(n):
            # Prepare a side-by-side visualization
            img = x[i].cpu().squeeze().numpy()
            gt = y[i].cpu().squeeze().numpy()
            pred = y_pred[i].cpu().squeeze().numpy()

            # Log as a W&B Image with a caption
            combined_array = np.concatenate([img, gt, pred], axis=1) # Stack horizontally
            combined = wandb.Image(
                combined_array, 
                caption=f"[{mode}] Batch {batch_idx}, Sample {i}: Input | Ground Truth | Pred"
            )
            images.append(combined)

        return images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx <= 1:
            images = self.logging_images("training", trainer, pl_module, outputs, batch, batch_idx)
            trainer.logger.experiment.log({"train_predictions": images})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx <= 1:
            images = self.logging_images("validation", trainer, pl_module, outputs, batch, batch_idx)
            trainer.logger.experiment.log({"val_predictions": images})