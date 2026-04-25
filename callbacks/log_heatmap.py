import cv2
import torch
import numpy as np
import wandb
from pytorch_lightning.callbacks import Callback

from losses.total_intensity import TotalIntensity

class LogHeatmapCallback(Callback):
    def __init__(self, num_samples=None, n_logs_per_batch=2):
        super().__init__()
        self.num_samples = num_samples
        self.n_logs_per_batch = n_logs_per_batch

    def logging_images(self, mode, trainer, pl_module, outputs, batch, batch_idx):
        try:
            x, y = batch
            pl_module.eval()
            with torch.no_grad():
                y_pred = pl_module.predict_step(x, batch_idx)
            
            images = []
            # Take up to num_samples from the batch
            if self.num_samples:
                n = min(x.shape[0], self.num_samples)
            else:
                n = x.shape[0]
            
            for i in range(n):
                # Prepare a side-by-side visualization
                img = x[i].cpu().numpy()
                gt = y[i].cpu().numpy()
                pred = y_pred[i].cpu().numpy()

                def _possible_monochrome_to_multi_channel(z):
                    if z.shape[0] == 1:
                        return np.repeat(z, 3, axis=0)

                    if len(z.shape) == 2:
                        return np.stack([z] * 3)
                    
                    return z

                # Possible monochrome images (in_channels unknown) -> colour (in_channels=3)
                img = _possible_monochrome_to_multi_channel(img)
                gt = _possible_monochrome_to_multi_channel(gt)
                pred = _possible_monochrome_to_multi_channel(pred)

                # if batch_idx == 0:
                #     print(img.shape)
                #     print(gt.shape)
                #     print(pred.shape)

                # Log as a W&B Image with a caption
                combined_array = np.concatenate([img, gt, pred], axis=-1) # Stack horizontally
                combined_array = np.transpose(combined_array, (1, 2, 0))
                # if batch_idx == 0:
                #     print(combined_array.shape)

                combined = wandb.Image(
                    combined_array, 
                    caption=f"[{mode}] Batch {batch_idx}, Sample {i}: Input | Ground Truth | Pred"
                )
                images.append(combined)

            return images
        
        except ValueError as e:
            print('---- [Diagnosis] ----')
            print('img.shape = ', img.shape)
            print('gt.shape = ', gt.shape)
            print('pred.shape = ', pred.shape)
            # print('combined_array.shape = ', combined_array.shape)
            raise e

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % (batch[0].shape[0] // self.n_logs_per_batch) == 0:
            images = self.logging_images("training", trainer, pl_module, outputs, batch, batch_idx)
            trainer.logger.experiment.log({"train_predictions": images})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % (batch[0].shape[0] // self.n_logs_per_batch) == 0:
            images = self.logging_images("validation", trainer, pl_module, outputs, batch, batch_idx)
            trainer.logger.experiment.log({"val_predictions": images})