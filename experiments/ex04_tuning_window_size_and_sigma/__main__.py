from torch.nn import L1Loss

from importlib import resources
import wandb
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner import Tuner

from datamodules.spiideo.gaussian import Spiideo2DGaussianMaskDataModule, Spiideo2DGaussianMaskDatasetArgs
from datamodules.base_datamodule import BaseDataModuleArgs
from models.base.unet_heatmap import UNetHeatmap
from models.player_detection.with_end_blur import UNetHeatmapWithEndBlur

from callbacks.log_heatmap import LogHeatmapCallback
from callbacks.log_config import LogConfigArtifactCallback

from losses.focal_loss import FocalLoss
from losses.total_intensity import TotalIntensity

with resources.files(__package__).joinpath('config.yaml').open('r', encoding='utf-8') as f: 
    config = yaml.safe_load(f)

spiideo_2d_gaussian_data_module = Spiideo2DGaussianMaskDataModule(
    Spiideo2DGaussianMaskDatasetArgs(**config['spiideo_dataset']), 
    BaseDataModuleArgs(**config['spiideo_datamodule']),
    debug=True
)

# unet_model = UNetHeatmap(
unet_model = UNetHeatmapWithEndBlur(
    lr=1e-5,
    loss_fns={
        'focal': FocalLoss(alpha=0.99, gamma=2),
        'intensity': TotalIntensity(),
    }, 
    lambdas=[0.99, 0.01],
    # lambdas=[1],
)

wandb_logger = WandbLogger(save_dir='wandb_logs', project='ISC-Football', name=config['experiment_name'])
trainer = pl.Trainer(
    **config['trainer'],
    logger=wandb_logger, 
    callbacks=[
        LogConfigArtifactCallback(config), 
        LogHeatmapCallback(), 
        # EarlyStopping('validation_loss', mode='min', patience=3),
    ]
)

# tuner = Tuner(trainer)

# lr_results = tuner.lr_find(model=unet_model, datamodule=spiideo_2d_gaussian_data_module)
# fig_lr_results = lr_results.plot(suggest=True)
# fig_lr_results.savefig('lr_find.png')
# wandb_logger.experiment.log_artifact('lr_find.png')

# artifact = wandb.Artifact(name="lr_results", type="analysis")
# artifact.add_file('lr_find.png')
# wandb_logger.experiment.log_artifact(artifact)

trainer.fit(model=unet_model, datamodule=spiideo_2d_gaussian_data_module)