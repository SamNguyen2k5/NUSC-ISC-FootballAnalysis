import torch
import numpy as np
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

class LogConfigArtifactCallback(Callback):
    def __init__(self, config):
        self.config = config

    def on_train_start(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger):
            config_data = [[k, str(v)] for k, v in self.config.items()]
            table = wandb.Table(columns=["Parameter", "Value"], data=config_data)
            trainer.logger.experiment.log({"config_table": table})
