from pydantic import BaseModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class BaseDataModuleArgs(BaseModel):
    batch_size: int = 4
    num_workers: int = 4
    persistent_workers: bool = True

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args: BaseDataModuleArgs):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.persistent_workers = args.persistent_workers

        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None

    def prepare_data(self):
        pass

    def dataset(self, mode):
        raise NotImplementedError()

    def setup(self, stage: str):
        match stage:
            case 'fit':
                self.train_ds = self.dataset(mode='train')
                self.valid_ds = self.dataset(mode='valid')
            case 'validate':
                self.valid_ds = self.dataset(mode='valid')
            case 'test':
                self.test_ds = self.dataset(mode='test')
            case 'predict':
                pass

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def teardown(self, stage: str):
        match stage:
            case 'fit':
                del self.train_ds
                del self.valid_ds
            case 'validate':
                del self.valid_ds
            case 'test':
                del self.test_ds
            case 'predict':
                pass
