import warnings
from typing import List
from pydantic import BaseModel

from dataobjects.pitch import Pitch
from datamodules.base_datamodule import BaseDataModule, BaseDataModuleArgs

from .calibration import CalibrationDataset, CalibrationDatasetArgs

class PitchCalibrationDatasetArgs(CalibrationDatasetArgs, BaseModel):
    selected_idxs: List[int] = None

class PitchKeypointCalibrationDatasetArgs(PitchCalibrationDatasetArgs):
    pass

class PitchCalibrationDataset(CalibrationDataset):
    def __init__(self, args: PitchCalibrationDatasetArgs, mode):
        super().__init__(args, mode)
        self.selected_idxs = []

        match args.selected_idxs:
            case None:
                for idx in range(len(self.keys)):
                    _, labels = super().__getitem__(idx)
                    pitch = Pitch.from_one_annotation(labels, return_none_if_bad_annotation=True)
                    if pitch is not None:
                        self.selected_idxs.append(idx)

                    if self.n_limit is not None and len(self.selected_idxs) >= self.n_limit:
                        break

            case list(selected_idxs):
                self.selected_idxs = selected_idxs

    @classmethod
    def from_folder(cls, **kwargs):
        warnings.warn('Depercated! Use the default constructor instead.')

    def __len__(self):
        return len(self.selected_idxs)

    def __getitem__(self, idx):
        idx = self.selected_idxs[idx]
        image, labels = super().__getitem__(idx)

        pitch = Pitch.from_one_annotation(labels, height=self.height, width=self.width, return_none_if_bad_annotation=True)
        if pitch is None:
            return None, None
        return image, pitch

class PitchKeypointCalibrationDataset(PitchCalibrationDataset):
    def __getitem__(self, idx):
        image, pitch = super().__getitem__(idx)
        if pitch is None:
            return None, None
        return image, pitch.keypoints

class PitchCalibrationDataModule(BaseDataModule):
    def __init__(self, args_dataset: PitchCalibrationDatasetArgs, args_datamodule: BaseDataModuleArgs):
        super().__init__(args_datamodule)
        self.args_dataset = args_dataset

    def dataset(self, mode: str):
        if mode not in ['train', 'valid', 'test']:
            raise NotImplementedError(f'Got mode = {mode}. Expected either train, valid or test.')
        return PitchCalibrationDataset(self.args_dataset, mode)

class PitchKeypointCalibrationDataModule(BaseDataModule):
    def __init__(self, args_dataset: PitchKeypointCalibrationDatasetArgs, args_datamodule: BaseDataModuleArgs):
        super().__init__(args_datamodule)
        self.args_dataset = args_dataset

    def dataset(self, mode: str):
        if mode not in ['train', 'valid', 'test']:
            raise NotImplementedError(f'Got mode = {mode}. Expected either train, valid or test.')
        return PitchKeypointCalibrationDataset(self.args_dataset, mode)
