import math
import multiprocessing
import os
from itertools import chain, cycle
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from enhancer.data.fileprocessor import Fileprocessor
from enhancer.utils import check_files
from enhancer.utils.config import Files
from enhancer.utils.io import Audio
from enhancer.utils.random import create_unique_rng


class TrainDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset.train__iter__()

    def __len__(self):
        return self.dataset.train__len__()


class ValidDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset.val__getitem__(idx)

    def __len__(self):
        return self.dataset.val__len__()


class TestDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset.test__getitem__(idx)

    def __len__(self):
        return self.dataset.test__len__()


class TaskDataset(pl.LightningDataModule):
    def __init__(
        self,
        name: str,
        root_dir: str,
        files: Files,
        valid_minutes: float = 0.20,
        duration: float = 1.0,
        stride=None,
        sampling_rate: int = 48000,
        matching_function=None,
        batch_size=32,
        num_workers: Optional[int] = None,
    ):
        super().__init__()

        self.name = name
        self.files, self.root_dir = check_files(root_dir, files)
        self.duration = duration
        self.stride = stride or duration
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.matching_function = matching_function
        self._validation = []
        if num_workers is None:
            num_workers = multiprocessing.cpu_count() // 2
        self.num_workers = num_workers
        if valid_minutes > 0.0:
            self.valid_minutes = valid_minutes
        else:
            raise ValueError("valid_minutes must be greater than 0")

    def setup(self, stage: Optional[str] = None):
        """
        prepare train/validation/test data splits
        """

        if stage in ("fit", None):

            train_clean = os.path.join(self.root_dir, self.files.train_clean)
            train_noisy = os.path.join(self.root_dir, self.files.train_noisy)
            fp = Fileprocessor.from_name(
                self.name, train_clean, train_noisy, self.matching_function
            )
            train_data = fp.prepare_matching_dict()
            train_data, self.val_data = self.train_valid_split(
                train_data, valid_minutes=self.valid_minutes, random_state=42
            )

            self.train_data = self.prepare_traindata(train_data)
            self._validation = self.prepare_mapstype(self.val_data)

            test_clean = os.path.join(self.root_dir, self.files.test_clean)
            test_noisy = os.path.join(self.root_dir, self.files.test_noisy)
            fp = Fileprocessor.from_name(
                self.name, test_clean, test_noisy, self.matching_function
            )
            test_data = fp.prepare_matching_dict()
            self._test = self.prepare_mapstype(test_data)

    def train_valid_split(
        self, data, valid_minutes: float = 20, random_state: int = 42
    ):

        valid_minutes *= 60
        valid_min_now = 0.0
        valid_indices = []
        random_indices = list(range(0, len(data)))
        rng = create_unique_rng(random_state, 0)
        rng.shuffle(random_indices)
        i = 0
        while valid_min_now <= valid_minutes:
            valid_indices.append(random_indices[i])
            valid_min_now += data[random_indices[i]]["duration"]
            i += 1

        train_data = [
            item for i, item in enumerate(data) if i not in valid_indices
        ]
        valid_data = [item for i, item in enumerate(data) if i in valid_indices]
        return train_data, valid_data

    def prepare_traindata(self, data):
        train_data = []
        for item in data:
            samples_metadata = []
            clean, noisy, total_dur = item.values()
            num_segments = self.get_num_segments(
                total_dur, self.duration, self.stride
            )
            for index in range(num_segments):
                start = index * self.stride
                samples_metadata.append(
                    ({"clean": clean, "noisy": noisy}, start)
                )
            train_data.append(samples_metadata)
        return train_data[:25]

    @staticmethod
    def get_num_segments(file_duration, duration, stride):

        if file_duration < duration:
            num_segments = 1
        else:
            num_segments = math.ceil((file_duration - duration) / stride) + 1

        return num_segments

    def prepare_mapstype(self, data):

        metadata = []
        for item in data:
            clean, noisy, total_dur = item.values()
            if total_dur < self.duration:
                metadata.append(({"clean": clean, "noisy": noisy}, 0.0))
            else:
                num_segments = round(total_dur / self.duration)
                for index in range(num_segments):
                    start_time = index * self.duration
                    metadata.append(
                        ({"clean": clean, "noisy": noisy}, start_time)
                    )
        return metadata

    def train_collatefn(self, batch):
        names = []
        output = {"noisy": [], "clean": []}
        for item in batch:
            output["noisy"].append(item["noisy"])
            output["clean"].append(item["clean"])
            names.append(item["name"])

        print(names)
        output["clean"] = torch.stack(output["clean"], dim=0)
        output["noisy"] = torch.stack(output["noisy"], dim=0)
        return output

    def worker_init_fn(self, _):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        worker_id = worker_info.id
        split_size = len(dataset.dataset.train_data) // worker_info.num_workers
        dataset.data = dataset.dataset.train_data[
            worker_id * split_size : (worker_id + 1) * split_size
        ]

    def train_dataloader(self):
        return DataLoader(
            TrainDataset(self),
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.train_collatefn,
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            ValidDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            TestDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class EnhancerDataset(TaskDataset):
    """
    Dataset object for creating clean-noisy speech enhancement datasets
    paramters:
    name : str
        name of the dataset
    root_dir : str
        root directory of the dataset containing clean/noisy folders
    files : Files
        dataclass containing train_clean, train_noisy, test_clean, test_noisy
        folder names (refer enhancer.utils.Files dataclass)
    duration : float
        expected audio duration of single audio sample for training
    sampling_rate : int
        desired sampling rate
    batch_size : int
        batch size of each batch
    num_workers : int
        num workers to be used while training
    matching_function : str
        maching functions - (one_to_one,one_to_many). Default set to None.
        use one_to_one mapping for datasets with one noisy file for each clean file
        use one_to_many mapping for multiple noisy files for each clean file


    """

    def __init__(
        self,
        name: str,
        root_dir: str,
        files: Files,
        valid_minutes=5.0,
        duration=1.0,
        stride=0.5,
        sampling_rate=48000,
        matching_function=None,
        batch_size=32,
        num_workers: Optional[int] = None,
    ):

        super().__init__(
            name=name,
            root_dir=root_dir,
            files=files,
            valid_minutes=valid_minutes,
            sampling_rate=sampling_rate,
            duration=duration,
            matching_function=matching_function,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.sampling_rate = sampling_rate
        self.files = files
        self.duration = max(1.0, duration)
        self.audio = Audio(self.sampling_rate, mono=True, return_tensor=True)
        self.stride = stride or duration

    def setup(self, stage: Optional[str] = None):

        super().setup(stage=stage)

    def random_sample(self, index):
        rng = create_unique_rng(self.model.current_epoch, index)
        return rng.sample(self.train_data, len(self.train_data))

    def train__iter__(self):
        return zip(
            *[
                self.get_stream(self.random_sample(i))
                for i in range(self.batch_size)
            ]
        )

    def get_stream(self, data):
        return chain.from_iterable(map(self.process_data, cycle(data)))

    def process_data(self, data):
        for item in data:
            yield self.prepare_segment(*item)

    @staticmethod
    def get_num_segments(file_duration, duration, stride):

        if file_duration < duration:
            num_segments = 1
        else:
            num_segments = math.ceil((file_duration - duration) / stride) + 1

        return num_segments

    def val__getitem__(self, idx):
        return self.prepare_segment(*self._validation[idx])

    def test__getitem__(self, idx):
        return self.prepare_segment(*self._test[idx])

    def prepare_segment(self, file_dict: dict, start_time: float):

        clean_segment = self.audio(
            file_dict["clean"], offset=start_time, duration=self.duration
        )
        noisy_segment = self.audio(
            file_dict["noisy"], offset=start_time, duration=self.duration
        )
        clean_segment = F.pad(
            clean_segment,
            (
                0,
                int(
                    self.duration * self.sampling_rate - clean_segment.shape[-1]
                ),
            ),
        )
        noisy_segment = F.pad(
            noisy_segment,
            (
                0,
                int(
                    self.duration * self.sampling_rate - noisy_segment.shape[-1]
                ),
            ),
        )
        return {
            "clean": clean_segment,
            "noisy": noisy_segment,
            "name": file_dict["clean"].split("/")[-1] + "->" + start_time,
        }

    def train__len__(self):

        return sum([len(item) for item in self.train_data])

    def val__len__(self):
        return len(self._validation)

    def test__len__(self):
        return len(self._test)
