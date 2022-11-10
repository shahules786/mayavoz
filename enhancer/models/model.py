import os
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any, List, Optional, Text, Union
from urllib.parse import urlparse

import numpy as np
import pytorch_lightning as pl
import torch
from huggingface_hub import cached_download, hf_hub_url
from pytorch_lightning.utilities.cloud_io import load as pl_load
from torch import nn
from torch.optim import Adam

from enhancer.data.dataset import EnhancerDataset
from enhancer.inference import Inference
from enhancer.loss import LOSS_MAP, LossWrapper
from enhancer.version import __version__

CACHE_DIR = ""
HF_TORCH_WEIGHTS = ""
DEFAULT_DEVICE = "cpu"


class Model(pl.LightningModule):
    """
    Base class for all models
    parameters:
        num_channels: int, default to 1
            number of channels in input audio
        sampling_rate : int, default 16khz
            audio sampling rate
        lr: float, optional
            learning rate for model training
        dataset: EnhancerDataset, optional
            Enhancer dataset used for training/validation
        duration: float, optional
            duration used for training/inference
        loss : string or List of strings or custom loss (nn.Module), default to "mse"
            loss functions to be used. Available ("mse","mae","Si-SDR")

    """

    def __init__(
        self,
        num_channels: int = 1,
        sampling_rate: int = 16000,
        lr: float = 1e-3,
        dataset: Optional[EnhancerDataset] = None,
        duration: Optional[float] = None,
        loss: Union[str, List] = "mse",
        metric: Union[str, List, Any] = "mse",
    ):
        super().__init__()
        assert (
            num_channels == 1
        ), "Enhancer only support for mono channel models"
        self.dataset = dataset
        self.save_hyperparameters(
            "num_channels", "sampling_rate", "lr", "loss", "metric", "duration"
        )
        if self.logger:
            self.logger.experiment.log_dict(
                dict(self.hparams), "hyperparameters.json"
            )

        self.loss = loss
        self.metric = metric

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):

        if isinstance(loss, str):
            loss = [loss]

        self._loss = LossWrapper(loss)

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, metric):
        self._metric = []
        if isinstance(metric, (str, nn.Module)):
            metric = [metric]

        for func in metric:
            if isinstance(func, str):
                if func in LOSS_MAP.keys():
                    if func in ("pesq", "stoi"):
                        self._metric.append(
                            LOSS_MAP[func](self.hparams.sampling_rate)
                        )
                    else:
                        self._metric.append(LOSS_MAP[func]())
                else:
                    ValueError(f"Invalid metrics {func}")

            elif isinstance(func, nn.Module):
                self._metric.append(func)
            else:
                raise ValueError("Invalid metrics")

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            torch.cuda.empty_cache()
            self.dataset.setup(stage)
            self.dataset.model = self

            print(
                "Total train duration",
                self.dataset.train_dataloader().dataset.__len__()
                * self.dataset.duration
                / 60,
                "minutes",
            )
            print(
                "Total validation duration",
                self.dataset.val_dataloader().dataset.__len__()
                * self.dataset.duration
                / 60,
                "minutes",
            )
            print(
                "Total test duration",
                self.dataset.test_dataloader().dataset.__len__()
                * self.dataset.duration
                / 60,
                "minutes",
            )

    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def val_dataloader(self):
        return self.dataset.val_dataloader()

    def test_dataloader(self):
        return self.dataset.test_dataloader()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx: int):

        mixed_waveform = batch["noisy"]
        target = batch["clean"]
        prediction = self(mixed_waveform)
        loss = self.loss(prediction, target)

        self.log(
            "train_loss",
            loss.item(),
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):

        metric_dict = {}
        mixed_waveform = batch["noisy"]
        target = batch["clean"]
        prediction = self(mixed_waveform)

        metric_dict["valid_loss"] = self.loss(target, prediction).item()
        for metric in self.metric:
            value = metric(target, prediction)
            metric_dict[f"valid_{metric.name}"] = value.item()

        self.log_dict(
            metric_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return metric_dict

    def test_step(self, batch, batch_idx):

        metric_dict = {}
        mixed_waveform = batch["noisy"]
        target = batch["clean"]
        prediction = self(mixed_waveform)

        for metric in self.metric:
            value = metric(target, prediction)
            metric_dict[f"test_{metric.name}"] = value

        self.log_dict(
            metric_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return metric_dict

    def test_epoch_end(self, outputs):

        test_mean_metrics = defaultdict(int)
        for output in outputs:
            for metric, value in output.items():
                test_mean_metrics[metric] += value.item()
        for metric in test_mean_metrics.keys():
            test_mean_metrics[metric] /= len(outputs)

        print("----------TEST REPORT----------\n")
        for metric in test_mean_metrics.keys():
            print(f"|{metric.upper()} | {test_mean_metrics[metric]} |")
        print("--------------------------------")

    def on_save_checkpoint(self, checkpoint):

        checkpoint["enhancer"] = {
            "version": {"enhancer": __version__, "pytorch": torch.__version__},
            "architecture": {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
            },
        }

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Union[Path, Text],
        map_location=None,
        hparams_file: Union[Path, Text] = None,
        strict: bool = True,
        use_auth_token: Union[Text, None] = None,
        cached_dir: Union[Path, Text] = CACHE_DIR,
        **kwargs,
    ):
        """
        Load Pretrained model

        parameters:
        checkpoint : Path or str
            Path to checkpoint, or a remote URL, or a model identifier from
            the huggingface.co model hub.
        map_location: optional
            Same role as in torch.load().
            Defaults to `lambda storage, loc: storage`.
        hparams_file : Path or str, optional
            Path to a .yaml file with hierarchical structure as in this example:
                drop_prob: 0.2
                dataloader:
                    batch_size: 32
            You most likely won’t need this since Lightning will always save the
            hyperparameters to the checkpoint. However, if your checkpoint weights
            do not have the hyperparameters saved, use this method to pass in a .yaml
            file with the hparams you would like to use. These will be converted
            into a dict and passed into your Model for use.
        strict : bool, optional
            Whether to strictly enforce that the keys in checkpoint match
            the keys returned by this module’s state dict. Defaults to True.
        use_auth_token : str, optional
            When loading a private huggingface.co model, set `use_auth_token`
            to True or to a string containing your hugginface.co authentication
            token that can be obtained by running `huggingface-cli login`
        cache_dir: Path or str, optional
            Path to model cache directory
        kwargs: optional
            Any extra keyword args needed to init the model.
            Can also be used to override saved hyperparameter values.

        Returns
        -------
        model : Model
            Model

        See also
        --------
        torch.load
        """

        checkpoint = str(checkpoint)
        if hparams_file is not None:
            hparams_file = str(hparams_file)

        if os.path.isfile(checkpoint):
            model_path_pl = checkpoint
        elif urlparse(checkpoint).scheme in ("http", "https"):
            model_path_pl = checkpoint
        else:

            if "@" in checkpoint:
                model_id = checkpoint.split("@")[0]
                revision_id = checkpoint.split("@")[1]
            else:
                model_id = checkpoint
                revision_id = None

            url = hf_hub_url(
                model_id, filename=HF_TORCH_WEIGHTS, revision=revision_id
            )
            model_path_pl = cached_download(
                url=url,
                library_name="enhancer",
                library_version=__version__,
                cache_dir=cached_dir,
                use_auth_token=use_auth_token,
            )

        if map_location is None:
            map_location = torch.device(DEFAULT_DEVICE)

        loaded_checkpoint = pl_load(model_path_pl, map_location)
        module_name = loaded_checkpoint["enhancer"]["architecture"]["module"]
        class_name = loaded_checkpoint["enhancer"]["architecture"]["class"]
        module = import_module(module_name)
        Klass = getattr(module, class_name)

        try:
            model = Klass.load_from_checkpoint(
                checkpoint_path=model_path_pl,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )
        except Exception as e:
            print(e)

        return model

    def infer(self, batch: torch.Tensor, batch_size: int = 32):
        """
        perform model inference
        parameters:
            batch : torch.Tensor
                input data
            batch_size : int, default 32
                batch size for inference
        """

        assert (
            batch.ndim == 3
        ), f"Expected batch with 3 dimensions (batch,channels,samples) got only {batch.ndim}"
        batch_predictions = []
        self.eval().to(self.device)
        with torch.no_grad():
            for batch_id in range(0, batch.shape[0], batch_size):
                batch_data = batch[batch_id : (batch_id + batch_size), :, :].to(
                    self.device
                )
                prediction = self(batch_data)
                batch_predictions.append(prediction)

        return torch.vstack(batch_predictions)

    def enhance(
        self,
        audio: Union[Path, np.ndarray, torch.Tensor],
        sampling_rate: Optional[int] = None,
        batch_size: int = 32,
        save_output: bool = False,
        duration: Optional[int] = None,
        step_size: Optional[int] = None,
    ):
        """
        Enhance audio using loaded pretained model.

        parameters:
            audio: Path to audio file or numpy array or torch tensor
                single input audio
            sampling_rate: int, optional incase input is path
                sampling rate of input
            batch_size: int, default 32
                input audio is split into multiple chunks. Inference is done on batches
                of these chunks according to given batch size.
            save_output : bool, default False
                weather to save output to file
            duration : float, optional
                chunk duration in seconds, defaults to duration of loaded pretrained model.
            step_size: int, optional
                step size between consecutive durations, defaults to 50% of duration
        """

        model_sampling_rate = self.hparams["sampling_rate"]
        if duration is None:
            duration = self.hparams["duration"]
        waveform = Inference.read_input(
            audio, sampling_rate, model_sampling_rate
        )
        waveform.to(self.device)
        window_size = round(duration * model_sampling_rate)
        batched_waveform = Inference.batchify(
            waveform, window_size, step_size=step_size
        )
        batch_prediction = self.infer(batched_waveform, batch_size=batch_size)
        waveform = Inference.aggreagate(
            batch_prediction,
            window_size,
            waveform.shape[-1],
            step_size,
        )

        if save_output and isinstance(audio, (str, Path)):
            Inference.write_output(waveform, audio, model_sampling_rate)

        else:
            waveform = Inference.prepare_output(
                waveform, model_sampling_rate, audio, sampling_rate
            )
            return waveform

    @property
    def valid_monitor(self):

        return "max" if self.loss.higher_better else "min"
