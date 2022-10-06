import numpy as np
import pytest
import torch

from enhancer.data.fileprocessor import Fileprocessor
from enhancer.utils.io import Audio


def test_io_channel():

    input_audio = np.random.rand(2, 32000)
    audio = Audio(mono=True, return_tensor=False)
    output_audio = audio(input_audio)
    assert output_audio.shape[0] == 1


def test_io_resampling():

    input_audio = np.random.rand(1, 32000)
    resampled_audio = Audio.resample_audio(input_audio, 16000, 8000)

    input_audio = torch.rand(1, 32000)
    resampled_audio_pt = Audio.resample_audio(input_audio, 16000, 8000)

    assert resampled_audio.shape[1] == resampled_audio_pt.size(1) == 16000


def test_fileprocessor_vctk():

    fp = Fileprocessor.from_name(
        "vctk",
        "tests/data/vctk/clean_testset_wav",
        "tests/data/vctk/noisy_testset_wav",
    )
    matching_dict = fp.prepare_matching_dict()
    assert len(matching_dict) == 2


@pytest.mark.parametrize("dataset_name", ["vctk", "dns-2020"])
def test_fileprocessor_names(dataset_name):
    fp = Fileprocessor.from_name(dataset_name, "clean_dir", "noisy_dir")
    assert hasattr(fp.matching_function, "__call__")


def test_fileprocessor_invaliname():
    with pytest.raises(ValueError):
        _ = Fileprocessor.from_name(
            "undefined", "clean_dir", "noisy_dir", 16000
        ).prepare_matching_dict()
