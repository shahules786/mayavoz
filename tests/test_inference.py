import pytest
import torch
import numpy as np

from enhancer.inference import Inference


@pytest.mark.parametrize("audio",["tests/data/vctk/clean_testset_wav/p257_166.wav",torch.rand(1,2,48000)])
def test_read_input(audio):

    read_audio = Inference.read_input(audio,48000,16000)
    assert isinstance(read_audio,torch.Tensor)
    assert read_audio.shape[0] == 1

def test_batchify():
    rand = torch.rand(1,1000)
    batched_rand = Inference.batchify(rand, window_size = 100, step_size=100)
    assert batched_rand.shape[0] == 12

def test_aggregate():
    rand = torch.rand(12,1,100)
    agg_rand = Inference.aggreagate(data=rand,window_size=100,total_frames=1000,step_size=100)
    assert agg_rand.shape[-1] == 1000


    