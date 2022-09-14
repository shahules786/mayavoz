import pytest
import torch
from enhancer import data

from enhancer.utils.config import Files
from enhancer.models import Demucs
from enhancer.data.dataset import EnhancerDataset


@pytest.fixture
def vctk_dataset():
    root_dir = "tests/data/vctk"
    files = Files(train_clean="clean_testset_wav",train_noisy="noisy_testset_wav",
         test_clean="clean_testset_wav", test_noisy="noisy_testset_wav")
    dataset = EnhancerDataset(name="vctk",root_dir=root_dir,files=files)
    return dataset
    


@pytest.mark.parametrize("batch_size,samples",[(1,1000)])
def test_forward(batch_size,samples):
    model = Demucs()
    model.eval()

    data = torch.rand(batch_size,1,samples,requires_grad=False)
    with torch.no_grad():
        _ = model(data)

    data = torch.rand(batch_size,2,samples,requires_grad=False)
    with torch.no_grad():
        with pytest.raises(TypeError):
            _ = model(data)


@pytest.mark.parametrize("dataset,channels,loss",
                        [(pytest.lazy_fixture("vctk_dataset"),1,["mae","mse"])])
def test_demucs_init(dataset,channels,loss):
    with torch.no_grad():
        model = Demucs(num_channels=channels,dataset=dataset,loss=loss)





    

