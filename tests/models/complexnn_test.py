import torch

from enhancer.models.complexnn.conv import ComplexConv2d


def test_complexconv2d():
    sample_input = torch.rand(1, 2, 256, 13)
    conv = ComplexConv2d(
        2, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)
    )
    with torch.no_grad():
        out = conv(sample_input)
    assert out.shape == torch.Size([1, 32, 128, 14])
