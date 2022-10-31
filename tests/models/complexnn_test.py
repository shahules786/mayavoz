import torch

from enhancer.models.complexnn.conv import ComplexConv2d, ComplexConvTranspose2d


def test_complexconv2d():
    sample_input = torch.rand(1, 2, 256, 13)
    conv = ComplexConv2d(
        2, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)
    )
    with torch.no_grad():
        out = conv(sample_input)
    assert out.shape == torch.Size([1, 32, 128, 14])


def test_complexconvtranspose2d():
    sample_input = torch.rand(1, 512, 4, 13)
    conv = ComplexConvTranspose2d(
        256 * 2,
        128 * 2,
        kernel_size=(5, 2),
        stride=(2, 1),
        padding=(2, 0),
        output_padding=(1, 0),
    )
    with torch.no_grad():
        out = conv(sample_input)

    assert out.shape == torch.Size([1, 256, 8, 14])
