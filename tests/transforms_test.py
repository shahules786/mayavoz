import torch

from enhancer.utils.transforms import ConviSTFT, ConvSTFT


def test_stft_istft():
    sample_input = torch.rand(1, 1, 16000)
    stft = ConvSTFT(window_len=400, hop_size=100, nfft=512)
    istft = ConviSTFT(window_len=400, hop_size=100, nfft=512)

    with torch.no_grad():
        spectrogram = stft(sample_input)
        waveform = istft(spectrogram)
    assert sample_input.shape == waveform.shape
