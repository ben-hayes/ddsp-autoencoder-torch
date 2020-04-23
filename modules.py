"""
Ben Hayes 2020

ECS7013P Deep Learning for Audio & Music

File: modules.py
Description: Modular components that are combined to form the full model.
"""
import torch.nn as nn
from torchaudio.transforms import MFCC

from utils import SAMPLE_LENGTH_IN_SECONDS, SAMPLE_RATE


class ZEncoder(nn.Module):
    """
    The recurrent encoder module.
    """
    def __init__(
            self,
            in_size_in_seconds=SAMPLE_LENGTH_IN_SECONDS,
            sr=SAMPLE_RATE,
            n_mfcc=30,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            rnn_dim=512,
            z_size=16):
        """
        Construct an instance of ZEncoder

        Args:
            in_size_in_seconds (float, optional): The length of the input in
                seconds. Defaults to SAMPLE_LENGTH_IN_SECONDS.
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
            n_mfcc (int, optional): Number of MFCCs. Defaults to 30.
            n_fft (int, optional): FFT size. Defaults to 1024.
            hop_length (int, optional): FFT hop length. Defaults to 256.
            n_mels (int, optional): Number of mel bands. Defaults to 128.
            rnn_dim (int, optional): Number of RNN states. Defaults to 512.
            z_size (int, optional): Size of latent dimension. Defaults to 16.
        """
        super().__init__()
        self.sr = sr
        self.in_size = sr * in_size_in_seconds
        self.mfcc = MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "f_min": 20.0,
                "f_max": 8000.0
            })

        self.time_dim = int(self.in_size // hop_length)
        self.norm = nn.LayerNorm((n_mfcc, self.time_dim))

        self.gru = nn.GRU(input_size=n_mfcc, hidden_size=rnn_dim)

        self.linear = nn.Linear(rnn_dim, z_size)

    def forward(self, x):
        """
        Encode an audio sample to a latent vector

        Args:
            x (torch.Tensor): Batch of audio samples

        Returns:
            torch.Tensor: Time distributed latent code
        """
        z = self.mfcc(x)[:, :, :self.time_dim]
        z = self.norm(z)
        z = z.permute(2, 0, 1)
        z, _ = self.gru(z)
        z = self.linear(z)
        return z


class MLP(nn.Module):
    """
    A wrapper class for a simple multi-layer perceptron.
    """
    def __init__(self, in_size=512, out_size=512):
        """
        Construct a Multi-Layer Perceptron.

        Args:
            in_size (int, optional): Input dimension. Defaults to 512.
            out_size (int, optional): Output dimension. Defaults to 512.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LayerNorm(out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
            nn.LayerNorm(out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
            nn.LayerNorm(out_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Pass a batch through the multi-layer perceptron

        Args:
            x (torch.Tensor): Batch of samples to pass through.

        Returns:
            torch.Tensor: Transformed batch of samples.
        """
        return self.model(x)


class ScaledSigmoid(nn.Module):
    """
    An implementation of the scaled sigmoid function described in the appendix
    to the DDSP paper.

    TODO: this would make more sense as a subclass of torch.autograd.Function.
    """
    def __init__(self):
        """
        Create a ScaledSigmoid instance.
        """
        super().__init__()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Process a batch with the scaled sigmoid function.

        Args:
            x (torch.Tensor): The input batch.

        Returns:
            torch.Tensor: The output batch.
        """
        #  magic numbers are the scale and exponentiation factors employed in
        #  the original DDSP paper
        return 2.0 * self.sig(x) ** (2.30258509) + 10e-7
