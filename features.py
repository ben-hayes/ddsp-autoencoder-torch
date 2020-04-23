"""
Ben Hayes 2020

ECS7013P Deep Learning for Audio & Music

File: features.py
Description: Feature extractors used on the front end of the network to create
             the conditioned latent tuple.
"""
import crepe
import librosa
import numpy as np
from pyworld import dio
import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram, AmplitudeToDB

from utils import SAMPLE_RATE, param_ranges


class LoudnessEncoder(nn.Module):
    """
    A feature extractor for A-weighted loudness.
    """
    def __init__(
            self,
            data_loud_mu,
            data_loud_std,
            sr=SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            out_size=250):
        """
        Construct an instance of LoudnessEncoder

        Args:
            data_loud_mu (torch.Tensor): The mean loudness of the dataset
            data_loud_std (torch.Tensor): The standard deviation of the
                loudness of the dataset
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
            n_fft (int, optional): FFT size. Defaults to 1024.
            hop_length (int, optional): FFT hop length. Defaults to 256.
            out_size (int, optional): Output length in frames. Defaults to 250.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.data_loud_mu = data_loud_mu
        self.data_loud_std = data_loud_std
        self.time_dim = int(out_size)

        self.stft = Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.db = AmplitudeToDB()
        self.centre_freqs = np.linspace(0, sr // 2, n_fft // 2 + 2)[:-1]
        self.register_buffer(
            "weightings",
            torch.tensor(
                librosa.core.A_weighting(self.centre_freqs)).view(-1, 1)
        )

    def forward(self, x):
        """
        Extract A-weighted loudness from an audio tensor

        Args:
            x (torch.Tensor): A tensor of audio samples.

        Returns:
            torch.Tensor: A tensor of frame-wise loudness
        """
        spec = self.stft(x)[:, :, :self.time_dim]
        db_spec = self.db(spec)
        weighted_db_spec =\
            self.weightings.repeat(x.shape[0], 1, db_spec.shape[-1]) + db_spec
        mag_spec = 10.0 ** (weighted_db_spec / 20.0)
        return ((self.db(mag_spec.mean(dim=-2)) + self.data_loud_mu)
                / self.data_loud_std).float().unsqueeze(-1)


class F0Encoder(nn.Module):
    """
    A wrapper for fundamental frequency feature extractors.
    """
    def __init__(
            self,
            sr=SAMPLE_RATE,
            min_freq=param_ranges["f0"]["min"],
            max_freq=param_ranges["f0"]["max"],
            hop_length=256,
            out_size=250):
        """
        Construct an instance of F0Encoder

        Args:
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
            min_freq (float, optional): Minimum output frequency. Defaults to
                param_ranges["f0"]["min"].
            max_freq (float, optional): Maximum output frequency. Defaults to
                param_ranges["f0"]["max"].
            hop_length (int, optional): Hop length in samples. Defaults to 256.
            out_size (int, optional): Output size in frames. Defaults to 250.
        """
        super().__init__()
        self.sr = sr
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.hop_length = hop_length
        self.time_dim = out_size

    def forward(self, x, known_f0=None, scale=True, mode='dio'):
        if known_f0 is not None:
            f0 = known_f0.view(-1, 1, 1)
            f0 = known_f0.repeat(
                1,
                x.shape[1] // self.hop_length,
                1).view(x.shape[0], -1, 1)

            if scale:
                f0 = f0 - self.min_freq
                f0 = f0 / (self.max_freq - self.min_freq)
            return f0

        if mode == 'crepe':
            freq_estimates = [
                torch.tensor(crepe.predict(
                    sample.cpu().numpy(),
                    self.sr,
                    viterbi=True,
                    verbose=0,
                    step_size=1000*self.hop_length/self.sr)[1]).float()
                for sample in x
            ]
            freq = torch.stack(freq_estimates, dim=0)
        elif mode == 'dio':
            freq_estimates = [
                torch.tensor(dio(
                    sample.double().cpu().numpy(),
                    self.sr,
                    frame_period=1000*self.hop_length/self.sr,
                    f0_floor=50,
                    f0_ceil=2000)[0]).float()
                for sample in x
            ]

            freq = torch.stack(freq_estimates, dim=0)

        if scale:
            freq = freq - self.min_freq
            freq = freq / (self.max_freq - self.min_freq)

        return torch.clamp(
            freq.unsqueeze(-1)[:, :self.time_dim, :]
                .to(device=x.device),
            0.0,
            1.0)
