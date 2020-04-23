"""
Ben Hayes 2020

ECS7013P Deep Learning for Audio & Music

File: autoencoder.py
Description: Defines the encoder, decoder, and full autoencoder modules that
             pull together all the other components. Implemented largely as in
             the original paper, with a few PyTorch specific tweaks.
"""
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from features import F0Encoder, LoudnessEncoder
from modules import ZEncoder, MLP
from synthesis import HarmonicOscillatorBank, FilteredNoise
from utils import SAMPLE_RATE, SAMPLE_LENGTH_IN_SECONDS


class DDSPEncoder(nn.Module):
    """
    Defines the encoder part of the DDSP autoencoder. The only trainable part
    of this is the so-called ZEncoder. The LoudnessEncoder and F0Encoder are
    simply feature extractors whose output is concatenated with the learnt
    embedding.
    """
    def __init__(
            self,
            sr=SAMPLE_RATE,
            in_size_in_seconds=SAMPLE_LENGTH_IN_SECONDS,
            z_size=16,
            n_mfcc=30,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            rnn_dim=512,
            data_loud_mu=0,
            data_loud_std=1):
        """
        Constructs a DDSPEncoder

        Args:
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
            in_size_in_seconds (float, optional): The length in seconds of the
                input samples. Defaults to SAMPLE_LENGTH_IN_SECONDS.
            z_size (int, optional): Size of latent dimension. Defaults to 16.
            n_mfcc (int, optional): Number of MFCCs. Defaults to 30.
            n_fft (int, optional): FFT size. Defaults to 1024.
            hop_length (int, optional): FFT hop length. Defaults to 256.
            n_mels (int, optional): Number of mels. Defaults to 128.
            rnn_dim (int, optional): Number of RNN states. Defaults to 512.
            data_mu (int, optional): Mean loudness of training dataset.
                Defaults to 0.
            data_std (int, optional): Standard deviation of loudness of
                training dataset. Defaults to 1.
        """

        super().__init__()

        time_dim = sr * in_size_in_seconds // hop_length
        self.f0_enc = F0Encoder(
            sr=sr,
            out_size=time_dim)

        self.loudness_enc = LoudnessEncoder(
            data_loud_mu,
            data_loud_std,
            sr=SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            out_size=time_dim)

        self.z_enc = ZEncoder(
            in_size_in_seconds=in_size_in_seconds,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            rnn_dim=rnn_dim,
            z_size=z_size)

    def forward(self, x, known_f0=None):
        """
        Forward a batch through the network.

        Args:
            x (torch.Tensor): Batch of training examples. Size: [B, N]
            known_f0 (torch.Tensor, optional): Skip F0 extraction with a
                precomputed F0 vector. Defaults to None.

        Returns:
            tuple: Latent tuple of F0 vector, loudness vector, and latent
                   embedding.
        """
        return self.f0_enc(x, known_f0),\
            self.loudness_enc(x),\
            self.z_enc(x)


class DDSPDecoder(nn.Module):
    """
    Defines the decoder of the DDSP autoencoder. Largely a recurrent model,
    with skip connections for F0 and loudness vectors.
    """
    def __init__(
            self,
            sr=SAMPLE_RATE,
            rnn_channels=512,
            out_size_in_seconds=SAMPLE_LENGTH_IN_SECONDS,
            n_oscillators=101,
            filter_size=256,
            z_size=16):
        """
        Construct a DDSPDecoder

        Args:
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
            rnn_channels (int, optional): Number of RNN hidden states. Defaults
                to 512.
            out_size_in_seconds (float, optional): Length of the output in
                seconds. Defaults to SAMPLE_LENGTH_IN_SECONDS.
            n_oscillators (int, optional): Number of oscillators in the
                harmonic oscillator bank. Defaults to 101.
            filter_size (int, optional): Length of the FIR filter impulse
                response. Defaults to 256.
            z_size (int, optional): Size of the latent dimension. Defaults to
                16.
        """
        super().__init__()
        self.out_size = out_size_in_seconds * sr
        self.f0_mlp = MLP(in_size=1)
        self.loudness_mlp = MLP(in_size=1)
        self.z_mlp = MLP(in_size=z_size)

        self.gru = nn.GRU(
            input_size=rnn_channels * 3,
            hidden_size=rnn_channels)

        self.final_mlp = MLP(in_size=rnn_channels * 3)

        self.H_out = nn.Linear(rnn_channels, filter_size // 2 + 1)
        self.amp_out = nn.Linear(rnn_channels, n_oscillators)

        self.output_activation_H = nn.Sigmoid()  # ScaledSigmoid()
        self.output_activation_amp = nn.Sigmoid()  # ScaledSigmoid()

    def forward(self, f0, loudness, z):
        """
        Decode a sound given a latent tuple.

        Args:
            f0 (torch.Tensor): A vector containing frame-wise F0 frequency.
            loudness (torch.Tensor): A vector of frame-wise A-weighted loudness
            z (torch.Tensor): A vector containing the learnt latent embedding.

        Returns:
            tuple: A tuple containing two torch.Tensors: one describing frame-
                wise amplitudes of harmonic sinusoids, and one describing the
                frame-wise transfer function of an FIR filter.
        """
        y_f0 = self.f0_mlp(f0.permute(1, 0, 2))
        y_loudness = self.loudness_mlp(loudness.permute(1, 0, 2))
        y_z = self.z_mlp(z)

        y = torch.cat((y_f0, y_loudness, y_z), dim=-1)
        y, _ = self.gru(y)
        y = torch.cat((y, y_f0, y_loudness), dim=-1)
        y = self.final_mlp(y)

        H = self.H_out(y)
        H = self.output_activation_H(H)
        H = H.permute(1, 2, 0)

        amp = self.amp_out(y)
        amp = self.output_activation_amp(amp)
        amp = amp.permute(1, 2, 0)

        return amp, H


class DDSPAutoencoder(nn.Module):
    """
    The full DDSP Autoencoder model.
    """
    def __init__(
            self,
            sr=SAMPLE_RATE,
            sample_size_in_seconds=SAMPLE_LENGTH_IN_SECONDS,
            z_size=16,
            n_mfcc=30,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            rnn_dim=512,
            filter_size=128,
            data_mu=0,
            data_std=1,
            data_loud_mu=0,
            data_loud_std=1,
            n_oscillators=101):
        """
        Construct a DDSPAutoencoder instance

        Args:
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
            sample_size_in_seconds (float, optional): The length of the input
                and output in seconds. Defaults to SAMPLE_LENGTH_IN_SECONDS.
            z_size (int, optional): Size of latent dimension. Defaults to 16.
            n_mfcc (int, optional): Number of MFCCs. Defaults to 30.
            n_fft (int, optional): FFT size. Defaults to 1024.
            hop_length (int, optional): FFT hop length. Defaults to 256.
            n_mels (int, optional): Number of mel bands. Defaults to 128.
            rnn_dim (int, optional): Number of RNN states. Defaults to 512.
            filter_size (int, optional): Length of filter impulse response.
                Defaults to 128.
            data_mu (int, optional): Mean of dataset. Defaults to 0.
            data_std (int, optional): Standard deviation of dataset. Defaults
                to 1.
            data_loud_mu (int, optional): Mean loudness of dataset. Defaults
                to 0.
            data_loud_std (int, optional): Standard deviation of loudness of
                dataset. Defaults to 1.
            n_oscillators (int, optional): Number of oscillators in harmonic
                oscillator bank. Defaults to 101.
        """
        super().__init__()

        self.data_mu = data_mu
        self.data_std = data_std
        self.out_size = sr * sample_size_in_seconds

        self.encoder = DDSPEncoder(
            sr=sr,
            in_size_in_seconds=sample_size_in_seconds,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=256,
            n_mels=128,
            rnn_dim=512,
            data_loud_mu=data_loud_mu,
            data_loud_std=data_loud_std,
            z_size=z_size)

        self.decoder = DDSPDecoder(
            sr=sr,
            rnn_channels=rnn_dim,
            out_size_in_seconds=sample_size_in_seconds,
            filter_size=filter_size,
            z_size=z_size)

        self.synth = HarmonicOscillatorBank(
            sr=sr,
            out_size_in_seconds=sample_size_in_seconds,
            n_oscillators=n_oscillators)
        self.noise = FilteredNoise(
            sr=sr,
            out_size_in_seconds=sample_size_in_seconds,
            ir_length=filter_size)

    def forward(self, x, known_f0=None, params=False):
        """
        Reconstruct an audio sample using the DDSP autoencoder

        Args:
            x (torch.Tensor): Batch of audio samples to reconstruct
            known_f0 (torch.Tensor, optional): Skip F0 extraction with a
                precomputed F0 vector. Defaults to None.
            params (bool, optional): If true, return amplitude and transfer
                function parameter vectors in lieu of synthesised audio.
                Defaults to False.

        Returns:
            torch.Tensor: Synthesised audio
                or
            tuple: Two torch.Tensors containing frame-wise amplitudes and
                FIR transfer functions.
        """
        norm_x = (x - self.data_mu) / self.data_std
        f0, loudness, z = self.encoder(norm_x, known_f0)
        amp, H = self.decoder(f0, loudness, z)

        if params is False:
            smooth_f0 = interpolate(
                f0.permute(0, 2, 1),
                self.out_size, mode="linear")
            smooth_amp = interpolate(amp, self.out_size, mode="linear")

            filtered_noise = self.noise(H)
            y = self.synth(smooth_f0, smooth_amp)
            return y + filtered_noise
        else:
            return amp, H
