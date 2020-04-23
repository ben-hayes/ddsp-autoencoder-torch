"""
Ben Hayes 2020

ECS7013P Deep Learning for Audio & Music

File: synthesis.py
Description: The differentiable synthesis modules used at the end of the model.
"""
import math

import torch
import torch.nn as nn

from utils import SAMPLE_LENGTH_IN_SECONDS, SAMPLE_RATE, param_ranges


class HarmonicOscillatorBank(nn.Module):
    """
    A differentiable harmonic oscillator bank, allowing gradients to
    backpropogate through a spectral modelling synthesis algorithm.
    """
    def __init__(
            self,
            sr=SAMPLE_RATE,
            out_size_in_seconds=SAMPLE_LENGTH_IN_SECONDS,
            n_oscillators=101,
            min_freq=param_ranges["f0"]["min"],
            max_freq=param_ranges["f0"]["max"]):
        """
        Construct a HarmonicOscillatorBank

        Args:
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
            out_size_in_seconds (float, optional): Size of the output in
                seconds. Defaults to SAMPLE_LENGTH_IN_SECONDS.
            n_oscillators (int, optional): Number of sinusoidal oscillators.
                Defaults to 101.
            min_freq (float, optional): Minimum oscillator frequency. Defaults
                to param_ranges["f0"]["min"].
            max_freq (float, optional): Maximum oscillator frequency. Defaults
                to param_ranges["f0"]["max"].
        """
        super().__init__()
        self.sr = sr
        self.out_size = int(sr * out_size_in_seconds)
        self.n_oscillators = n_oscillators
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.register_buffer("osc_indices", torch.linspace(
                1,
                self.n_oscillators,
                self.n_oscillators)
            .view(-1, 1))

    def forward(self, f0, amp):
        """
        Generate audio given fundamental frequency and amplitude parameter
        vectors.

        Args:
            f0 (torch.Tensor): Instantaneous fundamental frequency per sample.
            amp (torch.Tensor): Amplitude per sample per harmonic.

        Returns:
            torch.Tensor: Synthesised audio
        """
        scaled_f0 = f0 * (self.max_freq - self.min_freq) + self.min_freq
        freq_axis = (self.osc_indices * scaled_f0)
        omega_0 = math.tau * freq_axis / self.sr
        phase = torch.cumsum(omega_0, dim=-1)
        oscs = torch.cos(phase) * amp

        alias_filter = (freq_axis) > (0.5 * SAMPLE_RATE)
        oscs[alias_filter] = 0.0
        return oscs.sum(dim=1)


class FilteredNoise(nn.Module):
    """
    A differentiable filtered noise synthesiser. Consists of a noise generator
    and FIR filter parameterised by a series of frame-wise frequency domain
    transfer functions.
    """
    def __init__(
            self,
            sr=SAMPLE_RATE,
            out_size_in_seconds=SAMPLE_LENGTH_IN_SECONDS,
            hop_length=256,
            ir_length=256,
            attentuation=1):
        """
        Construct an instance of FilteredNoise

        Args:
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
            out_size_in_seconds (float, optional): Output size in seconds.
                Defaults to SAMPLE_LENGTH_IN_SECONDS.
            hop_length (int, optional): FFT hop size. Defaults to 256.
            ir_length (int, optional): Length of filter impulse response.
                Defaults to 256.
            attentuation (float, optional): Optional noise attenuation.
                Defaults to 1.
        """

        super().__init__()
        self.out_size = int(sr * out_size_in_seconds)
        self.hop_length = hop_length
        self.ir_length = ir_length
        self.attenuation = attentuation

        noise = torch.rand(self.out_size) * 2.0 - 1.0
        noise_frames = list(torch.chunk(noise, self.out_size // hop_length))
        noise_frames[-1] =\
            nn.functional.pad(
                noise_frames[-1],
                (0, noise_frames[0].shape[0] - noise_frames[-1].shape[0]))
        noise_frames = torch.stack(noise_frames, dim=0)
        noise_spec = torch.rfft(noise_frames, 1)

        self.register_buffer(
            "noise_spec",
            noise_spec.view(-1, 1, hop_length // 2 + 1, 2))

        window = torch.hann_window(ir_length)
        self.register_buffer("filter_window", window)

    def forward(self, H_mag):
        """
        Given the magnitude of a transfer function over several frames,
        generate a filtered noise signal.

        Args:
            H_mag (torch.Tensor): Magnitude of frame-wise FIR filter transfer
                function.

        Returns:
            torch.Tensor: Generated audio signal.
        """
        H_phase = torch.zeros_like(H_mag).to(device=H_mag.device)
        H_complex = torch.stack((H_mag, H_phase), dim=-1).permute(2, 0, 1, 3)

        h = torch.irfft(H_complex, 1, signal_sizes=(self.ir_length,))
        h = h.roll(self.ir_length // 2, -1)
        h_windowed = h * self.filter_window.view(1, 1, -1)
        h_padded = nn.functional.pad(
            h_windowed,
            (0, self.hop_length - self.ir_length))

        H_windowed = torch.rfft(h_padded, 1)

        H_re = H_windowed[:, :, :, 0]
        H_im = H_windowed[:, :, :, 1]
        X_re = self.noise_spec[:, :, :, 0]
        X_im = self.noise_spec[:, :, :, 1]

        Y_re = H_re * X_re - H_im * X_im
        Y_im = H_im * X_re + H_re * X_im
        Y = torch.stack((Y_re, Y_im), dim=-1)

        y = torch.irfft(Y, 1).view(H_mag.shape[0], -1)[:, :self.out_size]

        return y * self.attenuation
