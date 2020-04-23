import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram

class MultiscaleSpectralLoss(nn.Module):
    def __init__(self, scales=(2048, 1024, 512, 256, 128, 64)):
        super(MultiscaleSpectralLoss, self).__init__()
        self.scales = scales
        self.stfts = nn.ModuleList([
            Spectrogram(
                n_fft=scale,
                hop_length=(scale//4))
            for scale in scales
        ])
        self.l1s = nn.ModuleList([
            nn.L1Loss(reduction='mean')
            for scale in scales
        ])
    
    def forward(self, original, reconstructed):
        losses = []
        for stft, l1 in zip(self.stfts, self.l1s):
            spec_original = stft(original)
            spec_reconstructed = stft(reconstructed)
            diff = torch.abs(spec_original - spec_reconstructed)
            losses.append(diff.mean())
            
            log_original = torch.log(spec_original + 1e-7)
            log_reconstructed = torch.log(spec_reconstructed + 1e-7)
            diff = torch.abs(log_original - log_reconstructed)
            losses.append(diff.mean())
            
        return torch.stack(losses).sum()