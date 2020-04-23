"""
Ben Hayes 2020

ECS7013P Deep Learning for Audio & Music

File: data.py
Description: A Dataloader subclass wrapping a subset of the NSynth dataset, as
             well as a handful of utility functions.
"""
import json
import os

import librosa
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utils import SAMPLE_RATE


class NSynthSubset(Dataset):
    """
    A wrapper for a subset of the NSynth dataset.
    """
    def __init__(self, path_to_wavs, path_to_metadata, sr=SAMPLE_RATE):
        """
        Construct an instance of NSynthSubset

        Args:
            path_to_wavs (str): Path to NSynth WAV files
            path_to_metadata (str): Path to NSynth metadata JSON file
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        """
        self.files = os.listdir(path_to_wavs)
        self.path_to_wavs = path_to_wavs
        self.sr = sr
        with open(path_to_metadata, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        """
        Get dataset length.

        Returns:
            int: Length of dataset.
        """
        return len(self.files)

    def __getitem__(self, i):
        """
        Fetch an item from the dataset

        Args:
            i (int): Index of item in dataset

        Returns:
            dict: A dict containing the raw PCM audio samples of the selected
                item, as well as its fundamental frequency.
        """
        file_path = os.path.join(self.path_to_wavs, self.files[i])
        audio_data, sr = librosa.load(file_path, sr=self.sr)
        file_key, _ = os.path.splitext(self.files[i])
        midi_note = self.metadata[file_key]["pitch"]
        f0 = librosa.core.midi_to_hz(midi_note)
        return {
            "audio": torch.from_numpy(audio_data),
            "f0": f0
        }


def get_nsynth_loaders(
        path_to_wavs,
        path_to_metadata,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        train_split=0.8,
        val_split=0.1,
        random_seed=1):
    """
    Create train, validation, and test DataLoader instances from a subset of
    the NSynth dataset on disk.

    Args:
        path_to_wavs (str): Path to NSynth WAV files
        path_to_metadata (str): Path to NSynth metadata JSON file
        batch_size (int, optional): Batch size. Defaults to 8.
        shuffle (bool, optional): Whether to shuffle data. Defaults to True.
        num_workers (int, optional): Number of file loading worker threads.
            Defaults to 8.
        train_split (float, optional): Proportion of data to use for training.
            Defaults to 0.8.
        val_split (float, optional): Proportion of data to use for validation.
            Defaults to 0.1.
        random_seed (int, optional): Specify a random seed for data
            partitioning. This allows for reproducibility of the experiment.
            Defaults to 1.

    Returns:
        tuple: Contains train, validation, and test DataLoaders.
    """

    torch.manual_seed(random_seed)  # set this to be deterministic about splits
    data = NSynthSubset(path_to_wavs, path_to_metadata)
    train_count, val_count = int(len(data) * 0.8), int(len(data)*0.1)
    train, val, test = random_split(
        data,
        (
            train_count,
            val_count,
            len(data) - (train_count + val_count)
        ))

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    return train_loader, val_loader, test_loader


def get_data_norms(data_loader, precalc=False):
    """
    Calculate normalisation values from a given data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader from which to
            calculate
        precalc (bool, optional): If true, use pre-calculated values to save
            computation time. Defaults to False.

    Returns:
        [type]: [description]
    """
    if precalc:
        return {
            "data_loud_mu": torch.tensor(-19.9182),
            "data_loud_std": torch.tensor(17.4076),
            "data_mu": torch.tensor(0.0029),
            "data_std": torch.tensor(0.1776)
        }

    from features import LoudnessEncoder

    loud_enc = LoudnessEncoder(0, 1)
    loud_means = []
    loud_stds = []
    means = []
    stds = []

    for data in data_loader:
        audio = data["audio"]
        loudness = loud_enc(audio)
        means.append(audio.mean())
        stds.append(audio.std())
        loud_means.append(loudness.mean())
        loud_stds.append(loudness.std())

    return {
        "data_loud_mu": torch.mean(torch.tensor(loud_means)),
        "data_loud_std": torch.mean(torch.tensor(loud_stds)),
        "data_mu": torch.mean(torch.tensor(means)),
        "data_std": torch.mean(torch.tensor(stds))
    }
