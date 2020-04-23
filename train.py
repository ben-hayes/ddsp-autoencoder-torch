"""
Ben Hayes 2020

ECS7013P Deep Learning for Audio & Music

File: train.py
Description: A simple script for training the DDSP autoencoder.
"""
from argparse import ArgumentParser
import os

import torch

from autoencoder import DDSPAutoencoder
from data import get_nsynth_loaders, get_data_norms
from loss import MultiscaleSpectralLoss


def parse_args():
    parser = ArgumentParser("DDSP AutoEncoder for Torch Train Script")
    parser.add_argument("data_wavs")
    parser.add_argument("data_json")
    parser.add_argument("-o", "--model_save_path", type=str, default="models")
    parser.add_argument("-n", "--n_epochs", type=int, default=1000)
    parser.add_argument(
        "-e",
        "--early_stopping_patience",
        type=int,
        default=30)
    parser.add_argument(
        "-l",
        "--initial_learning_rate",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "-c",
        "--cuda_device",
        type=int,
        default=0
    )
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--calculate_data_norms", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--sample_size_in_seconds", type=float, default=4.0)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--lr_decay_interval", type=int, default=10000)

    return parser.parse_args()


def get_device(device_index):
    if device_index == -1:
        return torch.device('cpu')
    else:
        return torch.device('cuda:%d' % device_index)


def create_model(
        sr,
        sample_size_in_seconds,
        z_dim,
        device,
        data_mu,
        data_std,
        data_loud_mu,
        data_loud_std):

    net = DDSPAutoencoder(
        sr=sr,
        sample_size_in_seconds=sample_size_in_seconds,
        z_size=z_dim,
        data_mu=data_mu,
        data_std=data_std,
        data_loud_mu=data_loud_mu,
        data_loud_std=data_loud_std)
    return net.to(device)


def train(
        model,
        device,
        data_loader,
        optimizer,
        criterion,
        scheduler,
        lr_decay_interval,
        n_steps=0):

    model.train()
    total_loss = 0.0

    for batch, data in enumerate(data_loader):
        model.zero_grad()
        optimizer.zero_grad()

        audio = data["audio"].to(device)
        reconstruction = model(audio)

        loss = criterion(audio, reconstruction)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        n_steps += 1
        if n_steps % lr_decay_interval == lr_decay_interval - 1:
            scheduler.step()

        print("Batch #%d - train loss %.4f" % (batch, loss.item()), end="\r")

    epoch_loss = total_loss / (batch + 1)
    print("Batch #%d - train loss %.4f     " % (batch, epoch_loss))

    return epoch_loss, n_steps


def validate(
        model,
        device,
        data_loader,
        criterion,
        type="val"):

    model.eval()
    total_loss = 0.0

    for batch, data in enumerate(data_loader):
        audio = data["audio"].to(device)
        reconstruction = model(audio)

        loss = criterion(audio, reconstruction)
        total_loss += loss.item()

        print("Batch #%d - %s loss %.4f"
              % (batch, type, loss.item()), end="\r")

    val_loss = total_loss / (batch + 1)
    print("Batch #%d - %s loss %.4f     " % (batch, type, val_loss))

    return val_loss


def save_checkpoint(
        model,
        path,
        optimizer=None,
        scheduler=None,
        last_val_loss=None,
        last_train_loss=None,
        epoch=None,
        lowest_val=None,
        epochs_since_lowest=None):

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "last_val_loss": last_val_loss,
        "last_train_loss": last_train_loss,
        "lowest_val": lowest_val,
        "epochs_since_lowest": epochs_since_lowest
    }, path)


def run_train_loop(
        model,
        device,
        train_loader,
        val_loader,
        test_loader,
        n_epochs,
        lr,
        lr_decay,
        lr_decay_interval,
        model_save_path):

    criterion = MultiscaleSpectralLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

    n_steps = 0
    lowest_val_loss = 1e100
    epochs_since_lowest = 0
    for epoch in range(n_epochs):
        print("======== Epoch #%d ========" % epoch)

        train_loss, n_steps = train(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            lr_decay_interval,
            n_steps)
        val_loss = validate(
            model,
            device,
            val_loader,
            criterion
        )

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            epochs_since_lowest = 0

            save_checkpoint(
                model,
                os.path.join(model_save_path, "best_val_loss_model.pt"),
                optimizer,
                scheduler,
                val_loss,
                train_loss,
                epoch,
                lowest_val_loss,
                epochs_since_lowest)
        else:
            epochs_since_lowest += 1

        if epoch % 10 == 0:
            save_checkpoint(
                model,
                os.path.join(model_save_path, "epoch_%d.pt" % epoch),
                optimizer,
                scheduler,
                val_loss,
                train_loss,
                epoch,
                lowest_val_loss,
                epochs_since_lowest)

        if epochs_since_lowest >= 30:
            print(
                "Validation loss hasn't decreased for 30 epochs. Stopping...")
            save_checkpoint(
                model,
                os.path.join(model_save_path, "final_model.pt"),
                optimizer,
                scheduler,
                val_loss,
                train_loss,
                epoch,
                lowest_val_loss,
                epochs_since_lowest)

    validate(model, device, test_loader, criterion, type="test")


if __name__ == "__main__":
    args = parse_args()

    train_loader, val_loader, test_loader = get_nsynth_loaders(
        args.data_wavs,
        args.data_json,
        args.batch_size)
    data_norms = get_data_norms(
        train_loader,
        precalc=not args.calculate_data_norms)

    device = get_device(args.cuda_device)

    model = create_model(
        args.sample_rate,
        args.sample_size_in_seconds,
        args.z_dim,
        device,
        **data_norms)

    run_train_loop(
        model,
        device,
        train_loader,
        val_loader,
        test_loader,
        args.n_epochs,
        args.initial_learning_rate,
        args.lr_decay,
        args.lr_decay_interval,
        args.model_save_path)
