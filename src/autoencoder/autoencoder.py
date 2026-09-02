# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

import json
import os
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchsummary
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

MODELS_ROOT = os.environ.get("AE_MODELS", "./models")

class Autoencoder(torch.nn.Module, ABC):
    """Shared training, checkpointing and latent-space plotting for all AE variants.

    Subclasses set up encoder, decoder and loss terms in their own __init__ and
    override fit() where the loss needs extra terms. This class is deliberately
    not abstract about __init__: the variants differ in what they need to store
    before _make_model_name is called, so each one builds its own state.
    """

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device

    def _create_model_folder(self, folder=None):
        if folder is None:
            models_folder = MODELS_ROOT
        else:
            models_folder = folder
        self.storage_path = os.path.join(models_folder, self.model_name)
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            # print(f"folder made at {self.storage_path}")

    def _make_model_name(self, extras=None):
        current_time = datetime.now()
        time_string = current_time.strftime("%Y%m%d_%H%M")
        latent_space = self.latent_dim
        model_name = f"{time_string}_{self.model_type}_Lat{latent_space}"
        if extras is not None:
            model_name = f"{model_name}_{extras}"
        self.model_name = model_name
        return self.model_name

    def evaluate(self, eval_loader):
        self.to(self.device)
        self.eval()  # Set the model to evaluation mode
        eval_loss = 0.0
        for eval_data, _ in eval_loader:
            data = eval_data.to(self.device)
            with torch.no_grad():
                _, decoded = self(data)
                val_loss = self.reconstruction_loss_function(data, decoded)
            eval_loss += val_loss.item()

        eval_loss /= len(eval_loader)
        return eval_loss

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=10,
        device=None,
        lr=1e-3,
    ):
        if device is not None:
            self.device = device
        self.to(self.device)  # Move the model to the specified device
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        torch.cuda.empty_cache()
        n_tr_batches = len(train_loader)
        for epoch in range(epochs):
            reconstruction_error = 0.0
            self.train()  # Set the model to training mode
            progress_bar = tqdm(range(n_tr_batches), total=n_tr_batches, ncols=100)
            progress_bar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            for data, _ in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()  # Zero the parameter gradients
                _, decoded = self(data)  # Forward pass
                loss = self.reconstruction_loss_function(data, decoded)
                loss.backward()  # Backward pass to compute the gradient
                optimizer.step()  # Update the model weights
                reconstruction_error += loss.item()
                progress_bar.update()
                progress_bar.set_postfix(tr_loss=f"{loss.item():.5f}")
            reconstruction_error /= n_tr_batches
            # print(reconstruction_error)
            self.history["tr_re"].append(reconstruction_error)
            if val_loader is not None:
                # Evaluation phase
                eval_loss = self.evaluate(val_loader)
                self.history["val_re"].append(eval_loss)
                progress_bar.set_postfix(
                    tr_re=f"{reconstruction_error:.5f}",
                    val_re=f"{eval_loss:.5f}",
                )
            else:
                progress_bar.set_postfix(
                    tr_re=f"{reconstruction_error:.5f}",
                )
            progress_bar.close()
            # torch.cuda.empty_cache()
            if reconstruction_error <= self.best_loss:
                self.best_loss = reconstruction_error  # Update best loss
                self.save_checkpoint(optimizer, self.best_loss)
        return self.history

    def save_train_history(self):
        # Convert the dictionary to a JSON string
        dict_json = json.dumps(self.history, indent=4)  # 'indent' for pretty-printing
        history_path = os.path.join(self.storage_path, "train_history.json")
        # Write the JSON string to a file
        with open(history_path, "w") as file:
            file.write(dict_json)

    def save_checkpoint(self, optimizer, loss, path=None):
        """Save model checkpoint."""
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": loss,
        }
        if path is None:
            checkpoint_path = self.storage_path
        else:
            checkpoint_path = path
        checkpoint_path = os.path.join(checkpoint_path, "best_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved with loss: {loss}")

    def load_checkpoint(self, path=None, optimizer=None):
        if path is None:
            checkpoint_path = os.path.join(self.storage_path, "best_checkpoint.pth")
        else:
            checkpoint_path = path
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        self.best_loss = checkpoint["best_loss"]
        print(f"Checkpoint loaded with best loss: {self.best_loss}")

    def plot_training_history(self, store=False, show_plot=True):
        x_values = range(1, len(self.history["tr_re"]) + 1)
        for key, value in self.history.items():
            if value:  # If not empty
                plt.plot(x_values, value, label=key)
        plt.title("Training history")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Error")
        plt.grid(True)
        plt.legend()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if store:
            plot_path = os.path.join(self.storage_path, "train_history.svg")
            plt.savefig(plot_path, format="svg")
        if show_plot:
            plt.show()

    def present_latent(self, data_loader, store=False, show_plot=True):
        latent_data = []
        labels = []
        self.eval()  # Set the model to evaluation mode
        for data, label in data_loader:
            labels.append(label)  # add one to fix labels starting from
            data = data.to(self.device)
            with torch.no_grad():
                encoded, _ = self(data)
                latent_data.append(encoded)
        latent_data = torch.cat(latent_data, dim=0)
        latent_data = latent_data.view(len(latent_data), self.latent_dim)
        labels = torch.cat(labels, dim=0) + 1
        latent_data = latent_data.to("cpu")
        labels = labels.to("cpu")
        # Convert labels to a categorical variable for coloring
        label_colors = sns.color_palette("husl", len(np.unique(labels)))

        # Create dynamic column names based on the maximum length
        column_names = [f"latent_{i+1}" for i in range(self.latent_dim)]
        df = pd.DataFrame(latent_data, columns=column_names)
        df["Label"] = labels
        sns.pairplot(df, hue="Label", palette=label_colors)
        if store:
            plot_path = os.path.join(self.storage_path, "latent_space.svg")
            plt.savefig(plot_path, format="svg")
        if show_plot:
            plt.show()

    def get_labeled_latent(self, data_loader):
        latent_data = []
        labels = []
        self.eval()
        for data, label in data_loader:
            labels.append(label)
            data = data.to(self.device)
            with torch.no_grad():
                encoded, _ = self(data)
                latent_data.append(encoded)
        latent_data = torch.cat(latent_data, dim=0)
        latent_data = latent_data.view(len(latent_data), self.latent_dim)
        labels = torch.cat(labels, dim=0) + 1
        return latent_data, labels

    def summary(self):
        self.get_device()
        torchsummary.summary(self, input_size=self.input_shape)
        # torchsummary.summary(self.to(self.device), input_size=self.input_shape)

    def store_summary(self, save_path=None):
        if save_path is None:
            summary_path = self.storage_path
        else:
            summary_path = save_path
        summary_path = os.path.join(summary_path, "model_summary.txt")
        # Redirect the summary output to a text file
        with open(summary_path, "w") as f:
            with redirect_stdout(f):
                torchsummary.summary(self, input_size=self.input_shape)
                # torchsummary.summary(self.to(self.device), input_size=self.input_shape)
            print(self, file=f)

    def store(self, save_path=None):
        if save_path is None:
            model_path = self.storage_path
        else:
            model_path = save_path
        model_path = os.path.join(model_path, "entire_model.pth")
        torch.save(self, model_path)

    def visualize_reconstructions(self, data_loader, store=False, show_plot=True):
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # No need to track gradients
            for data, labels in data_loader:
                data = data.to(self.device)
                _, reconstructions = self(data)  # Get the reconstructed images
                data = data.cpu()
                reconstructions = reconstructions.cpu()

                # We'll plot one figure per class present in this batch
                unique_labels = labels.unique()

                for label in unique_labels:
                    fig, axs = plt.subplots(
                        2, (labels == label).sum().item(), figsize=(10, 3)
                    )
                    fig.suptitle(f"Grade {1+label.item()}")

                    # Filter images and reconstructions by class
                    class_images = data[labels == label]
                    class_recons = reconstructions[labels == label]

                    for i in range((labels == label).sum().item()):
                        if (
                            class_images.size(0) > 1
                        ):  # Check if there are multiple images
                            axs[0, i].imshow(class_images[i].permute(1, 2, 0))
                            axs[1, i].imshow(class_recons[i].permute(1, 2, 0))
                            axs[0, i].axis("off")
                            axs[1, i].axis("off")
                        else:  # If there's only one image, axs is not a 2D array
                            axs[0].imshow(class_images[i].permute(1, 2, 0))
                            axs[1].imshow(class_recons[i].permute(1, 2, 0))
                            axs[0].axis("off")
                            axs[1].axis("off")
                    plt.tight_layout()
                    if store:
                        plots_folder_path = os.path.join(self.storage_path, "test")
                        os.makedirs(plots_folder_path, exist_ok=True)
                        plot_path = os.path.join(
                            plots_folder_path, f"grade_{1+label.item()}.svg"
                        )
                        plt.savefig(plot_path, format="svg")
                    if show_plot:
                        plt.show()

    def make_report(self):
        report = {
            "name": self.model_name,
            "type": self.model_type,
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
            "best_loss": self.best_loss,
        }
        return report
