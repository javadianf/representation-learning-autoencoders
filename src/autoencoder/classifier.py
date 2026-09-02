# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from autoencoder.autoencoder import Autoencoder


class CenterLoss(torch.nn.Module):
    """Within-class centroid loss over the latent space.

    The centroids are model parameters rather than running statistics, so they
    are optimised jointly with the network and have to be handed to the same
    optimizer as the model weights (see fit). Only the cohesion term is
    implemented here; the between-class dispersion term described in the paper
    is not part of this version.
    """

    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.nn.Parameter(
            torch.randn(num_classes, feat_dim).to(self.device)
        )
        self.centers = self.centers.to(self.device)

    def forward(self, features, labels):
        labels = labels.to(self.device)
        features = features.to(self.device)

        centers_batch = self.centers.index_select(dim=0, index=labels.long())
        loss = (features - centers_batch).pow(2).sum() / 2.0 / features.size(0)
        return loss


class CDAE(Autoencoder):
    def __init__(
        self,
        input_shape=None,
        encoder=None,
        decoder=None,
        latent_dim=2,
        reconstruction_loss_function=torch.nn.MSELoss(),
        device=None,
        name_extras=None,
        model_type="NN",
        loss_weight={
            "reconstruction": 0.4,
            "classification": 0.5,
            "discrimination": 0.1,
        },
        num_classes=4,
        class_weights=None,
    ):
        super().__init__()
        assert latent_dim > 0
        self.input_shape = input_shape
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_function = reconstruction_loss_function
        self.loss_function = torch.nn.MSELoss()
        self.history = {"tr_loss": [], "tr_re": [], "val_loss": []}
        self.latent_dim = latent_dim
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        if class_weights is None:
            class_weights = [1] * self.num_classes
        self.weights = torch.tensor(class_weights, dtype=torch.float32)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()
        self.best_loss = float("inf")
        self.model_type = f"CDAE_{model_type}"  # Should be before name maker
        _ = self._make_model_name(extras=name_extras)
        self.make_classifier(num_class=self.num_classes)
        self._create_model_folder(folder=None)
        if device is None:
            self.get_device()
        else:
            self.device = device

    def make_classifier(self, num_class=4):
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, num_class), torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)
        return encoded, decoded, classification

    def evaluate(self, eval_loader):
        loss_function = torch.nn.NLLLoss()
        self.to(self.device)
        self.eval()  # Set the model to evaluation mode
        eval_loss = 0.0
        total_samples = 0
        for eval_data, labels in eval_loader:
            data = eval_data.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                _, _, classification = self(data)
            val_loss = loss_function(classification, labels)
            eval_loss += val_loss.item() * data.size(0)
            total_samples += data.size(0)

        average_loss = eval_loss / total_samples
        return average_loss

    def fit(self, train_loader, val_loader=None, epochs=10, device=None, lr=1e-3):
        if device is None:
            self.get_device()
        else:
            self.device = device
        self.to(self.device)
        criterion_reconstruction = torch.nn.MSELoss()
        criterion_classifier = torch.nn.NLLLoss(weight=self.weights)
        criterion_centerloss = CenterLoss(
            num_classes=self.num_classes, feat_dim=self.latent_dim, device=self.device
        )
        optimizer = torch.optim.Adam(
            list(self.parameters()) + list(criterion_centerloss.parameters()), lr=lr
        )
        # torch.cuda.empty_cache()
        n_tr_batches = len(train_loader)
        for epoch in range(epochs):
            # Training ----------------------------------------------
            # Set the model to training mode
            self.train()
            train_loss = 0.0
            reconstruction_error = 0.0
            progress_bar = tqdm(range(n_tr_batches), total=n_tr_batches, ncols=100)
            progress_bar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            for data, labels in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                encoded, decoded, classification = self(data)
                loss_reconstruction = criterion_reconstruction(decoded, data)
                loss_classification = criterion_classifier(classification, labels)
                loss_center = criterion_centerloss(encoded, labels)
                loss = (
                    self.loss_weight["reconstruction"] * loss_reconstruction
                    + self.loss_weight["classification"] * loss_classification
                    + self.loss_weight["discrimination"] * loss_center
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                reconstruction_error += loss_reconstruction.item()
                progress_bar.update()
                progress_bar.set_postfix(tr_loss=f"{loss.item():.5f}")
            # print(train_loss)
            train_loss /= n_tr_batches
            reconstruction_error /= n_tr_batches
            self.history["tr_loss"].append(train_loss)
            self.history["tr_re"].append(reconstruction_error)
            # Validation ----------------------------------------------
            if val_loader is not None:
                eval_loss = self.evaluate(val_loader)
                self.history["val_loss"].append(eval_loss)
                progress_bar.set_postfix(
                    tr_loss=f"{train_loss:.5f}",
                    tr_re=f"{reconstruction_error:.5f}",
                    val_re=f"{eval_loss:.5f}",
                )
            else:
                progress_bar.set_postfix(
                    tr_loss=f"{train_loss:.5f}",
                    tr_re=f"{reconstruction_error:.5f}",
                )
            progress_bar.close()
            # torch.cuda.empty_cache()
            if train_loss <= self.best_loss:
                self.best_loss = train_loss  # Update best loss
                self.save_checkpoint(optimizer, self.best_loss)
        return self.history

    def plot_confusion_matrix(self, dataloader, class_names=[1, 2, 3, 4]):
        self.eval()  # Set the model to evaluation mode
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data, labels in dataloader:
                # Assuming the model returns reconstruction and classification outputs
                _, _, outputs = self(data)
                predictions = torch.argmax(
                    outputs, dim=1
                )  # Convert logits to class predictions
                all_predictions.extend(predictions.cpu().numpy())  # Store predictions
                all_labels.extend(labels.cpu().numpy())  # Store true labels

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        # Compute the accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"accuracy = {accuracy}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        # plt.show()
        plot_path = os.path.join(self.storage_path, "confusion_matrix.pdf")
        # plt.savefig(plot_path, format="svg")
        plt.savefig(plot_path)
        plt.close()
