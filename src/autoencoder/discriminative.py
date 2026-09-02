# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

import torch
from tqdm import tqdm

from autoencoder.autoencoder import Autoencoder


def discriminative_loss(encoded, decoded, original, labels, margin=1.0):
    mse_loss = torch.nn.functional.mse_loss(decoded, original)

    # Prepare to select valid triplets
    labels = labels.unsqueeze(
        1
    )  # Expand labels to match encoded dimensions [batch_size, 1]
    mask = labels.eq(labels.T)  # Create a mask for positives [batch_size, batch_size]

    # Anchor for each sample in the batch
    anchor = encoded

    # Select positive and negative examples
    positives = []
    negatives = []
    for i in range(encoded.size(0)):
        # For positives, select embeddings from the same class but not the same instance
        positive_mask = (
            mask[i] & ~torch.eye(mask.size(0), dtype=torch.bool, device=mask.device)[i]
        )
        positives.append(encoded[positive_mask].mean(dim=0, keepdim=True))

        # For negatives, select embeddings from different classes
        negative_mask = ~mask[i]
        negatives.append(encoded[negative_mask].mean(dim=0, keepdim=True))

    positive = torch.cat(positives, dim=0)
    negative = torch.cat(negatives, dim=0)

    # Triplet margin loss
    triplet_loss = torch.nn.TripletMarginLoss(margin=margin)
    disc_loss = triplet_loss(anchor, positive, negative)

    # Combine the losses
    total_loss = mse_loss + disc_loss
    return total_loss, mse_loss


class DAE(Autoencoder):
    def __init__(
        self,
        input_shape=None,
        encoder=None,
        decoder=None,
        latent_dim=2,
        reconstruction_loss_function=torch.nn.MSELoss(),
        margin=1,
        device=None,
        name_extras=None,
        model_type="NN",
    ):
        super().__init__()
        assert latent_dim > 0
        self.input_shape = input_shape
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_function = reconstruction_loss_function
        self.loss_function = torch.nn.MSELoss()
        self.history = {"tr_loss": [], "tr_re": [], "val_re": []}
        self.latent_dim = latent_dim
        self.margin = margin
        self.best_loss = float("inf")
        self.model_type = f"DAE_{model_type}"  # Should be before name maker
        _ = self._make_model_name(extras=name_extras)
        self._create_model_folder(folder=None)
        if device is None:
            self.get_device()
        else:
            self.device = device

    def fit(self, train_loader, val_loader=None, epochs=10, device=None, lr=1e-3):
        if device is None:
            self.get_device()
        else:
            self.device = device
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        torch.cuda.empty_cache()
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
                # Require gradients for input
                data.requires_grad_()
                # Zero the parameter gradients
                optimizer.zero_grad()
                encoded, decoded = self(data)
                loss, r_error = discriminative_loss(
                    encoded, decoded, data, labels, margin=self.margin
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                reconstruction_error += r_error.item()
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
                self.history["val_re"].append(eval_loss)
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
