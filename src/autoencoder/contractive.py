# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

import torch
from tqdm import tqdm

from autoencoder.autoencoder import Autoencoder


# Define the Contractive Loss
def contractive_loss(encoded, decoded, original, lambda_c=0.01):
    mse_loss = torch.nn.functional.mse_loss(decoded, original)

    # Compute the Jacobian of the encoded output with respect to the input
    jacobian = torch.autograd.grad(
        outputs=encoded,
        inputs=original,
        grad_outputs=torch.ones(encoded.size()).to(original.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute the Frobenius norm of the Jacobian
    jacobian_norm = torch.sqrt(torch.sum(torch.square(jacobian)))

    # Total loss = Reconstruction loss + lambda_c * Contractive loss
    total_loss = mse_loss + lambda_c * jacobian_norm
    return total_loss, mse_loss


class CAE(Autoencoder):
    def __init__(
        self,
        input_shape=None,
        encoder=None,
        decoder=None,
        latent_dim=2,
        reconstruction_loss_function=torch.nn.MSELoss(),
        lambda_c=0.01,
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
        self.lambda_c = lambda_c
        self.best_loss = float("inf")
        self.model_type = f"CAE_{model_type}"  # Should be before name maker
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
            for data, _ in train_loader:
                data = data.to(self.device)
                # Require gradients for input
                data.requires_grad_()
                # Zero the parameter gradients
                optimizer.zero_grad()
                encoded, decoded = self(data)
                loss, r_error = contractive_loss(
                    encoded, decoded, data, lambda_c=self.lambda_c
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
