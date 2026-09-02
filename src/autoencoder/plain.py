# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

import torch

from autoencoder.autoencoder import Autoencoder


class PlainAE(Autoencoder):
    """Reconstruction-only baseline. No latent-space term of any kind.

    This is the AE row of the results table: whatever class structure appears in
    its latent space is a by-product of compressing the patch, not something the
    objective asked for.
    """

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
    ):
        super().__init__()
        assert latent_dim > 0
        self.input_shape = input_shape
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_function = reconstruction_loss_function
        self.history = {"tr_re": [], "val_re": []}
        self.latent_dim = latent_dim
        self.best_loss = float("inf")
        self.model_type = f"AE_{model_type}"  # Should be before name maker
        _ = self._make_model_name(extras=name_extras)
        self._create_model_folder(folder=None)
        if device is None:
            self.get_device()
        else:
            self.device = device


# Kept so that older scripts importing the previous name still resolve.
MLP_AE = PlainAE
