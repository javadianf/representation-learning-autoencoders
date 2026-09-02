# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn

from maker.handlers import get_activation_fcn


class MLP_model:
    def __init__(
        self,
        input_shape,
        enc_dims: Sequence[int],
        latent_dim: int = 2,
        dec_dims=None,
        use_bias=True,
        enc_act_fcn="relu",
        dec_act_fcn=None,
        output_act_fcn="sigmoid",
    ):
        assert latent_dim >= 2
        assert len(enc_dims) > 0
        assert all(d > 0 for d in enc_dims)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.enc_dims = enc_dims
        self.use_bias = use_bias
        if isinstance(enc_act_fcn, str):
            self.enc_act_fcn = [enc_act_fcn] * len(self.enc_dims)
        else:
            self.enc_act_fcn = enc_act_fcn
        if dec_dims is None:
            self.dec_dims = list(reversed(self.enc_dims))
        else:
            self.dec_dims = dec_dims
        if dec_act_fcn is None:
            self.dec_act_fcn = list(reversed(self.enc_act_fcn))
        else:
            self.dec_act_fcn = dec_act_fcn

        self.output_act_fcn = output_act_fcn
        self.encoder = None
        self.decoder = None

    def make(self):
        self._find_input_feature_size()
        self._build_encoder()
        self._build_decoder()
        return self.encoder, self.decoder

    def _find_input_feature_size(self):
        input_size = torch.prod(torch.tensor(self.input_shape))
        self.feature_size = input_size.item()

    def _build_encoder(self):
        input_layer = nn.Flatten()
        enc_layers = OrderedDict({"input_layer": input_layer})
        input_dims = [self.feature_size] + self.enc_dims[:-1]
        for idx in range(len(self.enc_dims)):
            enc_layers.update(
                {
                    f"enc_fc{idx+1}": nn.Linear(
                        input_dims[idx],
                        self.enc_dims[idx],
                        bias=self.use_bias,
                    )
                }
            )
            layer_act_fcn = get_activation_fcn(self.enc_act_fcn[idx])
            enc_layers.update({f"act_enc_fc{idx+1}": layer_act_fcn})
        # Latent space
        enc_layers.update(
            {
                f"latent_space": nn.Linear(
                    self.enc_dims[-1],
                    self.latent_dim,
                    bias=self.use_bias,
                )
            }
        )
        self.encoder = nn.Sequential(enc_layers)

    def _build_decoder(self):
        input_dims = [self.latent_dim] + self.dec_dims[:-1]
        #  Make layers
        dec_layers = OrderedDict()
        for idx in range(len(self.dec_dims)):
            dec_layers.update(
                {
                    f"dec_fc{idx+1}": nn.Linear(
                        input_dims[idx],
                        self.dec_dims[idx],
                        bias=self.use_bias,
                    )
                }
            )
            layer_act_fcn = get_activation_fcn(self.dec_act_fcn[idx])
            dec_layers.update({f"act_dec_fc{idx+1}": layer_act_fcn})
        # Latent space
        dec_layers.update(
            {
                f"output_flat_layer": nn.Linear(
                    self.dec_dims[-1],
                    self.feature_size,
                    bias=self.use_bias,
                )
            }
        )
        layer_act_fcn = get_activation_fcn(self.output_act_fcn)
        dec_layers.update({"output_flat_act_fcn": layer_act_fcn})
        dec_layers.update({"output_layer": nn.Unflatten(1, self.input_shape)})

        self.decoder = nn.Sequential(dec_layers)
