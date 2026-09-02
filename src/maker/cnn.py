# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn

import maker.handlers as handlers


def get_dimensions(layers, input_shape):
    temp_model = nn.Sequential(layers)
    # Create a dummy input tensor with desired shape
    dummy_input = torch.randn(1, *input_shape)
    # Call the model with the dummy input to get the output shape
    output_shape = temp_model(dummy_input).shape
    return output_shape


class CNN_model:
    def __init__(
        self, input_shape, layer_specs: dict, latent_dim: int = 2, dense_dim=None
    ):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        required_field = {
            "type": "CNN",
            "enc_num_channels": [16],
            "enc_kernels": [3],
            "enc_activations": ["relu"],
            "output_activation": "sigmoid",
        }
        self.layer_specs = handlers.prepare_specs(
            layer_specs, required_field, "enc_num_channels"
        )
        self.dense_dim = dense_dim
        self.make_inverse_spec()

    def make_inverse_spec(self):
        reverse_elements = ["_num_channels", "_kernels", "_activations"]
        for element in reverse_elements:
            self.layer_specs[f"dec{element}"] = self.layer_specs.get(
                f"dec{element}",
                self.layer_specs[f"enc{element}"][::-1],
            )

    def make(self):
        self._find_input_channels()
        self._build_encoder()
        self._build_decoder()
        return self.encoder, self.decoder

    def _find_input_channels(self):
        input_channels = self.input_shape[0]
        return input_channels

    def _build_encoder(self):
        input_channels = self._find_input_channels()
        enc_output_channels = self.layer_specs["enc_num_channels"]
        input_dims = [input_channels] + enc_output_channels
        #  Make layers
        enc_layers = OrderedDict()
        for idx in range(len(enc_output_channels)):
            enc_layers.update(
                {
                    f"enc_conv{idx+1}": nn.Conv2d(
                        input_dims[idx],
                        enc_output_channels[idx],
                        self.layer_specs["enc_kernels"][idx],
                    )
                }
            )
            # Activation function
            layer_act_fcn = handlers.get_activation_fcn(
                self.layer_specs["enc_activations"][idx]
            )
            enc_layers.update({f"enc_act{idx+1}": layer_act_fcn})
            # Batch normalization
            enc_layers.update(
                {
                    f"enc_bn{idx+1}": nn.BatchNorm2d(
                        enc_output_channels[idx],
                    )
                }
            )
        # Find number of features (nodes)
        self.shape_before_bottleneck = torch.tensor(
            get_dimensions(enc_layers, self.input_shape)
        )
        self.num_features = torch.prod(self.shape_before_bottleneck)
        self.num_features_to_latent = self.num_features
        enc_layers.update({f"enc_flatten": nn.Flatten()})
        # Linear layer before latent
        if self.dense_dim is not None and self.dense_dim != 0:
            enc_layers.update(
                {
                    f"linear_before": nn.Linear(
                        self.num_features,
                        self.dense_dim,
                    )
                }
            )
            self.num_features_to_latent = self.dense_dim
        # Latent space
        enc_layers.update(
            {
                f"latent_space": nn.Linear(
                    self.num_features_to_latent,
                    self.latent_dim,
                )
            }
        )
        self.encoder = nn.Sequential(enc_layers)
        self.enc_layers = enc_layers

    def _build_decoder(self):
        dec_layers = OrderedDict()
        #  Make layers
        # Linear layer before latent
        if self.dense_dim is not None and self.dense_dim != 0:
            dec_layers.update(
                {
                    f"linear_before": nn.Linear(
                        self.latent_dim,
                        self.num_features_to_latent,
                    )
                }
            )
        else:
            self.num_features_to_latent = self.latent_dim
        dec_layers.update(
            {
                f"dec_fc": nn.Linear(
                    self.num_features_to_latent,
                    self.num_features,
                )
            }
        )
        dec_layers.update(
            {
                f"dec_unflatten": nn.Unflatten(
                    1,
                    # self.num_features.item(),
                    self.shape_before_bottleneck.tolist()[1:],
                )
            }
        )
        dec_output_channels = self.layer_specs["dec_num_channels"]
        # Add channels before bottleneck to the number of input channels list
        dec_input_channels = [
            self.shape_before_bottleneck[1].item()
        ] + dec_output_channels

        for idx in range(len(dec_output_channels)):
            dec_layers.update(
                {
                    f"dec_fc{idx+1}": nn.ConvTranspose2d(
                        dec_input_channels[idx],
                        dec_output_channels[idx],
                        self.layer_specs["dec_kernels"][idx],
                    )
                }
            )
            # Activation function
            layer_act_fcn = handlers.get_activation_fcn(
                self.layer_specs["dec_activations"][idx]
            )
            dec_layers.update({f"dec_act{idx+1}": layer_act_fcn})
            # Batch normalization
            dec_layers.update(
                {
                    f"dec_bn{idx+1}": nn.BatchNorm2d(
                        dec_output_channels[idx],
                    )
                }
            )
        dec_layers.update(
            {
                f"output_layer": nn.Conv2d(
                    dec_output_channels[-1],
                    self.input_shape[0],
                    1,
                    # self.layer_specs["dec_kernels"][-1],
                )
            }
        )
        layer_act_fcn = handlers.get_activation_fcn(
            self.layer_specs["output_activation"]
        )
        dec_layers.update({"output_act": layer_act_fcn})
        self.decoder = nn.Sequential(dec_layers)
        self.dec_layers = dec_layers


if __name__ == "__main__":
    input_shape = [3, 79, 79]
    layer_specs = {
        "type": "CNN",
        "enc_num_channels": [32, 16, 8],
        "enc_kernels": [3],
        "enc_activations": ["relu"],
        "output_activation": "sigmoid",
    }
    latent_dimension = 3
    model_maker = CNN_model(
        input_shape, layer_specs, latent_dim=latent_dimension, dense_dim=0
    )
    encoder, decoder = model_maker.make()
    print(f"Encoder:\n{encoder}")
    print(f"Decoder:\n{decoder}")
