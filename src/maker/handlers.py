# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

import torch.nn as nn


def get_activation_fcn(act_fcn):
    match act_fcn:
        case "relu":
            return nn.ReLU()
        case "leakyrelu":
            return nn.LeakyReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()


def check_keys(specs, required) -> dict:
    for key, value in required.items():
        specs[key] = specs.get(key, value)
    return specs


def expand_specs(specs, base_spec) -> dict:
    skip_keys = ["type", base_spec, "output_activation"]
    num_layers = len(specs[base_spec])
    for key, value in specs.items():
        if key not in skip_keys:
            if len(value) == 1:
                specs[key] = [value[0]] * num_layers
            elif 1 < len(value) < num_layers:
                raise ValueError(
                    f"Mismatched layer data. {num_layers} layers given, but {key} has only {len(value)} elements!"
                )
    return specs


def prepare_specs(specs, required, base_spec) -> dict:
    full_specs = check_keys(specs, required)
    ready_specs = expand_specs(full_specs, base_spec)
    return ready_specs
