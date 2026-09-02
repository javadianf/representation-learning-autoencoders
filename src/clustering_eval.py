# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

import json
import os

import torch
from sklearn.metrics import davies_bouldin_score

from dataloaders import get_loaders

MODELS_ROOT = os.environ.get("AE_MODELS", "./models")


def get_clustering_metric(model, data_loader) -> float:
    """
    get_clustering_metric calculates the Davies Bouldin score for the latent space

    Args:
        model (Autoencoder): A full Autoencoder model
        data_loader (dataloader): a dataloader for the data used for the evaluation

    Returns:
        float: Davies Bouldin score of the latent space of the model on the given data
    """
    latent_data, labels = model.get_labeled_latent(data_loader)
    latent_data = latent_data.to("cpu")
    labels = labels.to("cpu")
    clustering_metric = davies_bouldin_score(latent_data, labels)
    return clustering_metric


def process_models(models_path, data_loader):
    results = {}
    for folder in os.listdir(models_path):
        folder_path = os.path.join(models_path, folder)
        if os.path.isdir(folder_path):
            model_path = os.path.join(folder_path, "entire_model.pth")
            if os.path.isfile(model_path):
                model = torch.load(model_path)
                clustering_metric = get_clustering_metric(model, data_loader)
                results[folder] = clustering_metric
    with open("clustering_metric.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    batch_size = 32
    train_loader, valid_loader, test_loader = get_loaders(batch_size=batch_size)
    models_path = MODELS_ROOT
    process_models(models_path, train_loader)
