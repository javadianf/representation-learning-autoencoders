# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.
# Load packages
import gc
import os
import torch
import autoencoder.classifier as cdae
from dataloaders import get_loaders
from maker.cnn import CNN_model
from reporting import append_to_json
MODELS_ROOT = os.environ.get("AE_MODELS", "./models")
def make_check_model(
    layer_specs,
    latent_dimension,
    model_type,
    learning_rate,
    epochs=5,
    name_extras="NN",
    batch_size=64,
    dense_dim=0,
    loss_weight={
        "reconstruction": 0.4,
        "classification": 0.5,
        "discrimination": 0.1,
    },
    class_weights=[1, 1, 1, 1],
):
    meta_report = {
        "enc_dims": len(layer_specs["enc_num_channels"]),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    # Make data loaders
    train_loader, valid_loader, test_loader = get_loaders(batch_size=batch_size)
    # Find the input shape
    inputs, _ = next(iter(train_loader))
    input_shape = inputs.shape[1:]  # Remove batch size

    # %%
    # Define and make model parts
    model_maker = CNN_model(
        input_shape, layer_specs, latent_dim=latent_dimension, dense_dim=dense_dim
    )
    encoder, decoder = model_maker.make()

    # %%
    # Make the model
    model = cdae.CDAE(
        input_shape=input_shape,
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dimension,
        name_extras=name_extras,
        model_type=model_type,
        loss_weight=loss_weight,
        class_weights=class_weights,
    )

    # %%
    # Train the model
    model.store_summary()
    _ = model.fit(
        train_loader, val_loader=valid_loader, epochs=epochs, lr=learning_rate
    )
    model.plot_training_history(store=True, show_plot=False)
    model.save_train_history()

    # %%
    # Evaluate the model
    # load the best performing epoch
    model.load_checkpoint()
    # model.present_latent(valid_loader, store=True, show_plot=False)
    # Store the best model
    model.store()
    # Visualize the reconstruction of test data
    # model.visualize_reconstructions(test_loader, store=True, show_plot=False)
    # get report data
    model_report = model.make_report()
    # %%
    # store reports
    report = meta_report | model_report
    append_to_json(os.path.join(MODELS_ROOT, "meta_report.json"), report)
    model.plot_confusion_matrix(valid_loader)

if __name__ == "__main__":
    # User settings
    batch_size = 12
    layer_dimensions = [
        [32, 16, 8],
    ]
    model_type = "CNN"
    latent_dimensions = [8]
    dense_dims = [8]

    learning_rate = 0.005
    epochs = 3
    loss_weight = {
        "reconstruction": 0.3,
        "classification": 0.7,
        "discrimination": 0,
    }
    class_weights = [1, 3, 3, 2]
    for enc_dims in layer_dimensions:
        for latent_dim in latent_dimensions:
            for dense_dim in dense_dims:
                layer_specs = {
                    "type": model_type,
                    "enc_num_channels": enc_dims,
                    "enc_kernels": [3],
                    "enc_activations": ["relu"],
                    "output_activation": "sigmoid",
                }
                name_extras = f"ly{len(enc_dims)}_d{dense_dim}_bs{batch_size}_lr{learning_rate:.4f}"
                make_check_model(
                    layer_specs,
                    latent_dim,
                    model_type,
                    learning_rate,
                    epochs=epochs,
                    name_extras=name_extras,
                    batch_size=batch_size,
                    dense_dim=dense_dim,
                    loss_weight=loss_weight,
                    class_weights=class_weights,
                )
                torch.cuda.empty_cache()
                gc.collect()
