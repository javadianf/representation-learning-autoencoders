# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.
# Load packages
import os
import autoencoder.contractive as cae
from dataloaders import get_loaders
from maker.cnn import CNN_model
from reporting import append_to_json

MODELS_ROOT = os.environ.get("AE_MODELS", "./models")


def make_check_model(
    layer_specs,
    latent_dimension,
    model_type,
    lambda_c,
    learning_rate,
    epochs=30,
    name_extras="NN",
    batch_size=64,
):
    meta_report = {
        "enc_dims": len(layer_specs["enc_num_channels"]),
        "lambda_c": lambda_c,
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
        input_shape,
        layer_specs,
        latent_dim=latent_dimension,
    )
    encoder, decoder = model_maker.make()

    # %%
    # Make the model
    model = cae.CAE(
        input_shape=input_shape,
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dimension,
        lambda_c=lambda_c,
        name_extras=name_extras,
        model_type=model_type,
    )

    # %%
    # Train the model
    # model.summary()
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
    model.present_latent(valid_loader, store=True, show_plot=False)
    # Store the best model
    model.store()
    # Visualize the reconstruction of test data
    model.visualize_reconstructions(test_loader, store=True, show_plot=False)
    # get report data
    model_report = model.make_report()
    # %%
    # store reports
    report = meta_report | model_report
    append_to_json(os.path.join(MODELS_ROOT, "meta_report.json"), report)


if __name__ == "__main__":
    # User settings
    batch_size = 64
    layer_dimensions = [
        [128, 64, 32, 16],
        [256, 128, 64, 32, 16],
        [256, 256, 128, 128, 64, 64, 32, 32, 16, 16],
    ]
    model_type = "CNN"
    latent_dimensions = [5, 10, 16]
    lambda_c = 0.5
    learning_rate = 0.002
    epochs = 30
    for enc_dims in layer_dimensions:
        for latent_dim in latent_dimensions:
            layer_specs = {
                "type": model_type,
                "enc_num_channels": enc_dims,
                "enc_kernels": [3],
                "enc_activations": ["relu"],
                "output_activation": "sigmoid",
            }
            name_extras = f"ly{len(enc_dims)}_lc{lambda_c:.3f}_bs{batch_size}_lr{learning_rate:.4f}"
            make_check_model(
                layer_specs,
                latent_dim,
                model_type,
                lambda_c,
                learning_rate,
                epochs=epochs,
                name_extras=name_extras,
                batch_size=batch_size,
            )
