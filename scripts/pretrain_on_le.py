from digital_experiments import current_directory, experiment

from src.data import (
    convert_to_loaders,
    evaluate_model,
    get_data,
    naïve_cv,
    shape_preserving_scaler,
)
from src.nn import NeuralNetwork, TrainOnSumsNetwork, get_trainer


@experiment(
    save_to="experiment-logs/pre-train", capture_logs=False, verbose=True, backend="csv"
)
def pretrain_then_train(
    fold=0,
    # data
    n_max=8,
    l_max=8,
    n_train=-1,
    # model
    hidden_size=200,
    n_layers=3,
    dropout=0.0,
    pre_dropout=0.0,
    # training
    activation="CELU",
    optimizer="AdamW",
    weight_decay=0.0,
    lr=1e-3,
    structures_per_batch=400,
    pre_structures_per_batch=400,
    pre_lr=1e-4,
    pre_weight_decay=1e-6,
    pre_epochs=-1,
):

    pre_epochs = int(pre_epochs)

    # PRE-TRAIN
    # calculate SOAP vectors for the GAP17 bulk_amo dataset
    # and extract per-atom local energies
    soaps, dft_energies, local_energies, _ = get_data(n_max, l_max)
    # shuffle and split data according to a naive CV policy
    # put each structure in one of the three splits so as to
    # avoid data leakage
    X, y_dft, y_local = naïve_cv(
        soaps,
        dft_energies,
        local_energies,
        n_train=n_train,
        fold=fold,
        ratio={"train": 0.8, "val": 0.1, "test": 0.1},
    )
    # standardize the soap vectors
    X = X | shape_preserving_scaler(X.train)

    pre_train_model = NeuralNetwork(
        [X.train.shape[-1], *([hidden_size] * n_layers), 1],
        activation,
        optimizer=optimizer,
        optimizer_kwargs={"lr": pre_lr, "weight_decay": pre_weight_decay},
        dropout=pre_dropout,
    )

    # pre-train the model on per-atom local energies
    train_data, val_data = convert_to_loaders(
        X, y_local, batch_size=pre_structures_per_batch
    )
    pre_trainer = get_trainer(
        current_directory(), patience=5, max_epochs=pre_epochs, file_suffix="-pre"
    )
    pre_trainer.fit(pre_train_model, train_data, val_data)
    pretrain_results = evaluate_model(pre_train_model, X, y_dft, y_local)

    # FINE-TUNE
    # fine-tune the model on total energies
    # same as in `train_on_dft.py`, but starting from the pre-trained model

    model = TrainOnSumsNetwork.load_from_checkpoint(
        pre_trainer.checkpoint_callback.best_model_path,
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
        dropout=dropout,
    )

    train_data, val_data = convert_to_loaders(X, y_dft, batch_size=structures_per_batch)
    trainer = get_trainer(current_directory(), patience=100)
    trainer.fit(model, train_data, val_data)
    fine_tune_results = evaluate_model(model, X, y_dft, y_local)

    return dict(pretrain=pretrain_results, fine_tune=fine_tune_results)


def run():
    for fold in range(10):
        pretrain_then_train(fold=fold)
