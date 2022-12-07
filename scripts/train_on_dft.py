from digital_experiments import current_directory, experiment

from src.data import (
    convert_to_loaders,
    evaluate_model,
    get_data,
    naïve_cv,
    shape_preserving_scaler,
)
from src.nn import TrainOnSumsNetwork, get_trainer


@experiment(save_to="experiment-logs/train-on-dft", capture_logs=False, backend="csv")
def train_nn_on_dft(
    fold: int,
    # data
    n_max=8,
    l_max=8,
    n_train=-1,
    # model
    hidden_size=200,
    n_layers=3,
    dropout=0.0,
    # training
    activation="CELU",
    optimizer="AdamW",
    weight_decay=0.0,
    lr=1e-3,
    structures_per_batch=64,
):
    structures_per_batch = int(structures_per_batch)
    soaps, dft_energies, local_energies, _ = get_data(n_max, l_max)
    # shuffle and split data according to a naive CV policy
    X, y_dft, y_local = naïve_cv(
        soaps,
        dft_energies,
        local_energies,
        n_train=n_train,
        fold=fold,
    )
    # standardize the soap vectors
    X = X | shape_preserving_scaler(X.train)

    model = TrainOnSumsNetwork(
        [X.train.shape[-1], *([hidden_size] * n_layers), 1],
        activation,
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
        dropout=dropout,
    )

    train_data, val_data = convert_to_loaders(X, y_dft, batch_size=structures_per_batch)
    get_trainer(current_directory(), patience=100).fit(model, train_data, val_data)

    return evaluate_model(model, X, y_dft, y_local)


def run():
    for fold in range(10):
        train_nn_on_dft(fold=fold)
