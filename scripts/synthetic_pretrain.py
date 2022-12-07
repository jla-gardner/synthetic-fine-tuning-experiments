from digital_experiments import current_directory, experiment

from src.data import (
    convert_to_loaders,
    evaluate_model,
    get_data,
    get_synthetic_data,
    naïve_cv,
    shape_preserving_scaler,
)
from src.nn import NeuralNetwork, TrainOnSumsNetwork, get_trainer


@experiment(save_to="experiment-logs/pre-train-on-synthetic", capture_logs=False, verbose=True)
def pretrain_then_train(
    fold,
    # data
    n_max=8,
    l_max=8,
    n_pretrain=-1,
    n_finetune=-1,
    # model
    hidden_size=100,
    n_layers=3,
    dropout=0.,
    pre_dropout=0,
    # training
    activation="CELU",
    optimizer="Adam",
    weight_decay=0,
    lr=3e-4,
    structures_per_batch=200,
    pre_structures_per_batch=50,
    pre_lr=3e-4,
    pre_weight_decay=0,
    all_data=False,
):


    # PRE-TRAIN

    synth_soaps, synth_local_energies, _ = get_synthetic_data(n_max, l_max, all_data)
    # synth_soaps, _, synth_local_energies, _ = get_data(n_max, l_max)
    # shuffle and split data according to a naive CV policy
    synth_X, synth_y_local = naïve_cv(
        synth_soaps,
        synth_local_energies,
        n_train=n_pretrain,
        fold=fold,
    )
    # standardize the soap vectors
    standardizer = shape_preserving_scaler(synth_X.train)
    synth_X = synth_X | standardizer

    # synth_X = synth_X | (lambda x: x.reshape(-1, x.shape[-1]))
    # synth_y_local = synth_y_local | (lambda x: x.reshape(-1, 1))

    print(synth_X.train.shape)

    pre_train_model = NeuralNetwork(
        [synth_X.train.shape[-1], *([hidden_size] * n_layers), 1],
        activation,
        optimizer=optimizer,
        optimizer_kwargs={"lr": pre_lr, "weight_decay": pre_weight_decay},
        dropout=pre_dropout,
    )

    train_data, val_data = convert_to_loaders(
        synth_X, synth_y_local, batch_size=pre_structures_per_batch
    )
    pre_trainer = get_trainer(current_directory(), patience=5, file_suffix="-pre")
    pre_trainer.fit(pre_train_model, train_data, val_data)
 
    # FINE-TUNE

    soaps, dft_energies, local_energies, _ = get_data(n_max, l_max)
    # shuffle and split data according to a naive CV policy
    X, y_dft, y_local = naïve_cv(
        soaps,
        dft_energies,
        local_energies,
        n_train=n_finetune,
        fold=fold,
    )
    # standardize the soap vectors
    X = X | standardizer
    
    print(X.train.shape)

    model = TrainOnSumsNetwork.load_from_checkpoint(
        pre_trainer.checkpoint_callback.best_model_path,
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
        dropout=dropout,
    )

    train_data, val_data = convert_to_loaders(X, y_dft, batch_size=structures_per_batch)
    trainer = get_trainer(current_directory(), patience=100)
    trainer.fit(model, train_data, val_data)
    
    fine_tune_results = evaluate_model(model, X, y_dft, y_local)

    return dict(fine_tune=fine_tune_results)


def run():
    for fold in range(10):
        pretrain_then_train(fold=fold)
