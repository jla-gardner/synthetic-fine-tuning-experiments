from digital_experiments import experiment
from digital_experiments.optmization import (
    Categorical,
    Integer,
    Real,
    optimize_step_for,
)
from digital_experiments.util import get_passed_kwargs_for

from scripts.train_on_dft import train_nn_on_dft

ROOT = "experiment-logs/train-on-dft/cv"


@experiment(save_to=ROOT, capture_logs=False, verbose=True, backend="csv")
def cv_train(**kwargs):
    results = []
    for fold in range(10):
        results.append(train_nn_on_dft(fold=fold, **kwargs))
    return results


space = {
    "lr": Real(5e-5, 5e-2, prior="log-uniform"),
    "weight_decay": Real(1e-6, 1, prior="log-uniform"),
    "dropout": Real(0, 0.4),
    "hidden_size": Integer(5, 2_000, prior="log-uniform"),
    "n_layers": Integer(1, 6),
    "activation": Categorical(["CELU", "ReLU", "Tanh"]),
    "structures_per_batch": Integer(5, 1_000, prior="log-uniform"),
}

def run():
    optimize_step_for(
        cv_train,
        loss_fn=lambda results: results["structure"]["rmse"]["val"]["mean"],
        n_random_points=200,
        space=space,
        root=ROOT,
    )
