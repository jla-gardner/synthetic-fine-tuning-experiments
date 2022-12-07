from digital_experiments import experiment
from digital_experiments.optmization import Integer, Real, optimize_step_for
from makefun import wraps

from scripts.pretrain_on_le import pretrain_then_train

ROOT = "experiment-logs/cv/pre-train"

space = {
    "pre_lr": Real(1e-4, 1e-1, prior="log-uniform"),
    "pre_weight_decay": Real(1e-6, 1, prior="log-uniform"),
    "pre_dropout": Real(0, 0.5),
    "pre_epochs": Integer(10, 100, prior="log-uniform"),
    "lr": Real(5e-4, 1e-3, prior="log-uniform"),
    "weight_decay": Real(1e-6, 1, prior="log-uniform"),
    "dropout": Real(0, 0.5),
    "hidden_size": Integer(20, 1_000, prior="log-uniform"),
    "n_layers": Integer(2, 4),
}


@experiment(save_to=ROOT, capture_logs=False, verbose=True)
@wraps(pretrain_then_train)
def cv_train(**kwargs):
    results = []
    for fold in range(1):
        results.append(pretrain_then_train(fold=fold, **kwargs))
    return results


def run():
    optimize_step_for(
        cv_train,
        loss_fn=lambda results: results["fine_tune"]["structure"]["rmse"]["val"][
            "mean"
        ],
        space=space,
        n_random_points=50,
        root=ROOT,
    )
