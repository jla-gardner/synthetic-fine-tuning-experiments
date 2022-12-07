from digital_experiments import experiment
from digital_experiments.optmization import Integer, Real, optimize_step_for

from scripts.synthetic_pretrain import pretrain_then_train

ROOT = "experiment-logs/cv/pre-train-on-synthetic"

space = {
    "pre_lr": Real(1e-4, 1e-1, prior="log-uniform"),
    "pre_weight_decay": Real(1e-6, 1, prior="log-uniform"),
    "pre_dropout": Real(0, 0.5),
    "lr": Real(5e-4, 1e-3, prior="log-uniform"),
    "weight_decay": Real(1e-6, 1, prior="log-uniform"),
    "dropout": Real(0, 0.5),
    "hidden_size": Integer(20, 1_000, prior="log-uniform"),
}


@experiment(save_to=ROOT, capture_logs=False, verbose=True)
def cv_train(*, pre_lr, pre_weight_decay, pre_dropout, lr, weight_decay, dropout, hidden_size):
    kwargs = dict(pre_lr=pre_lr, pre_weight_decay=pre_weight_decay, pre_dropout=pre_dropout, lr=lr, weight_decay=weight_decay, dropout=dropout, hidden_size=hidden_size)
    results = []
    for fold in range(10):
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
