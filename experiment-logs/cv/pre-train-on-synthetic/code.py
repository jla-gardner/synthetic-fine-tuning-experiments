@experiment(save_to=ROOT, capture_logs=False, verbose=True, backend="csv")
def cv_train(*, pre_lr, pre_weight_decay, pre_dropout, lr, weight_decay, dropout, hidden_size):
    kwargs = dict(pre_lr=pre_lr, pre_weight_decay=pre_weight_decay, pre_dropout=pre_dropout, lr=lr, weight_decay=weight_decay, dropout=dropout, hidden_size=hidden_size)
    results = []
    for fold in range(10):
        results.append(pretrain_then_train(fold=fold, **kwargs))
    return summarise(results)
