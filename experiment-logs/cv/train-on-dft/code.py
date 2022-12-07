@experiment(save_to=ROOT, capture_logs=False, verbose=True, backend="csv")
def cv_train(**kwargs):
    results = []
    for fold in range(10):
        results.append(train_nn_on_dft(fold=fold, **kwargs))
    return results