import operator
from typing import Sequence
from unicodedata import numeric

import numpy as np
import torch
from ase.io import read
from locache import persist
from quippy.descriptors import Descriptor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .util import _DATA_DIR


def all_equal(iterable):
    """check if all elements in an iterable are equal"""

    return len(set(iterable)) == 1


def count(things):
    """iterate over things and count them"""

    for i, thing in enumerate(things):
        print(f"Done: {i+1:>4}/{len(things):>4}", end="\r")
        yield thing
    print()


def det_shuffle(thing, seed=0):
    """Deterministically shuffle stuff"""

    return np.array(thing)[np.random.RandomState(seed).permutation(len(thing))]


def split(thing, ratio: Sequence[numeric] = (9, 1)):
    """Split stuff into <n> parts based on ratio"""

    n = len(thing)
    splits = np.cumsum(np.array(ratio) / sum(ratio) * n)
    splits = np.round(splits).astype(int)
    return [thing[lo:hi] for lo, hi in zip([0, *splits], splits)]


def mae(a, b):
    """
    The (componentwise) mean absolute error between two arrays
    """
    return np.abs(a - b).mean()


def rmse(a, b):
    """
    The (componentwise) root mean squared error between two arrays
    """
    return np.sqrt(((a - b) ** 2).mean())


class Data:
    """
    A simle utility class for storing data in a dict-like way
    The logical or operator `|` is overloaded to map a function over the data,
    like a pipe operator in other languages.

    e.g.
    >>> data = Data(x=1, y=2)
    >>> data.x
    1
    >>> data + 2
    Data(x=3, y=4)
    >>> data.map(lambda x: x + 2)
    Data(x=3, y=4)
    >>> data | lambda x: x * 2
    Data(x=2, y=4)
    """

    def __init__(self, **kwargs):
        self._props = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def as_dict(self):
        return self._props

    def __getitem__(self, key):
        return self._props[key]

    def map(self, func, other=None):
        if other is None:
            return self.__class__(**{k: func(v) for k, v in self._props.items()})
        return self._do(func, other)

    def __or__(self, func):
        return self.map(func)

    def _do(self, func, other):
        if isinstance(other, self.__class__):
            assert set(self._props) == set(other._props)
            return self.__class__(
                **{k: func(self._props[k], other._props[k]) for k in self._props}
            )
        return self.__class__(**{k: func(v, other) for k, v in self._props.items()})

    def __repr__(self) -> str:
        reps = [f"{k}={v}" for k, v in self._props.items()]
        has_newlines = any("\n" in r for r in reps)
        if not has_newlines:
            return "(" + ", ".join(reps) + ")"

        indent = max(len(k) for k in self._props) + 1
        ks = [f"{k}{' ' * (indent - 1 - len(k))}" for k in self._props]
        vals_split = [f"{v}".split("\n") for v in self._props.values()]
        vals = [
            "\n".join([v[0], *(" " * (indent + 2) + _v for _v in v[1:])])
            for v in vals_split
        ]

        reps = [f"{k}={v}" for k, v in zip(ks, vals)]
        return "(\n" + ",\n".join("  " + r for r in reps) + "\n)"

    def __add__(self, other):
        return self._do(operator.add, other)

    def __sub__(self, other):
        return self._do(operator.sub, other)

    def __mul__(self, other):
        return self._do(operator.mul, other)

    def __truediv__(self, other):
        return self._do(operator.truediv, other)

    def __pow__(self, other):
        return self._do(operator.pow, other)


def shape(x):
    return x.shape


def roll(array, percent):
    """roll an array by a given percentage"""

    return np.roll(array, int(len(array) * percent), axis=0)


def naïve_cv(*data, n_train=-1, fold=0, k=10, ratio=None) -> Data:
    """
    naïve cross-validation

    Procedure:
    1. Shuffle the data
    2. Roll the data by a percentage given by `fold / k`
    3. Split the data into the different sets based on `ratio`
    4. wrap in a `Data` object

    Parameters
    ----------
    data : array-like
        The data to split into folds
    n_train : int, optional
        The number of training examples to use, by default -1 i.e. all
    fold : int, optional
        The fold to use, by default 0
    k : int, optional
        The number of folds, by default 10
    ratio : dict, optional
        The ratio of the data to split into. Defaults to
        {"train": k - 2, "val": 1, "test": 1}

    Returns
    -------
    Data
        The data split into the different sets

    Examples
    --------
    >>> x = list(range(10))
    >>> y = list(range(10, 20))
    >>> data_x, data_y = naïve_cv(x, y, fold=0, k=5, ration={"train": 1, "test": 1})
    >>> data_x.train
    [2, 4, 7, 1, 9]
    >>> data_y.test
    [15, 11, 13, 17, 18]
    """

    if ratio is None:
        ratio = {"train": k - 2, "val": 1, "test": 1}

    assert all_equal(map(len, data))

    # deterministically shuffle the data
    data = [det_shuffle(d) for d in data]

    # split the data into k folds
    data = [roll(d, percent=fold / k) for d in data]

    # split the data into sets
    data = [split(d, ratio=list(ratio.values())) for d in data]

    # alter the amount of training data
    for d in data:
        d[0] = d[0][:n_train]

    # wrap in a Data object
    data = [Data(**dict(zip(ratio.keys(), d))) for d in data]

    if len(data) == 1:
        return data[0]
    return data


def numpy_dataset(*arrays, **loader_kwargs):
    """create a PyTorch DataLoader from (numpy) arrays"""

    return DataLoader(
        TensorDataset(*map(torch.FloatTensor, arrays)),
        **loader_kwargs,
    )


def shape_preserving_scaler(reference):
    """
    create a standard scaler that preserves the shape of the `reference` data
    """
    _size = reference.shape[-1]
    transform = StandardScaler().fit(reference.reshape(-1, _size)).transform

    def _scaler(x):
        assert x.shape[-1] == _size
        return transform(x.reshape(-1, _size)).reshape(x.shape)

    return _scaler


@persist
def get_data(n_max, l_max):
    """
    Calculate SOAP vectors for each structure in the GAP17 bulk_amo dataset
    and extract the per-cell DFT and per-atom local energies.

    Returns
    -------
    tuple
        - soaps (np.ndarray): the SOAP features
        - dft_energies (np.ndarray): the DFT energies
        - loca_energies (np.ndarray): the local energies
        - structures (list): the ASE structures
    """

    structures = [
        s for s in read(_DATA_DIR / "bulk_amo.extxyz", index=":") if len(s) == 64
    ]

    desc = Descriptor(f"soap cutoff=3.7 n_max={n_max} l_max={l_max} atom_sigma=0.5")
    for s in structures:
        s.arrays["soap"] = desc.calc(s)["data"]

    soaps = np.array([s.arrays["soap"] for s in structures])
    dft_engergies = np.array([s.get_potential_energy() for s in structures])
    loca_energies = np.array([s.arrays["gap17_energy"] for s in structures])

    return (
        soaps,
        dft_engergies.reshape(-1, 1) + 157 * 64,
        loca_energies[..., np.newaxis] + 157,
        structures,
    )


@persist
def get_synthetic_data(n_max, l_max, all_data=False):
    """
    Calculate SOAP vectors for each structure in the synthetic dataset
    and extract the per-atom local energies.

    Returns
    -------
    tuple
        - soaps (np.ndarray): the SOAP features
        - local_energies (np.ndarray): the local energies
        - structures (list): the ASE structures
    """

    structures = [
        s
        for s in read(
            "all_synthetic_data.extxyz" if all_data else "synthetic.extxyz",
            index="::5" if all_data else ":",
        )
    ]

    desc = Descriptor(f"soap cutoff=3.7 n_max={n_max} l_max={l_max} atom_sigma=0.5")
    for s in structures:
        s.arrays["soap"] = desc.calc(s)["data"]

    soaps = np.array([s.arrays["soap"] for s in structures])
    local_energies = np.array([s.arrays["gap17_energy"] for s in structures])

    return (
        soaps,
        local_energies[..., np.newaxis] + 157,
        structures,
    )


def convert_to_loaders(*data, batch_size=32):
    """convert Data objects to PyTorch DataLoader objects"""

    return [
        numpy_dataset(*(d[split] for d in data), batch_size=batch_size, shuffle=shuffle)
        for split, shuffle in zip(["train", "val"], [True, False])
    ]


def sum_along(axis=-1):
    return lambda x: x.sum(axis=axis)


def evaluate_model(model, X: Data, y_dft: Data, y_local: Data):
    """
    Evaluate a model on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    X : Data
        The per-atom SOAP vectors
    y_dft : Data
        The per-cell DFT energies
    y_local : Data
        The per-atom local energies

    Returns
    -------
    dict
        A dictionary containing the errors for the local and structure predictions
    """
    model.eval()

    metrics = [mae, rmse]

    with torch.no_grad():
        local_preds = X | torch.FloatTensor | model | np.asarray
    dft_preds = local_preds | sum_along(axis=1)

    local_errors = {
        metric.__name__: local_preds.map(metric, y_local).as_dict()
        for metric in metrics
    }
    structure_errors = {
        metric.__name__: dft_preds.map(metric, y_dft).as_dict() for metric in metrics
    }

    return dict(local=local_errors, structure=structure_errors)
