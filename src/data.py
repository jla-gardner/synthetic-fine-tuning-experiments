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
    return np.abs(a - b).mean()


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


class Data:
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
    """naïve cross-validation

    e.g.
    x, y = naive_cv(x, y, fold=0, k=10, ratio={train: 9, val: 1})
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
    _size = reference.shape[-1]
    transform = StandardScaler().fit(reference.reshape(-1, _size)).transform

    def _scaler(x):
        assert x.shape[-1] == _size
        return transform(x.reshape(-1, _size)).reshape(x.shape)

    return _scaler


@persist
def get_data(n_max, l_max):

    structures = [s for s in read( _DATA_DIR / "bulk_amo.extxyz", index=":") if len(s) == 64]

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

    structures = [s for s in read("all_synthetic_data.extxyz" if all_data else "synthetic.extxyz", index="::5" if all_data else ":")]

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

@persist
def get_surface_data(n_max, l_max):

    structures = [s for s in read("surface_amo.extxyz", index=":") if len(s) == 64]

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


def convert_to_loaders(*data, batch_size=32):
    """convert Data objects to PyTorch DataLoader objects"""

    return [
        numpy_dataset(*(d[split] for d in data), batch_size=batch_size, shuffle=shuffle)
        for split, shuffle in zip(["train", "val"], [True, False])
    ]


def sum_along(axis=-1):
    return lambda x: x.sum(axis=axis)


def evaluate_model(model, X: Data, y_dft: Data, y_local: Data):
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
