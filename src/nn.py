import warnings
from typing import Sequence

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.nn import Dropout, Identity, Linear, Sequential

# ignore useless warnings from PyTorch Lightning
warnings.filterwarnings("ignore", ".*which may be a bottleneck.*")
warnings.filterwarnings("ignore", ".*IProgress not found.*")
warnings.filterwarnings("ignore", ".*smaller than the logging interval.*")


def _neural_net(
    layers: Sequence[int],
    activation: str = "CELU",
    activate_last: bool = False,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """Create a neural network with given layers and activation function"""

    activation = getattr(torch.nn, activation)()
    layers = [
        Sequential(
            Linear(layers[i], layers[i + 1]),
            activation,
            Dropout(dropout) if dropout > 0 else Identity(),
        )
        for i in range(len(layers) - 1)
    ]
    if not activate_last:
        layers[-1] = layers[-1][0]

    return Sequential(*layers)


class RegressionModel(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: str = "mse_loss",
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.loss_fn = getattr(torch.nn.functional, loss_fn)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def training_step(self, batch, _):
        loss = self._step(batch)
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, _):
        loss = self._step(batch)
        self.log("val_loss", loss.item())

    def configure_optimizers(self):
        return getattr(torch.optim, self.optimizer)(
            self.model.parameters(), **self.optimizer_kwargs
        )


class NeuralNetwork(RegressionModel):
    def __init__(
        self,
        layers: Sequence[int],
        activation: str = "ReLU",
        activate_last: bool = False,
        loss_fn: str = "mse_loss",
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
        dropout: float = 0.0,
    ):
        model = _neural_net(layers, activation, activate_last, dropout)
        super().__init__(model, loss_fn, optimizer, optimizer_kwargs)


class TrainOnSumsNetwork(NeuralNetwork):
    def _step(self, batch):
        x, y = batch
        y_hat = self.forward(x).sum(axis=1)
        loss = self.loss_fn(y_hat, y)
        return loss


def get_trainer(directory, patience=50, log_every=10, max_epochs=-1, file_suffix=""):
    return Trainer(
        # enable_progress_bar=False,
        accelerator="auto",
        max_epochs=max_epochs,
        log_every_n_steps=log_every,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            ModelCheckpoint(
                dirpath=directory,
                monitor="val_loss",
                filename=f"best{file_suffix}",
                save_weights_only=True,
            ),
        ],
        logger=[TensorBoardLogger(directory / "tb"), CSVLogger(directory / "csv")],
    )
