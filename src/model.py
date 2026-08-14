"""Definición del MLP y su bucle de entrenamiento.

El modelo es un Multi-Layer Perceptron puro (requisito del proyecto): capas
totalmente conectadas + activación + normalización + dropout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from utils import rmse

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


@dataclass
class HParams:
    """Todos los hiperparámetros de una iteración en un solo objeto, para poder
    registrarlos junto con su RMSE en reports/experiments.csv."""

    hidden: tuple[int, ...] = (256, 128, 64)
    activation: str = "gelu"
    dropout: float = 0.2
    batch_norm: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 400
    patience: int = 60          # early stopping sobre RMSE de validación
    loss: str = "huber"         # "mse" | "huber"
    huber_delta: float = 1.0
    log_target: bool = True     # entrenar sobre log1p(SalePrice)
    scheduler: str = "cosine"   # "cosine" | "plateau" | "none"
    min_lr: float = 1e-5
    grad_clip: float = 1.0

    # Métrica con la que el early stopping elige el checkpoint.
    #   "rmse"     -> RMSE en escala original (la métrica de la competencia)
    #   "log_rmse" -> RMSE sobre log1p(precio)
    # Se usa log_rmse por defecto: el RMSE en escala original lo dominan una o
    # dos casas extremas por fold, así que como señal de selección es ruido y
    # produce paradas prematuras. El RMSE que se REPORTA sigue siendo el de la
    # escala original en ambos casos.
    es_metric: str = "log_rmse"

    def to_row(self) -> dict:
        d = asdict(self)
        d["hidden"] = "-".join(map(str, self.hidden))
        return d


class MLP(nn.Module):
    """MLP con bloques [Linear -> BatchNorm -> Activación -> Dropout]."""

    def __init__(self, n_features: int, hp: HParams):
        super().__init__()
        act = ACTIVATIONS[hp.activation]
        layers: list[nn.Module] = []
        prev = n_features
        for h in hp.hidden:
            layers.append(nn.Linear(prev, h))
            if hp.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if hp.dropout > 0:
                layers.append(nn.Dropout(hp.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        # He/Kaiming: apropiado para activaciones tipo ReLU y sus variantes.
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainResult:
    model: MLP
    best_val_rmse: float
    best_train_rmse: float
    best_epoch: int
    history: dict = field(default_factory=dict)


def _make_loss(hp: HParams) -> nn.Module:
    if hp.loss == "huber":
        # Huber es menos sensible a los precios extremos que MSE, lo que evita
        # que unas pocas casas de $700k dominen el gradiente.
        return nn.HuberLoss(delta=hp.huber_delta)
    return nn.MSELoss()


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hp: HParams,
    device: str = "cpu",
    verbose: bool = False,
) -> TrainResult:
    """Entrena un MLP y devuelve la mejor versión según RMSE de validación.

    El RMSE reportado SIEMPRE está en la escala original de SalePrice, aunque el
    entrenamiento ocurra en escala logarítmica: es la métrica de la competencia.
    """
    dev = torch.device(device)

    # Transformación del target. Si log_target, el modelo predice log1p(precio)
    # y se invierte con expm1 antes de medir.
    t_train = np.log1p(y_train) if hp.log_target else y_train.copy()
    t_val = np.log1p(y_val) if hp.log_target else y_val.copy()

    # Estandarizar el target estabiliza el entrenamiento cuando NO se usa log
    # (los precios crudos tienen escala ~1e5 y saturarían la red).
    t_mean, t_std = float(t_train.mean()), float(t_train.std())
    t_train_s = (t_train - t_mean) / t_std
    t_val_s = (t_val - t_mean) / t_std

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=dev)
    ytr = torch.tensor(t_train_s, dtype=torch.float32, device=dev)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=dev)

    model = MLP(X_train.shape[1], hp).to(dev)
    criterion = _make_loss(hp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr,
                                  weight_decay=hp.weight_decay)

    if hp.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp.epochs, eta_min=hp.min_lr)
    elif hp.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=hp.min_lr)
    else:
        scheduler = None

    def to_price(pred_scaled: np.ndarray) -> np.ndarray:
        t = pred_scaled * t_std + t_mean
        return np.expm1(t) if hp.log_target else t

    n = len(Xtr)
    history = {"epoch": [], "train_rmse": [], "val_rmse": [], "es_score": [], "lr": []}
    best = {"score": np.inf, "val": np.inf, "train": np.inf, "epoch": -1, "state": None}
    bad_epochs = 0

    for epoch in range(hp.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, hp.batch_size):
            idx = perm[i:i + hp.batch_size]
            if len(idx) < 2:  # BatchNorm necesita >=2 muestras
                continue
            optimizer.zero_grad()
            loss = criterion(model(Xtr[idx]), ytr[idx])
            loss.backward()
            if hp.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            tr_pred = to_price(model(Xtr).cpu().numpy())
            va_pred = to_price(model(Xva).cpu().numpy())
        tr_rmse = rmse(y_train, tr_pred)
        va_rmse = rmse(y_val, va_pred)

        # Señal de early stopping: estable (log) o directa (escala original).
        if hp.es_metric == "log_rmse":
            es_score = rmse(np.log1p(y_val), np.log1p(np.clip(va_pred, 0, None)))
        else:
            es_score = va_rmse

        history["epoch"].append(epoch)
        history["train_rmse"].append(tr_rmse)
        history["val_rmse"].append(va_rmse)
        history["es_score"].append(es_score)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            scheduler.step(es_score) if hp.scheduler == "plateau" else scheduler.step()

        if es_score < best["score"] - 1e-9:
            best.update(score=es_score, val=va_rmse, train=tr_rmse, epoch=epoch,
                        state={k: v.detach().clone()
                               for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= hp.patience:
                if verbose:
                    print(f"    early stopping en época {epoch}")
                break

        if verbose and epoch % 50 == 0:
            print(f"    ep {epoch:4d} | train {tr_rmse:9,.0f} | val {va_rmse:9,.0f}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    # Guardar los parámetros de la transformación del target dentro del modelo:
    # sin ellos las predicciones no se pueden devolver a escala de precio.
    model.target_mean_ = t_mean
    model.target_std_ = t_std
    model.log_target_ = hp.log_target

    return TrainResult(model=model, best_val_rmse=best["val"],
                       best_train_rmse=best["train"], best_epoch=best["epoch"],
                       history=history)


@torch.no_grad()
def predict(model: MLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Predice precios en escala original, deshaciendo las transformaciones."""
    model.eval()
    dev = torch.device(device)
    out = model(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
    t = out * model.target_std_ + model.target_mean_
    return np.expm1(t) if model.log_target_ else t
