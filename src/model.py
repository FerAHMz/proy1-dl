"""El MLP y su bucle de entrenamiento.

Es un Multi-Layer Perceptron puro, como pide el proyecto: capas densas con
activación, normalización y dropout.
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
    """Junto todos los hiperparámetros de una iteración acá para poder
    guardarlos con su RMSE en reports/experiments.csv."""

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

    # Con qué métrica el early stopping decide qué checkpoint guardar.
    #   "rmse"     -> RMSE en dólares (la métrica de la competencia)
    #   "log_rmse" -> RMSE sobre log1p(precio)
    # Dejo log_rmse por defecto. Con el RMSE en dólares, una o dos casas caras
    # por fold mandan sobre el resultado, así que como señal para elegir era
    # puro ruido y me cortaba el entrenamiento demasiado pronto. En los dos
    # casos el RMSE que reporto sigue siendo el de dólares.
    es_metric: str = "log_rmse"

    def to_row(self) -> dict:
        d = asdict(self)
        d["hidden"] = "-".join(map(str, self.hidden))
        return d


class MLP(nn.Module):
    """MLP armado con bloques [Linear -> BatchNorm -> Activación -> Dropout]."""

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
        # Kaiming va bien con ReLU y sus parientes, que es lo que uso.
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
        # Huber aguanta mejor los precios extremos que MSE: así unas pocas
        # casas de $700k no se llevan todo el gradiente.
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
    """Entrena un MLP y me devuelve la mejor versión según validación.

    Aunque entrene en escala logarítmica, el RMSE que reporto siempre está en
    dólares, que es la métrica con la que nos van a calificar.
    """
    dev = torch.device(device)

    # Si log_target está activo la red predice log1p(precio) y lo devuelvo a
    # dólares con expm1 antes de medir.
    t_train = np.log1p(y_train) if hp.log_target else y_train.copy()
    t_val = np.log1p(y_val) if hp.log_target else y_val.copy()

    # Estandarizo el target sobre todo para cuando NO uso log: los precios
    # crudos andan en 1e5 y me saturaban la red.
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
            if len(idx) < 2:  # con 1 sola muestra BatchNorm truena
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

        # La señal con la que decido si este checkpoint es el mejor.
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
    # Pego al modelo los parámetros con los que transformé el target; sin ellos
    # después no puedo devolver las predicciones a dólares.
    model.target_mean_ = t_mean
    model.target_std_ = t_std
    model.log_target_ = hp.log_target

    return TrainResult(model=model, best_val_rmse=best["val"],
                       best_train_rmse=best["train"], best_epoch=best["epoch"],
                       history=history)


@torch.no_grad()
def predict(model: MLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Predice en dólares, deshaciendo las transformaciones del target."""
    model.eval()
    dev = torch.device(device)
    out = model(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
    t = out * model.target_std_ + model.target_mean_
    return np.expm1(t) if model.log_target_ else t
