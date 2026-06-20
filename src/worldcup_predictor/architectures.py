from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED
from .model import apply_temperature


def _weights(frame: pd.DataFrame) -> np.ndarray:
    age_years = (frame["date"].max() - frame["date"]).dt.days.to_numpy() / 365.25
    return np.power(0.5, age_years / 8.0) * (
        0.65 + frame["importance"].to_numpy()
    )


def _fit_temperature(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    from sklearn.metrics import log_loss

    result = minimize_scalar(
        lambda value: log_loss(
            y_true,
            apply_temperature(probabilities, float(value)),
            labels=[0, 1, 2],
        ),
        bounds=(0.55, 1.8),
        method="bounded",
    )
    return float(result.x)


@dataclass
class ArchitectureMetadata:
    device: str
    training_seconds: float
    parameters: int
    temperature: float


class GpuXGBoostClassifier:
    def __init__(
        self,
        feature_columns: list[str],
        random_state: int = RANDOM_SEED,
        use_gpu: bool = True,
    ):
        self.feature_columns = feature_columns
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.temperature = 1.0
        self.model = None
        self.metadata: ArchitectureMetadata | None = None

    def fit(self, frame: pd.DataFrame) -> GpuXGBoostClassifier:
        import xgboost as xgb

        data = frame.dropna(subset=["target"]).sort_values("date")
        split = int(len(data) * 0.82)
        train, calibration = data.iloc[:split], data.iloc[split:]
        device = "cuda" if self.use_gpu else "cpu"
        self.model = xgb.XGBClassifier(
            n_estimators=650,
            max_depth=4,
            learning_rate=0.025,
            min_child_weight=8,
            subsample=0.82,
            colsample_bytree=0.78,
            reg_alpha=0.08,
            reg_lambda=3.0,
            gamma=0.03,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            device=device,
            random_state=self.random_state,
            n_jobs=8,
        )
        start = time.perf_counter()
        self.model.fit(
            train[self.feature_columns],
            train["target"].astype(int),
            sample_weight=_weights(train),
            eval_set=[
                (
                    calibration[self.feature_columns],
                    calibration["target"].astype(int),
                )
            ],
            verbose=False,
        )
        raw = self.model.predict_proba(calibration[self.feature_columns])
        self.temperature = _fit_temperature(
            raw, calibration["target"].astype(int).to_numpy()
        )
        self.metadata = ArchitectureMetadata(
            device=device,
            training_seconds=time.perf_counter() - start,
            parameters=int(
                sum(tree.count('"leaf"') for tree in self.model.get_booster().get_dump(dump_format="json"))
            ),
            temperature=self.temperature,
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(frame[self.feature_columns])
        return apply_temperature(raw, self.temperature)


class ResidualTabularNetwork:
    def __init__(
        self,
        feature_columns: list[str],
        random_state: int = RANDOM_SEED,
        use_gpu: bool = True,
    ):
        self.feature_columns = feature_columns
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.temperature = 1.0
        self.scaler = StandardScaler()
        self.network = None
        self.metadata: ArchitectureMetadata | None = None

    def fit(self, frame: pd.DataFrame) -> ResidualTabularNetwork:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        device = torch.device(
            "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        )
        data = frame.dropna(subset=["target"]).sort_values("date")
        split = int(len(data) * 0.82)
        train, calibration = data.iloc[:split], data.iloc[split:]
        x_train = self.scaler.fit_transform(train[self.feature_columns]).astype(
            np.float32
        )
        x_cal = self.scaler.transform(calibration[self.feature_columns]).astype(
            np.float32
        )
        y_train = train["target"].astype(int).to_numpy()
        y_cal = calibration["target"].astype(int).to_numpy()
        sample_weights = _weights(train).astype(np.float32)

        class ResidualBlock(nn.Module):
            def __init__(self, width: int, dropout: float):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(width, width),
                    nn.BatchNorm1d(width),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, width),
                    nn.BatchNorm1d(width),
                )
                self.activation = nn.SiLU()

            def forward(self, inputs):
                return self.activation(inputs + self.layers(inputs))

        class Network(nn.Module):
            def __init__(self, dimensions: int):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(dimensions, 128),
                    nn.BatchNorm1d(128),
                    nn.SiLU(),
                    ResidualBlock(128, 0.18),
                    ResidualBlock(128, 0.18),
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.Dropout(0.12),
                    nn.Linear(64, 3),
                )

            def forward(self, inputs):
                return self.model(inputs)

        self.network = Network(len(self.feature_columns)).to(device)
        dataset = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
            torch.from_numpy(sample_weights),
        )
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=8e-4, weight_decay=2e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=90
        )
        loss_function = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=0.025
        )
        calibration_x = torch.from_numpy(x_cal).to(device)
        calibration_y = torch.from_numpy(y_cal).to(device)
        best_loss = float("inf")
        best_state = None
        patience = 14
        stale = 0
        start = time.perf_counter()
        for _ in range(120):
            self.network.train()
            for batch_x, batch_y, batch_weights in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_weights = batch_weights.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    logits = self.network(batch_x)
                    loss = (
                        loss_function(logits, batch_y) * batch_weights
                    ).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 2.0)
                optimizer.step()
            scheduler.step()
            self.network.eval()
            with torch.no_grad():
                validation_loss = nn.functional.cross_entropy(
                    self.network(calibration_x), calibration_y
                ).item()
            if validation_loss < best_loss - 1e-4:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.network.load_state_dict(best_state)
        self.network.to(device).eval()
        raw = self._raw_probabilities(x_cal, device)
        self.temperature = _fit_temperature(raw, y_cal)
        self.metadata = ArchitectureMetadata(
            device=str(device),
            training_seconds=time.perf_counter() - start,
            parameters=sum(
                parameter.numel() for parameter in self.network.parameters()
            ),
            temperature=self.temperature,
        )
        return self

    def _raw_probabilities(self, values: np.ndarray, device) -> np.ndarray:
        import torch

        self.network.eval()
        output = []
        with torch.no_grad():
            for start in range(0, len(values), 4096):
                batch = torch.from_numpy(values[start : start + 4096]).to(device)
                output.append(
                    torch.softmax(self.network(batch), dim=1).cpu().numpy()
                )
        return np.vstack(output)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        device = next(self.network.parameters()).device
        values = self.scaler.transform(frame[self.feature_columns]).astype(
            np.float32
        )
        raw = self._raw_probabilities(values, device)
        return apply_temperature(raw, self.temperature)
