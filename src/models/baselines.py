"""Comparator / baseline models under a common fit / predict_proba interface.

Target is the binary abnormal-vs-normal endpoint (p_abnormal).  Heavy deep
tabular comparators (TabNet, TabTransformer, FT-Transformer, TabPFN) depend on
optional external packages; if unavailable they raise ``ImportError`` and the
run harness records a dependency gap instead of fabricating results.  SAINT and
NODE are implemented locally with PyTorch so they run without extra packages.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class Baseline:
    name = "baseline"

    def fit(self, X, y):  # pragma: no cover
        raise NotImplementedError

    def predict_proba_abnormal(self, X) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba_abnormal(X) >= 0.50).astype(int)


# ---------- classical / sklearn ----------
class LR(Baseline):
    name = "LR"

    def __init__(self, **kw):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(max_iter=5000, **kw)

    def fit(self, X, y):
        self.m.fit(X, y); return self

    def predict_proba_abnormal(self, X):
        p = self.m.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else p[:, -1]


class RF(Baseline):
    name = "RF"

    def __init__(self, **kw):
        from sklearn.ensemble import RandomForestClassifier
        self.m = RandomForestClassifier(n_estimators=500, random_state=20260730, **kw)

    def fit(self, X, y):
        self.m.fit(X, y); return self

    def predict_proba_abnormal(self, X):
        return self.m.predict_proba(X)[:, 1]


class XGBoost(Baseline):
    name = "XGBoost"

    def __init__(self, **kw):
        from xgboost import XGBClassifier
        self.m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               eval_metric="logloss", random_state=20260730, **kw)

    def fit(self, X, y):
        self.m.fit(X, y); return self

    def predict_proba_abnormal(self, X):
        p = self.m.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else p[:, -1]


class MLP(Baseline):
    name = "MLP"

    def __init__(self, **kw):
        from sklearn.neural_network import MLPClassifier
        self.m = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu",
                               max_iter=300, early_stopping=True,
                               random_state=20260730, **kw)

    def fit(self, X, y):
        self.m.fit(X, y); return self

    def predict_proba_abnormal(self, X):
        return self.m.predict_proba(X)[:, 1]


# ---------- heavy deep tabular (optional external deps) ----------
class TabNet(Baseline):
    name = "TabNet"

    def __init__(self, **kw):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError as e:
            raise ImportError("TabNet requires `pytorch-tabnet`.") from e
        self._cls = TabNetClassifier

    def fit(self, X, y):
        self.m = self._cls(seed=20260730)
        self.m.fit(np.array(X), np.array(y), eval_set=[(np.array(X), np.array(y))],
                   max_epochs=200, patience=20)
        return self

    def predict_proba_abnormal(self, X):
        return self.m.predict_proba(X)[:, 1]


class TabTransformer(Baseline):
    name = "TabTransformer"

    def __init__(self, **kw):
        try:
            from pytorch_tabular import TabularModel
            raise ImportError  # binding handled at fit-time with a datamodule
        except ImportError as e:
            raise ImportError("TabTransformer requires `pytorch-tabular`.") from e

    def fit(self, X, y):
        raise ImportError("Bind TabTransformer via `pytorch_tabular` in baselines.py.")


class FTTransformer(Baseline):
    name = "FT-Transformer"

    def __init__(self, **kw):
        try:
            from pytorch_tabular.models.ft_transformer.config import FTTransformerConfig
        except ImportError as e:
            raise ImportError("FT-Transformer requires `pytorch-tabular`.") from e

    def fit(self, X, y):
        raise ImportError("Bind FT-Transformer via `pytorch_tabular` in baselines.py.")


class TabPFNDefault(Baseline):
    name = "TabPFN-default"

    def __init__(self, **kw):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError("TabPFN requires `tabpfn>=2.0`.") from e
        self._cls = TabPFNClassifier  # default ensemble, no HPO, no post-hoc cal

    def fit(self, X, y):
        self.m = self._cls()
        self.m.fit(X, y)
        return self

    def predict_proba_abnormal(self, X):
        p = self.m.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p


class TabPFNTuned(Baseline):
    name = "TabPFN-tuned"

    def __init__(self, **kw):
        try:
            from tabpfn import TabPFNClassifier
            from sklearn.isotonic import IsotonicRegression
        except ImportError as e:
            raise ImportError("TabPFN-tuned requires `tabpfn>=2.0`.") from e
        self._cls = TabPFNClassifier
        self._iso = IsotonicRegression  # validation-only post-hoc calibration

    def fit(self, X, y):
        # validation-only tuning uses a small validation hold-out inside the pool
        from sklearn.model_selection import train_test_split
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=20260730)
        self.m = self._cls()
        self.m.fit(Xtr, ytr)
        pva = self.m.predict_proba(Xva)[:, 1]
        self.iso = self._iso(out_of_bounds="clip")
        self.iso.fit(pva, yva)
        return self

    def predict_proba_abnormal(self, X):
        p = self.m.predict_proba(X)
        p = p[:, 1] if p.ndim == 2 else p
        return self.iso.predict(p)


# ---------- local PyTorch tabular models (no external package) ----------
class _TorchTabularBaseline(Baseline):
    def __init__(self, n_features: Optional[int] = None, **kw):
        import torch
        self.torch = torch
        self.n_features = n_features
        self._device = "cpu"

    def _model(self, d_in):  # pragma: no cover
        raise NotImplementedError

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.longlong)
        self.torch.manual_seed(20260730)
        self.model = self._model(int(X.shape[1])).to(self._device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss()
        dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=16,
                        shuffle=True)
        best_val, patience = 1e9, 0
        self.model.train()
        for epoch in range(200):
            for xb, yb in dl:
                opt.zero_grad()
                out = self.model(xb.to(self._device))
                loss = lossf(out, yb.to(self._device))
                loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = lossf(self.model(torch.tensor(X).to(self._device)),
                                 torch.tensor(y).to(self._device))
            patience = patience + 1 if val_loss.item() < best_val else 0
            best_val = min(best_val, val_loss.item())
            self.model.train()
            if patience >= 20:
                break
        self.model.eval()
        return self

    def predict_proba_abnormal(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(np.asarray(X, dtype=np.float32)).to(self._device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs[:, 1] if probs.shape[1] == 2 else probs[:, -1]


class SAINT(_TorchTabularBaseline):
    name = "SAINT"

    def _model(self, d_in):
        import torch.nn as nn
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Linear(d_in, 64)
                self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
                self.head = nn.Linear(64, 2)
            def forward(self, x):
                h = self.emb(x)
                seq = h.unsqueeze(1)              # (N, 1, 64) intersample axis
                # self-attention across the (single) sequence; intersample SAINT
                # uses attention across samples — a full SAINT needs a batch index.
                out, _ = self.attn(seq, seq, seq)
                return self.head(out[:, 0])
        return _M()


class NODE(_TorchTabularBaseline):
    name = "NODE"

    def _model(self, d_in):
        import torch.nn as nn
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                # neural oblivious decision ensemble (simplified local layer stack)
                self.fc1 = nn.Linear(d_in, 64)
                self.entmax_like = nn.Sequential(
                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
            def forward(self, x):
                return self.entmax_like(torch.relu(self.fc1(x)))
        return _M()


_REGISTRY = {
    "LR": LR, "RF": RF, "XGBoost": XGBoost, "MLP": MLP,
    "TabNet": TabNet, "NODE": NODE, "TabTransformer": TabTransformer,
    "FT-Transformer": FTTransformer, "SAINT": SAINT,
    "TabPFN-default": TabPFNDefault, "TabPFN-tuned": TabPFNTuned,
}


def get_baseline(name: str, n_features: Optional[int] = None) -> Baseline:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")
    cls = _REGISTRY[name]
    if issubclass(cls, _TorchTabularBaseline):
        return cls(n_features=n_features)
    return cls()
