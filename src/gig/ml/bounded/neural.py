"""Families 36-39: bounded CPU sequence networks on admitted return histories.

Torch (or NumPy for the reservoir) trains small fixed architectures with a
fixed step budget and seed. Replay is saved NumPy algebra. Row t forecasts
observation t+1 from at most the last eight observations up to t.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from gig.ml.bounded.core import Family, register, unavailable
from gig.ml.bounded.supervised import vector

WINDOW, STEPS, RATE, DECAY = 8, 200, .01, 1e-3
CHANNELS, DILATIONS = 4, (1, 2, 4)
BLOCKS, UNITS = 2, 8
EMBED = 4
RESERVOIR, RADIUS, LEAK_INPUT, WASHOUT, RIDGE = 32, .9, .5, 10, 1e-2


def _windows(series):
    """Rows of the last WINDOW observations ending at t; earlier rows have no complete window."""
    return np.lib.stride_tricks.sliding_window_view(series, WINDOW)


def _pad(values):
    return np.vstack([np.full((WINDOW-1, values.shape[1]), np.nan), values]) if len(values) else values


def _train(parameters, forward, series):
    """Fixed-budget full-batch Adam on next-observation squared error; deterministic on one CPU thread."""
    import torch
    inputs = torch.tensor(_windows(series)[:-1].copy(), dtype=torch.float64)
    target = torch.tensor(series[WINDOW:], dtype=torch.float64)
    optimizer = torch.optim.Adam(parameters, lr=RATE, weight_decay=DECAY)
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = ((forward(inputs)-target)**2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        full = forward(torch.tensor(_windows(series).copy(), dtype=torch.float64)).numpy()
    return float(loss.detach()), _pad(full[:, None])


def _tensors(seed, shapes: dict[str, tuple[int, ...]]):
    import torch
    generator = torch.Generator().manual_seed(seed)
    return {name: (torch.randn(shape, generator=generator, dtype=torch.float64)/np.sqrt(shape[0] if shape else 1)).requires_grad_()
            for name, shape in shapes.items()}


def _export(tensors, loss) -> dict[str, Any]:
    return {**{name: value.detach().numpy().tolist() for name, value in tensors.items()}, "final_training_loss": loss}


def _replay(shapes, forward):
    def replay(payload, z):
        if len(z) < WINDOW:
            return np.full((len(z), 1), np.nan)
        weights = {name: vector(payload, name, shape) for name, shape in shapes.items()}
        return _pad(forward(weights, _windows(z[:, 0]), np)[:, None])
    return replay


def _fit(shapes, forward):
    def fit(z, _target, seed):
        import torch
        tensors = _tensors(seed, shapes)
        loss, expected = _train(list(tensors.values()), lambda x: forward(tensors, x, torch), z[:, 0])
        return _export(tensors, loss), expected
    return fit


def _relu(value, lib):
    return lib.clip(value, 0, None) if lib is np else lib.clamp(value, min=0)


_TCN = {"input": (CHANNELS,), "bias_0": (CHANNELS,),
        **{f"kernel_{i}": (2, CHANNELS, CHANNELS) for i in range(1, len(DILATIONS))},
        **{f"bias_{i}": (CHANNELS,) for i in range(1, len(DILATIONS))},
        "input_previous": (CHANNELS,), "readout": (CHANNELS,), "readout_bias": ()}


def _tcn_forward(w, windows, lib):
    """Causal dilated convolutions (kernel 2, dilations 1, 2, 4): receptive field exactly eight rows."""
    hidden = _relu(windows[:, 1:, None]*w["input"]+windows[:, :-1, None]*w["input_previous"]+w["bias_0"], lib)
    for index, dilation in enumerate(DILATIONS[1:], start=1):
        hidden = _relu(hidden[:, dilation:]@w[f"kernel_{index}"][1]+hidden[:, :-dilation]@w[f"kernel_{index}"][0]
                       + w[f"bias_{index}"], lib)
    return hidden[:, -1]@w["readout"]+w["readout_bias"]


_NBEATS = {k: v for block in range(BLOCKS) for k, v in {
    f"hidden_{block}": (WINDOW, UNITS), f"hidden_bias_{block}": (UNITS,), f"deep_{block}": (UNITS, UNITS),
    f"deep_bias_{block}": (UNITS,), f"backcast_{block}": (UNITS, WINDOW), f"forecast_{block}": (UNITS,),
    f"forecast_bias_{block}": ()}.items()}


def _nbeats_forward(w, windows, lib):
    """Generic doubly-residual blocks: each removes its backcast and adds a partial forecast."""
    residual, forecast = windows, 0.
    for block in range(BLOCKS):
        hidden = _relu(residual@w[f"hidden_{block}"]+w[f"hidden_bias_{block}"], lib)
        hidden = _relu(hidden@w[f"deep_{block}"]+w[f"deep_bias_{block}"], lib)
        residual = residual-hidden@w[f"backcast_{block}"]
        forecast = forecast+hidden@w[f"forecast_{block}"]+w[f"forecast_bias_{block}"]
    return forecast


_ATTENTION = {"embed": (EMBED,), "position": (WINDOW, EMBED), "query": (EMBED, EMBED), "key": (EMBED, EMBED),
              "value": (EMBED, EMBED), "readout": (EMBED,), "readout_bias": ()}


def _attention_forward(w, windows, lib):
    """One self-attention head; the newest observation queries the eight-row window."""
    tokens = windows[:, :, None]*w["embed"]+w["position"]
    scores = ((tokens[:, -1]@w["query"])[:, None, :]*(tokens@w["key"])).sum(-1)/np.sqrt(EMBED)
    scores = scores-scores.max(-1, keepdims=True) if lib is np else scores-scores.max(-1, keepdim=True).values
    weights = lib.exp(scores)
    weights = weights/(weights.sum(-1, keepdims=True) if lib is np else weights.sum(-1, keepdim=True))
    return ((weights[:, :, None]*(tokens@w["value"])).sum(1))@w["readout"]+w["readout_bias"]


def _states(payload, series):
    recurrent, entry, bias = (vector(payload, k, s) for k, s in
                              (("recurrent", (RESERVOIR, RESERVOIR)), ("input", (RESERVOIR,)), ("bias", (RESERVOIR,))))
    if np.max(np.abs(np.linalg.eigvals(recurrent))) > RADIUS+1e-9:
        raise ValueError("saved reservoir exceeds the fixed spectral radius; the echo-state property is unsupported")
    state, out = np.zeros(RESERVOIR), np.empty((len(series), RESERVOIR+1))
    for index, value in enumerate(series):
        state = np.tanh(recurrent@state+entry*value+bias)
        out[index] = np.r_[1., state]
    return out


def _esn_solve(design, target):
    penalty = RIDGE*np.eye(design.shape[1])
    penalty[0, 0] = 0.
    return np.linalg.solve(design.T@design+penalty, design.T@target)


def _esn_fit(z, _target, seed):
    rng = np.random.default_rng(seed)
    recurrent = rng.normal(size=(RESERVOIR, RESERVOIR))
    recurrent *= RADIUS/np.max(np.abs(np.linalg.eigvals(recurrent)))
    payload = {"recurrent": recurrent.tolist(), "input": (LEAK_INPUT*rng.normal(size=RESERVOIR)).tolist(),
               "bias": (.1*rng.normal(size=RESERVOIR)).tolist()}
    states = _states(payload, z[:, 0])
    payload["readout"] = _esn_solve(states[WASHOUT:-1], z[WASHOUT+1:, 0]).tolist()
    return payload, None


def _esn_replay(payload, z):
    return (_states(payload, z[:, 0])@vector(payload, "readout", (RESERVOIR+1,)))[:, None]


def _esn_check(payload, z, _target):
    """Only the readout is trained: it must solve the ridge normal equations on training states."""
    states = _states(payload, z[:, 0])
    if not np.allclose(states[WASHOUT:-1]@vector(payload, "readout", (RESERVOIR+1,)),
                       states[WASHOUT:-1]@_esn_solve(states[WASHOUT:-1], z[WASHOUT+1:, 0]), rtol=1e-6, atol=1e-8):
        raise ValueError("saved reservoir readout does not solve its training ridge problem")


_NOTE = (" Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged"
         " training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.")
_BUDGET = {"window": WINDOW, "steps": STEPS, "learning_rate": RATE, "weight_decay": DECAY}
_TORCH: dict[str, Any] = dict(kind="NEXT_OBSERVATION_RETURN_ESTIMATE", data="SERIES", requires=("torch",), min_train=40,
                              uncertainty=unavailable("POINT_NETWORK_HAS_NO_PREDICTIVE_DISTRIBUTION"), tolerance=1e-9)
register(Family(
    key="tcn", number=36, capability="temporal_convolution", label="Temporal convolution network · return history",
    scope="Bounded CPU temporal convolution network on each instrument's admitted return history: three causal dilated layers (kernel 2, dilations 1/2/4, 4 channels, 8-row receptive field), 200 full-batch Adam steps, seeded, one thread, 40–256 training rows, 16 fits." + _NOTE,
    fit=_fit(_TCN, _tcn_forward), replay=_replay(_TCN, _tcn_forward), backend="torch (causal dilated convolution) + torch.optim.Adam",
    parameters={**_BUDGET, "channels": CHANNELS, "dilations": list(DILATIONS), "kernel": 2}, **_TORCH))
register(Family(
    key="nbeats", number=38, capability="nbeats", label="N-BEATS generic blocks · return history",
    scope="Bounded CPU N-BEATS with two generic doubly-residual blocks (8-row lookback, two 8-unit ReLU layers each, one-step forecast): 200 full-batch Adam steps, seeded, one thread, 40–256 training rows, 16 fits. Trend/seasonality basis stacks are unavailable." + _NOTE,
    fit=_fit(_NBEATS, _nbeats_forward), replay=_replay(_NBEATS, _nbeats_forward), backend="torch (generic N-BEATS blocks) + torch.optim.Adam",
    parameters={**_BUDGET, "blocks": BLOCKS, "units": UNITS, "basis": "GENERIC"}, **_TORCH))
register(Family(
    key="attention", number=39, capability="attention_sequence", label="Single-head attention · return history",
    scope="Bounded CPU single-head self-attention over the last eight returns (4-dimensional embeddings, learned positions, newest observation as query): 200 full-batch Adam steps, seeded, one thread, 40–256 training rows, 16 fits. This is one attention layer, not a pretrained or multi-layer transformer." + _NOTE,
    fit=_fit(_ATTENTION, _attention_forward), replay=_replay(_ATTENTION, _attention_forward),
    backend="torch (single-head scaled dot-product attention) + torch.optim.Adam",
    parameters={**_BUDGET, "embedding": EMBED, "heads": 1, "layers": 1}, **_TORCH))
register(Family(
    key="esn", number=37, capability="echo_state_network", label="Echo state network · return history",
    scope="Bounded CPU echo state network on each instrument's admitted return history: fixed seeded 32-unit tanh reservoir at spectral radius 0.9, 10-row washout, ridge readout (0.01) solved in closed form, 40–256 training rows, 16 fits. Only the readout is trained." + _NOTE,
    kind="NEXT_OBSERVATION_RETURN_ESTIMATE", fit=_esn_fit, replay=_esn_replay, check=_esn_check, data="SERIES",
    backend="numpy.linalg (seeded reservoir, closed-form ridge readout)", requires=("numpy",), min_train=40,
    parameters={"reservoir": RESERVOIR, "spectral_radius": RADIUS, "input_scale": LEAK_INPUT, "washout": WASHOUT, "ridge": RIDGE},
    uncertainty=unavailable("POINT_NETWORK_HAS_NO_PREDICTIVE_DISTRIBUTION")))
