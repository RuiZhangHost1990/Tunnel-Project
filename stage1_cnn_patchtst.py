"""Shared temporal CNN followed by the package's channel-independent PatchTST.

Task adaptation of https://github.com/yuqinie98/PatchTST, not upstream weights.
Only observed history enters RevIN/CNN/patches; no recurrent layers are used.
"""
from __future__ import annotations

import torch
from torch import nn
from .stage1_patchtst import Stage1PatchTST, ARCHITECTURE as PATCH_ARCHITECTURE


ARCHITECTURE = {
    **PATCH_ARCHITECTURE,
    "model_type": "cnn_patchtst",
    "temporal_CNN": "shared per channel Conv1d(1,16,3,padding=1), GELU, Conv1d(16,1,3,padding=1)",
    "CNN_fusion": "normalized_history + CNN(normalized_history), before patching",
    "CNN_initialization": "Xavier uniform; final convolution gain 0.1; zero biases",
    "adaptation": "residual temporal CNN plus static/time covariate MLP forecast head",
}


class Stage1CNNPatchTST(Stage1PatchTST):
    """[B,25,30] + [B,20] + [B,1] -> [B,5,30].

    The same CNN and Transformer are applied to all monitoring channels.
    RevIN statistics are computed before CNN processing and reversed by the
    inherited forecast head, so predictions retain the external scaler's units.
    """
    def __init__(self, num_points=30, history_steps=25, future_steps=5):
        if history_steps < ARCHITECTURE["patch_length"]:
            raise ValueError("History must contain at least one full patch.")
        super().__init__(num_points, history_steps, future_steps)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )
        nn.init.xavier_uniform_(self.temporal_cnn[0].weight)
        nn.init.xavier_uniform_(self.temporal_cnn[2].weight, gain=0.1)
        nn.init.zeros_(self.temporal_cnn[0].bias)
        nn.init.zeros_(self.temporal_cnn[2].bias)

    def history_patches(self, history):
        batch, steps, channels = history.shape
        series = history.transpose(1, 2).reshape(batch * channels, 1, steps)
        local = (series + self.temporal_cnn(series)).reshape(batch, channels, steps)
        return local.unfold(-1, ARCHITECTURE["patch_length"], ARCHITECTURE["patch_stride"])


def check_model_structure():
    """Verify the CNN and attention both contribute and receive gradients."""
    with torch.random.fork_rng():
        torch.manual_seed(123)
        model = Stage1CNNPatchTST().eval()
        assert sum(isinstance(m, nn.Conv1d) for m in model.modules()) == 2
        assert not any(isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)) for m in model.modules())
        features, times, history = torch.randn(2, 20), torch.randn(2, 1), torch.randn(2, 25, 30)
        with torch.no_grad():
            prediction = model(features, times, history)
            changed = history.clone()
            changed[:, :, 1] += torch.linspace(-1, 1, 25)
            other = model(features, times, changed)
            assert torch.equal(prediction[:, :, 0], other[:, :, 0])
            assert not torch.equal(prediction[:, :, 1], other[:, :, 1])
            assert prediction.shape == (2, 5, 30) and torch.isfinite(prediction).all()
            weight = model.temporal_cnn[2].weight.clone()
            model.temporal_cnn[2].weight.zero_()
            ablated = model(features, times, history)
            assert not torch.allclose(prediction, ablated)
            model.temporal_cnn[2].weight.copy_(weight)
        model.train()
        model(features, times, history).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.temporal_cnn[0].weight.grad.abs().sum() > 0
        assert model.encoder[0].query.weight.grad.abs().sum() > 0
    return {"two_active_temporal_convolutions": True, "no_recurrent_layers": True,
            "channel_independence_in_evaluation": True, "CNN_and_attention_gradients": True,
            "finite_forward_backward": True}
