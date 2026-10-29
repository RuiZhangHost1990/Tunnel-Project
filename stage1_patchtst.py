"""Channel-independent PatchTST baseline with linear covariate adapters.

The temporal backbone follows the supervised PatchTST design: shared patch
projection, shared channel-independent encoder, residual attention, BatchNorm,
RevIN, and flattening over patches. A shared covariate-conditioned MLP head
retains the existing benchmark's static and time inputs without a CNN.
This is a task-adapted implementation, not an unmodified upstream model.
Reference: https://github.com/yuqinie98/PatchTST
"""
from __future__ import annotations

import math
import torch
from torch import nn

ARCHITECTURE = {
    "model_type": "PatchTST_without_CNN_with_linear_covariate_adapters",
    "channel_independent_temporal_encoder": True,
    "patch_length": 5,
    "patch_stride": 5,
    "patch_padding": "none; 25 observed steps form five complete patches",
    "d_model": 128,
    "encoder_layers": 3,
    "attention_heads": 8,
    "feedforward_dimension": 256,
    "activation": "GELU",
    "encoder_normalization": "BatchNorm1d, post-norm",
    "residual_attention": True,
    "attention_dropout": 0.0,
    "dropout": 0.2,
    "position_encoding": "learnable, uniform initialization [-0.02, 0.02]",
    "RevIN": True,
    "RevIN_affine": True,
    "RevIN_epsilon": 1e-5,
    "RevIN_statistics": "per case and monitoring channel, observed history only",
    "static_embedding_dimension": 64,
    "time_embedding_dimension": 16,
    "covariate_adapters": "separate Linear + ReLU projections; no convolution",
    "prediction_head": "shared across channels: 720 -> 128 (ReLU) -> 5",
    "patch_pooling": "flatten all five patch embeddings, feature-major order",
    "series_decomposition": False,
    "initialization": "Xavier uniform for Linear layers; BatchNorm/RevIN weights 1, biases 0",
    "upstream_reference": "https://github.com/yuqinie98/PatchTST",
    "adaptation": "linear static/time adapters and shared MLP replace the upstream linear forecast head",
}


class PatchEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        dim = ARCHITECTURE["d_model"]
        self.heads = ARCHITECTURE["attention_heads"]
        self.head_dim = dim // self.heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.2))
        self.attention_residual_dropout = nn.Dropout(0.2)
        self.attention_norm = nn.BatchNorm1d(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, dim)
        )
        self.feedforward_residual_dropout = nn.Dropout(0.2)
        self.feedforward_norm = nn.BatchNorm1d(dim)

    def forward(self, tokens, previous_scores=None):
        batch, patches, dim = tokens.shape
        projections = [
            layer(tokens).reshape(batch, patches, self.heads, self.head_dim).transpose(1, 2)
            for layer in (self.query, self.key, self.value)
        ]
        query, key, value = projections
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if previous_scores is not None:
            scores = scores + previous_scores
        attended = (scores.softmax(dim=-1) @ value).transpose(1, 2).reshape(batch, patches, dim)
        tokens = tokens + self.attention_residual_dropout(self.output(attended))
        tokens = self.attention_norm(tokens.transpose(1, 2)).transpose(1, 2)
        tokens = tokens + self.feedforward_residual_dropout(self.feedforward(tokens))
        tokens = self.feedforward_norm(tokens.transpose(1, 2)).transpose(1, 2)
        return tokens, scores


class Stage1PatchTST(nn.Module):
    def __init__(self, num_points=30, history_steps=25, future_steps=5):
        super().__init__()
        self.num_points = num_points
        self.history_steps = history_steps
        self.future_steps = future_steps
        length, stride = ARCHITECTURE["patch_length"], ARCHITECTURE["patch_stride"]
        if (history_steps - length) % stride:
            raise ValueError("This experiment uses complete patches without padding.")
        self.patch_count = 1 + (history_steps - length) // stride
        dim = ARCHITECTURE["d_model"]
        self.patch_projection = nn.Linear(length, dim)
        self.position_encoding = nn.Parameter(torch.empty(1, self.patch_count, dim))
        self.token_dropout = nn.Dropout(0.2)
        self.encoder = nn.ModuleList([PatchEncoderLayer() for _ in range(3)])
        self.static_projection = nn.Sequential(nn.Linear(20, 64), nn.ReLU())
        self.time_projection = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(self.patch_count * dim + 64 + 16, 128), nn.ReLU(), nn.Linear(128, future_steps)
        )
        self.revin_weight = nn.Parameter(torch.ones(1, 1, num_points))
        self.revin_bias = nn.Parameter(torch.zeros(1, 1, num_points))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.position_encoding, -0.02, 0.02)

    def history_patches(self, history):
        """[case, step, point] -> [case, point, patch, within-patch step]."""
        return history.transpose(1, 2).unfold(
            -1, ARCHITECTURE["patch_length"], ARCHITECTURE["patch_stride"]
        )

    def forward(self, features, time_steps, history):
        if history.shape[1:] != (self.history_steps, self.num_points):
            raise ValueError("Unexpected historical response dimensions.")
        batch = len(history)
        mean = history.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(history.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = ((history - mean) / std) * self.revin_weight + self.revin_bias
        patches = self.history_patches(normalized)
        tokens = self.patch_projection(patches).reshape(batch * self.num_points, self.patch_count, -1)
        tokens = self.token_dropout(tokens + self.position_encoding)
        scores = None
        for layer in self.encoder:
            tokens, scores = layer(tokens, scores)
        temporal = tokens.transpose(1, 2).reshape(batch, self.num_points, -1)
        context = torch.cat([self.static_projection(features), self.time_projection(time_steps)], dim=1)
        context = context.unsqueeze(1).expand(-1, self.num_points, -1)
        forecast = self.head(torch.cat([temporal, context], dim=-1)).transpose(1, 2)
        return ((forecast - self.revin_bias) / (self.revin_weight + 1e-10)) * std + mean


def check_model_structure():
    """Check defining properties before any training or test-set evaluation."""
    torch.manual_seed(123)
    model = Stage1PatchTST().eval()
    assert not any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.GRU, nn.LSTM)) for m in model.modules())
    ramp = torch.arange(25.0).reshape(1, 25, 1).expand(1, 25, 30)
    patches = model.history_patches(ramp)
    assert patches.shape == (1, 30, 5, 5)
    assert torch.equal(patches[0, 0].reshape(-1), torch.arange(25.0))
    features, time_steps, history = torch.randn(2, 20), torch.randn(2, 1), torch.randn(2, 25, 30)
    with torch.no_grad():
        prediction = model(features, time_steps, history)
        altered = history.clone()
        altered[:, :, 1] += torch.linspace(-1.0, 1.0, 25)
        changed_prediction = model(features, time_steps, altered)
    assert prediction.shape == (2, 5, 30) and torch.isfinite(prediction).all()
    # Another channel's history must not change channel 0's forecast in evaluation.
    assert torch.equal(prediction[:, :, 0], changed_prediction[:, :, 0])
    assert not torch.equal(prediction[:, :, 1], changed_prediction[:, :, 1])
    model.train()
    model(features, time_steps, history).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return {"no_CNN_or_recurrent_layers": True, "five_complete_history_patches": True,
            "channel_independence_in_evaluation": True, "forward_backward_finite": True}
