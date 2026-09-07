"""Minimal SSM-Inception implementation used for the DSADS experiment."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat


class TiedDropout(nn.Module):
    """Drop complete feature channels using one mask across time."""

    def __init__(self, probability: float, transposed: bool = True):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("probability must be in [0, 1)")
        self.probability = probability
        self.transposed = transposed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        if not self.transposed:
            x = rearrange(x, "b ... d -> b d ...")
        mask = torch.rand(x.shape[:2] + (1,) * (x.ndim - 2), device=x.device)
        x = x * (mask >= self.probability) / (1.0 - self.probability)
        if not self.transposed:
            x = rearrange(x, "b d ... -> b ... d")
        return x


class S4DKernel(nn.Module):
    """Generate the real convolution kernel of a diagonal S4 layer."""

    def __init__(
        self,
        d_model: int,
        state_size: int = 64,
        dt_min: float = 1.0e-3,
        dt_max: float = 1.0e-1,
    ):
        super().__init__()
        if state_size % 2:
            raise ValueError("state_size must be even")
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
        log_dt = log_dt + math.log(dt_min)
        c = torch.randn(d_model, state_size // 2, dtype=torch.cfloat)
        log_a_real = torch.log(0.5 * torch.ones(d_model, state_size // 2))
        a_imag = math.pi * repeat(
            torch.arange(state_size // 2), "n -> h n", h=d_model
        )

        self.c = nn.Parameter(torch.view_as_real(c))
        self.log_dt = nn.Parameter(log_dt)
        self.log_a_real = nn.Parameter(log_a_real)
        self.a_imag = nn.Parameter(a_imag)
        # These parameters used zero weight decay in the reported setup.
        for parameter in (self.log_dt, self.log_a_real, self.a_imag):
            parameter._optim = {"weight_decay": 0.0}

    def forward(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        c = torch.view_as_complex(self.c)
        a = -torch.exp(self.log_a_real) + 1j * self.a_imag
        dt_a = a * dt.unsqueeze(-1)
        time = torch.arange(length, device=a.device)
        vandermonde = dt_a.unsqueeze(-1) * time
        c = c * (torch.exp(dt_a) - 1.0) / a
        return 2.0 * torch.einsum("hn,hnl->hl", c, torch.exp(vandermonde)).real


class S4D(nn.Module):
    """FFT implementation of an S4D convolution, input shape [B, H, L]."""

    def __init__(self, d_model: int, state_size: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.skip = nn.Parameter(torch.randn(d_model))
        self.kernel = S4DKernel(d_model, state_size)
        self.activation = nn.GELU()
        self.dropout = TiedDropout(dropout)
        self.output_projection = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        kernel = self.kernel(length)
        kernel_fft = torch.fft.rfft(kernel, n=2 * length)
        input_fft = torch.fft.rfft(x, n=2 * length)
        y = torch.fft.irfft(input_fft * kernel_fft, n=2 * length)[..., :length]
        y = y + x * self.skip.unsqueeze(-1)
        return self.output_projection(self.dropout(self.activation(y)))


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class DSInceptionSE(nn.Module):
    """Five-branch depthwise-separable Inception block with SE attention."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels % 4:
            raise ValueError("out_channels must be divisible by four")
        branch_channels = out_channels // 4
        self.reduce = nn.Conv1d(in_channels, branch_channels, kernel_size=1)
        self.temporal_branches = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(branch_channels, kernel_size)
                for kernel_size in (3, 5, 7, 9)
            ]
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_projection = nn.Conv1d(
            in_channels, branch_channels, kernel_size=1
        )
        concatenated_channels = 5 * branch_channels
        self.attention = nn.Sequential(
            nn.Linear(concatenated_channels, concatenated_channels // 4),
            nn.ReLU(),
            nn.Linear(concatenated_channels // 4, concatenated_channels),
            nn.Sigmoid(),
        )
        self.compression = nn.Conv1d(
            concatenated_channels, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        branches = [branch(reduced) for branch in self.temporal_branches]
        branches.append(self.pool_projection(self.pool(x)))
        concatenated = torch.cat(branches, dim=1)
        weights = self.attention(concatenated.mean(dim=2)).unsqueeze(2)
        return self.compression(concatenated * weights)


class SSMInception(nn.Module):
    """SSM-Inception classifier for channel-first sensor windows."""

    def __init__(
        self,
        num_channels: int = 45,
        num_classes: int = 19,
        embedding_dim: int = 32,
        state_size: int = 32,
        s4d_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(num_channels, embedding_dim)
        self.s4d = S4D(embedding_dim, state_size, s4d_dropout)
        self.local_block_1 = nn.Sequential(
            DSInceptionSE(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.local_block_2 = nn.Sequential(
            DSInceptionSE(64, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("expected input shape [batch, channels, time]")
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2)
        x = self.s4d(x)
        x = self.local_block_1(x)
        x = self.local_block_2(x)
        x = self.output_norm(self.global_pool(x).squeeze(-1))
        return self.classifier(x)

