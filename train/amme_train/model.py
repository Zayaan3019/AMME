"""DeepLOB: CNN + Inception + LSTM + Attention over LOB snapshots.

Zhang, Zohren & Roberts (2019), *"DeepLOB: Deep Convolutional Neural Networks
for Limit Order Books"*, IEEE Transactions on Signal Processing 67(11).

The architecture is not arbitrary -- each stage exploits a specific structure in
the ``(T, 4L)`` input, and being able to say *why* each one is there is the
point of building it rather than importing it:

**Stage 1, convolutions across the feature axis.** The 40 columns are not
interchangeable: they are ``[ask_p, ask_v, bid_p, bid_v]`` repeated per level.
The first ``(1,2)`` stride-2 convolution therefore sees exactly one
``(price, volume)`` pair at a time and learns a per-level summary; the second
halves again to combine ask-side and bid-side of the same level; the third,
width ``(1,10)``, integrates across all levels. Padding is *deliberately absent*
on the horizontal axis -- padding would mix a level's price with its neighbour's
volume, which is meaningless.

**Stage 2, convolutions along time** ``(4,1)``. Local temporal patterns within
each learned feature map.

**Stage 3, an Inception module.** Parallel ``3x1`` / ``5x1`` / pooled branches
capture dynamics at several time scales at once, instead of committing to one
receptive field. Concatenated to 192 channels.

**Stage 4, LSTM.** Long-range temporal dependence the convolutions cannot reach.

**Stage 5, attention** (our addition; the paper takes the final hidden state).
Taking only ``h_T`` forces the entire 100-step history through one vector and
throws away *where* in the window the informative event happened. Additive
attention pools all hidden states instead, and the weights are inspectable --
which matters for a model whose output is allowed to move quotes.

Output: 3 classes (down / flat / up) over the smoothed-label horizon.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """One horizontal (feature-axis) conv followed by two temporal convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        leaky_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(leaky_slope),
            nn.BatchNorm2d(out_channels),
            # (4,1) with padding (3,0) then cropping keeps the time axis length
            # fixed while remaining strictly causal-agnostic in width.
            nn.Conv2d(out_channels, out_channels, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(leaky_slope),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(leaky_slope),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InceptionModule(nn.Module):
    """Multi-scale temporal features: 3x1, 5x1 and a pooled branch."""

    def __init__(self, in_channels: int, branch_channels: int = 64,
                 leaky_slope: float = 0.01) -> None:
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(leaky_slope),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(leaky_slope),
            nn.BatchNorm2d(branch_channels),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(leaky_slope),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(leaky_slope),
            nn.BatchNorm2d(branch_channels),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(leaky_slope),
            nn.BatchNorm2d(branch_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.branch3(x), self.branch5(x), self.branch_pool(x)], dim=1)


class AdditiveAttention(nn.Module):
    """Bahdanau-style pooling over LSTM hidden states.

    ``score_t = v^T tanh(W h_t)``, softmax over ``t``, context is the weighted
    sum. Replaces "take the last hidden state", which discards the position of
    the informative event inside the window.
    """

    def __init__(self, hidden_size: int, attention_size: int = 32) -> None:
        super().__init__()
        self.project = nn.Linear(hidden_size, attention_size)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """``hidden`` is ``(B, T, H)``; returns ``(context (B,H), weights (B,T))``."""
        energy = torch.tanh(self.project(hidden))     # (B, T, A)
        scores = self.score(energy).squeeze(-1)       # (B, T)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return context, weights


class DeepLOB(nn.Module):
    """The full model. Input ``(B, 1, T, 4L)``, output ``(B, 3)`` logits."""

    def __init__(
        self,
        levels: int = 10,
        lookback: int = 100,
        lstm_hidden: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.lookback = lookback
        self.use_attention = use_attention

        # 4L -> 2L -> L -> 1 across the three horizontal kernels.
        self.conv1 = ConvBlock(1, 32, kernel=(1, 2), stride=(1, 2))
        self.conv2 = ConvBlock(32, 32, kernel=(1, 2), stride=(1, 2))
        self.conv3 = ConvBlock(32, 32, kernel=(1, levels), stride=(1, 1))

        self.inception = InceptionModule(32, branch_channels=64)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=192, hidden_size=lstm_hidden, batch_first=True)
        self.attention = AdditiveAttention(lstm_hidden) if use_attention else None
        self.head = nn.Linear(lstm_hidden, num_classes)

    def _crop_time(self, x: Tensor) -> Tensor:
        """Trim the padding-induced overhang back to ``lookback`` steps."""
        return x[:, :, : self.lookback, :]

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:                      # (B, T, F) -> add the channel axis
            x = x.unsqueeze(1)
        x = self._crop_time(self.conv1(x))
        x = self._crop_time(self.conv2(x))
        x = self._crop_time(self.conv3(x))
        x = self._crop_time(self.inception(x))     # (B, 192, T, 1)

        x = x.squeeze(-1).permute(0, 2, 1)        # (B, T, 192)
        x = self.dropout(x)
        hidden, _ = self.lstm(x)                  # (B, T, H)

        if self.attention is not None:
            context, _weights = self.attention(hidden)
        else:
            context = hidden[:, -1, :]
        return self.head(context)

    @torch.no_grad()
    def attention_weights(self, x: Tensor) -> Tensor | None:
        """Attention distribution over the window. Diagnostic / explainability."""
        if self.attention is None:
            return None
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._crop_time(self.conv1(x))
        x = self._crop_time(self.conv2(x))
        x = self._crop_time(self.conv3(x))
        x = self._crop_time(self.inception(x))
        hidden, _ = self.lstm(x.squeeze(-1).permute(0, 2, 1))
        _context, weights = self.attention(hidden)
        return weights


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "AdditiveAttention",
    "ConvBlock",
    "DeepLOB",
    "InceptionModule",
    "count_parameters",
]
