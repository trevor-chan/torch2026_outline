"""Dataset views that preserve local sequence ordering."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class OrderedTripletDataset(Dataset):
    """Expose centered, equally spaced triplets from an ordered frame dataset."""

    def __init__(self, dataset: Dataset, frame_stride: int = 1) -> None:
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if len(dataset) <= 2 * frame_stride:
            raise ValueError(
                f"Dataset needs more than {2 * frame_stride} items for stride {frame_stride}"
            )
        self.dataset = dataset
        self.frame_stride = frame_stride

    def __len__(self) -> int:
        return len(self.dataset) - 2 * self.frame_stride

    def __getitem__(self, index: int):
        items = [
            self.dataset[index],
            self.dataset[index + self.frame_stride],
            self.dataset[index + 2 * self.frame_stride],
        ]
        first = items[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            return tuple(torch.stack(values) for values in zip(*items, strict=True))
        return torch.stack(items)
