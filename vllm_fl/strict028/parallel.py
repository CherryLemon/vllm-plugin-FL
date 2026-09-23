# SPDX-License-Identifier: Apache-2.0
"""Rank axes for local attention TP, request DP and global expert sharding."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelLayout:
    tensor_size: int
    data_size: int
    global_rank: int

    def __post_init__(self):
        if self.tensor_size < 1 or self.data_size < 1:
            raise ValueError("parallel dimensions must be positive")
        if not 0 <= self.global_rank < self.world_size:
            raise ValueError("global rank is outside the parallel layout")

    @property
    def world_size(self):
        return self.tensor_size * self.data_size

    @property
    def tensor_rank(self):
        return self.global_rank % self.tensor_size

    @property
    def data_rank(self):
        return self.global_rank // self.tensor_size

    def groups(self, axis):
        if axis == "tp":
            return [
                list(range(i * self.tensor_size, (i + 1) * self.tensor_size))
                for i in range(self.data_size)
            ]
        if axis == "dp":
            return [
                list(range(i, self.world_size, self.tensor_size))
                for i in range(self.tensor_size)
            ]
        if axis == "ep":
            return [list(range(self.world_size))]
        raise ValueError(f"unknown parallel axis: {axis}")

    def members(self, axis):
        return next(group for group in self.groups(axis) if self.global_rank in group)
