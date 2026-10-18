from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterator, List

class GroupedBatchSampler:
    # 按 group_col（默认 npz_path）分组，在组内切 batch，减少一个 batch 解压多个 npz 引发的 MemoryError
    def __init__(self, df: pd.DataFrame, batch_size: int, group_col: str = "npz_path",
                 seed: int = 42, drop_last: bool = True, shuffle: bool = True):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.group_col = group_col
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.groups = {k: np.array(list(v), dtype=np.int64) for k, v in df.groupby(group_col).groups.items()}

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self.groups.keys())
        if self.shuffle:
            rng.shuffle(keys)

        batches: List[List[int]] = []
        for k in keys:
            idxs = self.groups[k].copy()
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                chunk = idxs[i:i+self.batch_size].tolist()
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)

        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        for b in batches:
            yield b

    def __len__(self) -> int:
        n = 0
        for idxs in self.groups.values():
            if self.drop_last:
                n += len(idxs) // self.batch_size
            else:
                n += (len(idxs) + self.batch_size - 1) // self.batch_size
        return n
