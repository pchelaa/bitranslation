# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class EvenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token):
        super().__init__(dataset)
        self.token = token
        self._sizes = np.where(dataset.sizes % 4 == 0, dataset.sizes, dataset.sizes + (4 - (dataset.sizes % 4)))

    def __getitem__(self, idx):
        item = self.dataset[idx]
        print(self.dataset[idx])
        while len(item) % 4 != 0:
            item = torch.cat([item.new([self.token]), item])
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        if n % 4 == 0:
            return n

        return n + 4 - (n % 4)

    def size(self, index):
        n = self.dataset.size(index)
        if n % 4 == 0:
            return n

        return n + 4 - (n % 4)
