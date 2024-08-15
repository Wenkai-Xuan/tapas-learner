import torch
from torch.utils.data import Dataset, DataLoader
from typing import Mapping, List

from tapas_utils import *

class TapasDataset(Dataset):
    def __init__(self,
                 samples
                 ):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        to_ret = {
            "observation": get_raw_features_seq(self.samples[index]).type(torch.float32),
            "src_key_padding_mask": hf_get_src_key_padding_mask(self.samples[index]),
            "targets": hf_get_flat_label(self.samples[index]).type(torch.float32),
            "indexes": index
        }
        return to_ret

    def get_seq(self, index):
        return self.samples[index].sequence
