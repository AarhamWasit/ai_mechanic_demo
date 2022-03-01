from torchvision import datasets, transforms
from base import BaseDataLoader

from datasets import CascadeSplitDataset

class CascadeSplitDataLoader(BaseDataLoader):
    
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir
        self.dataset = CascadeSplitDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
    def get_label_map(self):
        return self.dataset.label_map

    def get_unique_vals(self):
        return self.dataset.unique_vals

    def get_target_dict(self):
        return self.dataset.target_dict
    