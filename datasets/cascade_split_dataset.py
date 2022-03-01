import os
import librosa
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', True)
pd.set_option('max_colwidth', 100)

pd.options.display.float_format = '{:,.2f}'.format

class CascadeSplitDataset(Dataset):

    def __init__(self, root_dir, sample_rate=48000):

        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.df_cascade = pd.read_csv('datasets/cascade_metadata.csv')
        self.df_out = self.df_cascade
        self.file_list = []
        
        self.target_dict = {'fuel': 0, 'config': 1, 'cyl': 2,
                 'turbo': 3, 'misfire': 4, 'loc': 5, 'idle': 6,
                 'make': 7 ,'oem': 8, 'disp-cls': 9, 'hp-cls': 10}
        
        self.label_map = {}
        self.unique_vals = {}

        self.get_file_list()
        self.get_labels()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_path = self.file_list[idx]
        
        if '.wav' in file_path:
            signal, sample_rate1 = librosa.load(file_path, sr=self.sample_rate)
        else:
            signal = np.load(self.file_list[idx])
        
        signal = signal[np.newaxis, ...]

        file_parts = self.file_list[idx].split('/')[-1].split('_')
        filename_prefix = file_parts[0]
        filename_prefix = filename_prefix.replace('-','')
        all_targets = self.df_out.loc[self.df_out['filename_prefix'] == filename_prefix]
        
        targets = all_targets[['fuel', 'config', 'cyl', 'turbo', 'misfire', 'loc', 'idle', 'make', 'oem', 'disp-cls', 'hp-cls']].to_numpy()

        if targets.shape[0] != 1:

            print('Broken', self.file_list[idx])
            print(filename_prefix)
            print()

            targets = np.repeat(-100, 11)
            targets = targets[np.newaxis, ...]

        #targets = targets[np.newaxis, ...]

        return signal, targets


    def get_file_list(self):
        
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self.root_dir)):
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                self.file_list.append(file_path)


    def get_labels(self):

        self.df_out['disp-cls'] = self.df_out['disp']
        self.df_out['hp-cls'] = self.df_out['hp']

        # we use tilde (~) since it is last in ASCII table
        self.df_out = self.df_out.replace('Unknown', '~')

        cols = [col for col in self.df_out.columns 
                    if col not in ['order', 'train_val_test', 'filename', 
                                   'filename_prefix', 'length', 'clips', 'disp', 'hp']]

        for col in cols:

            self.df_out[[col]] = self.df_out[[col]].astype(str)

            unique_vals = sorted(pd.unique(self.df_out[col]))

            self.label_map[col] = unique_vals

            self.unique_vals[col] = len(unique_vals)

            col_map = {k: v for v, k in enumerate(unique_vals)}

            col_map['~'] = -100
            
            self.df_out[col] = self.df_out[col].apply(lambda x: col_map[x])

            self.df_out[col] = self.df_out[col].astype('int64')

        self.df_out.drop(columns=['order', 'train_val_test', 'filename', 
                                   'length', 'clips', 'disp', 'hp'], inplace=True)

