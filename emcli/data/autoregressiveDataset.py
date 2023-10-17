#@title code: autoregressive dataset

import numpy as np
import torch
from torch.utils.data import Dataset

class AutoregressiveDataset(Dataset):
    def __init__(self, X, Y, t_pushforward=3, split='train',
        dtype=np.float32):
        '''
        todo:
          - load data lazy
        Args:
          X  list(xarray.core.dataset.Dataset{
              'CO2': xarray.core.dataarray.DataArray(n_time)} * n_scenarios):
              globally averaged co2 over time of multiple scenarios
          Y list(xarray.core.dataset.Dataset{
              'tas': xarray.core.dataarray.DataArray(n_time, n_lat, n_lon)} *
              n_scenarios):
              global surface temperature field of multiple scenarios
          t_pushforward int: Number of time steps that target is ahead of input.
              E.g., [t=0]->[t=3] for t_pushforward = 3
          split str
          dtype np.dtype: Data type for in-/output and model
        Returns
          data list(input_state: np.array(t_pushforward, in_channels, n_lat, n_lon),
              np.array(out_channels, n_lat, n_lon)):
              Concatenated list of individual data samples
        '''
        # index data into list
        self.data = []
        self.dtype = dtype

        n_lat, n_lon = Y[0]['tas'].shape[1:] # e.g., 96, 144
        n_scenarios = len(X)
        ## Concatenate data from all scenarios
        for s in range(n_scenarios):
          n_tsteps = X[s]['CO2'].shape[0]
          for t in range(n_tsteps):
            if (t + t_pushforward) < n_tsteps:
              # Return all time steps from first to t_pushforward as input
              co2 = X[s]['CO2'][t:t+t_pushforward]
              co2 = co2.to_numpy().astype(self.dtype)
              # Broadcast mean co2 to global field
              co2 = co2[:,None,None] * np.ones((t_pushforward, n_lat, n_lon), dtype=self.dtype)
              tas = Y[s]['tas'][t:t+t_pushforward].to_numpy().astype(self.dtype)
              input = np.concatenate((tas[:,None,...], co2[:,None,...]), axis=1)
              # Return t_pushforward'th time step as target: tas_field. Model
              # will forecast t_pushforward steps autoregressively before loss
              # is applied
              target = Y[s]['tas'][t+t_pushforward,...]
              target = target.to_numpy().astype(self.dtype)
              target = target[None,...]

              self.data.append((input, target))

        self.split = split
        # todo: self.transform = Compose([
        #     Resize((cfg['image_size'])),
        #     ToTensor()
        # ])

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)


    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            inputs torch.Tensor(
                in_channels, n_lat, n_lon):
            targets torch.Tensor(
                out_channels, n_lat, n_lon):
        '''
        input, target = self.data[idx]

        # todo: load image lazily here.
        # todo: apply data augmentation transforms here.

        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        return input, target