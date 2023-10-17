"""
Utils for data analysis and processing
"""

from tqdm import tqdm
import numpy as np
import torch

def calc_statistics(data=None, var_key='tas'):
    """
    Calculates min, max, mean, std dev over all scenarios in dataset
    todo:
        Currently calculates the standard deviation over all cmip scenarios by just taking the average across scenarios
        Currently calculates the spatial mean by equally weighting every grid point
    Args:
        data list(n_scenarios * xarray.DataArray([n_time, n_lon, n_lat]): List of xarray DataArray that contain, e.g., global CO2 emissions over time or local surface temperatures over time
        var_key string: Key to variable in data for which the statistics should be calculated 
    """
    min = 1.e10
    max = -1.e10
    mean = 0.
    std  = 0.
    n_scenarios = len(data)
    for scenario_id in range(n_scenarios):
        min = np.min((min, data[scenario_id][var_key].data.min()))
        max = np.max((max, data[scenario_id][var_key].data.max()))
        mean += data[scenario_id][var_key].data.mean()
        std += data[scenario_id][var_key].data.std()

    mean /= float(n_scenarios)
    std /= float(n_scenarios)
    return min, max, mean, std

def create_histogram(data=None, var_key='tas', n_bins=2, normalize=True):
    """ Creates a histogram of all values in data
    Args:
        data list(xarray.DataArray([n_time, n_lon, n_lat])
        var_key string: Key to variable in data for which the histogram should be calculated 
        n_bins int: number of bins in the histogram
        normalize bool: if True will normalize the histogram to get relative frequency
    """
    hist = torch.zeros(n_bins)
    n_scenarios = len(data)
    
    min,max,_,_ = calc_statistics(data, var_key=var_key)

    # Loop over all scenarios in the data
    for scenario_id in tqdm(range(n_scenarios)):
        # Get tas (n_time, n_lon, n_lat) where time is treated as batch dimension
        tas = torch.from_numpy(data[scenario_id][var_key].data)

        # Calculate the histogram of each image using torch.histogram function [^2^][2]
        h, bin_values = torch.histogram(tas, bins=n_bins, range=(min,max))

        # Add the histogram values to the hist array
        hist += h
    # Normalize the hist array to get the relative frequency of each pixel value
    if normalize:
        hist = hist / hist.sum()

    hist = hist.numpy()
    bin_values = bin_values.numpy()
    
    return hist, bin_values
