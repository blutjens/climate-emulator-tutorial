import numpy as np
from xskillscore import rmse
from emcli.dataset.interim_to_processed import calculate_global_weighted_average

def calculate_nrmse(Y_true, Y_pred):
    """
    Args:
        Y_true xr.DataArray(lat, lon, time)
        Y_pred xr.DataArray(lat, lon, time)
    """
    lats = Y_true.latitude
    lons = Y_true.longitude

    # Take temporal average over eval period
    Y_true_mean = Y_true.sel(time=slice(2080,None)).mean('time') 
    Y_pred_mean = Y_pred.sel(time=slice(2080,None)).mean('time')

    # Take spatial average and rmse
    weights = np.cos(np.deg2rad(lats)).expand_dims(longitude=len(lons)).assign_coords(longitude=lons)
    Y_rmse = rmse(Y_true_mean, Y_pred_mean, weights=weights).data

    # Normalize
    Y_true_abs = np.abs(calculate_global_weighted_average(Y_true_mean)).data
    Y_nrmse = Y_rmse / Y_true_abs

    # np.abs(global_mean(Y_true_mean)data)
    return Y_nrmse
