import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # plot_tas_annual_local_err_map
import matplotlib.colors as colors # plot_tas_annual_local_err_map

def plot_tas_annual_local_err_map(tas_true, tas_pred):
    """
    Plots three maps of surface temperature anomalies (tas).
    This includes the ground-truth tas, predicted tas,
    and error of ground-truth - predicted. 
    Args:
        tas_true xr.DataArray(n_t, n_lat, n_lon): Ground-truth 
            annual mean surface temperature anomalies over the 
            globe in °C
        tas_pred xr.DataArray(n_t, n_lat, n_lon): Predicted tas
    Returns:
        axs: matplotlib axes object
    """
    # Get coordinates
    try:
        lon = tas_true.longitude.data
        lat = tas_true.latitude.data
    except:
        lon = tas_true.lon.data
        lat = tas_true.lat.data

    # Compute temporal average of target and prediction over evaluation timeframe
    tas_true_t_avg = tas_true.sel(time=slice("2081", "2100")).mean(dim="time")
    tas_pred_t_avg = tas_pred.sel(time=slice("2081", "2100")).mean(dim="time")

    # Compute error of prediction minus target
    err_pattern_scaling = tas_pred_t_avg - tas_true_t_avg

    # Create figure with PlateCarree projection
    fig, axs = plt.subplots(1, 3, figsize=(12, 9), 
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        dpi=300)

    # Plot ground-truth surface temperature anomalies
    cnorm = colors.TwoSlopeNorm(vmin=tas_true_t_avg.min(), vcenter=0, vmax=tas_true_t_avg.max()) # center colorbar around zero
    mesh = axs[0].pcolormesh(lon, lat, tas_true_t_avg.data, cmap='coolwarm',norm=cnorm)
    cbar = plt.colorbar(mesh, ax=axs[0], orientation='horizontal', shrink=0.95, pad=0.05)
    cbar.set_label('True surface temperature anomalies, \naveraged over 2080-2100 in °C')
    cbar.ax.set_xscale('linear')
    axs[0].coastlines()

    # Plot predicted surface temperature anomalies
    mesh = axs[1].pcolormesh(lon, lat, tas_pred_t_avg.data, cmap='coolwarm',norm=cnorm)
    cbar = plt.colorbar(mesh, ax=axs[1], orientation='horizontal', shrink=0.95, pad=0.05)
    cbar.set_label('Predicted surf. temp. anom., \navg 2080-2100 in °C')
    cbar.ax.set_xscale('linear')
    axs[1].coastlines()

    # Plot error of pattern scaling
    bounds = np.hstack((np.linspace(err_pattern_scaling.min(), -0.25, 5), np.linspace(0.25, err_pattern_scaling.max(), 5)))
    divnorm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    mesh = axs[2].pcolormesh(lon, lat, err_pattern_scaling.data, cmap='coolwarm', norm=divnorm)
    cbar = plt.colorbar(mesh, ax=axs[2], orientation='horizontal', shrink=0.95, pad=0.05)
    cbar.set_label('Error (pred-target) in surf. temp. \nanom. avg 2080-2100 in °C')
    cbar.ax.set_xscale('linear')
    axs[2].coastlines()
    
    return axs

def plot_histogram(hist, bin_values, 
        xlabel="Annual Mean Local Surface Temp. Anomaly, tas, in K",
        ylabel="Relative frequency",
        title="Histogram of surface temp. in train set",
        ax=None):
    """
    Args:
        hist: occurence frequencies for every section in bin_values
        bin_values: x-values of edges of histogram bins
    """
    # Plot the histogram using matplotlib's bar function
    if ax is None:
        fig, ax = plt.subplots(figsize =(4, 2))
    w = abs(bin_values[1]) - abs(bin_values[0])
    ax.bar(bin_values[:-1], hist, width=w, alpha=0.5, align='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax