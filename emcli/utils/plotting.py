import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # plot_tas_annual_local_err_map
import matplotlib.colors as colors # plot_tas_annual_local_err_map

from emcli.dataset.interim_to_processed import calculate_global_weighted_average


def plot_co2_tas_global_over_time(X_train, 
          X_test, 
          Y_train, 
          Y_test, 
          scenarios_train=['ssp126','ssp370','ssp585','hist-GHG','hist-aer'],
          scenarios_test=['ssp245'],
          preds=None):
    """
    Creates three plots to highlight linear relationship between co2 and 
    temperature. First, plots the globally averaged cumulative CO2 
    emissions over time. Second, plots the globally-averaged surface
    temperature anomaly, tas. Third, creates a scatter plot of tas vs. co2. 
    Args:
        X_train list(n_scenarios_train * xarray.Dataset{
              key: xarray.DataArray(n_time)},
              key: xarray.DataArray(n_time, lat, lon)}, 
              ...)
        X_test similar format to X_train
        Y_train similar format to X_train
        Y_test similar format to X_train
        scenarios_train
        scenarios_test
    Returns:
        axs
    """
    fig, axs = plt.subplots(1,3, figsize =(11,4))
    
    # Plot global cumulative CO2 emissions over time
    for idx, scenario in enumerate(scenarios_train):
        axs[0].plot(X_train[idx].time, X_train[idx]['CO2'], label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        axs[0].plot(X_test[idx].time, X_test[idx]['CO2'], color='black', label=scenario)
    axs[0].set_xlabel("Time in years")
    axs[0].set_ylabel("Cumulative anthropogenic CO2 \n emissions since 1850 (GtCO2)")
    axs[0].set_title("Cumulative CO2 emissions")
    axs[0].legend()
    
    # Plot global surface temperature over time
    for idx, scenario in enumerate(scenarios_train):
        axs[1].plot(Y_train[idx].time, calculate_global_weighted_average(Y_train[idx]['tas']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        axs[1].plot(Y_test[idx].time, calculate_global_weighted_average(Y_test[idx]['tas']), color='black', label=scenario)
        if preds is not None:
            axs[1].plot(Y_test[idx].time, preds, color='black', linestyle='--', alpha=0.8, label=scenario+'-pred')
    axs[1].set_xlabel("Time in years")
    axs[1].set_ylabel("Annual Global Surface \n Temperate Anomaly in °C")
    axs[1].set_title("Surface temperature, 'tas'")
    axs[1].legend()

    # Plot global surface temperature over cum. co2 emissions
    for idx, scenario in enumerate(scenarios_train):
        axs[2].plot(X_train[idx]['CO2'], calculate_global_weighted_average(Y_train[idx]['tas']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        axs[2].plot(X_test[idx]['CO2'], calculate_global_weighted_average(Y_test[idx]['tas']), color='black', label=scenario)
        if preds is not None:
            axs[2].plot(X_test[idx]['CO2'], preds, linestyle='--', color='black', label=scenario+'-pred')
    axs[2].set_xlabel("Cum. CO2 Emissions")
    axs[2].set_ylabel("Annual Global Surface \n Temperate Anomaly in °C")
    axs[2].set_title("CO2 vs. tas")
    
    plt.tight_layout()

    return axs

def plot_all_vars_global_avg(X_train, 
            X_test, 
            Y_train, 
            Y_test, 
            scenarios_train=['ssp126','ssp370','ssp585','hist-GHG','hist-aer'],
            scenarios_test=['ssp245']):
    """
    Plots the global averages over time of all input and output variables. 
    Args:
        X_train list(n_scenarios_train * xarray.Dataset{
              key: xarray.DataArray(n_time)},
              key: xarray.DataArray(n_time, lat, lon)}, 
              ...)
        X_test similar format to X_train
        Y_train similar format to X_train
        Y_test similar format to X_train
        scenarios_train
        scenarios_test
    """
    fig, axs = plt.subplots(4,2, figsize =(12,8))
    
    # Plot global cumulative CO2 emissions over time
    ax = axs[0,0]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(X_train[idx].time, X_train[idx]['CO2'], label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(X_test[idx].time, X_test[idx]['CO2'], linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Cumulative anthropogenic CO2 \n emissions since 1850 (GtCO2)")
    ax.set_title("Cumulative CO2 emissions")
    ax.legend()
    
    # Plot global CH4 emissions over time
    ax = axs[0,1]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(X_train[idx].time, X_train[idx]['CH4'], label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(X_test[idx].time, X_test[idx]['CH4'], linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Anthropogenic CH4 \n emissions (GtCH4/year)")
    ax.set_title("CH4 emissions")
    
    # Plot globally-averaged SO2 emissions over time
    ax = axs[1,0]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(X_train[idx].time, calculate_global_weighted_average(X_train[idx]['SO2']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(X_test[idx].time, calculate_global_weighted_average(X_test[idx]['SO2']), linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Anthropogenic SO2 \n emissions (?SO2/yr)")# todo check units (TgSO2/year)")
    ax.set_title("SO2 emissions")
    
    # Plot globally-averaged BC emissions over time
    ax = axs[1,1]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(X_train[idx].time, calculate_global_weighted_average(X_train[idx]['BC']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(X_test[idx].time, calculate_global_weighted_average(X_test[idx]['BC']), linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Anthropogenic BC \n emissions (?BC/yr)")# todo check units (TgBC/year)")
    ax.set_title("BC emissions")
    
    # Plot global surface temperature over time
    ax = axs[2,0]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(Y_train[idx].time, calculate_global_weighted_average(Y_train[idx]['tas']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(Y_test[idx].time, calculate_global_weighted_average(Y_test[idx]['tas']), linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Annual Global Surface \n Temperate Anomaly in °C")
    ax.set_title("Surface temperature")
    
    # Plot global diurnal temperature range over time
    ax = axs[2,1]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(Y_train[idx].time, calculate_global_weighted_average(Y_train[idx]['diurnal_temperature_range']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(Y_test[idx].time, calculate_global_weighted_average(Y_test[idx]['diurnal_temperature_range']), linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Annual Global Diurnal \n Temperature Range in °C")
    ax.set_title("Diurnal Temperature Range")
    
    # Plot global precipitation over time
    ax = axs[3,0]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(Y_train[idx].time, calculate_global_weighted_average(Y_train[idx]['pr']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(Y_test[idx].time, calculate_global_weighted_average(Y_test[idx]['pr']), linestyle='--', color='black', label=scenario)
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Annual Global Precipitation\n (mm/day)")
    ax.set_title("Precipitation")
    
    # Plot global extreme precipitation over time
    ax = axs[3,1]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(Y_train[idx].time, calculate_global_weighted_average(Y_train[idx]['pr90']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(Y_test[idx].time, calculate_global_weighted_average(Y_test[idx]['pr90']), linestyle='--', color='black', label=scenario)
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Annual Global Extreme \n Precipitation (mm/day)")
    ax.set_title("Extreme Precipitation")
    
    plt.tight_layout()

    return axs

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
    projection = ccrs.Robinson(central_longitude=0)
    transform = ccrs.PlateCarree(central_longitude=0)
    fig, axs = plt.subplots(1, 3, figsize=(12, 9), 
        subplot_kw=dict(projection=projection),
        dpi=300)

    # Plot ground-truth surface temperature anomalies
    cnorm = colors.TwoSlopeNorm(vmin=tas_true_t_avg.min(), vcenter=0, vmax=tas_true_t_avg.max()) # center colorbar around zero
    mesh = axs[0].pcolormesh(lon, lat, tas_true_t_avg.data, cmap='coolwarm',norm=cnorm, transform=transform)
    cbar = plt.colorbar(mesh, ax=axs[0], orientation='horizontal', shrink=0.95, pad=0.05)
    cbar.set_label('True surface temperature anomalies, \naveraged over 2080-2100 in °C')
    cbar.ax.set_xscale('linear')
    axs[0].coastlines()

    # Plot predicted surface temperature anomalies
    mesh = axs[1].pcolormesh(lon, lat, tas_pred_t_avg.data, cmap='coolwarm',norm=cnorm, transform=transform)
    cbar = plt.colorbar(mesh, ax=axs[1], orientation='horizontal', shrink=0.95, pad=0.05)
    cbar.set_label('Predicted surf. temp. anom., \navg 2080-2100 in °C')
    cbar.ax.set_xscale('linear')
    axs[1].coastlines()

    # Plot error of pattern scaling
    bounds = np.hstack((np.linspace(err_pattern_scaling.min(), -0.25, 5), np.linspace(0.25, err_pattern_scaling.max(), 5)))
    divnorm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    mesh = axs[2].pcolormesh(lon, lat, err_pattern_scaling.data, cmap='coolwarm', norm=divnorm, transform=transform)
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