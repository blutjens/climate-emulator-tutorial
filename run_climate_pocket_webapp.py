"""
Webapp to host climate emulator. From anthropogenic greenhouse gases
 to surface temperature anomalies.

Call via 
$ streamlit run run_climate_pocket_webapp.py
"""
import pickle # store and load compressed data
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # plot_tas_annual_local_err_map
import matplotlib.colors as colors # plot_tas_annual_local_err_map

from emcli.dataset.climatebench import load_climatebench_data
import emcli.models.pattern_scaling.model as ps
from emcli.dataset.interim_to_processed import calculate_global_weighted_average

# If True, loads raw climatebench data and stores a compressed
#  version that can be stored on git and loaded on webapp
# If False, load processed compressed data. Set False for deployment.
RUN_OFFLINE = False 

st.write("""
# BC3 Climate Pocket
Mapping CO2 to temperature
""")

# Store compressed version of data that can be loaded on external app
def store_compressed_data(dir_compressed_data, filename_compressed_data):
    cache = dict()

    # Load climatebench data to get dimensions and baseline
    cache['baseline_scenario'] = 'ssp245'
    len_historical = 165
    data_path = './data/raw/climatebench/' # use 'climate-emulator-tutorial/' on colab and '../' on local machine
    X_input, Y_target = load_climatebench_data(
    simus=[cache['baseline_scenario']], len_historical=len_historical, 
    data_path=data_path)

    # Get lat, lon for map plotting
    cache['lat'] = Y_target[0]['tas'].latitude.data
    cache['lon'] = Y_target[0]['tas'].longitude.data
    # Get colormap bounds
    cache['tasmin'] = Y_target[0]['tas'].min().data.item() # Convert to python 'class' float
    cache['tasmax'] = Y_target[0]['tas'].max().data.item()
    # Define slider bounds
    cache['co2min'] = X_input[0]['CO2'].min().data.item() # Convert to python 'class' float
    cache['co2max'] = 2 * X_input[0]['CO2'].max().data.item()

    # Get baseline
    tas_baseline = Y_target[0]['tas'].sel(time=slice("2081", "2100")).mean(dim="time")
    cache['tas_baseline'] = tas_baseline.data
    cache['tas_global_baseline'] = calculate_global_weighted_average(tas_baseline)

    # Store data
    Path(dir_compressed_data).mkdir(parents=True, exist_ok=True)
    path = dir_compressed_data + filename_compressed_data
    with open(path,'wb') as f:
        pickle.dump(cache,f)

    return cache

# Load data into cache
@st.cache_data
def load_data():
    # Load data of baseline scenario.
    dir_compressed_data = './data/processed/climatebench/webapp/'
    filename_compressed_data = 'cache_climatebench_ssp245_baseline.pkl'
    if RUN_OFFLINE:
        cache = store_compressed_data(
            dir_compressed_data,
            filename_compressed_data)
    else:
        # Load data from file
        path = dir_compressed_data + filename_compressed_data
        with open(path, 'rb') as f:
            cache = pickle.load(f)

    # Load model global co2 to global tas
    path_model = Path('./runs/pattern_scaling/default/models/global_co2_to_global_tas.pkl')
    cache['model_global_co2_to_global_tas'] = ps.load(dir=str(path_model.parent)+'/', 
                                             filename=str(path_model.name))

    # Load model global tas to local tas
    path_model = Path('./runs/pattern_scaling/default/models/global_tas_to_local_tas.pkl')
    cache['model_global_tas_to_local_tas'] = ps.load(dir=str(path_model.parent)+'/', filename=str(path_model.name))
    
    # Create figure with PlateCarree projection
    cache['fig'], cache['axs'] = plt.subplots(figsize=(6,4), 
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        dpi=200)

    return cache

# Get the data from the cache
cache = load_data()

# Create a slider retrieve input selection, e.g., global co2
co2_selected = st.slider('### Cumulative CO2 emissions (GtCO2)', cache['co2min'], cache['co2max'], cache['co2max'])

# Map selected global co2 from input onto global tas
co2_selected = np.array([[co2_selected,]])# (n_time,in_channels)
preds_global_tas = cache['model_global_co2_to_global_tas'].predict(co2_selected) # (n_time, out_channels)
preds_global_tas = preds_global_tas[:,0] # 

# Map global tas onto local tas
preds = cache['model_global_tas_to_local_tas'].predict(preds_global_tas) # (n_time, n_lat, n_lon)
preds = preds.mean(axis=0) # (n_lat, n_lon)

# Compute difference to baseline
diff = preds - cache['tas_baseline']

# Plot ground-truth surface temperature anomalies using cartopy
st.write(f"""
#### Global temperature increase:
#### Predicted: {preds_global_tas[0]:.2f}°C vs. Baseline:  {cache['tas_global_baseline']:.2f}°C 
""")
cnorm = colors.TwoSlopeNorm(vmin=cache['tasmin'], vcenter=0, vmax=cache['tasmax']) # center colorbar around zero
mesh = cache['axs'].pcolormesh(cache['lon'], cache['lat'], diff, cmap='coolwarm',norm=cnorm)
cbar = plt.colorbar(mesh, ax=cache['axs'], orientation='horizontal', shrink=0.95, pad=0.05)
cbar.set_label('Difference (predicted - baseline) surface temperature \nanomalies, averaged over 2080-2100 in °C')
cbar.ax.set_xscale('linear')
cache['axs'].coastlines()

# Display the map on the webdemo using streamlit
st.pyplot(cache['fig'])

st.write(f"""
- The input  is the cumulative CO2 emissions (GtCO2). Averaged over 2080-2100.
- The baseline are surface temperature anomalies wrt. 1850. Averaged over 2080-2100. Averaged globally using a cosine weights. Averaged over 3 ensemble members. Uses the {cache['baseline_scenario']} scenario and NorESM2 model.
""")
