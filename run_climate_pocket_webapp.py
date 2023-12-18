"""
Webapp to host climate emulator. From anthropogenic greenhouse gases
 to surface temperature anomalies.

Call via 
$ conda activate emcli
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
from emcli.dataset.interim_to_processed import calculate_global_weighted_average_np

def run_model(co2_selected, 
    model_global_co2_to_global_tas, 
    model_global_tas_to_local_tas):
    """
    Runs model to compute a map of predicted surface temperature anomalies
    given the selected co2.
    Args:
        co2_selected float: Selected cumulative co2 emissions
        model_global_co2_to_global_tas: model that maps global co2 to global tas
        model_global_tas_to_local_tas: model that maps global tas to local tas)
    Returns:
        preds numpy.array(n_lat, n_lon): predicted surface temperature anomaly since 1850 
            under the selected cumulative co2 emissions
        preds_global_tas (1): predicted global surface temperature anomaly since 1850
    """
    # Run model to map selected global co2 from input onto global tas
    co2_selected = np.array([[co2_selected,]])# (n_time,in_channels)
    preds_global_tas = model_global_co2_to_global_tas.predict(co2_selected) # (n_time, out_channels)
    preds_global_tas = preds_global_tas[:,0] # 

    # Map global tas onto local tas
    preds = model_global_tas_to_local_tas.predict(preds_global_tas) # (n_time, n_lat, n_lon)
    preds = preds.mean(axis=0) # (n_lat, n_lon)

    return preds

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
    # Define input slider bounds
    cache['co2min'] = X_input[0]['CO2'].min().data.item() # Convert to python 'class' float
    cache['co2max'] = 2 * X_input[0]['CO2'].max().data.item()
    cache['co2today'] = X_input[0]['CO2'].sel(time=slice("2010", "2020")).mean(dim="time").data.item()

    # Get baseline
    tas_baseline = Y_target[0]['tas'].sel(time=slice("1850", "1880")).mean(dim="time")
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
def load_cache():
    # Set False for deployment. If False, loads compressed baseline data. 
    # Set True for development. If True, loads raw baseline data 
    #   (from climatebench) and stores a compressed version that can be 
    #   stored on git and loaded on webapp. Note, that changes in 
    #   store_compressed_data() are not automatically recognized. Workaround
    #   is to add a dummy print statement into load_cache whenever store_
    #   compressed_data is changed.
    RUN_OFFLINE = False

    # Load data of baseline scenario.
    dir_compressed_data = './data/processed/climatebench/webapp/'
    filename_compressed_data = 'cache_climatebench_ssp245_baseline.pkl'
    if RUN_OFFLINE == True:
        st.write('++++++++++++++++++++++++++++++++++++++++++')
        st.write('# WARNING running offline.')
        st.write('Change RUN_OFFLINE before commit to external.')
        st.write('++++++++++++++++++++++++++++++++++++++++++')
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
    
    # Get the bounds of the colormap given the min, max values of the slider
    tasmin = run_model(cache['co2min'],
        cache['model_global_co2_to_global_tas'],
        cache['model_global_tas_to_local_tas'])
    cache['tasmin'] = tasmin.min().item()
    tasmax = run_model(cache['co2max'],
        cache['model_global_co2_to_global_tas'],
        cache['model_global_tas_to_local_tas'])
    tasmax = np.min((tasmax.max(), 10.)) # Cut-off max temp at 10°C because I saw 
        # outliers at 14°C in the data and it skews the colormap
    cache['tasmax'] = tasmax.item() # convert to class 'float'

    # Create figure with cartopy projection
    cache['fig'], cache['axs'] = plt.subplots(figsize=(6,4), 
        subplot_kw=dict(projection=ccrs.Robinson()),
        dpi=200)
    return cache

if __name__ == "__main__":
    # Main function, which is run for every change on the webpage

    st.write("""
    # BC3 Climate Pocket: CO2 -> Temp.
    This app is NOT ready for deployment and just a demo of ML-based climate emulators.
    """)

    # Get the data from the cache
    cache = load_cache()

    st.write(f"""
    In 2019, the global temperature increase from 1850 was approx. 1.07°C and we had already emitted 2390 gigatons of carbondioxide (GtCO2) [Src: IPCC AR6].
    ##### Set the cumulative CO2 emissions since 1850 (in GtCO2) to
    """)

    # Create a slider to retrieve input selection, e.g., global co2
    co2_selected = st.slider('cumulative CO2 emissions (GtCO2)', 
        min_value=cache['co2min'], 
        max_value=cache['co2max'], 
        value=cache['co2today'],
        label_visibility='collapsed')

    preds = run_model(co2_selected,
        cache['model_global_co2_to_global_tas'],
        cache['model_global_tas_to_local_tas'])

    # Compute tas difference to baseline
    diff = preds - cache['tas_baseline']

    # Calculate global avg of predicted map
    preds_global_tas = calculate_global_weighted_average_np(preds[None,...], cache['lat'])

    # Plot ground-truth surface temperature anomalies using cartopy
    st.write(f"""
    ##### and the model predicts a temperature increase until 2100 of: {preds_global_tas[0]:.2f}°C:
    """)

    # Create discrete colorbar, centered around 0
    color_boundaries_neg = np.round(np.linspace(cache['tasmin'], -0.20, 7)*5.)/5. # , e.g., *4/4 rounds to closest 0.25
    color_boundaries_pos = np.round(np.linspace(0.5, cache['tasmax'], 7)*2.)/2. # *2/2 rounds to closest 0.5 
    color_boundaries = np.hstack((color_boundaries_neg, color_boundaries_pos))
    # print('color_boundaries', color_boundaries, color_boundaries_neg.shape, color_boundaries_pos.shape)
    cnorm = colors.BoundaryNorm(boundaries=color_boundaries, ncolors=256)

    # Plot values
    mesh = cache['axs'].pcolormesh(cache['lon'], cache['lat'], diff, cmap='coolwarm', norm=cnorm, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(mesh, ax=cache['axs'], orientation='horizontal', shrink=0.95, pad=0.05, spacing='uniform', extend='max')
    # cbar.set_label('Difference (2100 - today) global temperature increase in °C')
    cbar.ax.set_xscale('linear')
    cache['axs'].coastlines()

    # En-roads prefers no caption in image.

    # Display the map on the webdemo using streamlit
    st.pyplot(cache['fig'])

    st.write(f"""
    - About:
        - Developer: Björn Lütjens, MIT EAPS, blutjens.github.io
        - Funded by BC3 Bringing Computation to the Climate Challenge Grant at MIT
        - There is a tutorial to this demo at: https://github.com/blutjens/climate-emulator-tutorial/
        - This demo is NOT displaying true climate data. We refer to the IPCC report for the most up-to-date climate data.
    - Data description:
        - The input is the cumulative CO2 emissions (GtCO2) since 1850.
        - The plot shows the surface temperature anomalies with respec to a baseline temperature. The baseline temperature is the average over 1850-1880. 
        - The model is made of two linear models. The first model maps global co2 to global tas. The second model is a linear pattern scaling model that maps global tas onto local tas. The model was trained on historical data until 2014 and the ensemble average of 3 NorESM2 model runs of {cache['baseline_scenario']} scenario for 2015 and onwards. The model was train on the scenarios ssp126, ssp370, ssp585, hist-GHG, hist-aer.
        - The global averages use cosine weights to approximate the grid cell area.
    - Known errors:
        - The underlying data from only three NorESM2 realizations contains too much internal variability. A better model needs to be fit on an average of CMIP6 models and more ensemble members.
        - The predictive model is linear and as such imperfect. For example, in the ssp245 test scenario the model underpredicts the cooling in the North Atlantic Warming hole. The model also slightly overpredicts warming in the Russian Arctic.
        - ProjError: transform error: Invalid coordinate: This error occurs when the slider is moved repeatidly before the calculation is finished. No known fix.
        - Doesn't work on safari or edge.
    """)