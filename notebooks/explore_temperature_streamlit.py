"""
Webapp to host climate emulator. From anthropogenic greenhouse gases
 to surface temperature anomalies.

Call via 
$ streamlit run notebooks/explore_temperature_streamlit.py
"""
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # plot_tas_annual_local_err_map
import matplotlib.colors as colors # plot_tas_annual_local_err_map

from emcli.dataset.climatebench import load_climatebench_data

st.write("""
# BC3 Climate Pocket
Exploring temperature over time
""")


# Load data into cache
@st.cache_data
def load_data():
    # Load data
    scenarios = ['ssp370']
    len_historical = 165
    data_path = './data/raw/climatebench/' # use 'climate-emulator-tutorial/' on colab and '../' on local machine

    _, Y_target = load_climatebench_data(
    simus=scenarios, len_historical=len_historical, 
    data_path=data_path)
    scenario_id = 0
    data = Y_target[scenario_id]['tas']

    # Create figure with PlateCarree projection
    fig, axs = plt.subplots(figsize=(4,3), 
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        dpi=50)

    return data, fig, axs

# Get the data from the cache
data,fig,axs = load_data()

# Get dimensions
lon = data.longitude.data
lat = data.latitude.data
# Create a slider to select the time index
tmin = data.time.data.min()
tmax = data.time.data.max()
time_select = st.slider('Select the time index', tmin, tmax, tmax)

# Extract the surface temperature at the selected time
data_slice = data.sel(time=time_select, method='nearest')
data_slice = data_slice.drop('time')

# Plot ground-truth surface temperature anomalies using cartopy
cnorm = colors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max()) # center colorbar around zero
mesh = axs.pcolormesh(lon, lat, data_slice.data, cmap='coolwarm',norm=cnorm)
cbar = plt.colorbar(mesh, ax=axs, orientation='horizontal', shrink=0.95, pad=0.05)
cbar.set_label('True surface temperature anomalies, \naveraged over 2080-2100 in °C')
cbar.ax.set_xscale('linear')
axs.coastlines()

# Display the map on the webdemo using streamlit
st.pyplot(fig)