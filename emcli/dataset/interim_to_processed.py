"""
Data processing function for predict climate variables, such as
temperature and pressure, from greenhouse forcings and past 
climate variables.

The interim dataset that is used as input has the following format:
    dict(key: np.array(n_scenarios, n_tsteps, n_lat, n_lon))

The processed dataset that is the output will have the following format:

inputs - batch, c, lat (height), lon (width)
    e.g., global forcings             + local forcings at t 
    e.g., global forcings over t-10:t + local forcings at t
    e.g., global forcings             + local forcings at t + local state at t-1
    e.g., global forcings             + local forcings at t + local state at t-3:t-1
    e.g., global state at t
outputs - batch, c, lat, lon
    e.g., global state at t
    e.g., local state at t
    e.g., local state at t+3
"""
from pathlib import Path
import numpy as np

def calculate_global_weighted_average_np(np_data, latitude):
  """ Calculates global weighted average approximating
  grid cell area with cos(latitude). Assumes input is
  numpy array.
  Args:
    np_data np.array(time, latitude, longitude)
    latitude np.array(n_latitude): latitude values in deg
  Returns:
    global_avg np.array(time)
  """
  # weighted average across latitude
  weights = np.cos(np.deg2rad(latitude))
  global_avg = np.average(np_data, axis=1, weights=weights)
  # average across longitude
  global_avg = np.mean(global_avg, axis=-1)

  return global_avg

def calculate_global_weighted_average(xr_data):
  """ Calculates global weighted average approximating
  grid cell area with cos(latitude).
  Args:
    xr_data xarray.DataArray(time, latitude, longitude)
  Returns:
    global_avg xarray.DataArray(time)
  """
  if 'latitude' in xr_data.dims:
    weights = np.cos(np.deg2rad(xr_data.latitude))
    global_avg = xr_data.weighted(
      weights).mean(dim=("latitude", "longitude"))
  elif 'lat' in xr_data.dims:
    weights = np.cos(np.deg2rad(xr_data.lat))
    global_avg = xr_data.weighted(
      weights).mean(dim=("lat", "lon"))
  return global_avg

def keys_exist_in_list_of_datasets(xr_list_of_datasets, required_keys=['CO2']):
  """ Checks if all requested data variables are in given dataset
  Args:
    xr_list_of_datasets list(n_scenarios * xarray.Dataset{
      key: xarray.DataArray()})
  """
  for xr_dataset in xr_list_of_datasets:
    # Check all input variables, e.g., 'CO2'
    for key in required_keys:
      if key not in xr_dataset.data_vars: 
        raise AttributeError(f'argument {key} not in dataset keys')
  return True

def average_globally_and_stack_in_time_and_channel(xr_list_of_datasets, keys):
    """
    Iterates through list of xarray datasets. Converts all (time, lat, lon)
    variables into (time) variables by taking global average. Only grabs
    variables mentioned in keys. Converts xarray to numpy. Returns 
    data that is stacked along time and channel dimension. 
    Args:
      xr_list_of_datasets  list(n_elements * xarray.Dataset{
      key: xarray.DataArray(n_time)},
      key: xarray.DataArray(n_time, lat, lon)}): list of datasets
    Returns
      stacked_data np.array(time, channels, lat, lon)
    """
    stacked_data = []
    # Add data from every dataset in the dataArray, e.g., every scenario
    for xr_dataset in xr_list_of_datasets:
      variables_np = [] 
      # Add data from every required key, e.g., 'CO2'
      for key in keys:
        variable_xr = xr_dataset[key].copy()
        # Take global average of locally-resolved variables
        if 'latitude' in variable_xr.dims or 'lat' in variable_xr.dims:
          variable_xr = calculate_global_weighted_average(variable_xr)
        # Convert to numpy
        variable_np = variable_xr.data # (n_time,)
        # Inflate dimensions onto all axes
        variable_np = variable_np[...,None,None,None] # (n_time, 1, lat, lon)
        variables_np.append(variable_np) # [n_channels * (n_time, 1, lat, lon)]
      # stack all variables_np along channel dimension
      variables_np = np.concatenate(variables_np, axis=1) # (n_time, n_channels, lat, lon)
      stacked_data.append(variables_np)
    # stack all data points in time dimensions
    stacked_data = np.concatenate(stacked_data, axis=0)
    return stacked_data

def interim_to_global_global(X_global_local, Y_global_local,
  input_keys=['CO2'],
  target_keys=['tas'],
  save_dir='data/processed/global_global/'):
  """
  Reshapes the interim dataset to map global averages at t to 
  global averages at t, e.g., cum. co2 at t -> globally 
  averaged temperature at t.

  Args:
    X_global_local  list(n_scenarios * xarray.Dataset{
      'CO2': xarray.DataArray(n_time), # globally averaged forcings over time, e.g., co2
      'BC':  xarray.DataArray(n_time, latitude, longitude)}) # locally-resolved forcings, e.g., bc
    Y_global_local list(n_scenarios * xarray.Dataset{
      'tas': xarray.DataArray(n_time, n_lat, n_lon)}):
      locally-resolved surface temperature field of multiple scenarios
    input_keys list(str): List of data keys that are used as input
    target_keys list(str): list of data keys that are used outputs
    save_dir str: directory path for storing files
  Returns:
    input : np.array(n_samples, in_channels, 1, 1) # input
        global forcings at t, e.g., cum. co2, ch4, and 
        globally averaged bc and so2 at t. Saved at save_dir/input.py
    target: np.array(n_samples, out_channels, 1, 1) # target global state at t, 
        e.g., globally-average temperature. Saved at save_dir/target.npy
	"""
  n_scenarios = len(X_global_local)
  
  assert keys_exist_in_list_of_datasets(X_global_local, input_keys)
  assert keys_exist_in_list_of_datasets(Y_global_local, target_keys)

  # todo: discard duplicate data from historical sequences, similar to 
  # for scenario in scenarios:
  #   X_global_local[scenario] = X_global_local[scenario].drop(time == historical) 
  #   Y_global_local[scenario] = X_global_local[scenario].drop(time == historical) 

  input = average_globally_and_stack_in_time_and_channel(X_global_local, input_keys)
  target = average_globally_and_stack_in_time_and_channel(Y_global_local, target_keys)

  # Check that in/outputs have same number of tsteps
  assert input.shape[0] == target.shape[0], 'Error: inputs and target have different number of time steps'

  # Store input and target files as .npy
  print('Saving processed data at: ', save_dir)
  Path(save_dir).mkdir(parents=True, exist_ok=True)
  np.save(save_dir + 'input.npy', input)
  np.save(save_dir + 'target.npy', target)

  # Deprecated: Create list of time steps. Every time step contains one data dictionary.
  # data: list(
  #    'input': np.array(n_samples, in_channels, 1, 1) 
  #    'target': np.array(n_samples, out_channels, 1, 1) 
  # data = []
  # n_tsteps = input.shape[0]
  # for t in range(n_tsteps):
  #   data.append({'input': input[t,...], 'target': target[t,...]})

  return input,target

def interim_to_pushforward(X_global, Y_local, 
	n_t_pushforward=3,
  dtype=np.float32):
  '''
  Args:
    X_global  list(n_scenarios * xarray.Dataset{
      'CO2': xarray.DataArray(n_time)}):
      globally averaged co2 over time of multiple scenarios
    Y_local list(n_scenarios * xarray.Dataset{
      'tas': xarray.DataArray(n_time, n_lat, n_lon)}):
      locally-resolved surface temperature field of multiple scenarios
    n_t_pushforward int: Number of time steps that target is ahead of input.
      E.g., [t=0]->[t=3] for n_t_pushforward = 3
    dtype np.dtype: Data type for in-/output and model
  Returns
    data list(
    	np.array(n_t_pushforward, in_channels, n_lat, n_lon), # Inputs
      np.array(out_channels, n_lat, n_lon)): # Targets
        Concatenated list of individual data samples
  '''
  # index data into list
  self.data = []
  self.dtype = dtype

  n_lat, n_lon = Y_local[0]['tas'].shape[1:] # e.g., 96, 144
  n_scenarios = len(X_global)

  ## Concatenate data from all scenarios along time axis
  for s in range(n_scenarios):
    n_tsteps = X_global[s]['CO2'].shape[0]
  # Iterate over every n_t_pushforward snippet with overlap 
    for t in range(n_tsteps): 
      if (t + n_t_pushforward) < n_tsteps:
        # Add inputs -- global forcings
        # The input contains forcings along the full 
        # snippet (from t to t+n_t_pushforward) such that the 
        # autoregressive model can take them as input.
        co2 = X_global[s]['CO2'][t:t+n_t_pushforward]
        co2 = co2.to_numpy().astype(self.dtype)
        # Broadcast mean co2 to global field
        co2 = co2[:,None,None] * np.ones((n_t_pushforward, n_lat, n_lon), dtype=self.dtype)
        
        # Add inputs -- autoregressive state
        tas = Y_local[s]['tas'][t:t+n_t_pushforward]
        tas = tas.to_numpy().astype(self.dtype)
        input = np.concatenate((tas[:,None,...], co2[:,None,...]), axis=1)

        # Add targets
        # Return n_t_pushforward'th time step as target: tas_field. Model
        # will forecast n_t_pushforward steps autoregressively before loss
        # is applied
        target = Y_local[s]['tas'][t+n_t_pushforward,...]
        target = target.to_numpy().astype(self.dtype)
        target = target[None,...]

        self.data.append((input, target))

  # todo: store data
  np.store(self.data)