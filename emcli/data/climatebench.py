import xarray as xr

def load_climatebench_train_data(
    simus = ['ssp126','ssp370','ssp585','hist-GHG','hist-aer'],
    len_historical = 165,
    data_path = 'data/'):
    """ Load ClimateBench Training Data
    Loads the scenario passed in <simus> from data_path into memory. 
    The data from scenarios, e.g., ssp126 is concatenated with historical 
    data.
    Returns:
        X_train list(xarray.Dataset[len_historical (+ len_future), lon, lat]) 
            with CO2, SO2, CH4, BC in float64
        Y_train list(xarray.Dataset[len_historical (+ len_future), lon, lat]) 
            with diurnal_temperature_range & tas in float32 and pr & pr90 in float64
    """
    # to-do: load only co2 and temp
    #        make possible to load rest later on
    # do lazy loading to highlight benefit of netCDF

    X_train = []
    Y_train = []

    for i, simu in enumerate(simus):

        input_name = 'inputs_' + simu + '.nc'
        output_name = 'outputs_' + simu + '.nc'

        if 'hist' in simu:
            # load inputs
            input_xr = xr.open_dataset(data_path + input_name)

            # load outputs
            output_xr = xr.open_dataset(data_path + output_name).mean(dim='member')
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                        "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude',
                                                                                'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])

        # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
        else:
            # load inputs
            input_xr = xr.open_mfdataset([data_path + 'inputs_historical.nc',
                                        data_path + input_name]).compute()

            # load outputs
            output_xr = xr.concat([xr.open_dataset(data_path + 'outputs_historical.nc').mean(dim='member'),
                                xr.open_dataset(data_path + output_name).mean(dim='member')],
                                dim='time').compute()
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                        "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude',
                                                                                'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])

        print(input_xr.dims, simu)

        # Append to list
        X_train.append(input_xr)
        Y_train.append(output_xr)

    return X_train, Y_train
