import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle # saving model
from pathlib import Path

class PatternScaling(object):
    """
    Does pattern scaling. Here we fit one linear model per 
    grid point. The linear model maps a global variable, 
    e.g., temperature, to the grid point's local value. 
    This model captures temporal patterns in each grid cell.
    The model is local, i.e., it will be independent of 
    neighboring grid points. The model is linear in time, 
    i.e., it assumes no non-linearly amplifying feedbacks 
    between the global and local variable.
    """
    def __init__(self, deg=1):
        """
        Args:
            deg int: degree of polynomial fit. Default is 1 
                for linear fit.
        """
        self.deg = deg
        self.coeffs = None

    def train(self, in_global, out_local):
        """
        Args:
            in_global np.array((n_t,)): The model input is a 
                global variable, e.g., annual global mean surface 
                temperature anomalies of in °C
            out_local np.array((n_t,n_lat,n_lon)): The model 
                output is a locally-resolved variable. E.g., annual 
                mean surface temperature anomalies at every lat,lon
                in °C
        Sets:
            coeffs np.array((deg+1, n_lat, n_lon))
        """
        n_t, n_lat, n_lon = out_local.shape
        
        # Preprocess data, by flattening in space
        out_local = out_local.reshape(n_t,-1) # (n_t, n_lat*n_lon)

        # Fit linear regression coefficients to every grid point
        self.coeffs = np.polyfit(in_global, out_local, deg=self.deg) # (2, n_lat*n_lon)

        # Reshape coefficients onto locally-resolved grid
        self.coeffs = self.coeffs.reshape(-1, n_lat, n_lon) # (2, n_lat, n_lon)

    def predict(self, in_global):
        """
        Args:
            in_global np.array((n_t,))
        Returns:
            preds np.array((n_t, n_lat, n_lon))
        """
        n_lat = self.coeffs.shape[1]
        n_lon = self.coeffs.shape[2]

        # Predict by applying pattern scaling coefficients on locally-resolved grid
        in_global = np.tile(in_global[:,np.newaxis, np.newaxis], reps=(1,n_lat,n_lon)) # repeat onto local grid to get shape (n_t, n_lat, n_lon)
        preds = np.polyval(self.coeffs, in_global) # (n_t, n_lat, n_lon)

        return preds
    
def save(model, 
         dir='./runs/pattern_scaling/default/models/', 
         filename='model.pkl'):
    """
    Saves model at dir
    """
    Path(dir).mkdir(parents=True, exist_ok=True)
    path = dir + filename
    with open(path,'wb') as f:
        pickle.dump(model,f)

def load(dir='./runs/pattern_scaling/default/models/', 
        filename='model.pkl'):
    """
    Loads model from file.
    """
    path = dir + filename
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def fit_linear_regression_global_global(
    data_dir='data/interim/global_global/train/', 
    plot=False):
    """
    Fits a linear regression model from globally-averaged GHG
    forcings at t to global average variables at t. E.g.,
    global co2 at t -> global tas at t. Accepts multiple in-/
    and output channels.

    Args:
        data_dir str: Path to directory with input and target data. See 
            emcli.dataset.interim_to_processed.interim_to_global_global for
            data format.
        plot bool: if True, plots training data and fit.
    Returns:
        model sklearn.LinearRegression: linear regression model fit to data.
    """
    # Load processed data
    input = np.load(data_dir + 'input.npy') # (n_samples, in_channels, lat, lon)
    target = np.load(data_dir + 'target.npy')  # (n_samples, out_channels, lat, lon)
    
    # Process data for linear regression
    input_lr = np.reshape(input, input.shape[:2]) # (n_samples, in_channels)
    target_lr = np.reshape(target, target.shape[:2]) # (n_samples, out_channels)

    # Initialize and fit a LinearRegression model
    model = LinearRegression()
    model.fit(input_lr, target_lr)

    if plot:
        fig, axs = plt.subplots(1,1, figsize =(4,4))
        axs.plot(input_lr, target_lr, '.', label='train')
        axs.plot(input_lr, model.predict(input_lr), color='black', label='pred')
        axs.set_xlabel("input")
        axs.set_ylabel("target")
        axs.set_title("Linear Regression on train data")
        axs.legend()

    return model

def predict_linear_regression_global_global(model, 
    data_dir='data/interim/global_global/test/',
    plot=False):
    """
    Predict linear regression model on global-averages variables
    to global average variables.
    Args:
        model sklearn.LinearRegression model
        data_dir str: Path to directory with input and target data. See 
            emcli.dataset.interim_to_processed.interim_to_global_global for
            data format.
        plot bool: if True, plots test data and fit.
    Returns
        preds np.array(n_samples,out_channels): predictions
    """
    # Load processed data
    input_test = np.load(data_dir + 'input.npy') # (n_samples, in_channels, lat, lon)
    target_test = np.load(data_dir + 'target.npy')  # (n_samples, out_channels, lat, lon)

    # Process data for linear regression
    input_test_lr = np.reshape(input_test, input_test.shape[:2]) # (n_samples, in_channels)
    target_test_lr = np.reshape(target_test, target_test.shape[:2]) # (n_samples, out_channels)
    
    # Predict linear regression model on test data
    preds = model.predict(input_test_lr)
    
    if plot:
        fig, axs = plt.subplots(1,1, figsize =(4,4))
        axs.plot(input_test_lr, target_test_lr, '.', label='true')
        axs.plot(input_test_lr, preds, color='black', label='pred')
        axs.set_xlabel("input")
        axs.set_ylabel("target")
        axs.set_title("Linear regression on test data")
        axs.legend()

    return preds
