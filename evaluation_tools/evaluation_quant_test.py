import numpy as np
import torch
from pipeedge.quantization.clamp_op import _clamp_factor_gelu

PLOT_BINS = 200

def normalized(a, axis=-1, order=1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return (a / np.expand_dims(l2, axis)).flatten()

def mse(ori_x,x, is_norm=False):
    if isinstance(ori_x, torch.Tensor):
        ori_x = ori_x.numpy()
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    mse = (np.square(ori_x-x)).mean()
    if is_norm:
        mse = mse/((ori_x)**2).mean()
    return mse

def Laplace(x, mu, beta, bin_len):
    return np.exp(-np.abs(x-mu)/beta)/(2*beta)

def Gaussian(x, mu, sigma, bin_len):
    return np.exp(-0.5*np.power((x-mu)/sigma, 2))*bin_len/(sigma*2.50662827463)

def estimate_alpha(x):
    return np.power(x, 2).sum()/x.size

def estimate_beta(x):
    return np.abs(x).sum()/x.size

def RML_test(x):
    alpha = estimate_alpha(x)
    beta = estimate_beta(x)
    T = np.log(np.sqrt(alpha)) - np.log(beta) - 0.27421
    return T, (alpha, beta)

def fitting_with_search(norm_hist, theo_xaxi, fitting_parameter, fitting_type = 'laplace', search_steps=100):
    if fitting_type == 'laplace':
        fitting_curve = Laplace(theo_xaxi, 0.0, fitting_parameter, (theo_xaxi.max()-theo_xaxi.min())/PLOT_BINS)
    else:
        fitting_curve = Gaussian(theo_xaxi, 0.0, np.sqrt(fitting_parameter), (theo_xaxi.max()-theo_xaxi.min())/PLOT_BINS)
    norm_fitting_curve = normalized(fitting_curve)
    search_direction = -1.0 if norm_hist.max() > norm_fitting_curve.max() else 1.0

    # old version: to specify search range by hand
    # step_len = search_range * fitting_parameter / search_steps
    # new version: search untill the max of fitting curve reach to the that of real data
    b_real = 1/(2*norm_hist.max())
    b_fitting = 1/(2*norm_fitting_curve.max())
    step_len = np.abs(b_real-b_fitting)*fitting_parameter/(search_steps*b_fitting)
    best_parameter, best_mse = 0.0, 10000.0
    for i in range(search_steps+1):
        temp_parameter = fitting_parameter + i*step_len*search_direction
        if fitting_type == 'laplace':
            fitting_hist = Laplace(theo_xaxi, 0.0, temp_parameter, (theo_xaxi.max()-theo_xaxi.min())/PLOT_BINS)
        else:
            fitting_hist = Gaussian(theo_xaxi, 0.0, np.sqrt(temp_parameter), (theo_xaxi.max()-theo_xaxi.min())/PLOT_BINS)
        fitting_hist = normalized(fitting_hist)
        temp_mse = mse(norm_hist, fitting_hist)
        if best_mse > temp_mse:
            best_mse = temp_mse
            best_parameter = temp_parameter
    return best_parameter


def clamp_with_minimalMSE(input, bit, layer_type='normal'):
    input_numpy = input.numpy()
    hist_x, bins_x = np.histogram(input_numpy, bins = PLOT_BINS)
    theo_xaxi = np.linspace(bins_x.min(), bins_x.max(), PLOT_BINS)
    norm_hist_x = normalized(hist_x)
    rml, (alpha_hat, beta_hat) = RML_test(input_numpy)
    #enforce using of laplace
    rml = 1
    if rml>0:
        #Laplace
        fitting_type = 'laplace'
        fitting_parameter = beta_hat
    else:
        #Gaussain
        fitting_type = 'gaussian'
        fitting_parameter = alpha_hat
    # near-neighbor search
    best_parameter = fitting_with_search(norm_hist_x, theo_xaxi, fitting_parameter, fitting_type=fitting_type)
    optimal_clamp_range = _clamp_factor_gelu(bit) * best_parameter
    # optimal_clamp_range = _clamp_factor_gelu[layer_type][fitting_type][bit] * fitting_parameter
    optimal_clamp_range = torch.tensor(optimal_clamp_range).float()

    # clamp
    result = torch.where(torch.abs(input)<optimal_clamp_range, input, optimal_clamp_range)

    return result, optimal_clamp_range