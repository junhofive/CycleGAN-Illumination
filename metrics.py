import math
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage.color import rgb2lab
from colour import delta_E
from pytorch_msssim import ms_ssim


# Function to calculate the mean squared error
def mean_squared_error(true, pred):
    return ((true - pred) ** 2).mean().item()


# Function to calculate the PSNR
def psnr(mse, max_val=1.0):
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def LPIPS_val(loss_fn, output, target):
    dist = loss_fn.forward(output, target)
    return dist.item()


def ciede2000(output, target):
    # Convert RGB images to Lab color space
    output_image_lab = rgb2lab(output)
    target_image_lab = rgb2lab(target)

    # Compute CIEDE2000 color difference for each pixel
    delta_e = delta_E(output_image_lab, target_image_lab, method='cie2000')

    # Compute mean CIEDE2000 color difference for the entire image
    mean_delta_e = np.mean(delta_e)

    return mean_delta_e


def ms_ssim_metric(output, target):
    # Compute MS-SSIM
    ms_ssim_value = ms_ssim(output, target, data_range=1, size_average=True)

    return ms_ssim_value.item()


def calculate_metrics(target, output, target_tensor, output_tensor, lpips_model):
    mse_value = mean_squared_error(target, output)
    psnr_value = psnr(mse_value, max_val=1.0)
    ssim_value = ssim(target, output, channel_axis=-1, data_range=1.0)

    lpips_value = LPIPS_val(lpips_model, output_tensor, target_tensor)
    ciede_value = ciede2000(output, target)

    return {'mse': mse_value, 'ssim': ssim_value,
            'psnr': psnr_value, 'lpips': lpips_value,
            'ciede2000': ciede_value}