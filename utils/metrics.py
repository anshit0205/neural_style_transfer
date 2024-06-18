import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import lpips
from tensorflow_gan.python.eval import kid_score

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compute_ssim(img1, img2):
    img1_gray = tf.image.rgb_to_grayscale(img1)
    img2_gray = tf.image.rgb_to_grayscale(img2)
    return ssim(img1_gray.numpy().squeeze(), img2_gray.numpy().squeeze(), data_range=img2_gray.numpy().max() - img2_gray.numpy().min())

def compute_lpips(img1, img2):
    loss_fn = lpips.LPIPS(net='vgg')
    return loss_fn(img1, img2).item()

def compute_kid(img1, img2):
    kid = kid_score(img1, img2)
    return kid
