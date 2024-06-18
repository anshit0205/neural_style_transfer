import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.image_utils import load_img_and_preprocess, unpreprocess, scale_img
from utils.metrics import psnr, compute_ssim, compute_lpips, compute_kid
from utils.loss_utils import VGG16_Avgpool, minimize_with_lbfgs

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content_img_path', type=str, required=True, help='Path to the content image')
    parser.add_argument('--style_img_path', type=str, required=True, help='Path to the style image')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    content_img_path = args.content_img_path
    style_img_path = args.style_img_path

    content_img = load_img_and_preprocess(content_img_path)
    h, w = content_img.shape[1:3]
    style_img = load_img_and_preprocess(style_img_path, (h, w))

    batch_shape = content_img.shape
    shape = content_img.shape[1:]

    # Define content and style layers
    content_layer_name = 'block4_conv2'
    style_layer_names = [
        'block1_conv2',
        'block2_conv2',
        'block3_conv3',
        'block4_conv1',
        'block5_conv2'
    ]

    # Create VGG model with average pooling
    vgg = VGG16_Avgpool(shape)

    # Set up content model
    content_layer = vgg.get_layer(content_layer_name).output
    content_model = tf.keras.models.Model(vgg.input, content_layer)
    content_target = tf.constant(content_model.predict(content_img))

    # Set up style model
    style_layers = [vgg.get_layer(layer_name).output for layer_name in style_layer_names]
    style_model = tf.keras.models.Model(vgg.input, style_layers)
    style_layers_outputs = [tf.constant(output) for output in style_model.predict(style_img)]

    # Optimize the image
    final_img = minimize_with_lbfgs(content_model, style_model, content_target, style_layers_outputs, content_img, style_img, 11, batch_shape)

    content_img_np = unpreprocess(content_img[0])
    style_img_np = unpreprocess(style_img[0])

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(scale_img(content_img_np))
    plt.title('Content Image')

    plt.subplot(1, 2, 2)
    plt.imshow(scale_img(style_img_np))
    plt.title('Style Image')

    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(scale_img(final_img[0]))
    plt.title('Result Image')
    plt.show()

    ssim_index = compute_ssim(content_img_np, final_img[0])
    print(f'SSIM: {ssim_index}')

    psnr_value = psnr(content_img_np, final_img[0])
    print(f'PSNR: {psnr_value} dB')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(content_img_np.ravel(), bins=256, color='red', alpha=0.5)
    plt.title('Content Image Histogram')

    plt.subplot(1, 3, 2)
    plt.hist(style_img_np.ravel(), bins=256, color='blue', alpha=0.5)
    plt.title('Style Image Histogram')

    plt.subplot(1, 3, 3)
    plt.hist(final_img.ravel(), bins=256, color='green', alpha=0.5)
    plt.title('Result Image Histogram')

    plt.show()

    lpips_score = compute_lpips(content_img_np, final_img[0])
    print(f'LPIPS: {lpips_score}')

    kid_score = compute_kid(content_img_np, final_img[0])
    print(f'KID: {kid_score[0]} ± {kid_score[1]}')
