import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import sys

def main():
    network_dir = '/home/tkuipers/pgan_gender_results/00004-pgan-chestxray_labeled-cond-g4k-grpc-preset-v2-4gpus-fp32/network-snapshot-008700.pkl'
    output_dir = '/home/tkuipers/pgan_gender_results/females'
    label_ids = np.array([1])
    img_name = 'fake_findings'

    image_count = 15

    tflib.init_tf()

    print('Loading network...')
    with open(network_dir, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    # Create random latent vectors
    rnd = np.random.RandomState()
    latents = rnd.randn(image_count, Gs.input_shape[1])

    # Create labels
    if 1:
        labels = np.zeros([1] + Gs.input_shapes[1][1:])
        labels[0][label_ids] = 1
        labels = np.repeat(labels, image_count, axis=0)
    if 0:
        labels = np.zeros((image_count, len(label_ids)))
        labels[label_ids, label_ids] = 1.0

    print(labels)

    # Generate image.
    print('Generating images...')
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, labels, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save image.
    print('Saving images...')
    for i, image in enumerate(images):
        PIL.Image.fromarray(image[:,:,0]).save(output_dir + '_' + img_name + '_' + str(i) + '.png')

if __name__ == '__main__':
    main()