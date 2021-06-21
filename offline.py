# Script to make absorption images from saved data instead of live updates

import numpy as np

from image_reconstruction.image_averager import ImageAverager
from image_reconstruction.cpu_reconstructor import CPUReconstructor

N_bg = N_probe = 100

averager = ImageAverager(N_bg)
reconstructor = CPUReconstructor(N_probe)

for i in range(N_bg):
    image = np.load(f'saved_images/dark_{i:04d}.npy')
    averager.add_ref_image(image)


for i in range(N_probe):
    image = np.load(f'saved_images/probe_{i:04d}.npy')
    reconstructor.add_ref_image(image)

dark = averager.get_average()

i = 0
absorption = np.load(f'saved_images/atoms_{i:04d}.npy')
recon_probe, _ = reconstructor.reconstruct(absorption)
absorbed_fraction = 1 - (absorption.astype(float) - dark) / (recon_probe - dark)
