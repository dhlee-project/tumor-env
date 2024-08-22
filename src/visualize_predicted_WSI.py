import pathlib
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

here = pathlib.Path(__file__).parent.resolve()
tissue_dir = glob.glob(str(here / './data/tissue/NCT-CRC-HE-100K/*'))
tissue_dir = sorted(tissue_dir)
tissue_class = {os.path.basename(i): idx for idx, i in enumerate(tissue_dir)}
tissue_color = {0:(128, 128, 128),
                1:(255, 255, 255),
                2:(128, 0, 128),
                3:(30, 144, 255),
                4:(255, 228, 196),
                5:(34,139,34),
                6:(210,105,30),
                7:(255,165,0),
                8:(255, 0, 0)}

filename = 'pred-TCGA-5M-AATE-01A-01-TS1.900EF8FE-361C-4C61-8761-328026D2F627'
result = np.load(f'./data/result_tcga/{filename}.npy')
result = np.argmax(result,2)


vis = np.zeros((*result.shape, 3))

for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        vis[i, j] = tissue_color[result[i, j]]

vis = vis/255.
vis = cv2.resize(vis, None, fx=3., fy=3.)

plt.imshow(vis,  interpolation='quadric')
plt.axis('off')
plt.imsave(f'./data/result/{filename}.png', vis, dpi=300)
plt.close()