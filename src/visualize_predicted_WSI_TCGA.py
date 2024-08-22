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
                5:(34, 139, 34),
                6:(210, 105, 30),
                7:(255, 165, 0),
                8:(255, 0, 0)}

site = 'Rectum'
site = 'rectosigmoid_junction'
site = 'colon'
filename_list = glob.glob(f'./data/result_tcga_{site}/*.npy')
filename_list = [os.path.splitext(os.path.basename(i))[0] for i in filename_list]
res = []
tum_env_list = ['tum_adi', 'tum_deb', 'tum_lym', 'tum_muc', 'tum_mus', 'tum_str']

for ii in filename_list:
    filename = ii
    result = np.load(f'./data/result_tcga_{site}/{filename}.npy')
    result = np.argmax(result, 2)

    ##
    y = np.where(result == 8)[0]
    x = np.where(result == 8)[1]

    tum_str = 0;    tum_adi = 0;
    tum_deb = 0;    tum_lym = 0;
    tum_muc = 0;    tum_mus = 0;
    for k in range(len(y)):
        y1 = max(y[k]-1, 0)
        y2 = min(y[k]+2, result.shape[0])
        x1 = max(x[k]-1, 0)
        x2 = min(x[k]+2, result.shape[1])
        tum_str += np.sum(result[y1:y2, x1:x2] == 7)
        tum_adi += np.sum(result[y1:y2, x1:x2] == 0)
        tum_deb += np.sum(result[y1:y2, x1:x2] == 2)
        tum_lym += np.sum(result[y1:y2, x1:x2] == 3)
        tum_muc += np.sum(result[y1:y2, x1:x2] == 4)
        tum_mus += np.sum(result[y1:y2, x1:x2] == 5)
    tum_env = [tum_adi, tum_deb, tum_lym, tum_muc, tum_mus, tum_str]
    sum_patch = [np.sum(result == tissue_class[i]) for i in list(tissue_class.keys())]
    res.append([filename.replace('pred-','')] + sum_patch + tum_env)

import pandas as pd
result = pd.DataFrame(res,  columns=['filename'] + list(tissue_class.keys())+tum_env_list)
result['patient-id'] = result['filename'].str.slice(start=0,stop=12)
result['patient-id'] = result['patient-id'].str.replace('-', '.')
result = result[['patient-id', 'filename'] + list(tissue_class.keys())+tum_env_list]
result.to_csv(f'/home/dong/Desktop/Fib/tcga_{site}_result.csv', index=False)





import matplotlib.pyplot as plt
import numpy as np

result = np.load('/home/dong/Desktop/pred-S14-25634-3U-HIGH.npy')
aa.shape
