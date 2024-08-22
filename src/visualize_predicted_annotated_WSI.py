import glob
import os
import pathlib
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

here = pathlib.Path(__file__).parent.resolve()

tissue_dir = glob.glob(str(here / './data/tissue/NCT-CRC-HE-100K/*'))
tissue_dir = sorted(tissue_dir)
tissue_class = {os.path.basename(i): idx for idx, i in enumerate(tissue_dir)}
tissue_color = {0: (128, 128, 128),
                1: (255, 255, 255),
                2: (128, 0, 128),
                3: (30, 144, 255),
                4: (255, 228, 196),
                5: (34, 139, 34),
                6: (210, 105, 30),
                7: (255, 165, 0),
                8: (255, 0, 0)}

fildir = 'result_stomach_part' ## file directory name
resolution = 224
slide_list = glob.glob(f'./data/{fildir}/*.npy')
# slide_data = np.load('./data/slide/KKM2020_annotation_0805.npy',
#                      allow_pickle=True).item()
slide_data = np.load('./data/slide/stomach_annotation_0717.npy',
                     allow_pickle=True).item()


tsr_result = pd.DataFrame([],
                          index= range(len(slide_list)),
                          columns=['slidename',
                                   'part',
                                   'label',
                                   'Tumor-Stroma-Ratio',
                                   'Tumor-Tissue-Ratio',
                                   'Tumor&Normal-Stroma-Ratio',
                                   'Tumor&Normal-Tissue-Ratio']
                          )


for i in range(len(slide_list)):
    slide_result_path = slide_list[i]
    result = np.load(slide_result_path, allow_pickle=True)
    annotated = np.load(slide_result_path, allow_pickle=True)
    slidename = os.path.splitext(os.path.basename(slide_result_path))[
        0].replace('pred-', '')
    part_no = slidename.split('-')[-1]
    slidename = '-'.join(slidename.split('-')[:-1])

    slide_data[slidename].keys()
    coord_x = slide_data[slidename][int(part_no)]['coord_x']
    coord_y = slide_data[slidename][int(part_no)]['coord_y']

    coord = [[i // resolution, j // resolution] for i, j in zip(coord_x, coord_y)]

    vis = np.zeros((*result.shape, 3))
    for ii in range(result.shape[0]):
        for jj in range(result.shape[1]):
            vis[ii, jj] = tissue_color[result[ii, jj]]
    vis = vis / 255.

    vis = cv2.polylines(vis, [np.array(coord)], isClosed=True, color=(0, 0, 0))
    vis = cv2.resize(vis, None, fx=3., fy=3.)
    #plt.imshow(vis,  interpolation='quadric')
    #plt.axis('off')
    #plt.imsave(f'./data/{fildir}/{slidename}.png', vis, dpi=300)
    #plt.close()

    mask = np.zeros_like(result)
    mask = cv2.fillPoly(mask, [np.array(coord)], color=(1)) == 1

    hotspot = result[mask]
    hotspot = hotspot[hotspot != 1]  # omit background

    tsr_result.iloc[i, 0] = slidename
    tsr_result.iloc[i, 1] = part_no
    tsr_result.iloc[i, 2] = slide_data[slidename]['label']
    # Tumor-Stroma-Ratio
    cnt_stroma = np.sum(hotspot == 7)
    cnt_tumor = np.sum(hotspot == 8)
    tsr_result.iloc[i, 3] = cnt_stroma / (cnt_stroma + cnt_tumor)

    # Tumor-Tisue-Ratio
    cnt_tisue = len(hotspot)
    cnt_tumor = np.sum(hotspot == 8)
    tsr_result.iloc[i, 4] = (cnt_tisue-cnt_tumor) / (cnt_tisue)
    # Tumor&Normal-Stroma-Ratio
    cnt_stroma = np.sum(hotspot == 7)
    cnt_tumor = np.sum(np.logical_or(hotspot == 8, hotspot == 6))
    tsr_result.iloc[i, 5] = cnt_stroma / (cnt_stroma + cnt_tumor)
    # Tumor&Normal-Tisue-Ratio
    cnt_tisue = len(hotspot)
    cnt_tumor = np.sum(np.logical_or(hotspot == 8, hotspot == 6))
    tsr_result.iloc[i, 6] = (cnt_tisue-cnt_tumor) / (cnt_tisue)

tsr_result.to_csv(f'./data/result_stomach_part.csv')

