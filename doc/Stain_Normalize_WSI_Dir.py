import sys
import openslide
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(r'D:\ASAP\bin') #install asap
import multiresolutionimageinterface as mir
import stainNorm_Reinhard
import stain_utils
from skimage import color
import os

if __name__ =='__main__':

    tile_size = 512
    wsi_ori_dir = r'E:\Projects\Colon\Data\Samsung_Colon\Stain_Normalization\wsi_ori'
    wsi_sn_dir = r'E:\Projects\Colon\Data\Samsung_Colon\Stain_Normalization\wsi_sn'

    ref_img_path = r'E:\Projects\Colon\Data\Samsung_Colon\Stain_Normalization\ref_imgs\1.jpg'  # 1.jpg, 2.png, 3.jpg
    ref_image = stain_utils.read_image(ref_img_path)
    n = stainNorm_Reinhard.Normalizer()
    n.fit(ref_image)

    wsi_names = [f for f in os.listdir(wsi_ori_dir)]

    for wsi_name in wsi_names:

        slide_p = openslide.OpenSlide(os.path.join(wsi_ori_dir, wsi_name))
        level_dims = slide_p.level_dimensions[0]
        level_ds = slide_p.level_downsamples[0]

        writer = mir.MultiResolutionImageWriter()

        writer.openFile(os.path.join(wsi_sn_dir, wsi_name))  ## .tif, .svs

        writer.setTileSize(tile_size)
        writer.setCompression(mir.JPEG)  # JPEG, LZW, JPEG2000
        # writer.setJPEGQuality(90)
        # writer.setInterpolation(mir.NearestNeighbor)
        writer.setDataType(mir.UChar)
        writer.setColorType(mir.RGB)
        # if there is blank region in the wsi (size mismatch), it won't be opened in qupath
        WSI_Width = level_dims[0] // tile_size * tile_size
        WSI_Height = level_dims[1] // tile_size * tile_size
        writer.writeImageInformation(WSI_Width, WSI_Height)
        for x in range(0, WSI_Width, tile_size):
            for y in range(0, WSI_Height, tile_size):
                patch_img = np.array(
                    slide_p.read_region((x, y), 0, (tile_size, tile_size)).convert('RGB')).astype("ubyte")

                patch_img_gray = (color.rgb2gray(patch_img) * 255).astype('uint8')
                patch_img_binary = patch_img_gray > 220

                # .mrxs transform black region into white
                if np.count_nonzero(patch_img_gray == 0) > np.size(patch_img_gray) * 0.01:
                    patch_img_sn = np.ones((tile_size, tile_size, 3)).astype("ubyte") * 255
                elif np.mean(patch_img_binary) > 0.7:
                    patch_img_sn = patch_img
                else:
                    patch_img_sn = n.transform(patch_img)
                    if isinstance(patch_img_sn, bool):  #######################################
                        if patch_img_sn == False:
                            patch_img_sn = patch_img
                writer.writeBaseImagePartToLocation(patch_img_sn.flatten(), x=int(x), y=int(y))

        slide_p.close()
        writer.finishImage()

