import argparse
import glob
import os
import pathlib

import cv2
import numpy as np
import openslide
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tifffile import memmap


def load_model(pretrained):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(in_features=512, out_features=9, bias=True)
    return model


def isBG(Img, BG_Thres, BG_Percent):
    Gray_Img = np.uint8(rgb2gray(Img) * 255)
    White_Percent = np.mean((Gray_Img > BG_Thres))
    Black_Percent = np.mean((Gray_Img < 255 - BG_Thres))

    if Black_Percent > BG_Percent or White_Percent > BG_Percent or Black_Percent + White_Percent > BG_Percent:
        return True
    else:
        return False


def get_region(grid_x, image_w, grid_w, margin_w):
    '''
    Return the base and offset pair to read from the image.
    :param grid_x: grid index on the image
    :param image_w: image width (or height)
    :param grid_w: grid width (or height)
    :param margin: margin width (or height)

    :return: the base index and the width on the image to read
    '''
    image_x = grid_x * grid_w

    image_l = min(image_x, image_w - grid_w)
    image_r = image_l + grid_w - 1

    read_l = max(0, image_l - margin_w)
    read_r = min(image_r + margin, image_w - 1)
    #    read_l = min(image_x - margin_w, image_w - (grid_w + margin_w))
    #    read_r = min(read_l + grid_w + (margin_w << 1), image_w) - 1
    #    image_l = max(0,read_l + margin_w)
    #    image_r = min(image_l + grid_w , image_w) - 1
    return read_l, image_l, image_r, read_r


def resize_region(im_l, im_r, scale_factor):
    sl = im_l // scale_factor
    sw = (im_r - im_l + 1) // scale_factor
    sr = sl + sw - 1
    return sl, sr

# Testing settings
parser = argparse.ArgumentParser(description='tumor-env')
parser.add_argument('--dataset', default='./data/wsi', help='test datset')
parser.add_argument('--result-dir', default='./data/result', help='result dir')
parser.add_argument('--model', type=str, default='unet', help='model name')
parser.add_argument('--nepochs', type=int, default=200,
                    help='saved model of which epochs')
parser.add_argument('--cuda', default=True, action='store_true',
                    help='use cuda')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=38315)
opt = parser.parse_args()
print(opt)
device = torch.device("cuda:0" if opt.cuda else "cpu")

margin = 0
local_size = 224

# load data
here = pathlib.Path(__file__).parent.resolve()

tissue_dir = glob.glob(str(here / './data/tissue/NCT-CRC-HE-100K/*'))
tissue_dir = sorted(tissue_dir)
tissue_class = {os.path.basename(i): idx for idx, i in enumerate(tissue_dir)}

testset_dir = opt.dataset
testset_name = glob.glob(os.path.join(testset_dir, '*.svs'))

info = 'model'
#model_path = "./data/checkpoint/norm_renet18_b256_lr1e-05/model-resnet18-201022-20.pth"
model_path = "./data/checkpoint/nonorm_renet18_b256_lr1e-05_transform/model-resnet18-201022-20.pth"
model = load_model(pretrained=False).to(device)
model.load_state_dict(torch.load(model_path))
model = model.eval()


for i in range(0, len(testset_name)):

    slide_path = testset_name[i]
    filename = os.path.splitext(os.path.basename(slide_path))[0]
    HE_SLIDE = openslide.open_slide(slide_path)
    slide_width, slide_height = HE_SLIDE.dimensions
    HE_SLIDE.level_dimensions

    num_w = slide_width // local_size + 1
    num_h = slide_height // local_size + 1

    # ROI_w, ROI_h = ROI_region
    result_name = f'pred-{filename}.npy'
    result_path = os.path.join(opt.result_dir, result_name)
    # get interest tile location
    A = np.ones((num_w, num_h))  #
    iter_list = [[i[0][0],
                  i[0][1]
                  ] for i in np.ndenumerate(A)]  #
    len_itr = len(iter_list)
    tsp_map = np.zeros((num_h, num_w))

    for itr, [iter_w, iter_h] in enumerate(iter_list):

        l, im_l, im_r, r = get_region(iter_w, slide_width, local_size, margin)
        t, im_t, im_b, b = get_region(iter_h, slide_height, local_size, margin)
        HE_patch_raw = HE_SLIDE.read_region((l, t), 0, (224*2, 224*2))
        HE_patch_raw = np.array(HE_patch_raw)[:,:,:3]
        HE_patch_raw = cv2.resize(HE_patch_raw, (224, 224))
        # if type(normalizer.transform(HE_patch_raw)) is bool:
        #    tsp_map[iter_h, iter_w] = 1
        #    continue
        # HE_patch_raw = normalizer.transform(HE_patch_raw)
        # if isBG(HE_patch_raw, 240, 0.95) == True:
        #     continue
        # HE_patch_resized = cv2.resize(HE_patch_raw,
        #                               None,
        #                               fx=1 / img_resize_factor,
        #                               fy=1 / img_resize_factor)
        HE_patch_tensor = transforms.ToTensor()(HE_patch_raw)
        HE_patch_tensor = HE_patch_tensor.view(1, *HE_patch_tensor.shape)
        input_ = HE_patch_tensor.to(device).type(torch.float32)
        output = model(input_).detach().cpu().numpy().copy()
        output = np.argmax(output)
        # label = list(tissue_class.keys())[label]
        tsp_map[iter_h, iter_w] = output

        if itr % 100 == 0:
            print('Done {}/{}'.format(itr, len_itr))
            np.save(result_path, tsp_map)