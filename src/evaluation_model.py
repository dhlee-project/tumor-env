import argparse
import glob
import os
import pathlib

import cv2
import numpy as np
import openslide
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tifffile import memmap
from PIL import Image


def load_model(pretrained):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(in_features=512, out_features=9, bias=True)
    return model


class TissueDataset(Dataset):
    def __init__(self, test_patch, label):
        self.test_patch = test_patch
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = Image.open(self.test_patch[idx]).convert('RGB')
        target = self.label[idx]

        image = transforms.ColorJitter(brightness=0.25,  # 0.25
                               contrast=0.75,  # 0.75
                               saturation=0.75,  # 0.25
                               hue=0.5)(image)

        image = transforms.ToTensor()(image)
        return {'image': image, 'target': target}


# Testing settings
parser = argparse.ArgumentParser(description='tumor-env')
parser.add_argument('--dataset', default='./data/tissue/CRC-VAL-HE-7K/*', help='test datset')
parser.add_argument('--result-dir', default='./data/result_test', help='result dir')
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

# load data
here = pathlib.Path(__file__).parent.resolve()
tissue_dir = glob.glob(str(here / './data/tissue/NCT-CRC-HE-100K/*'))
tissue_dir = sorted(tissue_dir)
tissue_class = {os.path.basename(i): idx for idx, i in enumerate(tissue_dir)}
testset_dir = opt.dataset
testset_name = glob.glob(os.path.join(testset_dir, '*.tif'))


test_name = glob.glob(os.path.join(testset_dir, '*.tif'))
info = 'model'
#model_path = "./data/checkpoint/nonorm_renet18_b256_lr1e-05_transform/model-resnet18-201022-90.pth"
# model_path = "./data/checkpoint/norm_renet18_b256_lr1e-05/model-resnet18-201022-20.pth"
# model_path = "./data/checkpoint/nonorm_renet18_b256_lr1e-05/model-resnet18-201022-20.pth"
model_path = "./data/checkpoint/nonorm_renet18_b256_lr1e-05_transform_201103_3/model-resnet18-201103-20.pth"

model = load_model(pretrained=False).to(device)
model.load_state_dict(torch.load(model_path))
model = model.eval()
label = [os.path.basename(os.path.dirname(i)) for i in testset_name]

dataset = TissueDataset(testset_name, label)
dataloader = DataLoader(dataset, 256, num_workers=6)
total_len = 0
total_true = 0

for idx, batch in enumerate(dataloader):
    batch_images = batch['image']
    batch_target = batch['target']
    input_ = batch['image'].to(device).type(torch.float32)
    output = model(input_).detach().cpu().numpy().copy()
    output = np.argmax(output, axis=1)
    target = [tissue_class[i] for i in batch_target]
    total_len += len(target)
    total_true += np.sum(target == output)

batch_images.shape
acc = total_true/total_len
print(acc)
