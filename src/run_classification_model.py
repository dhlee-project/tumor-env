import glob
import os
import pathlib
import logging.handlers

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


def load_model():
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(in_features=512, out_features=9, bias=True)
    return resnet


class TissueDataset(Dataset):

    def __init__(self, root_dir, csv_file, type='Train', transform=None):
        self.dataset = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

        if type == 'Train':
            self.dataset = self.dataset[self.dataset['type'] == 'Train']
            self.dataset.reset_index(drop=True, inplace=True)
        else:
            self.dataset = self.dataset[self.dataset['type'] == 'Test']
            self.dataset.reset_index(drop=True, inplace=True)

        self.dummy_label = pd.get_dummies(self.dataset['label'])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                self.dataset.iloc[idx, 0])
        #image = cv2.imread(img_path)[:, :, [2, 1, 0]]
        #image = image.transpose(2, 0, 1) / 255.
        image = Image.open(img_path).convert('RGB')
        label = self.dataset.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        image = transforms.ToTensor()(image)

        sample = {'image': image, 'label': label}
        return sample


here = pathlib.Path(__file__).parent.resolve()
data_type = 'nonorm'
epoch_size = 100
batch_size = 256
epoch = 1
threads = 6
lr_rate = 0.00001
info = f'{data_type}_renet18_b{batch_size}_lr{lr_rate}_transform_201104_3'
if not os.path.exists(os.path.join(str(here / '../data/checkpoint'), info)):
    os.mkdir(os.path.join(str(here / '../data/checkpoint'), info))

log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler(str(here /'../data/checkpoint/{}/log.txt'.format(info)))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
log.addHandler(fileHandler)
log.addHandler(streamHandler)
log.info(f'info: Training Tumor env {info} \n \n')
log.info('info: Loading datasets')
# load data
tissue_dir = glob.glob(str(here / '../data/tissue/NCT-CRC-HE-100K/*'))
tissue_dir = sorted(tissue_dir)
tissue_class = {os.path.basename(i): idx for idx, i in enumerate(tissue_dir)}

root_dir = str(here / '../data/tissue')
dataset_dir = f'data-{data_type}.csv'

transform = transforms.ColorJitter(brightness=0.75,#0.25
                                   contrast=0.75,#0.75
                                   saturation=0.75,#0.25
                                   hue=(0.4, 0.5))

train_dataset = TissueDataset(root_dir, dataset_dir, 'Train', transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=threads)
test_dataset = TissueDataset(root_dir, dataset_dir, 'Test')
test_dataloader = DataLoader(test_dataset, batch_size=32,
                             shuffle=True, num_workers=threads)
# load model
model = load_model().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr_rate)

# train info
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(epoch_size):
    running_loss = 0
    running_acc = 0
    model = model.train()
    for idx, batch in enumerate(train_dataloader):
        image = batch['image'].to(device, dtype=torch.float)
        label = batch['label'].to(device, dtype=torch.long)
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        running_acc += np.sum(pred == label.cpu().numpy()) / pred.shape[0]

        if (idx + 1) % 100 == 0:
            mean_loss = running_loss / (idx + 1)
            mean_acc = running_acc / (idx + 1)
            log.info('[{}/{}][{}/{}] loss: {:.3f} acc: {:.3f}'.format(
                epoch + 1, epoch_size, (idx + 1) * batch_size,
                len(train_dataloader) * batch_size, mean_loss, mean_acc))

    if (epoch + 1) % 1 == 0:
        model = model.eval()
        eval_loss = 0
        eval_acc = 0
        for idx, batch in enumerate(test_dataloader):
            image = batch['image'].to(device, dtype=torch.float)
            label = batch['label'].to(device, dtype=torch.long)
            output = model(image)
            loss = criterion(output, label)
            eval_loss += loss.item()

            pred = np.argmax(output.detach().cpu().numpy(), axis=1)
            eval_acc += np.sum(pred == label.cpu().numpy()) / pred.shape[0]

        mean_loss = eval_loss / (idx + 1)
        mean_acc = eval_acc / (idx + 1)
        log.info(f'TEST Loss : {mean_loss:.3f}, ACC : {mean_acc:.3f}')

    if (epoch + 1) % 1 == 0:
        torch.save(
            model.state_dict(),
            str(here / f'../data/checkpoint/{info}/model-resnet18-201103-{epoch+1}.pth')
        )
