import sys
import os.path
from os.path import exists
from pathlib import Path
#Path(__file__).resolve().parent.parent

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim

from skimage import io, transform

import json
import os

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

from time import time

from itertools import chain
import operator

plt.ion()
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

########################################################################################################################

def clean_data(data_list):
	for i in range(len(data_list)):
		bbox = data_list[i]['bbox']
		bb_x1 = int(bbox[0])
		bb_y1 = int(bbox[1])

		data_list[i].pop('visibility', None)

		x, y = data_list[i]['landmarks'][0::2], data_list[i]['landmarks'][1::2]

		x = np.array(x) - bb_x1
		y = np.array(y) - bb_y1

		land = np.array([x, y]).T.flatten()

		data_list[i].update({'landmarks': land})

	return data_list


connections = [[0,2],[2,1],[3,4],[4,5],[5,6],[6,7],[4,8],[8,9],[9,10],[4,11],[11,12],[12,13],[11,14],[14,15],[11,16]]

def brid_pl(landmarks):

	landmarks = np.reshape(landmarks,(-1,2)).tolist()
	connec = []
	for j, k in connections:
		connec.append([landmarks[j],landmarks[k]])
	connec_pl = chain(*np.array(connec).transpose((0,2,1)).tolist())

	return np.array(connec).astype('float'), connec_pl

########################################################################################################################

with open('data/openmonkeychallenge/train_annotation.json', 'r') as f:
	train_ann = json.load(f)
train_ann = train_ann['data']
size_train = len(train_ann)

train_ann = clean_data(train_ann)
especies = list(set(map(operator.itemgetter('species'), train_ann)))


train_d = pd.DataFrame(train_ann)
train_d['file'] = 'data/openmonkeychallenge/train/train/'+train_d['file']

########################################################################################################################


class Landmarks_Dataset(Dataset):

	def __init__(self, dataset, transform=None):
		self.dataset = dataset
		self.transform = transform

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		if torch.is_tensor(index):
			idx = idx.tolist()
		# Para ajustar margenes de la imagen
		bbox = self.dataset.loc[index, 'bbox']
		bb_x1 = int(bbox[0])
		bb_y1 = int(bbox[1])
		bb_x2 = int(bbox[0] + bbox[2])
		bb_y2 = int(bbox[1] + bbox[3])

		img_name = os.path.join(self.dataset.loc[index, 'file'])
		image = io.imread(img_name)
		image = image[bb_y1:bb_y2, bb_x1:bb_x2]
		specie, fil = self.dataset.loc[index, 'species'], self.dataset.loc[index, 'file']
		landmarks = self.dataset.loc[index, 'landmarks']
		landmarks = landmarks.astype('float').flatten()
		sample = {'image': image, 'landmarks': landmarks}

		sample.update({'species': specie, 'file': fil})

		if self.transform:
			sample = self.transform(sample)

		return sample


class Rescale(object):

	def __init__(self, output_size):

		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, landmarks, specie, fil = sample.values()

		h, w = image.shape[:2]

		if isinstance(self.output_size, int):
			new_h, new_w = self.output_size, self.output_size
		else:
			new_h, new_w = self.output_size
		new_h, new_w = int(new_h), int(new_w)

		img = transform.resize(image, (new_h, new_w))
		landmarks = (landmarks.reshape((-1, 2)) * [new_w / w, new_h / h]).flatten()

		return {'image': img,
			'landmarks': landmarks,
			'species': specie,
			'file': fil}


class ToTensor(object):

	def __call__(self, sample):
		image, landmarks, specie, fil = sample.values()

		img = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(img),
			'landmarks': torch.from_numpy(landmarks),
			'species': specie,
			'file': fil}



train_data = Landmarks_Dataset(train_d, transform=transforms.Compose([Rescale(200),
                                                                      ToTensor()]))


num_workers = 2
dataset_train = DataLoader(train_data, batch_size=64,
                           shuffle=True, num_workers=num_workers)

class DoubleConv(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, bias=True, dtype=torch.float64),
			nn.BatchNorm2d(mid_channels, dtype=torch.float64),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, bias=True, dtype=torch.float64),
			nn.BatchNorm2d(out_channels, dtype=torch.float64),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.Max_Pool = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.Max_Pool(x)


class Up(nn.Module):

	def __init__(self, in_channels):
		super().__init__()
		self.Ups = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2, dtype=torch.float64)
		self.conv = DoubleConv(in_channels, in_channels // 2)

	def forward(self, x1, x2):
		x1 = self.Ups(x1)
		crp = transforms.CenterCrop((x1.shape[-2], x1.shape[-1]))
		x2 = crp(x2)

		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class UNet_pred(nn.Module):

	def __init__(self):
		super().__init__()
		#U-Net
		self.D1 = DoubleConv(3, 64)
		self.D2 = Down(64, 128)
		self.D3 = Down(128, 256)
		self.D4 = Down(256, 512)
		self.U1 = Up(512)
		self.U2 = Up(256)
		self.U3 = Up(128)
		self.U4 = nn.Conv2d(64, 2, 1, dtype=torch.float64)

		# Mlp for point estimation
		self.mlp = nn.Sequential(
			nn.Flatten(),
			nn.Linear(23328, 5000, dtype=torch.float64),
			nn.LeakyReLU(negative_slope=0.5, inplace=True),
			nn.Linear(5000, 500, dtype=torch.float64),
			nn.LeakyReLU(negative_slope=0.5, inplace=True),
			nn.Linear(500, 34, dtype=torch.float64),
			nn.Sigmoid()
		)

	def forward(self, x):
		x1 = self.D1(x)
		x2 = self.D2(x1)
		x3 = self.D3(x2)
		x4 = self.D4(x3)
		x5 = self.U1(x4, x3)
		x6 = self.U2(x5, x2)
		x7 = self.U3(x6, x1)
		x8 = self.U4(x7)
		x9 = self.mlp(x8)

		return x9*200

U = UNet_pred().to(device = DEVICE)

def avg(listt):
	return sum(listt)/len(listt)


########################################################################################################################


def train(model, datatrain, epochs, l_r, device):
	loss_function = nn.L1Loss()
	optimizer = optim.Adam(model.parameters(), lr=l_r)
	train_loss = []

	for j in range(epochs):
		model.train()
		av = []
		for batch in datatrain:

			imgs_input = batch['image']
			imgs_input = imgs_input.to(device=device, dtype=torch.float64)
			ldm_target = batch['landmarks']
			ldm_target = ldm_target.to(device=device, dtype=torch.float64)

			optimizer.zero_grad()

			prediction = model(imgs_input)

			loss = loss_function(prediction, ldm_target)

			loss.backward()

			optimizer.step()

			av.append(loss.item())

		train_loss.append(avg(av))
		print(avg(av)) 
	return train_loss


OPP = train(U, dataset_train, 15, 0.0001, DEVICE)

torch.save(U.state_dict(), f'Uw.pt')
pickle.dump(OPP, open('loss.pickle', 'wb'))
