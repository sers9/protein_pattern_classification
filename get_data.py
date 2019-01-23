import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import pandas as pd
from skimage.io import imread
import numpy as np

def get_one_hot_encoding(label):
	one_hot_label = np.zeros((28,), dtype=int)
	for ii in label:
		one_hot_label[int(ii)] = 1
	return one_hot_label

class HumanAtlasDataset(Dataset):
	def __init__(self, data_dir, label_file, n_class, transform=None):
		"""
		Args:
			data_dir (string): Path to the png files.
			label_file (string): csv file listing the labels for the images.
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self.data_dir = data_dir
		self.df = pd.read_csv(label_file)
		self.image_names = self.df['Id'].values
		self.n_class = n_class

		label_list = [val.split(' ') for val in self.df['Target'].values]

		# convert labels to one-hot encoding:
		self.labels = np.zeros((len(label_list), self.n_class), dtype=int) # 28 labels
		for index, lbl in enumerate(label_list):
			self.labels[index] = get_one_hot_encoding(lbl)

		self.transform = transform

	def __getitem__(self, index):
		image_name = os.path.join(self.data_dir, self.image_names[index])
		label = self.labels[index]

		# protein of interest
		green = np.expand_dims(imread(f'{image_name}_green.png'), axis=2)

		# cellular landmarks:
		blue = np.expand_dims(imread(f'{image_name}_blue.png'), axis=2) # nucleus 
		red = np.expand_dims(imread(f'{image_name}_red.png'), axis=2) # microtubules 
		yellow = np.expand_dims(imread(f'{image_name}_yellow.png'), axis=2) # endoplasmic reticulum

		# concatenate  images:
		whole_image = np.concatenate((blue, red, yellow, green), axis=2)
		
		if self.transform:
			whole_image = self.transform(whole_image)

		return whole_image, torch.FloatTensor(label)
 
	def __len__(self):
		return len(self.image_names)



class HumanAtlasDatasetTest(Dataset):
	def __init__(self, data_dir, image_list, n_class, transform=None):
		"""
		Args:
			data_dir (string): Path to the png files.
			image_list (string): csv file listing the images for test.
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self.data_dir = data_dir
		self.df = pd.read_csv(image_list)
		self.image_names = self.df['Id'].values
		self.n_class = n_class

		self.transform = transform

	def __getitem__(self, index):
		image_name = os.path.join(self.data_dir, self.image_names[index])

		# protein of interest
		green = np.expand_dims(imread(f'{image_name}_green.png'), axis=2)

		# cellular landmarks:
		blue = np.expand_dims(imread(f'{image_name}_blue.png'), axis=2) # nucleus 
		red = np.expand_dims(imread(f'{image_name}_red.png'), axis=2) # microtubules 
		yellow = np.expand_dims(imread(f'{image_name}_yellow.png'), axis=2) # endoplasmic reticulum

		# concatenate  images:
		whole_image = np.concatenate((blue, red, yellow, green), axis=2)

		if self.transform:
			whole_image = self.transform(whole_image)

		return self.image_names[index], whole_image
 
	def __len__(self):
		return len(self.image_names)

