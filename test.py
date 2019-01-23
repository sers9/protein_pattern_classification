from tqdm import tqdm
import argparse
from get_data import HumanAtlasDatasetTest, HumanAtlasDataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from networks import DenseNet121
import os
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from options import test_options

opt = test_options()

# get DataLoader:
test_dataset = HumanAtlasDataset(data_dir=opt.data_dir, label_file=opt.image_list,  n_class=opt.n_class, 
										transform = transforms.Compose([
										transforms.ToTensor()
										]))
# get dataloader
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True)

model_0 = DenseNet121(opt.n_class).cuda()
model_1 = DenseNet121(opt.n_class).cuda()
model_2 = DenseNet121(opt.n_class).cuda()

checkpoint_0 = torch.load(f"{opt.chckpnt_dir}/{opt.chckpnt_folder}/model_0_{opt.model_type}.pth.tar")
checkpoint_1 = torch.load(f"{opt.chckpnt_dir}/{opt.chckpnt_folder}/model_1_{opt.model_type}.pth.tar")
checkpoint_2 = torch.load(f"{opt.chckpnt_dir}/{opt.chckpnt_folder}/model_2_{opt.model_type}.pth.tar")

model_0.load_state_dict(checkpoint_0['model'])
model_1.load_state_dict(checkpoint_1['model'])
model_2.load_state_dict(checkpoint_2['model'])

# switch to evaluate mode
model_0.eval()	
model_1.eval()	
model_2.eval()	

counter = 0
tp = np.zeros(opt.n_class) # True Positive array
fp = np.zeros(opt.n_class) # False Positive array
fn = np.zeros(opt.n_class) # False Negative array

loader = tqdm(test_loader, total=len(test_loader))
for _, (images, labels) in enumerate(loader):

	images = Variable(images.cuda())
	output_0 = model_0(images)    
	output_1 = model_1(images)    
	output_2 = model_2(images)    

	pred_arr_0 = output_0.cpu().detach().numpy()
	pred_arr_1 = output_1.cpu().detach().numpy()
	pred_arr_2 = output_2.cpu().detach().numpy()

	label_arr = labels.cpu().detach().numpy()

	pred_arr = pred_arr_0 + pred_arr_1 + pred_arr_2
	pred_arr = (pred_arr > 1.0).astype(np.int32)

	tp_batch = label_arr * pred_arr # multiplication of the label and pred arrays
	tp += tp_batch.sum(axis=0) # sum for all samples throughout the batch

	fp_batch = (label_arr * pred_arr + pred_arr) % 2 
	fp += fp_batch.sum(axis=0)

	fn_batch = (label_arr * pred_arr + label_arr) % 2 
	fn += fn_batch.sum(axis=0)

	counter += 1

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
# calculate f1 score for all classes and average for all classes as unweighted:
macro_f = (2 * precision * recall / (precision + recall + 1e-18)).mean()

print(f'Final Macro-F score is: {macro_f:.4f}')