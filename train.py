import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import argparse
from tqdm import tqdm
import time
import os

from get_data import HumanAtlasDataset
from networks import DenseNet121
import shutil
from options import training_options

opt = training_options()

def adjust_learning_rate(optimizer, global_step, decay_steps, learning_rate, alpha=0.0):
    """
    cosine annealing:
    https://www.tensorflow.org/api_docs/python/tf/train/cosine_decay
    """
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed

    for param_group in optimizer.param_groups:
        param_group['lr'] = decayed_learning_rate

def forward_propagate(loader, model, criterion, epoch, optimizer=None, mode='train'):

    tp = np.zeros(opt.n_class) # True Positive array
    fp = np.zeros(opt.n_class) # False Positive array
    fn = np.zeros(opt.n_class) # False Negative array

    counter = 0
    running_loss = 0
    running_corrects = 0
    running_f_macro = 0

    if mode == 'train':
        model.train()
    elif mode == 'validate':
        model.eval()

    ep_st = time.time()

    loader_tqdm = tqdm(loader, total=len(loader))
    for itr, (image, label) in enumerate(loader_tqdm):

        image = image.cuda()
        label = label.cuda()        
        
        output = model(image)
        loss = criterion(output, label)        
        
        if mode == 'train':
            adjust_learning_rate(optimizer, global_step=itr, decay_steps=len(loader), learning_rate=opt.lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        label_arr = label.cpu().detach().numpy().astype(np.int)
        pred_arr = output.cpu().detach().numpy() > 0.5
        tp_batch = label_arr * pred_arr # multiplication of the label and pred arrays
        tp += tp_batch.sum(axis=0) # sum for all samples throughout the batch

        fp_batch = (label_arr * pred_arr + pred_arr) % 2 
        fp += fp_batch.sum(axis=0)

        fn_batch = (label_arr * pred_arr + label_arr) % 2 
        fn += fn_batch.sum(axis=0)

        counter += 1

    # calculate the precision and the recall for the epoch
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    # calculate f1 score for all classes and average for all classes as unweighted:
    macro_f = (2 * precision * recall / (precision + recall + 1e-18)).mean()

    epoch_loss = running_loss / counter

    print(f'{mode} epoch: {epoch} | loss: {epoch_loss:.4f} | macro-F: {macro_f:.4f} | time: {time.time()-ep_st:.4f} sec')

    return macro_f, epoch_loss


# get DataLoader:
dataset = HumanAtlasDataset(data_dir=opt.data_dir, label_file=opt.images_list_file, n_class=opt.n_class, 
                                transform = transforms.Compose([transforms.ToTensor(), 
                                                                ]))

dataset_size = len(dataset)
all_ind = list(range(dataset_size)) # all indices for all dataset
split = int(np.floor(opt.val_ratio * dataset_size))

TR_LOADERS = []
VAL_LOADERS = []

ind = all_ind
for ii in range(opt.num_val_splits):

    for _ in range(10):
        np.random.shuffle(ind)

    # sampled indices for training and validation splits
    if ii == 0:
        tr, val = ind[split:], ind[:split] 
    else:
        _, val = ind[split:], ind[:split]
        tr = [j for j in all_ind if j not in val] 

    tr_sam = SubsetRandomSampler(tr) 
    val_sam = SubsetRandomSampler(val) 
    print('# tr, # val: ', len(tr), len(val))

    # train loader:
    tr_ld = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=tr_sam, 
                                                num_workers=opt.workers, pin_memory=True)

    # validation loader:
    val_ld = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=val_sam, 
                                                    num_workers=opt.workers, pin_memory=True)

    TR_LOADERS.append(tr_ld)
    VAL_LOADERS.append(val_ld)

    # validation splits does not overlap with each other, so new val split is sampled from train indices only.
    ind = tr


MODELS = []
OPTIMIZERS = []
criterion = nn.BCELoss()
for ii in range(opt.num_val_splits):
    model = DenseNet121(opt.n_class).cuda()
    optimizer = optim.Adam(model.parameters(), opt.lr)
        
    MODELS.append(model)
    OPTIMIZERS.append(optimizer)

print('\n\n# train iterations: ', len(TR_LOADERS[0]))
print('# valid iterations: ', len(VAL_LOADERS[0]))

for epoch in range(opt.start_epoch, opt.max_epochs+1):

    print_and_save_msg('\n\n', opt.log_file)
    for ii in range(opt.num_val_splits):
        print_and_save_msg(f'training of model {ii+1} in epoch {epoch}:', opt.log_file)
        train_loader = TR_LOADERS[ii]
        valid_loader = VAL_LOADERS[ii]
        model = MODELS[ii]
        optimizer = OPTIMIZERS[ii]

        # train
        macro_f_tr, loss_tr = forward_propagate(train_loader, model, criterion, epoch, optimizer, mode='train')
        # validation
        macro_f_val, loss_val = forward_propagate(valid_loader, model, criterion, epoch, mode='validate')
        
        # save the model
        state = {'epoch': epoch, 'macro_f': macro_f_val, 'loss':loss_val, 'model': model.state_dict(), 'optim' : optimizer.state_dict()}
        chck_path = os.path.join(opt.chckpnt_dir, f'model_{ii+1}_last.pth.tar')
        torch.save(state, chck_path)
        if macro_f_val > opt.best_macro_fs[ii]:
            shutil.copyfile(chck_path, os.path.join(opt.chckpnt_dir, f'model_{ii+1}_best.pth.tar'))
            opt.best_macro_fs[ii] = macro_f_val

