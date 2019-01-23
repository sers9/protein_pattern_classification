import argparse
from datetime import datetime
import os
import torch.optim as optim


def training_options():
	parser = argparse.ArgumentParser()
	parser.add_argument("--max_epochs", type=int, default=50, help="")
	parser.add_argument("--start_epoch", type=int, default=1, help="")
	parser.add_argument("--lr", type=float, default=1e-3, help="")
	parser.add_argument("--data_dir", type=str, default='./human_atlas/train', help="")
	parser.add_argument("--images_list_file", type=str, default='./train_split.csv', help="")
	parser.add_argument("--val_ratio", type=float, default=0.2, help="fraction of the validation dataset")
	parser.add_argument("--num_val_splits", type=int, default=3, help="# validation splits")	
	parser.add_argument('--no_shuffle', action='store_true', help='if true, doesn\'t shuffle the dataset')
	parser.add_argument('--random_seed', type=int, default=123, help='')
	parser.add_argument("--batch_size", type=int, default=16, help="")
	parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
	parser.add_argument("--n_class", type=int, default=28, help="")
	parser.add_argument("--chckpnt_dir", type=str, default='./checkpoints', help="")
	opt = parser.parse_args()
	opt.chckpnt_dir = './checkpoints'
	os.mkdir(opt.chckpnt_dir)
	opt.log_file = '/log_train.txt' # training log file
	opt.best_macro_fs = [0.0, 0.0, 0.0]

	return opt


def test_options():
	parser = argparse.ArgumentParser()
	parser.add_argument("--chckpnt_dir", type=str, default='./chckpnt_dir', help="experiment directory")
	parser.add_argument("--data_dir", type=str, default='./data/human_atlas/train', help="")
	parser.add_argument("--image_list", type=str, default='./test_split.csv', help="")
	parser.add_argument("--batch_size", type=int, default=4, help="")
	parser.add_argument("--n_class", type=int, default=28, help="")
	parser.add_argument("--chckpnt_folder", type=str, default='./checkpoints', help="")
	parser.add_argument("--model_type", type=str, default='best', help="")
	opt = parser.parse_args()
	opt.log_file = "log_test.txt"
	opt.result_dir = "./results"

	return opt