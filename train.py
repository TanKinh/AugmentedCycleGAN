import sys
import os
from shutil import copyfile
import glob
import random
import torch
import numpy as np
from options.train_options import TrainOptions
from data_loader.data_loader import DataLoader
from models.model import AugmentedCycleGAN

def write_log_to_file(out_f, message):
    out_f.write(message+"\n")
    out_f.flush()
    print(message)

def train_model():
    opt = TrainOptions().parse(sub_dirs=['vis_multi','vis_cycle','vis_latest','train_vis_cycle'])
    out_f = open("%s/results.txt" % opt.expr_dir, 'w')
    use_gpu = len(opt.gpu_ids) > 0

    if opt.seed is not None:
        print ("using random seed:", opt.seed)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if use_gpu:
            torch.cuda.manual_seed_all(opt.seed)

    train_data_loader = DataLoader(opt, subset='train', unaligned=True, batchSize=opt.batchSize)
    test_data_loader = DataLoader(opt, subset='val', unaligned=False, batchSize=opt.batchSize) # test set
    dev_data_loader = DataLoader(opt, subset='dev', unaligned=False, batchSize=opt.batchSize)
    dev_cycle_loader = DataLoader(opt, subset='dev', unaligned=False, batchSize=opt.batchSize)

    train_dataset = train_data_loader.load_data()
    dataset_size = len(train_data_loader)
    write_log_to_file(out_f, '#training images = %d' % dataset_size)
    vis_inf = False

    test_dataset = test_data_loader.load_data()
    write_log_to_file(out_f, '#test images = %d' % len(test_data_loader))

    dev_dataset = dev_data_loader.load_data()
    write_log_to_file(out_f, '#dev images = %d' % len(dev_data_loader))

    model = AugmentedCycleGAN(opt)

    
if __name__ == "__main__":
    train_model()
