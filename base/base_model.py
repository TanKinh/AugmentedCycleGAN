import os
import torch
from collections import OrderedDict
import numpy as np 

class BaseModel():
    
    def aaa():
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    # load and print networks
    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [networks.get_cheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain orpt.continute_train:
            print('cotinute with epoch', opt.which_epoch)
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def create_fake_B(self):
        pass

    # using when testign time, no_grad()
    def test(self):
        with torch.no_grad():
            self.forward()

    def test_fake_B(self):
        with torch.no_grad():
            self.create_fake_B()

    # get image paths 
    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        print('learnign rate before update :', self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %7f' % lr)

    # save model to the disk
    def save_network(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_name)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_fileName = '%s_net_%s.pth' % (whoch_epoch, name)
                save_path = os.path.join(slef.save_dir, save_filename)
                print('filename :', save_path)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                state_dict = torch.load(save_path, map_location=str(self.device))
                for key in list(state_dict.keys()):
                    self.__path_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        print('---------- Networks initialized ----------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters :%3 M' %(name, num_params / 1e6))
        print('------------------------------------------')
