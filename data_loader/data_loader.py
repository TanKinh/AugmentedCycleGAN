from base.base_data_loader import BaseDataLoader
import os.path
from PIL import Image
import random
import torch
import torch.utils.data
import torchvision.transforms as transforms
import io

class DataLoader(BaseDataLoader):
    def __init__(self, opt, subset, unaligned, batchSize,
                 shuffle=False, fraction=1., load_in_mem=True, drop_last=False):
        self.opt = opt
        self.dataset = Font2Hand(opt, subset, unaligned, fraction, load_in_mem)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=shuffle,
            num_workers=0,
            drop_last=drop_last)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

class Font2Hand(object):
    def __init__(self, opt, subset, unaligned, fraction, load_in_mem):
        self.opt = opt
        self.root = opt.dataroot
        self.subset = subset
        self.unaligned = unaligned
        self.fraction = fraction
        self.load_in_mem = load_in_mem

        if subset in ['dev', 'train']:
            self.dir_A = os.path.join(self.root, 'trainA')
            self.dir_B = os.path.join(self.root, 'trainB')
            self.image_path = os.path.join(self.root, 'valB')
            self.image_path = sorted(make_dataset(self.image_path))
        elif subset == 'val':
            self.dir_A = os.path.join(self.root, 'valA')
            self.dir_B = os.path.join(self.root, 'valB')
            self.image_path = os.path.join(self.root, 'valB')
            self.image_path = sorted(make_dataset(self.image_path))

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

        # return only fraction of subset
        subset_size = int(len(self.A_paths)*fraction)
        self.A_paths = self.A_paths[:subset_size]
        self.B_paths = self.B_paths[:subset_size]

        subset_size = int(len(self.A_paths) * fraction)
        self.A_paths = self.A_paths[:subset_size]
        self.B_paths = self.B_paths[:subset_size]

        if load_in_mem:
            mem_A_paths = []
            mem_B_paths = []
            for A, B in zip(self.A_paths, self.B_paths):
                with open(A, 'rb') as fa:
                    mem_A_paths.append(io.BytesIO(fa.read()))
                with open(B, 'rb') as fb:
                    mem_B_paths.append(io.BytesIO(fb.read()))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.unaligned:#shuffe
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_B = index % self.A_size
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        if self.opt.input_nc == 1:# RGB to gray
            tmp = A_img[0, ...] * 0.299 + A_img[1, ...] * 0.587 + A_img[2, ...] * 0.114
            A_img = tmp.unsqueeze(0)

        if self.opt.output_nc == 1:
            tmp = B_img[0, ...] * 0.299 + B_img[1, ...] * 0.587 + B_img[2, ...] * 0.114
            B_img = tmp.unsqueeze(0)
        
        return {'A': A_img, 'B': B_img, 'A_paths' : A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

def get_transform(opt):
    transform_list = [transforms.Resize([64, 64], Image.BICUBIC)]
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    # print(os.path.isdir(dir), '%s is not a valid directory' % dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images