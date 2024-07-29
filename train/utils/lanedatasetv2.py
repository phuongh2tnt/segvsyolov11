import collections
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch
import os

class LaneDataset(Dataset):
    def __init__(self, dataset_dir='segatten/train/dataset', subset='train', img_size=480):
        """
        :param dataset_dir: directory containing the dataset
        :param subset: subset that we are working on ('train'/'valid'/'test')
        :param img_size: image size
        """
        super(LaneDataset, self).__init__()
        self.img_size = img_size
        self.resize_img = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_gt = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)
        self.subset = subset
        
        if subset in ['test', 'valid']:
            self.data_path = os.path.join(dataset_dir, subset)
            text_file = os.path.join(dataset_dir, f"train/{subset}.txt")
        else:
            self.data_path = 'segmentattention/test'
            text_file = 'segmentattention/test/test.txt'
        
        with open(text_file, 'r') as f:
            self.filenames = f.read().splitlines()
        print(f'Loaded {subset} subset with {len(self.filenames)} images')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(os.path.join(self.data_path, 'images', 'DSC_' + filename + '.JPG')).convert('RGB')
        gt = Image.open(os.path.join(self.data_path, 'groundtruth', 'DSC_' + filename + '.png'))
        img = self.resize_img(img)
        gt = self.resize_gt(gt)
        img = T.ToTensor()(img)
        gt = torch.from_numpy(np.asarray(gt))
        return img, gt
