#Không báo lỗi cảnh bảo numpy
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
            if subset == 'test':
                text_file = 'E:/thanh/ntu_group/phuong/segatten/train/dataset/test/test.txt'
            elif subset == 'valid':
                text_file = 'E:/thanh/ntu_group/phuong/segatten/train/dataset/valid/test.txt'
        else:
            self.data_path = 'segatten/test'
            text_file = 'segatten/test/test.txt'
        
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
        
        # Convert ground truth to a NumPy array and ensure it is writable
        gt_array = np.asarray(gt)
        gt_array = np.copy(gt_array)  # Ensure the array is writable
        
        gt = torch.from_numpy(gt_array).long()  # Convert to PyTorch tensor
        
        return img, gt
