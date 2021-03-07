
import torch.utils.data as data

import random
from PIL import Image
from . import preprocess
from . import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return rp.readPFM(path)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader, augment = False):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.augment = augment

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]


        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)

        if self.training:

            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]


            # sample = {}
            # sample['left'] = left_img  # [H, W, 3]
            # sample['right'] = right_img
            # sample['disp'] = dataL
            #
            #
            processed = preprocess.get_transform(256, 512, augment=False)
            # sample = processed(sample)
            # left_img = sample['left']
            # right_img = sample['right']
            # dataL = sample['disp']

            left_img   = processed(left_img)
            right_img  = processed(right_img)
            # dataL = processed(right_img)

            return left_img, right_img, dataL
        else:

            processed = preprocess.get_transform(256, 512, augment=False)
            left_img       = processed(left_img)
            right_img      = processed(right_img) 
            return left_img, right_img, dataL


    def __len__(self):
        return len(self.left)
