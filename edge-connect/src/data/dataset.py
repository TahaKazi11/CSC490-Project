import glob
import os
import random
import scipy
import torch
import torchvision
import imageio.v3 as iio
import numpy as np
import torchvision.transforms.functional as F

from PIL import image
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from torch.utils.data import DataLoader

'''
https://github.com/AndreFagereng/polyp-GAN/blob/main/edge-connect/src/dataset.py
'''
class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = cfg.SIGMA
        self.edge = cfg.EDGE
        self.mask = cfg.MASK
        self.nms = cfg.NMS

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if cfg.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return 0
        
    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size
        if self.mask == 6:
            size = 256
        
        # load image
        img = iio.imread(self.data[index], index=None)

        # ensure img is RGB
        if (img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if need be
        if size != 0:
            img = self.resize(img, size, size)

        img_gray = rgb2gray(img)

        # load maskilen
        mask_filename = self.load_name(index)
        mask_path = '/'.join(self.mask_data[0].split('/')[0:-1]) + '/' + mask_filename
        mask = self.load_mask(img, index, mask_filename)

        # load edge
        edge = self.load_edge(img_gray, index, mask)

        # TODO: work on new augments
        # if self.augment:
            
        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)
    
    def load_edge(self, img, index, mask):
        sigma = self.sigma

        mask = None

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)
            return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = iio.imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            #if self.nms == 1:
            #    edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge

    def create_mask(self, width, height, mask_width, mask_height, x=None, y=None):
        mask = np.zeros((height, width))
        mask_x = x if x is not None else random.randint(0, width - mask_width)
        mask_y = y if y is not None else random.randint(0, height - mask_height)
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return mask

    def load_mask(self, img, index, f):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        if f is not None:
            mask = iio.imread(f)
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return self.create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return self.create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]
        
        return Image.fromarray(img).resize(size=(height, width))

    def load_flist(self, flist):

        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
       

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


