#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes_btcv import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os


class NiftiImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, depth_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder, '*.nii.gz'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot_samples(self, n_slice=15, n_row=4):
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                sample = sample[0]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice])
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d= img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 14, #13 masks (no background)+ noise
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            pairs.append((input_file, target_file))
        return pairs

    def label2value(self, masked_img):
        result_img = masked_img.copy()
        result_img[result_img == LabelEnum.BACKGROUND.value] = 0.0
        result_img[result_img == LabelEnum.SPLEEN.value] = 1/13
        result_img[result_img == LabelEnum.RKID.value] = 2/13
        result_img[result_img == LabelEnum.LKID.value] = 3/13
        result_img[result_img == LabelEnum.GALL.value] = 4/13
        result_img[result_img == LabelEnum.ESO.value] = 5/13
        result_img[result_img == LabelEnum.LIVER.value] = 6/13
        result_img[result_img == LabelEnum.STO.value] = 7/13
        result_img[result_img == LabelEnum.AORTA.value] = 8/13
        result_img[result_img == LabelEnum.IVC.value] = 9/13
        result_img[result_img == LabelEnum.VEINS.value] = 10/13
        result_img[result_img == LabelEnum.PANCREAS.value] = 11/13
        result_img[result_img == LabelEnum.RAD.value] = 12/13
        result_img[result_img == LabelEnum.LAD.value] = 1.0
        #result_img = self.scaler.fit_transform(result_img.reshape(-1, result_img.shape[-1])).reshape(result_img.shape)
        return result_img

    def label2masks(self, masked_img):
        result_img = np.zeros(masked_img.shape  + ( self.input_channel - 1,)) 
        result_img[masked_img == LabelEnum.SPLEEN.value, 0] = 1
        result_img[masked_img == LabelEnum.RKID.value, 1] = 1
        result_img[masked_img == LabelEnum.LKID.value, 2] = 1
        result_img[masked_img == LabelEnum.GALL.value, 3] = 1
        result_img[masked_img == LabelEnum.ESO.value, 4] = 1
        result_img[masked_img == LabelEnum.LIVER.value, 5] = 1
        result_img[masked_img == LabelEnum.STO.value, 6] = 1
        result_img[masked_img == LabelEnum.AORTA.value, 7] = 1
        result_img[masked_img == LabelEnum.IVC.value, 8] = 1
        result_img[masked_img == LabelEnum.VEINS.value, 9] = 1
        result_img[masked_img == LabelEnum.PANCREAS.value, 10] = 1
        result_img[masked_img == LabelEnum.RAD.value, 11] = 1
        result_img[masked_img == LabelEnum.LAD.value, 12] = 1
        return result_img

    def combine_mask_channels(self, masks): # masks: (2, H, W, D), 2->mask channels
        result_img = np.zeros(masks.shape[1:])
        result_img += LabelEnum.SPLEEN.value * masks[0]
        result_img += LabelEnum.RKID.value * masks[1]
        result_img += LabelEnum.LKID.value * masks[2]
        result_img += LabelEnum.GALL.value * masks[3]
        result_img += LabelEnum.ESO.value * masks[4]
        result_img += LabelEnum.LIVER.value * masks[5]
        result_img += LabelEnum.STO.value * masks[6]
        result_img += LabelEnum.AORTA.value * masks[7]
        result_img += LabelEnum.IVC.value * masks[8]
        result_img += LabelEnum.VEINS.value * masks[9]
        result_img += LabelEnum.PANCREAS.value * masks[10]
        result_img += LabelEnum.RAD.value * masks[11]
        result_img += LabelEnum.LAD.value * masks[12]
        return result_img

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file)
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)

        target_img = self.read_image(target_file)
        target_img = self.resize_img(target_img)

        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input':input_img, 'target':target_img}