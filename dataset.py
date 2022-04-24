
import numpy as np
import megengine as mge
import glob
import os
from PIL import Image
import random
from megengine.data.dataset import Dataset
import megengine.data.transform as T
import megengine.data as data

def build_dataset(batch_size,workers,lr_path,gt_path,crop_size=64):
    train_dataset = Dataset_Denoise(lr_path=lr_path,gt_path=gt_path,crop_size=crop_size)
    train_sampler = data.Infinite(
        data.RandomSampler(train_dataset, batch_size=batch_size, drop_last=True)
    )
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
#         transform=T.Compose(
#         [  # Baseline Augmentation for small models
#            T.RandomResizedCrop(32),
#            T.RandomHorizontalFlip(), 
#         ]),
        num_workers=workers,
    )

    return train_dataloader

def build_dataset_all(batch_size,workers,lr_path,gt_path,crop_size=64):
    train_dataset = Dataset_Denoise_all(lr_path=lr_path,gt_path=gt_path,crop_size=crop_size)
    train_sampler = data.Infinite(
        data.RandomSampler(train_dataset, batch_size=batch_size, drop_last=True)
    )
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        num_workers=workers,
    )
    val_dataset = Dataset_val(lr_path=lr_path, gt_path=gt_path)
    val_dataloader = data.DataLoader(
        val_dataset,
        num_workers=0,
    )
    return train_dataloader,val_dataloader


class Crop_Transforms(object):
    """
    图像变换.
    """

    def __init__(self, crop_size):


        self.crop_size = crop_size


    def __call__(self, lr,gt):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """

        # start_x = random.randint(0, (lr.shape[0]-self.crop_size)/4)
        # start_y = random.randint(0, (lr.shape[1] - self.crop_size)/4)
        # start_x = start_x * 4
        # start_y = start_y * 4
        start_x = random.randint(0, lr.shape[0] - self.crop_size)
        start_y = random.randint(0, lr.shape[1] - self.crop_size)
        lr = lr[start_x:start_x+self.crop_size,start_y:start_y+self.crop_size]
        gt = gt[start_x:start_x + self.crop_size, start_y:start_y + self.crop_size]

        return lr,gt

class Random_x_filp(object):
    """
    图像变换.
    """

    def __init__(self):
        1
    def __call__(self,lr,gt):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """
        key = random.randint(0, 1)
        if(key == 1):
            lr = lr[:, ::-1]
            gt = gt[:, ::-1]
        return lr,gt

class Random_y_filp(object):
    """
    图像变换.
    """

    def __init__(self):
        1
    def __call__(self,lr,gt):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """
        key = random.randint(0, 1)
        if(key == 1):
            lr = lr[::-1, :]
            gt = gt[::-1, :]
        return lr,gt

#
# def pack_raw(im):
#     # pack Bayer image to 4 channels
#
#
#     im = np.expand_dims(im, axis=0)
#     img_shape = im.shape
#     H = img_shape[0]
#     W = img_shape[1]
#
#
#     out = np.concatenate((im[:,0:H:2, 0:W:2],
#                           im[:,0:H:2, 1:W:2],
#                           im[:,1:H:2, 1:W:2],
#                           im[:,1:H:2, 0:W:2]), axis=0)
#     return out

# class Pack_raw(object):
#     """
#     图像变换.
#     """
#
#     def __init__(self):
#         1
#
#
#     def __call__(self, im):
#         im = np.expand_dims(im, axis=0)
#         img_shape = im.shape
#         print("img_shape=",img_shape)
#         H = img_shape[1]
#         W = img_shape[2]
#
#
#         out = np.concatenate((im[:,0:H:2, 0:W:2],
#                               im[:,0:H:2, 1:W:2],
#                               im[:,1:H:2, 1:W:2],
#                               im[:,1:H:2, 0:W:2]), axis=0)
#         print("out=", out.shape)
#         return out


class BrightnessContrast(object):
    def __init__(self, norm_num, prob=0.5):
        self.prob = prob
        self.norm_num = norm_num

    def __call__(self, sample, sample_gt):
        h, w = sample.shape[1:]
        if random.random() < self.prob:
            alpha = random.random() + 0.5
            beta = (random.random() * 150 + 50) / self.norm_num
            bbeta = np.full((4, h, w), beta)
            sample = alpha * sample + bbeta
            if sample_gt is not None:
                sample_gt = alpha * sample_gt + bbeta
        return sample, sample_gt


class Dataset_Denoise(Dataset):
    r"""An abstract base class for all datasets.

    __getitem__ and __len__ method are aditionally needed.
    """


    def __init__(self,lr_path,gt_path,crop_size=64):
        content = open(lr_path, 'rb').read()
        self.samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        content = open(gt_path, 'rb').read()
        self.samples_gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

        self.samples_ref = self.samples_ref[1000:-1]
        self.samples_gt = self.samples_gt[1000:-1]

        self.crop = Crop_Transforms(crop_size=crop_size)
        self.filp_x = Random_x_filp()
        self.filp_y = Random_y_filp()
        # self.pack_raw = Pack_raw()
        print("len(self.samples_ref)=",len(self.samples_ref))

    def __getitem__(self, index):
        input = np.float32(self.samples_ref[index, :, :]) * np.float32(1 / 65536)
        gt = np.float32(self.samples_gt[index, :, :]) * np.float32(1 / 65536)
        input, gt = self.crop(input, gt)

        input, gt = self.filp_x(input, gt)
        input, gt = self.filp_y(input, gt)
        # input = mge.tensor(input)
        # gt = mge.tensor(gt)

        input = np.expand_dims(input,axis=0)
        gt = np.expand_dims(gt, axis=0)


        return input,gt


    def __len__(self):
        return len(self.samples_ref)



class Dataset_val(Dataset):
    r"""An abstract base class for all datasets.

    __getitem__ and __len__ method are aditionally needed.
    """


    def __init__(self,lr_path,gt_path):
        content = open(lr_path, 'rb').read()
        self.samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        content = open(gt_path, 'rb').read()
        self.samples_gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

        self.samples_ref_val = self.samples_ref[:1000]
        #print("self.samples_ref_val=",self.samples_ref_val.shape)
        self.samples_gt_val = self.samples_gt[:1000]

        #print("val数据集:len(self.samples_ref)=",len(self.samples_ref_val))

    def __getitem__(self, index):
        input = np.float32(self.samples_ref_val[index, :, :]) * np.float32(1 / 65536)
        gt = np.float32(self.samples_gt_val[index, :, :]) * np.float32(1 / 65536)
        # input = mge.tensor(input)
        # gt = mge.tensor(gt)
        input = np.expand_dims(input,axis=0)
        gt = np.expand_dims(gt, axis=0)

        return input,gt


    def __len__(self):
        #print("len(self.samples_ref_val)=",len(self.samples_ref_val))
        return len(self.samples_ref_val)


class Dataset_Denoise_all(Dataset):
    r"""An abstract base class for all datasets.

    __getitem__ and __len__ method are aditionally needed.
    """


    def __init__(self,lr_path,gt_path,crop_size=64):
        content = open(lr_path, 'rb').read()
        self.samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        content = open(gt_path, 'rb').read()
        self.samples_gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

        # self.samples_ref = self.samples_ref[1000:-1]
        # self.samples_gt = self.samples_gt[1000:-1]

        self.crop = Crop_Transforms(crop_size=crop_size)
        self.filp_x = Random_x_filp()
        self.filp_y = Random_y_filp()
        print("len(self.samples_ref)=",len(self.samples_ref))

    def __getitem__(self, index):
        input = np.float32(self.samples_ref[index, :, :]) * np.float32(1 / 65536)
        gt = np.float32(self.samples_gt[index, :, :]) * np.float32(1 / 65536)
        input, gt = self.crop(input, gt)
        input, gt = self.filp_x(input, gt)
        input, gt = self.filp_y(input, gt)
        # input = mge.tensor(input)
        # gt = mge.tensor(gt)
        input = np.expand_dims(input,axis=0)
        gt = np.expand_dims(gt, axis=0)

        return input,gt


    def __len__(self):
        return len(self.samples_ref)
