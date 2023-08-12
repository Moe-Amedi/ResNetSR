import torch
import numpy as np
from torchvision.transforms import ToTensor
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import pytorch_msssim


def psnr(predicted, target):
    mse = torch.mean((predicted - target) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr

def ssim_loss(predicted, target):
    return pytorch_msssim.ssim(predicted, target, data_range=1, size_average=True)

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_filenames = os.listdir(hr_dir)
        self.lr_filenames = os.listdir(lr_dir)

    def __getitem__(self, index):
        hr_path = os.path.join(self.hr_dir, self.hr_filenames[index])
        lr_path = os.path.join(self.lr_dir, self.lr_filenames[index])
        hr_img = ToTensor()(Image.open(hr_path))
        lr_img = ToTensor()(Image.open(lr_path))
        return lr_img, hr_img

    def __len__(self):
        return len(self.hr_filenames)

def imshow(predicted_images, hr_images):
    predicted_images = predicted_images.cpu().detach()
    hr_images = hr_images.cpu().detach()
    # Denormalize the image data
    predicted_images = predicted_images / 2 + 0.5
    hr_images = hr_images / 2 + 0.5

    predicted_npimg = predicted_images.numpy()
    hr_npimg = hr_images.numpy()

    predicted_npimg = np.transpose(predicted_npimg, (0, 2, 3, 1))
    hr_npimg = np.transpose(hr_npimg, (0, 2, 3, 1))

    grid = make_grid(torch.cat([predicted_images, hr_images], dim=0), nrow=2)
    npgrid = grid.cpu().detach().numpy()

    npgrid = np.transpose(npgrid, (1, 2, 0))

    plt.imshow(npgrid)
    plt.show()
