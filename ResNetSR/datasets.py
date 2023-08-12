import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

def get_lr(directory, new_directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            image_hr = Image.open(os.path.join(directory, filename))
            
            # Convert the image to a tensor
            image_hr = transforms.ToTensor()(image_hr)

            # Downsample the image to create a low-resolution version
            image_lr = F.interpolate(image_hr.unsqueeze(0), scale_factor=0.25, mode='bicubic', align_corners=False)

            # Convert the low-resolution tensor back to an image
            image_lr = transforms.ToPILImage()(image_lr.squeeze(0))

            image_lr.save(os.path.join(new_directory, filename))

def resize(directory, new_directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = Image.open(os.path.join(directory, filename))
            
            image = image.resize((1024, 1024))

            image.save(os.path.join(new_directory, filename))


dir = 'dataset/DIV2K'
d = 'dataset/DIV2K/Valid/valid_hr'
nd = 'dataset/DIV2K/Valid/valid_lr'
get_lr(directory=d, new_directory=nd)
# resize(directory=d, new_directory=nd)

# preprocess_div2k(folder_path=dir, image_size=2048)