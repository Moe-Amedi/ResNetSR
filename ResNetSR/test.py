import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from ResNetArch import ResNetSR
from utils import psnr, ssim_loss
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = ResNetSR().to(device)
model.load_state_dict(torch.load('best_model_weights.pth'))

if os.path.exists('best_model_weights.pth'):
    print("weights loaded")

model.eval()

hr_image = Image.open('dataset/DIV2K/Valid/valid_hr/0882.png')
hr_image_tensor = transforms.ToTensor()(hr_image).unsqueeze(0).to(device)

test_image_lr = Image.open('test_Images/0882.png').convert('RGB')
test_image_lr = test_image_lr.resize((256, 256))
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_image_lr_tensor = transform(test_image_lr).unsqueeze(0).to(device)
test_image_lr = transforms.ToPILImage()(test_image_lr_tensor.cpu().squeeze())

transform_hr = transforms.Compose([
    transforms.ToTensor(),
])
test_image_hr_tensor = transform_hr(hr_image).unsqueeze(0).to(device)

test_image_sr = model(test_image_lr_tensor).squeeze(0).clamp(0, 1).to(device)

# print(test_image_sr.size)
# print(test_image_hr_tensor.size())

test_image_hr_tensor = test_image_hr_tensor.squeeze().cpu()

psnr_val = psnr(test_image_sr, hr_image_tensor).item()
print('PSNR: {:.4f} dB'.format(psnr_val))

ssim_val = ssim_loss(test_image_sr.unsqueeze(0), hr_image_tensor).item()
print('SSIM: {:.4f}'.format(ssim_val))

test_image_sr = transforms.ToPILImage()(test_image_sr.cpu())

test_image_sr.save('results/image82.png')
