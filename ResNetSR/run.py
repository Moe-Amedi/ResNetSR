import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
from ResNetArch import ResNetSR
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

weights_path = os.path.join(os.path.dirname(__file__), 'best_model_weights.pth')

model = ResNetSR().to(device)
model.load_state_dict(torch.load(weights_path))

if os.path.exists('best_model_weights.pth'):
    print("weights loaded")

model.eval()

global completed
completed = False

def process_image(input_path: str, completed_callback=None) -> Tuple[str, str]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    weights_path = os.path.join(os.path.dirname(__file__), 'best_model_weights.pth')

    model = ResNetSR().to(device)
    model.load_state_dict(torch.load(weights_path))

    if os.path.exists('best_model_weights.pth'):
        print("weights loaded")

    model.eval()

    test_image_lr = Image.open(input_path).convert('RGB')
    test_image_lr = test_image_lr.resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_image_lr_tensor = transform(test_image_lr).unsqueeze(0).to(device)
    test_image_lr = transforms.ToPILImage()(test_image_lr_tensor.cpu().squeeze())

    test_image_sr = model(test_image_lr_tensor).squeeze(0).clamp(0, 1).to(device)

    test_image_sr = transforms.ToPILImage()(test_image_sr.cpu())

    output_path = os.path.splitext(input_path)[0] + '_HR.png'
    test_image_sr.save(output_path)
    print(f"Output image saved to {output_path}")
    
    global completed
    completed = True
    
    return output_path
