import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ResNetArch import ResNetSR
from utils import *

if __name__ == '__main__':
    train_hr_dir = 'dataset/DIV2K/Train/Train_hr'
    train_lr_dir = 'dataset/DIV2K/Train/Train_lr'
    val_hr_dir = 'dataset/DIV2K/Valid/valid_hr'
    val_lr_dir = 'dataset/DIV2K/Valid/valid_lr'
    batch_size = 1
    learning_rate = 0.0001
    num_epochs = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    train_dataset = SuperResolutionDataset(train_hr_dir, train_lr_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    val_dataset = SuperResolutionDataset(val_hr_dir, val_lr_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    
    model = ResNetSR().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if os.path.exists('best_model_weights.pth'):
        model.load_state_dict(torch.load('best_model_weights.pth'))
        print("Weights loaded")
        best_weights = model.state_dict()
    else:
        print('No weights detected')
        best_weights = None
    
    log_interval = 100
    train_losses = []
    val_losses = []
    best_metric = 0.248941
    
    for epoch in range(num_epochs):
        model.train()
    
        for batch_idx, (lr_images, hr_images) in enumerate(train_loader):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
    
            optimizer.zero_grad()
    
            predicted_images = model(lr_images)
            loss = ssim_loss(predicted_images, hr_images)
            psnr_val = psnr(predicted_images, hr_images)
    
            loss.backward()
    
            optimizer.step()
    
            
            if batch_idx % log_interval == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},\tPSNR: {:.6f}'.format(
                    epoch, batch_idx * len(lr_images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), psnr_val.item()))
    
                #print('Low-resolution images T')
                #imshow(predicted_images, hr_images)
    
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_psnr = 0
            for lr_images, hr_images in val_loader:
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
                predicted_images = model(lr_images)
                val_loss += ssim_loss(predicted_images, hr_images).item() * lr_images.size(0)
                val_psnr += psnr(predicted_images, hr_images).item() * lr_images.size(0)
    
                #print('Low-resolution images V')
                # imshow(predicted_images, hr_images)
    
            avg_loss = val_loss / len(val_loader.dataset)
            avg_psnr = val_psnr / len(val_loader.dataset)
    
            print('Validation Loss: {:.6f}, Validation PSNR: {:.6f}'.format(avg_loss, avg_psnr))
    

        if avg_loss < best_metric:
            best_metric = avg_loss
            best_weights = model.state_dict()
            torch.save(best_weights, 'best_model_weights.pth')
            print('Saved model weights')
        else:
            print("model did not save")
    
    
    print('Training complete')
