import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = self.upconv_block(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder with proper upsampling and concatenation
        upconv4 = self.upconv4(bottleneck)
        dec4 = torch.cat((F.interpolate(upconv4, size=enc4.shape[2:], mode='bilinear', align_corners=True), enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        upconv3 = self.upconv3(dec4)
        dec3 = torch.cat((F.interpolate(upconv3, size=enc3.shape[2:], mode='bilinear', align_corners=True), enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        upconv2 = self.upconv2(dec3)
        dec2 = torch.cat((F.interpolate(upconv2, size=enc2.shape[2:], mode='bilinear', align_corners=True), enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        upconv1 = self.upconv1(dec2)
        dec1 = torch.cat((F.interpolate(upconv1, size=enc1.shape[2:], mode='bilinear', align_corners=True), enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final Convolution
        return self.final_conv(dec1)



class ImagePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.input_images = sorted(glob.glob(os.path.join(root_dir, '*_in.jpg')))
        self.output_images = sorted(glob.glob(os.path.join(root_dir, '*_out.jpg')))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = self.input_images[idx]
        output_image_path = self.output_images[idx]

        input_image = Image.open(input_image_path).convert('L')  # Convert to grayscale
        output_image = Image.open(output_image_path).convert('L')  # Convert to grayscale

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

# Transforms
transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
])

# Dataset
dataset = ImagePairDataset(root_dir='/home/ais/USAN/src/PySocialForce/images/in_out', transform=transform)

# Calculate lengths for training and validation datasets
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# Model, Loss, and Optimizer
model = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

output_dir = os.path.join(os.path.dirname('/home/ais/USAN/src/PySocialForce/images/'), 'EncDec_validation_outputs')
os.makedirs(output_dir, exist_ok=True)

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    running_train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * inputs.size(0)
    
    epoch_train_loss = running_train_loss / len(train_loader.dataset)





    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_val_loss += loss.item() * inputs.size(0)
            
            # Save the outputs as images
            for j in range(inputs.size(0)):
                input_img = inputs[j].cpu().numpy().squeeze()
                target_img = targets[j].cpu().numpy().squeeze()
                output_img = outputs[j].cpu().numpy().squeeze()
                
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(input_img, cmap='gray')
                ax[0].set_title('Input')
                ax[1].imshow(target_img, cmap='gray')
                ax[1].set_title('Target')
                ax[2].imshow(output_img, cmap='gray')
                ax[2].set_title('Output')
                
                for a in ax:
                    a.axis('off')
                
                plt.savefig(os.path.join(output_dir, f'val_epoch{epoch+1}_batch{i}_img{j}.png'))
                plt.close()
    
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

# Save the Model
torch.save(model.state_dict(), 'unet_model.pth')
