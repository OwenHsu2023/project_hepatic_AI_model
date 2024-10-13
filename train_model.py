import io,os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Display PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Display CUDA version
print(f"CUDA version: {torch.version.cuda}")

# Display cuDNN version
print(f"cuDNN version: {torch.backends.cudnn.version()}")

# Example usage: Move a tensor to the selected device
tensor = torch.tensor([1.0, 2.0, 3.0])
tensor = tensor.to(device)
print(f"Tensor device: {tensor.device}")


# Path setting

train_image_folder = os.path.join('dataset', 'train', 'images' )
train_mask_folder = os.path.join('dataset', 'train', 'masks' )
test_image_folder = os.path.join('dataset', 'test', 'images' )
test_mask_folder = os.path.join('dataset', 'test', 'masks' )

# print( os.listdir(test_image_folder))


class MRIDataset(Dataset):
    def __init__(self, image_folder, mask_folder):
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.mask_paths = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return image, mask
    


# Create datasets and dataloaders
train_dataset = MRIDataset(train_image_folder, train_mask_folder)
test_dataset = MRIDataset(test_image_folder, test_mask_folder)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define 3D Unet model
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(64, 1, kernel_size=3, padding=1),
            nn.Conv3d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = UNet3D().cuda()



# Define Dice Loss Function
def dice_loss(pred, target):
    smooth = 1.
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)



# Training and selection model
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter()


num_epochs = 10
best_dice = 0.0
best_model_path = 'best_unet3d_model_20241003_2.pt'

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for images, masks in progress_bar:
        images, masks = images.cuda(), masks.cuda()  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(images)
        
        # Resize mask to match output size if necessary
        if outputs.shape != masks.shape:
            masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
        
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    writer.add_scalar('Loss/train', epoch_loss/len(train_loader), epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}')

    model.eval()
    val_dice_scores = []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.cuda(), masks.cuda()  # Move data to GPU
            outputs = model(images)
            
            # # Resize mask to match output size if necessary
            # if outputs.shape != masks.shape:
            #     masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
            
            dice = 1 - dice_loss(outputs, masks)
            val_dice_scores.append(dice.item())
    avg_val_dice = np.mean(val_dice_scores)
    writer.add_scalar('Dice/val', avg_val_dice, epoch)
    print(f'Validation Dice Score: {avg_val_dice}')

    # Save the best model
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved Best Model with Dice Score: {best_dice}')

writer.close()



# Validation
# model.load_state_dict(torch.load(best_model_path))
# model.eval()
# dice_scores = []
# with torch.no_grad():
#     for images, masks in test_loader:
#         images, masks = images.cuda(), masks.cuda()  # Move data to GPU
#         outputs = model(images)
        
#         # Resize mask to match output size if necessary
#         if outputs.shape != masks.shape:
#             masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
        
#         dice = 1 - dice_loss(outputs, masks)
#         dice_scores.append(dice.item())
# print(f'Average Dice Score on Test Dataset: {np.mean(dice_scores)}')


