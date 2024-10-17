import sys
import io,os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datetime import date, datetime
import csv

# Make sure call optimizer.py with the parameter alpha, beta, epochs, lr, start datetime
if len(sys.argv) != 6:
    print("Usage: python optimizer.py <alpha> <beta> <epochs> <lr> <datetime>")
    sys.exit(1)

# Get parameter from BATCH and transfer it to float point
alpha_val = float(sys.argv[1])
beta_val = float(sys.argv[2])
epochs_val = int(sys.argv[3])
lr_val = float(sys.argv[4])
CSVfiledate = str(sys.argv[5])
current_datetime = datetime.now().strftime("%m%d_%H%M")
print(f'alpha= {alpha_val} beta= {beta_val} epoch= {epochs_val} lr= {lr_val} date= {CSVfiledate}')

# Path setting
#train_image_folder = os.path.join('dataset', 'train', 'images' )
#train_mask_folder = os.path.join('dataset', 'train', 'masks' )
#test_image_folder = os.path.join('dataset', 'test', 'images' )
#test_mask_folder = os.path.join('dataset', 'test', 'masks' )

train_image_folder = os.path.join('dataset', 'train_20241014_53cases', 'images' )
train_mask_folder = os.path.join('dataset', 'train_20241014_53cases', 'masks' )
test_image_folder = os.path.join('dataset', 'test_20241014_53cases', 'images' )
test_mask_folder = os.path.join('dataset', 'test_20241014_53cases', 'masks' )
# print( os.listdir(test_image_folder))



#Naming CSV 
CSV_filename =  f'dice_table_{CSVfiledate}.csv'

# Open the CSV file in write mode
CSVfile_exists = os.path.isfile(CSV_filename)
CSVfile = open(CSV_filename, mode='a', newline='')
CSVwriter = csv.writer(CSVfile)
# Write the header
if not CSVfile_exists:
    CSVwriter.writerow([{train_image_folder}])
    CSVwriter.writerow([{train_mask_folder}])
    CSVwriter.writerow([{test_image_folder}])
    CSVwriter.writerow([{test_mask_folder}])
    CSVwriter.writerow(['alpha', 'beta', 'epoch', 'LR', 'Dice', 'Loss', 'Datetime'])

# Check if temp model file exists
temp_model_path = os.path.join('model', 'Unet3D_temp.pt')
if os.path.exists(temp_model_path):
    # If it exists, delete it
    os.remove(temp_model_path)
    print(f'old temp model {temp_model_path} has been deleted.')



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
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            # nn.Conv3d(1, 64, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            # nn.Conv3d(64, 64, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            # nn.Conv3d(64, 64, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, padding=1),
            # nn.Conv3d(64, 1, kernel_size=7, padding=1),
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


# Loss function
# criterion = nn.BCELoss()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Flatten the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()
        
        # Calculate Tversky index
        Tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Tversky loss
        loss = 1 - Tversky_index
        
        return loss


# Optimizer
#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=lr_val)
writer = SummaryWriter()


#num_epochs = 1
num_epochs = epochs_val
best_dice = 0.0

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
            
#       criterion = TverskyLoss(alpha=0.9, beta=0.9)
        criterion = TverskyLoss(alpha=alpha_val, beta=beta_val)
        loss = criterion(outputs, masks)
        
        # loss = dice_loss(outputs, masks)
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
        torch.save(model.state_dict(), temp_model_path)
        print(f'Saved Best Model with Dice Score: {best_dice}')



# Write the value to CSV file
CSVwriter.writerow([alpha_val, beta_val, epochs_val, lr_val, best_dice, epoch_loss/len(train_loader), current_datetime])
# Close the file
CSVfile.close()

# Naming the model file with dice_score, alpha, beta, epochs, lr and date
#best_model_path = f"Unet3D_Dice={alpha_val}_A{alpha_val}_B{beta_val}_E{epochs_val}_LR{lr_val}_{current_datetime}.pt"
best_model_path = os.path.join('model', f"Unet3D_Dice={round(best_dice,3)}_A{alpha_val}_B{beta_val}_E{epochs_val}_LR{lr_val}_{current_datetime}.pt" )
os.rename(temp_model_path, best_model_path)
print(f"model name : {best_model_path}")

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


