import io,os
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import io,os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import nrrd


best_model_path = 'best_unet3d_model_20241003_2.pt'

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
model = UNet3D().cuda()


def inference(model, image_path, output_path):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    image = nib.load(image_path).get_fdata()
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()  # Add batch and channel dimensions, move to GPU
    with torch.no_grad():
        output = model(image)
    output = output.squeeze().cpu().numpy()  # Move output to CPU and convert to numpy
    output = output>0.5  #bool
    output = output.astype(np.int16) # mask
    output_img = nib.Nifti1Image(output, np.eye(4))
    
    

    # print("Output_Type:", type(output))
    # print("Output_Shape:", output.shape)
    # print("Output_image:", type(output_img))
    # print("Output_image_Shape:", output_img.shape)
    
    # binary_mask = (output >0.5)
    
    # # plt.plot(output[:,:,20])
    # plt.imshow(output_img, cmap='gray')
    # plt.show()

    nib.save(output_img, output_path)

# Example usage
cases_file_name = str('17_20240809_14036330')
path_inference_images = os.path.join( 'dataset', 'inference', 'images', cases_file_name+'.nii.gz')
path_inference_masks = os.path.join( 'dataset', 'inference', 'masks', cases_file_name+'_mask.nii.gz')
inference( model, path_inference_images , path_inference_masks )




# nii_file = os.path.join( 'dataset', 'inference', 'masks',cases_file_name+'_mask.nii.gz' )
# nii_img = nib.load(nii_file)
# nii_data = nii_img.get_fdata()

# header = nii_img.header

# header_dict = { key:header[key] for key in header.keys() }

# nrrd_file = os.path.join( 'dataset', 'inference', 'masks', cases_file_name+'_mask.seg.nrrd')
# nrrd.write(nrrd_file, nii_data, header_dict)






















