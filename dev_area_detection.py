import io,os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


# ---------------------------------------------------------
# Sample binary image data (1s and 0s)
# binary_image = np.array([[...]], dtype=np.uint8)  # Replace with your binary image data

# Path
path_mask = os.path.join( 'dataset', 'inference', 'masks', '17_20240809_14036330_mask_AI_inference.nii.gz')
path_raw_image = os.path.join( 'dataset', 'inference', 'images', '17_20240809_14036330.nii.gz')

# Load nii file
mask = nib.load(path_mask).get_fdata()
raw_image = nib.load(path_raw_image).get_fdata()
binary_image = np.array(mask, dtype=np.uint8)  #3D


# OpenCV Find contours, CV2
# contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(binary_image[:,:,15], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]
# Find the largest contour
max_area_index = np.argmax(areas)
largest_contour = contours[max_area_index]

# Draw contours on the original binary image
image_with_contours = cv2.cvtColor(binary_image[:,:,15], cv2.COLOR_GRAY2BGR)

# image_contours_all = cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)
image_contours_max_area = cv2.drawContours(image_with_contours, [largest_contour], -1, (255, 0, 0), 1)



# Fill contour


# Save as nii.gz



# Count Dice Score vs raw image






# Show the result using matplotlib
# plt.subplot(131)
# # plt.imshow(raw_image)
# plt.subplot(132)
# plt.imshow(image_contours_all)
plt.subplot(133)
plt.imshow(image_contours_max_area)
plt.title("Detected Closed Areas")
plt.show()



