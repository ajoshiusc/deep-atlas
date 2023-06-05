import os
import numpy as np
import torch
import monai
from monai.data import Dataset, NiftiSaver
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized
from monai.networks.layers import Norm
from monai.networks.nets import UNet
import glob
import torch.nn as nn

# Define the training dataset
test_image_dir_t1 = glob.glob("/home/ajoshi/HCP_1200/*/T1w/T1*")
test_image_dir_t2 = glob.glob("/home/ajoshi/HCP_1200/*/T1w/T2*")
test_image_dir_t1 = test_image_dir_t1[:100]
test_image_dir_t2 = test_image_dir_t2[:100]

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(test_image_dir_t1, test_image_dir_t2)
]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

infer_data = Dataset(
    data=data_dicts,
    transform=Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"], spatial_size=[64, 64, 64]),
            ScaleIntensityd(keys=["image", "label"]),
        ]
    ),
)

# Define the output directory to save the converted T2 images
output_dir = 'outT2file'
os.makedirs(output_dir, exist_ok=True)

# Create a NiftiSaver to save the converted T2 images
saver = NiftiSaver(output_dir)

# Define the network model with 1x1 convolutions for contrast conversion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContrastConversionModel(nn.Module):
    def __init__(self):
        super(ContrastConversionModel, self).__init__()
        self.conv1x1_10 = nn.Conv3d(1, 10, kernel_size=1, stride=1)
        self.conv1x1 = nn.Conv3d(10, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1x1_10(x)
        x = self.conv1x1(x)
        return x

# Load the trained model
#model = ContrastConversionModel()

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 1)
        self.fc1 = nn.Linear(2**3*64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.LeakyReLU(),
        #nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.relu(out)
        out = self.conv_layer2(out)
        out = self.relu(out)

        # out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.batch(out)
        # out = self.drop(out)
        # out = self.fc2(out)
        
        return out



model = CNNModel().to(device) #ContrastConversionModel().to(device)

model.load_state_dict(torch.load('t12t2_model.pth'))
model.to(device)
model.eval()

# Run inference on the input images
for i in range(len(infer_data)):
    input_image = infer_data[i]['image'].to(device)  # Get the input image tensor
    t2_image = infer_data[i]['label'].to(device)  # Get the input image tensor

    output = model(input_image)  # Perform inference

    output_image = monai.utils.first(output).detach().cpu().numpy()  # Convert the output tensor to numpy array
    t2_image = monai.utils.first(t2_image).detach().cpu().numpy()  # Convert the output tensor to numpy array

    # Save the output as a NIfTI image
    pth,fname = os.path.split(infer_data.data[i]['image'])
    image_name = os.path.basename(fname)
    output_path = os.path.join(output_dir, f"{image_name}_T2.nii.gz")
    saver.save(output_image[None,], meta_data={'filename_or_obj':'t2_converted'})
    saver.save(t2_image[None,], meta_data={'filename_or_obj':'t2'})