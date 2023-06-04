import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import monai
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChannel,
    ScaleIntensityd,
    ToTensor,
    EnsureChannelFirstd,
    Resized,
)
from monai.networks.layers import Norm
from monai.networks.nets import UNet
import glob

# Set random seed for reproducibility
torch.manual_seed(123)

# Define the training dataset
# Assuming you have T1 and T2 images stored in separate directories
train_image_dir_t1 = glob.glob("/home/ajoshi/HCP_1200/*/T1w/T1*")
train_image_dir_t2 = glob.glob("/home/ajoshi/HCP_1200/*/T1w/T2*")
train_image_dir_t1 = train_image_dir_t1[:100]
train_image_dir_t2 = train_image_dir_t2[:100]

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_image_dir_t1, train_image_dir_t2)
]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

train_data = Dataset(
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

# Create a DataLoader for training data
batch_size = 2
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=4
)

# Define the network model with 1x1 convolutions for contrast conversion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContrastConversionModel(nn.Module):
    def __init__(self):
        super(ContrastConversionModel, self).__init__()
        self.unet = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.conv1x1 = nn.Conv3d(1, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.unet(x)
        x = self.conv1x1(x)
        return x


model = ContrastConversionModel().to(device)

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_data in train_loader:
        inputs, targets = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss}")

# Save the trained model
torch.save(model.state_dict(), "path_to_save_model.pth")


for batch_data in train_loader:
    inputs, targets = batch_data["image"].to(device), batch_data["label"].to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    epoch_loss += loss.item()

    loss.backward()
    optimizer.step()

avg_epoch_loss = epoch_loss / len(train_loader)
print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss}")

