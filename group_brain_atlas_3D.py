#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import monai
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from monai.networks.nets import unet
from monai.networks.blocks import Warp, DVF2DDF
from monai.config import USE_COMPILED
from tqdm import tqdm
from torch.nn import MSELoss
import numpy as np

# from skimage.io import imshow

from glob import glob


# In[2]:


import nilearn.image as ni


# In[3]:

nofixed = 10
max_epochs = 500000


pretrained = False
epoch_file = "/home/ajoshi/group_reg_200000.pt"
start_epoch = 200000

sub_files = glob("../../HCP_1200/*/T1w/T1w_acpc_dc_restore_brain.nii.gz")[:nofixed]

# In[4]:


num_sub = len(sub_files)
data = np.zeros((num_sub, 64, 64, 64))


# In[5]:


# import cv2
from skimage.transform import resize

for i, sub in enumerate(tqdm(sub_files)):
    v = ni.load_img(sub)
    v = resize(v.get_fdata(), (64, 64, 64), mode="constant")
    data[i] = v

fixed = np.float32(data)
moving = np.float32(0 * data[1, :, :, :])


# In[6]:


# plt.imshow(fixed[1, :, :, 40])


# In[7]:


# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import torch
from torch.nn.modules.loss import _Loss

from monai.utils import LossReduction


def spatial_gradient(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Calculate gradients on single dimension of a tensor using central finite difference.
    It moves the tensor along the dimension to calculate the approximate gradient
    dx[i] = (x[i+1] - x[i-1]) / 2.
    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)

    Args:
        x: the shape should be BCH(WD).
        dim: dimension to calculate gradient along.
    Returns:
        gradient_dx: the shape should be BCH(WD)
    """
    slice_1 = slice(1, -1)
    slice_2_s = slice(2, None)
    slice_2_e = slice(None, -2)
    slice_all = slice(None)
    slicing_s, slicing_e = [slice_all, slice_all], [slice_all, slice_all]
    while len(slicing_s) < x.ndim:
        slicing_s = slicing_s + [slice_1]
        slicing_e = slicing_e + [slice_1]
    slicing_s[dim] = slice_2_s
    slicing_e[dim] = slice_1  # slice_2_e
    return x[slicing_s] - x[slicing_e]  # / 2.0


class GradEnergyLoss(_Loss):
    """
    Calculate the Grad energy based on first-order differentiation of pred using forward finite difference.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        normalize: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            normalize:
                Whether to divide out spatial sizes in order to make the computation roughly
                invariant to image scale (i.e. vector field sampling resolution). Defaults to False.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.normalize = normalize

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BCH(WD)

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if pred.ndim not in [3, 4, 5]:
            raise ValueError(
                f"Expecting 3-d, 4-d or 5-d pred, instead got pred of shape {pred.shape}"
            )
        for i in range(pred.ndim - 2):
            if pred.shape[-i - 1] <= 4:
                raise ValueError(
                    f"All spatial dimensions must be > 4, got spatial dimensions {pred.shape[2:]}"
                )
        if pred.shape[1] != pred.ndim - 2:
            raise ValueError(
                f"Number of vector components, {pred.shape[1]}, does not match number of spatial dimensions, {pred.ndim-2}"
            )

        # first order gradient
        first_order_gradient = [
            spatial_gradient(pred, dim) for dim in range(2, pred.ndim)
        ]

        # spatial dimensions in a shape suited for broadcasting below
        if self.normalize:
            spatial_dims = torch.tensor(pred.shape, device=pred.device)[2:].reshape(
                (1, -1) + (pred.ndim - 2) * (1,)
            )

        energy = 0

        for dim in range(len(first_order_gradient)):
            energy += first_order_gradient[dim] ** 2

        if self.reduction == LossReduction.MEAN.value:
            energy = torch.mean(energy)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            energy = torch.sum(energy)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        return energy


# In[8]:


def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def show_image_list(
    list_images,
    list_titles=None,
    list_cmaps=None,
    grid=True,
    num_cols=2,
    figsize=(20, 10),
    title_fontsize=30,
    out_filename="outpng.png",
):
    """
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    """

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), "%d imgs != %d titles" % (
            len(list_images),
            len(list_titles),
        )

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), "%d imgs != %d cmaps" % (
            len(list_images),
            len(list_cmaps),
        )

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else "Image %d" % (i)
        # cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        im = list_axes[i].imshow(img, cmap="viridis", vmin=0, vmax=850)

        plt.colorbar(im, fraction=0.046, pad=0.04)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_filename)
    plt.close()


# In[9]:


list_images = [moving[:, :, 32]]
name = ["Fixed Image"]
list_titles = list(np.repeat(name, 5))
for i in range(5):
    list_images.append(fixed[i, :, :, 32])
show_image_list(
    list_images=list_images,
    list_titles=["Moving Image"] + list_titles,
    num_cols=2,
    figsize=(10, 10),
    grid=False,
    title_fontsize=15,
    out_filename="Sample_Images.png",
)


# In[10]:


sum = 0
for i in range(nofixed):
    sum += fixed[i]

avg_fixed = sum / nofixed
plt.imshow(avg_fixed[:, :, 32])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title("Average Image")
plt.savefig("Average_image.png")
plt.close()


# In[11]:


reg = unet.UNet(
    spatial_dims=3,  # spatial dims
    in_channels=2,
    out_channels=3,  # output channels (to represent 3D displacement vector field)
    channels=(16, 32, 32, 32, 32),  # channel sequence
    strides=(1, 2, 2, 4),  # convolutional strides
    dropout=0.2,
    norm="batch",
).to("cuda")


if USE_COMPILED:
    warp_layer = Warp(3, padding_mode="zeros").to("cuda")
else:
    warp_layer = Warp("bilinear", padding_mode="zeros").to("cuda")

dvf_2_ddf = DVF2DDF()

reg.train()
optimizerR = torch.optim.Adam(reg.parameters(), lr=0.01)

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, img):
        super(Model, self).__init__()
        self.moving_img = nn.Parameter(img)

    def forward(self):
        return self.moving_img


moving_net = Model(torch.tensor(0 * moving)).to("cuda")
optimizerM = torch.optim.Adam(moving_net.parameters(), lr=0.1)
moving_net.train()


# In[12]:


image_loss = MSELoss()
regularization_loss = GradEnergyLoss()
image_loss.to("cuda")


# In[13]:


print(moving.shape)
print(fixed.shape)
fixedo = fixed.copy()
movingo = moving.copy()


# In[14]:


moving = moving_net()

print(moving[None, :, :, :].shape)
print(torch.tile(moving[None, :, :, :], (fixed.shape[0], 1, 1, 1)).shape)
print(fixed.shape)


# In[15]:


fixed = fixedo[:, None, :, :]
# moving = np.tile(movingo,(fixed.shape[0],1,1,1))


loss_array = []
epoch_array = []

fixed = torch.tensor(fixed).to("cuda")


if pretrained:
    checkpoint = torch.load(epoch_file)
    reg.load_state_dict(checkpoint["reg_dict"])
    moving_net.load_state_dict(checkpoint["moving_net_dict"])

else:
    start_epoch = 0


for epoch in range(start_epoch, max_epochs):
    optimizerM.zero_grad()
    optimizerR.zero_grad()

    moving1 = moving_net()
    moving = torch.tile(moving1[None, :, :, :], (fixed.shape[0], 1, 1, 1))
    moving = moving[:, None, :, :, :]
    input_data = torch.cat((moving, fixed), dim=1)

    dvf = reg(input_data)
    ddf = dvf_2_ddf(dvf)
    moved = warp_layer(moving, ddf)

    imgloss = image_loss(moved, fixed) + 30000 * regularization_loss(ddf)

    imgloss.backward()
    optimizerR.step()
    optimizerM.step()

    loss_value = imgloss.item()
    loss_array.append(loss_value)
    epoch_array.append(epoch)
    # print(f"Epoch: {epoch}, Loss: {loss_value}")

    if np.mod(epoch + 1, 10000) == 0:
        torch.save(
            {
                "reg_dict": reg.state_dict(),
                "moving_net_dict": moving_net.state_dict(),
                "optimizerR_dict": optimizerR.state_dict(),
                "optimizerM_dict": optimizerM.state_dict(),
                "moving1": moving1,
                "loss_array": loss_array,
            },
            f"group_reg_{epoch+1}.pt",
        )
        print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {loss_value}")

        output_images = [
            moving.detach()[0, 0, 22].to("cpu").numpy(),
            fixed[0, 0, 22].to("cpu").numpy(),
            moved[0, 0, 22].detach().to("cpu").numpy(),
            ddf[0, 0, 22].detach().to("cpu").numpy(),
            ddf[0, 1, 22].detach().to("cpu").numpy(),
        ]
        show_image_list(
            list_images=output_images,
            list_titles=[
                "Learnt Moving Image",
                "Fixed Image",
                "Moved Image",
                "Deformation Field",
                "Deformation Field",
            ],
            num_cols=3,
            figsize=(10, 10),
            grid=False,
            title_fontsize=15,
            out_filename=f"output_{epoch}.png",
        )

grp_atlas = moving.detach()[0, 0].to("cpu").numpy()

# In[16]:


output_images = [
    moving.detach()[0, 0, 22].to("cpu").numpy(),
    fixed[0, 0, 22].to("cpu").numpy(),
    moved[0, 0, 22].detach().to("cpu").numpy(),
    ddf[0, 0, 22].detach().to("cpu").numpy(),
    ddf[0, 1, 22].detach().to("cpu").numpy(),
]
show_image_list(
    list_images=output_images,
    list_titles=[
        "Learnt Moving Image",
        "Fixed Image",
        "Moved Image",
        "Deformation Field",
        "Deformation Field",
    ],
    num_cols=3,
    figsize=(10, 10),
    grid=False,
    title_fontsize=15,
)


# In[ ]:
