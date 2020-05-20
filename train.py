# -*- coding: utf-8 -*-

"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering
PHYSICS OF DATA - Department of Physics and Astronomy
COGNITIVE NEUROSCIENCE AND CLINICAL NEUROPSYCHOLOGY - Department of Psychology

A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti

Author: Dr. Matteo Gadaleta

Lab. 02 - Linear regression with artificial neurons

"""

### Dependencies

from torch.utils.data import DataLoader
from torch import nn
import torch

from src.autoencoder import Autoencoder
from src.dataset import MNIST

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import sys
import os


### Set random seed
np.random.seed(42)
torch.manual_seed(42)


### Create dataset

# Retrieve input MNIST dataset
mnist_dataset = MNIST('./data/MNIST.mat')

# Split input MNIST dataset
train_dataset, test_dataset = MNIST.train_test_split(mnist_dataset, test_size=0.2)

# Add noise to train dataset
output_labels, output_images, input_images = train_dataset.data
train_dataset.add_gaussian_noise(output_labels, output_images, input_images, mu=0.5, sigma=0.3)
train_dataset.add_salt_pepper_noise(output_labels, output_images, input_images, prc=0.3)
train_dataset.add_occlusion_noise(output_labels, output_images, input_images, prc=0.3)
train_dataset.clip(scale=(0.0, 1.0))

# Add noise to test dataset
output_labels, output_images, input_images = test_dataset.data
test_dataset.add_gaussian_noise(output_labels, output_images, input_images, mu=0.5, sigma=0.3)
test_dataset.add_salt_pepper_noise(output_labels, output_images, input_images, prc=0.3)
test_dataset.add_occlusion_noise(output_labels, output_images, input_images, prc=0.3)
test_dataset.clip(scale=(0.0, 1.0))

print('Training dataset has %d rows' % len(train_dataset))
print('Test dataset has %d rows' % len(test_dataset))

# train_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# test_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# train_dataset = MNIST(data_root_dir, train=True,  download=True, transform=train_transform)
# test_dataset  = MNIST(data_root_dir, train=False, download=True, transform=test_transform)

### Plot some sample
plt.close('all')
fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    out_label, out_image, in_image = random.choice(train_dataset)
    ax.imshow(in_image.squeeze().numpy(), cmap='gist_gray')
    ax.set_title('Label: %d' % out_label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

### Initialize the network

# Latent space features
encoded_space_dim = 6
# Convolutional encoder
encoder_cnn = nn.Sequential(
    nn.Conv2d(1, 8, 3, stride=2, padding=1),
    nn.ReLU(True),
    nn.Conv2d(8, 16, 3, stride=2, padding=1),
    nn.ReLU(True),
    nn.Conv2d(16, 32, 3, stride=2, padding=0),
    nn.ReLU(True)
)
# Linear encoder
encoder_lin = nn.Sequential(
    nn.Linear(3 * 3 * 32, 64),
    nn.ReLU(True),
    nn.Linear(64, encoded_space_dim)
)
# Linear decoder
decoder_lin = nn.Sequential(
    nn.Linear(encoded_space_dim, 64),
    nn.ReLU(True),
    nn.Linear(64, 3 * 3 * 32),
    nn.ReLU(True)
)
# Convolutional decoder
decoder_cnn = nn.Sequential(
    nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
    nn.ReLU(True),
    nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
    nn.ReLU(True),
    nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
)
# Instantiate the network
net = Autoencoder(
    encoder_cnn=encoder_cnn, encoder_lin=encoder_lin,
    decoder_lin=decoder_lin, decoder_cnn=decoder_cnn,
    lin_to_cnn=(32, 3, 3)
)
# Show the networ
print(net)

### Some examples
# Take an input image (remember to add the batch dimension)
img = test_dataset[0][1].unsqueeze(0)
print('Original image shape:', img.shape)
# Encode the image
img_enc = net.encode(img)
print('Encoded image shape:', img_enc.shape)
# Decode the image
dec_img = net.decode(img_enc)
print('Decoded image shape:', dec_img.shape)


#%% Prepare training

### Define dataloader
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

### Define a loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer
lr = 1e-3 # Learning rate
optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

# Get reference to cpu default device
cpu = torch.device('cpu')
# Select a device among CUDA and CPU
device = torch.device('cuda') if torch.cuda.is_available() else cpu
# Move all the network parameters to the selected device (if they are already on that device nothing happens)
net.to(device)


#%% Network training

# ### Training function
# def train_epoch(net, dataloader, loss_fn, optimizer):
#     # Training
#     net.train()
#     for sample_batch in dataloader:
#         # Extract data and move tensors to the selected device
#         image_batch = sample_batch[0].to(device)
#         # Forward pass
#         output = net(image_batch)
#         loss = loss_fn(output, image_batch)
#         # Backward pass
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         # Print loss
#         print('\t partial train loss: %f' % (loss.data))


# ### Testing function
# def test_epoch(net, dataloader, loss_fn, optimizer):
#     # Validation
#     net.eval() # Evaluation mode (e.g. disable dropout)
#     with torch.no_grad(): # No need to track the gradients
#         conc_out = torch.Tensor().float()
#         conc_label = torch.Tensor().float()
#         for sample_batch in dataloader:
#             # Extract data and move tensors to the selected device
#             image_batch = sample_batch[0].to(device)
#             # Forward pass
#             out = net(image_batch)
#             # Concatenate with previous outputs
#             conc_out = torch.cat([conc_out, out.cpu()])
#             conc_label = torch.cat([conc_label, image_batch.cpu()])
#         # Evaluate global loss
#         val_loss = loss_fn(conc_out, conc_label)
#     return val_loss.data

### Training cycle

# Define 10 sample images
sampled_test_indices = np.random.choice(len(test_dataset), 7)

# Loop through each training epoch
training = True
num_epochs = 10
if training:
    for epoch in range(num_epochs):

        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training
        net.train_epoch(dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim, device=device)

        ### Validation
        val_loss = net.test_epoch(dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim, device=device)
        # Print Validationloss
        print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, num_epochs, val_loss))

        ### Plot progress

        # Initialize plot
        fig, axs = plt.subplots(2, len(sampled_test_indices), figsize=(21, 4))
        # Loop through each column
        for j in range(len(sampled_test_indices)):
            # Get a random row in test dataset
            k = sampled_test_indices[j]
            # Ret j-th row of test dataset
            out_label, out_image, in_image = test_dataset[k]
            # Reshape input image
            in_image = in_image.unsqueeze(0).to(device)
            # Set the network in evaluation mode
            net.eval()
            with torch.no_grad():
                # Make prediction
                pred_image = net(in_image)
            # Make original (not noised) image plot
            axs[0, j].set_title('Original image (label=%d)' % out_label.item())
            axs[0, j].imshow(in_image.to(cpu).squeeze().numpy(), cmap='gist_gray')
            # Make reconstructed image plot
            axs[1, j].set_title('Reconstructed image')
            axs[1, j].imshow(pred_image.to(cpu).squeeze().numpy(), cmap='gist_gray')
        # Make complete plot
        fig.suptitle('Epoch nr %d' % (epoch + 1))
        plt.tight_layout()
        plt.pause(0.1)
        # Save figures
        os.makedirs('autoencoder_progress_%d_features' % encoded_space_dim, exist_ok=True)
        plt.savefig('autoencoder_progress_%d_features/epoch_%d.png' % (encoded_space_dim, epoch + 1))
        plt.show()
        plt.close()

        # Save network parameters
        torch.save(net.state_dict(), 'net_params.pth')


### Network analysis

# Put network on CPU
net.to(cpu)
# Load network parameters
net.load_state_dict(torch.load('net_params.pth', map_location='cpu'))

### Get the encoded representation of the test samples
encoded_samples = []
for sample in tqdm(test_dataset):
    out_label, out_image, in_image = sample
    in_image = in_image.unsqueeze(0)
    # Encode image
    net.eval()
    with torch.no_grad():
        encoded_image  = net.encode(in_image)
    # Append to list
    encoded_samples.append((encoded_image.flatten().numpy(), out_label.item()))


### Visualize encoded space
color_map = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
        5: '#8c564b',
        6: '#e377c2',
        7: '#7f7f7f',
        8: '#bcbd22',
        9: '#17becf'
        }

# Plot just 1k points
encoded_samples_reduced = random.sample(encoded_samples, 1000)
plt.figure(figsize=(12,10))
for enc_sample, label in tqdm(encoded_samples_reduced):
    plt.plot(enc_sample[0], enc_sample[1], marker='.', color=color_map[label])
plt.grid(True)
plt.legend([plt.Line2D([0], [0], ls='', marker='.', color=c, label=l) for l, c in color_map.items()], color_map.keys())
plt.tight_layout()
plt.show()

if encoded_space_dim == 2:
    ### Generate samples

    encoded_value = torch.tensor([8.0, -12.0]).float().unsqueeze(0)

    net.eval()
    with torch.no_grad():
        new_image  = net.decode(encoded_value)

    plt.figure(figsize=(12,10))
    plt.imshow(new_image.squeeze().numpy(), cmap='gist_gray')
    plt.show()
