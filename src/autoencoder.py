# Dependencies
from torch import nn
import torch

# from tqdm import tqdm
import numpy as np
# import json


class Autoencoder(nn.Module):

    def __init__(
        self, encoder_cnn, encoder_lin,
        decoder_lin, decoder_cnn, lin_to_cnn
    ):
        # Parent constructor (mandatory)
        super().__init__()

        # Store linear to convolutional reshape size
        self.lin_to_cnn = lin_to_cnn

        # Encoder layer deconstructs image and extracts features
        # Decoder layer reconstructs image from features
        # Encoder and decoder are specular

        # Encoder - convolutional
        self.encoder_cnn = encoder_cnn
        # self.encoder_cnn = nn.Sequential(
        #     nn.Conv2d(1, 8, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 16, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, 3, stride=2, padding=0),
        #     nn.ReLU(True)
        # )
        # Encoder - linear
        self.encoder_lin = encoder_lin
        # self.encoder_lin = nn.Sequential(
        #     nn.Linear(3 * 3 * 32, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, encoded_space_dim)
        # )

        # Decoder - linear
        self.decoder_lin = decoder_lin
        # self.decoder_lin = nn.Sequential(
        #     nn.Linear(encoded_space_dim, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 3 * 3 * 32),
        #     nn.ReLU(True)
        # )
        # Decoder - convolutional
        self.decoder_cnn = decoder_cnn
        # self.decoder_cnn = nn.Sequential(
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        # )

    @property
    def lin_to_cnn(self):
        return self.lin_to_cnn_

    @lin_to_cnn.setter
    def lin_to_cnn(self, lin_to_cnn):
        self.lin_to_cnn_ = list(lin_to_cnn)

    @property
    def cnn_to_lin(self):
        return np.prod(self.lin_to_cnn)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([-1, self.cnn_to_lin])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x

    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, *self.lin_to_cnn])
        # Apply transposed convolutions
        x = self.decoder_cnn(x)
        x = torch.sigmoid(x)
        return x

    # Train network on a single epoch
    def train_epoch(self, dataloader, loss_fn, optimizer, device):
        # Set network in training mode
        self.train()
        # Loop through each batch in DataLoader
        for sample_batch in dataloader:
            # Extract images and associated label
            out_labels, out_images, in_images = sample_batch
            # Forward pass
            pred_images = self.__call__(in_images.to(device))
            loss = loss_fn(pred_images, out_images.to(device))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print loss
            print('\t partial train loss: %f' % (loss.data))


    # Test network on a single epoch
    def test_epoch(self, dataloader, loss_fn, optimizer, device):
        # Set network in evaluation mode (e.g. disable dropout)
        self.eval()
        # Disable gradient computation
        with torch.no_grad():
            # Initialize true and predicted images containers
            conc_true = torch.Tensor().float().cpu()
            conc_pred = torch.Tensor().float().cpu()
            # # Initialize validation loss container
            # valid_loss = list()
            # Loop through each batch in dataloader
            for i, sample_batch in enumerate(dataloader):
                # Extract data and move tensors to the selected device
                out_labels, out_images, in_images = sample_batch
                # Forward pass
                pred_images = self.__call__(in_images.to(device))
                # Concatenate with previous outputs
                conc_pred = torch.cat([conc_pred, pred_images.cpu()])
                conc_true = torch.cat([conc_true, out_images.cpu()])
                # print('Test batch %d' % i)
                # # Append validation losses for current batch
                # print(loss_fn(pred_images.cpu(), out_images.cpu()))
                # valid_loss = valid_loss +  loss_fn(pred_images.cpu(), out_images.cpu(), reduction='none').tolist()
            # Evaluate global loss
            val_loss = loss_fn(conc_pred, conc_true)
        # Return mean loss
        return val_loss.data












# class Autoencoder(nn.Module):
#     """
#     Autoencoder class is composed of:
#     1. Convolutional encoder: turns 2D image into linear features, mantaining
#     spatial correlation
#     2. Linear encoder: learns encoding from linearized fetaures
#     3. Linear decoder: learns decoding from linearized features
#     4. Convolutional decoder: turns linearized features back to 2D image,
#     trying to rebuild it
#
#     Conv2d/ConvTranspose2d first/last layer has input channel = 1, since it
#     takes a grey-channel images as input. Other layers are set according to
#     previous layers, instead
#
#     Input shape is a tuple (batch size, num channels, height, width)
#     """
#
#     # Constructor
#     def __init__(
#         self, encoder_cnn=None, encoder_lin=None,
#         decoder_lin=None, decoder_cnn=None, lin_to_cnn=None
#     ):
#         # Call parent constructor
#         super().__init__()
#         # Set neural network topology
#         self.encoder_cnn = encoder_cnn
#         self.encoder_lin = encoder_lin
#         self.decoder_lin = decoder_lin
#         self.decoder_cnn = decoder_cnn
#         # Define linear to convolutional shape transformation
#         self.lin_to_cnn = lin_to_cnn
#
#     @property
#     def lin_to_cnn(self):
#         return self.lin_to_cnn_
#
#     @lin_to_cnn.setter
#     def lin_to_cnn(self, lin_to_cnn):
#         self.lin_to_cnn_ = list(lin_to_cnn)
#
#     @property
#     def cnn_to_lin(self):
#         return np.prod(self.lin_to_cnn)
#
#     # Go through encoding (convolutional, linear) layers
#     def encode(self, x):
#         # Apply convolutions to input tensor
#         x = self.encoder_cnn(x)
#         # Flatten tensor to input linear layers (0-th size is batch size)
#         x = x.view([-1, self.cnn_to_lin])
#         # Apply linear layers
#         x = self.encoder_lin(x)
#         # Return output
#         return x
#
#     # Go through decoding (linear, convolutional) layers
#     def decode(self, x):
#         # Apply linear layers
#         x = self.decoder_lin(x)
#         # Reshape
#         x = x.view([-1, *self.lin_to_cnn])
#         # Apply transposed convolutions
#         x = self.decoder_cnn(x)
#         # Apply activation
#         x = torch.sigmoid(x)
#         return x
#
#     # Make forward call: first encode then decode
#     def forward(self, x):
#         # Encode
#         x = self.encode(x)
#         # Decode
#         x = self.decode(x)
#         # Return decoded result
#         return x
#
#     # Train autodecoder on an images batch
#     def train_batch(
#         self, batch, loss_fn, optimizer,
#         device=torch.device('cpu')
#     ):
#         # Extract images batch (X) and labels batch (y)
#         X, y = batch
#         # Move batch to correct device
#         X = X.to(device)
#         # Make forward pass
#         out = self(X[1, :, :, :])  # Compute output wrt noised image
#         loss = loss_fn(out, X[0, :, :, :])  # Compute loss wrt original image
#         # Backward pass
#         optimizer.zero_grad()  # Delete previous gradient
#         loss.backward()  # Compute loss gradient
#         optimizer.step()  # Update weights
#         # Return loss
#         return out, loss
#
#     # Test autodecoder on an images batch
#     def test_batch(self, batch, loss_fn, device=torch.device('cpu')):
#         # Disable gradient computation
#         with torch.no_grad():
#             # Extract images batch (X) and labels batch (y)
#             X, y = batch
#             # Move batch to correct device
#             X = X.to(device)
#             # Make forward pass
#             out = self(X[1, :, :, :])  # Compute output wrt noised image
#             loss = loss_fn(out, X[0, :, :, :])  # Compute loss wrt original image
#             # Return loss
#             return out, loss
#
#     # Train autoencoder on a dataset for a given number of epochs
#     def train_epochs(
#         self, num_epochs, train_dataloader, test_dataloader,
#         optimizer, loss_fn, verbose=False,
#         model_path=None, epochs_path=None, save_epochs=0,
#         device=torch.device('cpu'),
#     ):
#         # Define train and test loss
#         train_loss, test_loss = list(), list()
#         # Define epochs iterable
#         epochs_iterable = list(range(num_epochs))
#         # Verbose case: use tqdm instead of simple list
#         if verbose:
#             epochs_iterable = tqdm(range(num_epochs))
#         # Loop through each epoch
#         for epoch in epochs_iterable:
#             # Initialize current epoch training and test loss
#             epoch_train_loss, epoch_test_loss = list(), list()
#             # Set the network in training mode
#             self.train()
#             # Loop through every batch in training dataset
#             for batch in train_dataloader:
#                 # Train the autoencoder on current batch
#                 _, loss = self.train_batch(
#                     batch=batch, loss_fn=loss_fn, optimizer=optimizer,
#                     device=device
#                 )
#                 # Store current training loss
#                 epoch_train_loss.append(loss.item())
#                 # # Verbose output
#                 # if verbose:
#                 #     epochs_iterable.write(
#                 #         'Current training loss is {:.04f}'.format(loss.item())
#                 #     )
#             # Set the network in evaluation/test mode
#             self.eval()
#             # Loop through every batch in evaluation dataset
#             for batch in test_dataloader:
#                 # Test the autoencoder on current batch
#                 _, loss = self.test_batch(
#                     batch=batch, loss_fn=loss_fn, device=device
#                 )
#                 # Store current training loss
#                 epoch_test_loss.append(loss.item())
#                 # # Verbose output
#                 # if verbose:
#                 #     print(
#                 #         'Current test loss is {:.04f}'.format(loss.item())
#                 #     )
#             # Save current epoch train and test losses
#             train_loss = train_loss + epoch_train_loss
#             test_loss = test_loss + epoch_test_loss
#             # Save parameters and loss after a certain number of epochs
#             if save_epochs and ((epoch + 1) % save_epochs == 0):
#                 # Save model parameters
#                 torch.save(self.state_dict(), model_path)
#                 # Save losses
#                 with open(epochs_path, 'w') as epochs_file:
#                     # Store either the loss and the number of epochs until now
#                     json.dump({
#                         'train_loss': train_loss,
#                         'test_loss': test_loss,
#                         'num_epochs': epoch + 1
#                     }, epochs_file)
#         # Return training and testing loss
#         return train_loss, test_loss
