# Dependencies
from torch import nn
import torch

from tqdm import tqdm
import numpy as np
import json


class Autoencoder(nn.Module):
    """
    Autoencoder class is composed of:
    1. Convolutional encoder: turns 2D image into linear features, mantaining
    spatial correlation
    2. Linear encoder: learns encoding from linearized fetaures
    3. Linear decoder: learns decoding from linearized features
    4. Convolutional decoder: turns linearized features back to 2D image,
    trying to rebuild it

    Conv2d/ConvTranspose2d first/last layer has input channel = 1, since it
    takes a grey-channel images as input. Other layers are set according to
    previous layers, instead

    Input shape is a tuple (batch size, num channels, height, width)
    """

    # Constructor
    def __init__(
        self, encoder_cnn=None, encoder_lin=None,
        decoder_lin=None, decoder_cnn=None, lin_to_cnn=None
    ):
        # Call parent constructor
        super().__init__()
        # Set neural network topology
        self.encoder_cnn = encoder_cnn
        self.encoder_lin = encoder_lin
        self.decoder_lin = decoder_lin
        self.decoder_cnn = decoder_cnn
        # Define linear to convolutional shape transformation
        self.lin_to_cnn = lin_to_cnn

    @property
    def lin_to_cnn(self):
        return self.lin_to_cnn_

    @lin_to_cnn.setter
    def lin_to_cnn(self, lin_to_cnn):
        self.lin_to_cnn_ = list(lin_to_cnn)

    @property
    def cnn_to_lin(self):
        return np.prod(self.lin_to_cnn)

    # Go through encoding (convolutional, linear) layers
    def encode(self, x):
        # Apply convolutions to input tensor
        x = self.encoder_cnn(x)
        # Flatten tensor to input linear layers (0-th size is batch size)
        x = x.view([-1, self.cnn_to_lin])
        # Apply linear layers
        x = self.encoder_lin(x)
        # Return output
        return x

    # Go through decoding (linear, convolutional) layers
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, *self.lin_to_cnn])
        # Apply transposed convolutions
        x = self.decoder_cnn(x)
        # Apply activation
        x = torch.sigmoid(x)
        return x

    # Make forward call: first encode then decode
    def forward(self, x):
        # Encode
        x = self.encode(x)
        # Decode
        x = self.decode(x)
        # Return decoded result
        return x

    # Train autodecoder on an images batch
    def train_batch(
        self, batch, loss_fn, optimizer,
        device=torch.device('cpu')
    ):
        # Extract images batch (X) and labels batch (y)
        X, y = batch
        # Move batch to correct device
        X.to(device)
        # Make forward pass
        out = self(X)  # Compute output
        loss = loss_fn(out, X)  # Compute loss
        # Backward pass
        optimizer.zero_grad()  # Delete previous gradient
        loss.backward()  # Compute loss gradient
        optimizer.step()  # Update weights
        # Return loss
        return out, loss

    # Test autodecoder on an images batch
    def test_batch(self, batch, loss_fn, device=torch.device('cpu')):
        # Disable gradient computation
        with torch.no_grad():
            # Extract images batch (X) and labels batch (y)
            X, y = batch
            # Move batch to correct device
            X.to(device)
            # Make forward pass
            out = self(X)  # Compute output
            loss = loss_fn(out, X)  # Compute loss
            # Return loss
            return out, loss

    # Train autoencoder on a dataset for a given number of epochs
    def train_epochs(
        self, num_epochs, train_dataloader, test_dataloader,
        optimizer, loss_fn, verbose=False,
        model_path=None, epochs_path=None, save_epochs=0,
        device=torch.device('cpu'),
    ):
        # Define train and test loss
        train_loss, test_loss = list(), list()
        # # Define train and test time
        # train_time, test_time = list(), list()
        # Define epochs wrapper
        epochs_fn = (lambda x: x) if not verbose else tqdm
        # Loop through each epoch
        for epoch in epochs_fn(range(num_epochs)):
            # Set the network in training mode
            self.train()
            # Loop through every batch in training dataset
            for batch in train_dataloader:
                # Train the autoencoder on current batch
                _, loss = self.train_batch(batch, loss_fn, optimizer, device)
                # Store current training loss
                train_loss.append(loss.item())
            # Set the network in evaluation/test mode
            self.eval()
            # Loop through every batch in evaluation dataset
            for batch in test_dataloader:
                # Test the autoencoder on current batch
                _, loss = self.test_batch(batch, loss_fn, device)
                # Store current training loss
                test_loss.append(loss.item())
            # Save parameters and loss after a certain number of epochs
            if epoch + 1 % save_epochs == 0:
                # Save model parameters
                torch.save(self.state_dict(), model_path)
                # Save losses
                with open(epochs_path, 'w') as epochs_file:
                    # Store either the loss and the number of epochs until now
                    json.dump({
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'num_epochs': epoch + 1
                    }, epochs_file)
        # Return training and testing loss
        return train_loss, test_loss
