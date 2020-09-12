# Dependencies
from src.dataset import MNIST, ToTensor, train_test_split
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, binary_cross_entropy
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time
import os


class AutoEncoder(nn.Module):
    """ Auto Encoder (AE)

    Classic auto encoder with convolutional layers and symmetrical encoder and
    decoder sides, useful for digits reconstruction.
    """

    def __init__(self, latent_dim=2):
        """ Constructor

        Args
        latent_dim (int)        Dimension of the encoded latent space

        Raise
        (ValueError)            In case latent dimension is not valid (not
                                integer or lower than one)
        """
        # Call parent constructor
        super().__init__()

        # Check latent space dimension
        if not isinstance(latent_dim, int) or latent_dim < 1:
            # Raise exception
            raise ValueError('given latent space dimension is not valid')

        # Define encoding convolutional layer
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        # Define encoding linear layer
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim)
        )

        # Define decoding linear layer
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True)
        )

        # Define decoding convolutional layer
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

        # Store latent space dimension
        self.latent_dim = latent_dim

    def forward(self, x):
        """ Fed input to the network

        Args
        x (torch.Tensor)        Input images batch as 3d tensor

        Return
        (torch.tensor)          Reconstructed input images batch as 3d tensor
        """
        # Go through encoding side
        x = self.encode(x)
        # Go through decoding side
        x = self.decode(x)
        # Return reconstructed images batch
        return x

    def encode(self, x):
        """ Fed raw input to encoding side

        Args
        x (torch.Tensor)        Input images batch as 3d tensor

        Return
        (torch.tensor)          Encoded input images batch as 2d tensor
        """
        # Feed input through convolutional layers
        x = self.encoder_cnn(x)
        # Flatten convolutional layers output
        x = torch.flatten(x, 1)
        # x = x.view([x.size(0), -1])
        # Feed input through linear layers
        return self.encoder_lin(x)

    def decode(self, x):
        """ Fed encoded input to decoding side

        Args
        x (torch.Tensor)        Encoded input images batch as 2d tensor

        Return
        (torch.tensor)          Decoded input images batch as 3d tensor
        """
        # Feed encoded input through linear decoder
        x = self.decoder_lin(x)
        # Reshape linear decoder output
        x = x.view([-1, 32, 3, 3])
        # Feed convolutional decoder with reshaped input
        x = self.decoder_cnn(x)
        # Apply decision layer (sigmoid)
        return torch.sigmoid(x)

    @property
    def device(self):
        return next(self.parameters()).device

    def train_batch(self, batch, loss_fn, optim=None, ret_images=False, eval=False):
        """ Train network on a single batch

        Args
        batch (tuple)           Tuple containing output labels tensor, input
                                images tensor and output images tensor
        loss_fn (nn.Module)     Loss function instance
        optim (nn.Module)       Optimizer used during weights update
        ret_images (bool)       Wether to return test images either
        eval (bool)             Wether to do training or evaluation (test)

        Return
        (float)                 Current batch loss
        (torch.Tensor)          Eventually return reconstructed images either

        Raise
        (ValueError)            In case training mode has been chosen without
                                defining an optimizer instance
        """
        # Check that optimizer has been set in training mode
        if (not eval) and (optim is None):
            # Raise exception
            raise ValueError('optimizer must be set for training')

        # Retrieve device
        device = self.device
        # Retrieve output labels, input image and output image
        out_labels, in_images, out_images = batch
        # Move input and output images to device
        in_images, out_images = in_images.to(device), out_images.to(device)

        # Make forward pass
        net_images = self(in_images)
        # Compute loss
        loss = loss_fn(net_images, out_images)

        # Training mode
        if not eval:
            # Clean previous optimizer state
            optim.zero_grad()
            # Make backward pass (update weights)
            loss.backward()
            # Update weights
            optim.step()

        # Case images have been required
        if ret_images:
            # Return either loss and images
            return float(loss.data), net_images

        # Return loss
        return float(loss.data)

    def test_batch(self, batch, loss_fn, ret_images=False):
        """ Test network on a single batch

        Args
        batch (tuple)           Tuple containing output labels tensor, input
                                images tensor and output images tensor
        loss_fn (nn.Module)     Loss function instance
        ret_images (bool)       Wether to return test images either

        Return
        (float)                 Current batch loss
        (torch.Tensor)          Eventually return reconstructed images either
        """
        return self.train_batch(batch, loss_fn, optim=None, ret_images=ret_images, eval=True)

    def train_epoch(self, dataloader, loss_fn, optim=None, eval=False):
        """ Train network on all the batches in a single epoch

        Args
        dataloader (Dataloader) Dataloader allowing to iterate over batches
        loss_fn (nn.Module)     Loss function instance
        optim (nn.Module)       Optimizer used during weights update
        eval (bool)             Wether to do training or evaluation (test)

        Return
        (float)                 Current epoch mean loss
        (float)                 Current epoch total time, in seconds

        Raise
        (ValueError)            In case training mode has been chosen without
                                defining an optimizer instance
        """
        # Initialize losses and times
        epoch_losses, epoch_times = [], []
        # Set network in training/evaluation mode
        self.eval() if eval else self.train()
        # Loop through each batch in given dataloader
        for batch in dataloader:
            # Initialize batch timer
            batch_start = time.time()
            # Get current batch loss
            batch_loss = self.train_batch(batch, loss_fn=loss_fn, optim=optim, eval=eval)
            # Store current batch loss
            epoch_losses.append(batch_loss)
            # Store current batch time
            epoch_times.append(time.time() - batch_start)
        # Return mean loss and total time
        return sum(epoch_losses) / len(epoch_losses), sum(epoch_times)

    def test_epoch(self, dataloader, loss_fn):
        """ Test network on all the batches in a single epoch

        Args
        dataloader (Dataloader) Dataloader allowing to iterate over batches
        loss_fn (nn.Module)     Loss function instance

        Return
        (float)                 Current epoch mean loss
        (float)                 Current epoch total time, in seconds

        Raise
        (ValueError)            In case training mode has been chosen without
                                defining an optimizer instance
        """
        return self.train_epoch(dataloader, loss_fn, optim=None, eval=True)


class VariationalAutoEncoder(AutoEncoder):
    """ Variational Auto Encoder (VAE)

    This AutoEncoder is similar to the default one, except that it encodes to
    a distribution in the latent space, not to a point. This should provide
    latent space with either continuity and completeness, while reducing
    overfitting.
    """

    def __init__(self, latent_dim=2):
        """ Constructor

        Args
        latent_dim (int)        Dimension of the encoded latent space

        Raise
        (ValueError)            In case latent dimension is not valid (not
                                integer or lower than one)
        """
        # Call parent constructor
        super().__init__(latent_dim=latent_dim)
        # Remove linear encoder
        del self.encoder_lin
        # Define linear encoder for mean
        self.encoder_mu = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim)
        )
        # Define linear encoder for (log transformed) variance
        self.encoder_logvar = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        """ Fed input to the network

        Args
        x (torch.Tensor)        Input images batch as 3d tensor

        Return
        (torch.Tensor)          Reconstructed input images batch as 3d tensor
        (torch.Tensor)          Latent space encoded mean
        (torch.Tensor)          Latent space encoded (log transformed) variance
        """
        # Encode either mean and (log transformed) variance
        mu, logvar = self.encode(x)
        # Apply reparametrisation trick
        z = self.reparameterize(mu, logvar)
        # Return either decoded
        return self.decode(z), mu, logvar

    def encode(self, x):
        """ Fed raw input to encoding side

        Args
        x (torch.Tensor)        Input images batch as 3d tensor

        Return
        (torch.Tensor)          Encoded input mean
        (torch.Tensor)          Encoded input (log transformed) variance
        """
        # Feed input through convolutional layers
        x = self.encoder_cnn(x)
        # Flatten convolutional layers output
        x = torch.flatten(x, 1)
        # Encode mean
        mu = self.encoder_mu(x)
        # Encode (log transformed) variance
        logvar = self.encoder_logvar(x)
        # Return encoded mean and variance
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """ Reparametrisation trick

        Args
        mu (torch.Tensor)          Values of the mean
        logvar (torch.Tensor)      Values of the (log transformed) variance

        Return
        (torch.Tensor)             Values sampled according to given parameters
        """
        # Compute standard deviation (Monte-Carlo expectation approximation)
        std = torch.exp(0.5 * logvar)
        # Sample from standard normalnormal (eps shape will match std one)
        eps = torch.randn_like(std)
        # Compute objective function
        return mu + eps * std

    @staticmethod
    def loss_fn(x_pred, x_true, mu, logvar):
        """ Variational autoencoder loss function

        Args
        x_pred (torch.Tensor)       Reconstructed image tensor
        x_true (torch.Tensor)       Original image tensor
        mu (torch.Tensor)           Mean terms
        logvar (torch.Tensor)       Variance terms (log transformed)

        Return
        (torch.Tensor)              Computed losses
        """
        # Reconstruction loss: BCE
        bce = binary_cross_entropy(x_pred, x_true, reduction='sum')
        # # Reconstruction loss: MSE
        # mse = mse_loss(x_pred, x_true, reduction='mean')
        # KLD: Kullback-Leibler divergence (regularization)
        kld = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Return loss
        return bce - kld
        # return mse - kld

    def train_batch(self, batch, loss_fn, optim=None, ret_images=False, eval=False):
        """ Train network on a single batch

        Args
        batch (tuple)           Tuple containing output labels tensor, input
                                images tensor and output images tensor
        loss_fn (nn.Module)     Loss function instance
        optim (nn.Module)       Optimizer used during weights update
        ret_images (bool)       Wether to return test images either
        eval (bool)             Wether to do training or evaluation (test)

        Return
        (float)                 Current batch loss
        (torch.Tensor)          Eventually return reconstructed images either

        Raise
        (ValueError)            In case training mode has been chosen without
                                defining an optimizer instance
        """
        # Check that optimizer has been set in training mode
        if (not eval) and (optim is None):
            # Raise exception
            raise ValueError('optimizer must be set for training')

        # Retrieve device
        device = self.device
        # Retrieve output labels, input image and output image
        out_labels, in_images, out_images = batch
        # Move input and output images to device
        in_images, out_images = in_images.to(device), out_images.to(device)

        # Make forward pass
        net_images, mu, logvar = self(in_images)

        # Case loss function is not the regularized one (e.g. MSE)
        if loss_fn != self.loss_fn:
            # Give only reconstructed images to loss
            loss = loss_fn(net_images, out_images)
        # Case loss function is the regularized one
        if loss_fn == self.loss_fn:
            # Compute loss using either distribution parameters
            loss = loss_fn(net_images, out_images, mu, logvar)

        # Training mode
        if not eval:
            # Clean previous optimizer state
            optim.zero_grad()
            # Make backward pass (update weights)
            loss.backward()
            # Update weights
            optim.step()

        # Case images have been required
        if ret_images:
            # Return either loss and images
            return float(loss.data), net_images

        # Return loss
        return float(loss.data)


def train_test_epoch(net, loss_fn, optim, train_data, test_data):
    """ Train and test the network over a single epoch

    Args
    net (nn.Module)             Network to be trained and tested
    loss_fn (nn.Module)         Loss function instance
    optim (nn.Module)           Optimizer used during weights update
    train_data (DataLoader)     Training dataset loader
    test_data (DataLoader)      Test dataset loader

    Return
    (float)                     Training loss for current epoch
    (float)                     Training time for current epoch
    (float)                     Test loss for current epoch
    (float)                     Test time for current epoch
    """
    # # Check given number of epochs
    # if not isinstance(num_epochs, int) or num_epochs < 1:
    #     # Raise exception
    #     raise ValueError('given number of epochs is not valid')

    # # Check givem step
    # if not isinstance(step_epochs, int) or step_epochs < 1:
    #     # Raise exception
    #     raise ValueError('given epochs step is not valid')

    # # Loop through each epoch
    # for i in range(0, num_epochs, step_epochs):
    #     # Initialize list of epoch training losses and times
    #     train_losses, train_times = [], []
    #     # Initialize list of epoch test losses and times
    #     test_losses, test_times = [], []

    # # Loop through each epoch in current step
    # for j in range(i, min(num_epochs, i + step_epochs)):
    # Make training, retrieve mean loss and total time
    train_loss, train_time = net.train_epoch(
        dataloader=train_data,
        loss_fn=loss_fn,
        optim=optim
    )
    # # Store training loss and time
    # train_losses.append(train_loss)
    # train_times.append(train_time)

    # Disable gradient computation
    with torch.no_grad():
        # Make evaluation, retrieve mean loss and total time
        test_loss, test_time = net.test_epoch(
            dataloader=test_data,
            loss_fn=loss_fn
        )
    # # Store test loss and time
    # test_losses.append(test_loss)
    # test_times.append(test_time)

    # # Yield results
    # yield j, train_losses, train_times, test_losses, test_times

    # Return results
    return train_loss, train_time, test_loss, test_time


# Test
if __name__ == '__main__':

    # Define project root path
    ROOT_PATH = os.path.dirname(__file__) + '/..'
    # Define data folder path
    DATA_PATH = ROOT_PATH + '/data'
    # Define MNIST dataset path
    MNIST_PATH = DATA_PATH + '/MNIST.mat'

    # Retrieve best device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.cpu()

    # Load dataset
    dataset = MNIST.from_mat(MNIST_PATH)
    # Add transformation
    dataset.transform = ToTensor()
    # Split dataset in training and test
    train_iter, test_iter = train_test_split(dataset, train_perc=0.6)
    # Make lists out of iterators
    train_dataset, test_dataset = list(train_iter), list(test_iter)

    # Define autoencoder instance
    net = AutoEncoder(latent_dim=2)
    net.to(device)
    # Define loss function
    loss_fn = nn.MSELoss()
    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    # Define a sample image index
    k = np.random.choice(len(train_dataset))
    # Retrieve image and label
    label, image, _ = train_dataset[k]
    # Add batch size, move to device
    image = image.unsqueeze(0).to(device)
    # Show shape
    print('Retrieved image has shape (AE):', image.shape)
    # Try encoding (add batch size)
    encoded = net.encode(image)
    # Show encoded shape
    print('Encoded image shape (AE):', encoded.shape)
    # Try decoding
    decoded = net.decode(encoded)
    # Show decoded shape
    print('Decoded image shape (AE):', decoded.shape)
    print()

    # # Initialize results table
    # results = {'train_loss': [], 'train_time': [], 'test_loss': [], 'test_time': []}
    # # Define iterator
    # step_iter = tqdm.tqdm(desc='Training', iterable=train_test_epochs(
    #     net=net, loss_fn=loss_fn, optim=optimizer, num_epochs=10,
    #     train_data=DataLoader(train_dataset, batch_size=1000, shuffle=True),
    #     test_data=DataLoader(test_dataset, batch_size=1000, shuffle=False)
    # ))
    # # Make training and evaluation
    # for step_results in step_iter:
    #     # Store current results
    #     results['train_loss'] += step_results[0]
    #     results['train_time'] += step_results[1]
    #     results['test_loss'] += step_results[2]
    #     results['test_time'] += step_results[3]
    #
    # # Initialize plot: show loss
    # fig, ax = plt.subplots(figsize=(25, 5))
    # # Retrieve y train and y test
    # y_train = results['train_loss']
    # y_test = results['test_loss']
    # # Plot train loss
    # ax.plot(range(1, len(y_train) + 1), y_train, '-')
    # ax.plot(range(1, len(y_test) + 1), y_test, '-')
    # # Add title and labels
    # ax.set_title('Loss per epoch: train vs test')
    # ax.set_ylabel('Loss')
    # ax.set_xlabel('Epoch')
    # # Show plot
    # plt.show()

    # Define variational autoencoder
    vae = VariationalAutoEncoder(latent_dim=2)
    vae.to(device)

    # Define a sample image index
    k = np.random.choice(len(train_dataset))
    # Retrieve image and label
    label, image, _ = train_dataset[k]
    # Add batch size, move to device
    image = image.unsqueeze(0).to(device)
    # Show shape
    print('Retrieved image has shape (VAE):', image.shape)
    # Encode to mean and variance (add batch size)
    mu, logvar = vae.encode(image)
    # Encode to value using reparametrisation tirck
    encoded = vae.reparameterize(mu, logvar)
    # Show encoded shape
    print('Encoded mean has shape (VAE):', mu.shape)
    print('Encoded (log transformed) variance has shape (VAE):', logvar.shape)
    print('Sampled point in latent space has shape (VAE):', encoded.shape)
    # Decode
    decoded = vae.decode(encoded)
    # Show decoded shape
    print('Decoded image shape (VAE):', decoded.shape)
    # Compute loss
    loss = vae.loss_fn(decoded, image, mu, logvar).cpu()
    # Show computed loss
    print('Computed regularized loss (VAE):', loss.item())
    print()
