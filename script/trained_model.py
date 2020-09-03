# Dependencies
from src.dataset import RandomNoise, GaussianNoise, SaltPepperNoise, OcclusionNoise, Clip, ToTensor, Compose
from src.dataset import MNIST
from src.network import VariationalAutoEncoder as VAE
from src.network import AutoEncoder as AE
from torch.utils.data import DataLoader
from torch import nn
import argparse
import torch
import os

# Main
if __name__ == '__main__':

    # Define project root path
    ROOT_PATH = os.path.dirname(__file__)
    # Define data folder path
    DATA_PATH = ROOT_PATH + '/data'

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Test trained model for digits reconstruction')
    # Add argument: number of latent space dimensions
    parser.add_argument(
        '--num_latent', type=int, default=2,
        help='Number of latent space dimensions'
    )
    # Add argument: type of autoencoder, if variational or not
    parser.add_argument(
        '--variational', type=int, default=1,
        help='Wether to use variational autoencoder or not'
    )
    # Add argument: path to model (best autoencoder)
    parser.add_argument(
        '--model_path', type=str, default=DATA_PATH+'best_model.pth',
        help='Path to trained model weights'
    )
    # Add argument: test data
    parser.add_argument(
        '--mnist_path', type=str, default=DATA_PATH+'/MNIST.mat',
        help='Path to test dataset'
    )
    # Add argument: noise type
    parser.add_argument(
        '--add_noise', type=str, nargs='+', default=['none', 'gaussian', 'occlusion', 'saltpepper'],
        help='Add noise to dataset'
    )
    # Add argument: batch size
    parser.add_argument(
        '--batch_size', type=int, default=1000,
        help='Batch size to use during MNIST dataset evaluation'
    )
    # Add argument: device
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to be used during MNIST dataset evaluation'
    )
    # Parse arguments
    args = parser.parse_args()

    # Define device
    device = torch.device(args.device)

    # Define noise dictionary
    noise_dict = {
        # No Noise
        'none': Clip(),
        # Gaussian noise
        'gaussian': Compose([GaussianNoise(0, 0.3), Clip()]),
        # Occlusion noise
        'occlusion': Compose([OcclusionNoise(), Clip()]),
        # Salt and pepper noise
        'saltpepper': Compose([SaltPepperNoise(), Clip()])
    }

    # Load dataset
    mnist = MNIST.from_mat(args.mnist_path, transform=Compose([
        # Apply noise randomly
        RandomNoise([noise_dict[k] for k in set(args.add_noise)]),
        # Cast images to tensor
        ToTensor()
    ]))
    # Make dataloader
    loader = DataLoader(mnist, batch_size=args.batch_size, shuffle=True)

    # Define number of latent space dimensions
    latent_dim = int(args.num_latent)
    # Define autoencoder
    net = VAE(latent_dim) if args.variational else AE(latent_dim)
    # Load weights from given model path
    net.load_state_dict(torch.load(args.model_path))
    # Move autoencoder to given device
    net.to(device)

    # Define loss function
    loss_fn = torch.nn.MSELoss()

    # Set network in evaluation mode
    net.eval()
    # Disable gradient computation
    with torch.no_grad():
        # Retrieve mean evaluation loss and total time over the whole dataset
        test_loss, test_time = net.test_epoch(loader, loss_fn=loss_fn)
        # Show test loss
        print(test_loss)
