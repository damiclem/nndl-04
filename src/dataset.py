from torch.utils.data import Dataset
import torch

import numpy as np
import scipy.io

class MNIST(Dataset):

    # Constructor
    def __init__(self, in_path=None):

        # Initialize empty attributes
        output_labels = None
        output_images = None
        output_images = None

        # Case input path is set: initialize dataset
        if in_path is not None:

            # Load MNIST.mat dictionary from file
            data_dict = scipy.io.loadmat(in_path)
            # Get (correct) labels
            output_labels = data_dict['output_labels']
            # Get (correct) images
            output_images = data_dict['input_images']

            # Define n rows and m columns
            n, m = output_images.shape
            # Reshape linearized images to squared ones
            output_images = output_images.reshape((n, int(m**0.5), int(m**0.5)))

        # Save dataset attributes
        self.output_labels = output_labels
        self.output_images = output_images
        self.input_images = output_images

    # Return data (output labels, output images, input images)
    @property
    def data(self):
        return (self.output_labels, self.output_images, self.input_images)

    # Define number of images inside dataset
    def __len__(self):
        # Return number of rows in dataset
        return self.output_images.shape[0]

    # Retrieve a single item in dataset
    def __getitem__(self, i):
        # Return either the output/correct label, the output/correct image
        # along with a an eventually modified input image as tensor
        output_label = torch.tensor(self.output_labels[i]).long()
        output_image = torch.tensor(self.output_images[i, :, :]).unsqueeze(0).float()
        input_image = torch.tensor(self.input_images[i, :, :]).unsqueeze(0).float()
        # Return tuple
        return output_label, output_image, input_image

    # Clip input and output images in a values interval
    def clip(self, scale=(0.0, 1.0)):
        self.output_images = np.clip(self.output_images, 0.0, 1.0)
        self.input_images = np.clip(self.input_images, 0.0, 1.0)

    # Add output labels, output images and input images to current dataset
    def append(self, output_labels, output_images, input_images):
        self.output_labels = np.concatenate((self.output_labels, output_labels))
        self.output_images = np.concatenate((self.output_images, output_images))
        self.input_images = np.concatenate((self.input_images, input_images))

    # Add gaussian noised images
    def add_gaussian_noise(self, output_labels, output_images, input_images, mu=0.0, sigma=1.0):
        # Apply gaussian noise over rows axis
        input_images = np.array([
            MNIST.gaussian_noise(input_images[i], mu, sigma)
            for i in range(input_images.shape[0])
        ])
        # Append to current dataset
        self.append(output_labels, output_images, input_images)

    # Add salt and pepper noised images
    def add_salt_pepper_noise(self, output_labels, output_images, input_images, prc=0.1, scale=(0.0, 1.0)):
        # Apply gaussian noise over rows axis
        input_images = np.array([
            MNIST.salt_pepper_noise(input_images[i], prc, scale)
            for i in range(input_images.shape[0])
        ])
        # Append to current dataset
        self.append(output_labels, output_images, input_images)

    # Add occusion noised images
    def add_occlusion_noise(self, output_labels, output_images, input_images, prc=0.1, scale=(0.0, 1.0)):
        # Apply occlusion noise over rows axis
        input_images = np.array([
            MNIST.occlusion_noise(input_images[i], prc, scale)
            for i in range(input_images.shape[0])
        ])
        # Append to current dataset
        self.append(output_labels, output_images, input_images)

    # Apply gaussian noise to numpy image (ndarray with shape=(n, m))
    @staticmethod
    def gaussian_noise(image, mu=0.0, sigma=1.0):
        # Compute and add gaussian noise
        return image + np.random.normal(mu, sigma, image.shape)

    # Apply salt and pepper noise to numpy image (ndarray with shape=(n, m))
    @staticmethod
    def salt_pepper_noise(image, prc=0.1, scale=(0.0, 1.0)):
        # Get image shape
        n, m = image.shape
        # Flatten the image
        image = image.flatten()
        # Get some indices of the image
        chosen = np.random.choice(n * m, int(n * m * prc), replace=False)
        # Compute the index for selecting half of the chosen index
        half = int(round(len(chosen)/2))
        # Set half chosen pixels to minimum
        image[chosen[:half]] = scale[0]
        # Set the other half to maximum
        image[chosen[half:]] = scale[1]
        # Return reshaped image
        return image.reshape((n, m))

    # Apply occlusion noise to numpy image (ndarray with shape=(n, m))
    @staticmethod
    def occlusion_noise(image, prc=0.1, scale=(0.0, 1.0)):
        # Get image shape
        n, m = image.shape
        # Define block height (prc * n) and width (prc * m)
        h, w = int(prc * n), int(prc * m)
        # Choose at random an index on the rows
        i = np.random.choice(n - h, replace=False)
        # Choose at random an index of the columns
        j = np.random.choice(m - w, replace=False)
        # Fully deactivate the selected region (minimum value)
        image[i:i+h, j:j+w] = scale[0]
        # Return noised image
        return image

    # Split a dataset in train and test
    @staticmethod
    def train_test_split(in_dataset, test_size, train_size=None):

        # Define number of examples in input dataset as n
        n = len(in_dataset)

        # Set train size complementary to test size
        if train_size is None:
            train_size = (1 - test_size)

        # Set train and test size in terms of rows
        train_size = int(round(train_size * n))
        test_size = int(round(test_size * n))

        # Get random train and test datasets indices
        train_indices = set(np.random.choice(n, train_size, replace=False))
        test_indices = set(np.arange(n)) - train_indices

        # Set train and test indices back to lists instead of sets
        train_indices = list(train_indices)
        test_indices = list(test_indices)

        # Define a new train dataset
        train_dataset = MNIST()
        train_dataset.output_labels = in_dataset.output_labels[train_indices]
        train_dataset.output_images = in_dataset.output_images[train_indices, :, :]
        train_dataset.input_images = in_dataset.input_images[train_indices, :, :]

        # Define a new test dataset
        test_dataset = MNIST()
        test_dataset.output_labels = in_dataset.output_labels[test_indices]
        test_dataset.output_images = in_dataset.output_images[test_indices, :, :]
        test_dataset.input_images = in_dataset.input_images[test_indices, :, :]

        # Return the two splitted datasets
        return train_dataset, test_dataset
