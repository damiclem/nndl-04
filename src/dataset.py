# Dataset
import torch.utils.data as tud
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os


class MNIST(tud.Dataset):
    """ MNIST dataset

    This class handles a user define MNIST dataset, whose digits are turned.
    It allows to define some transformations to digit images once they are
    accessed, such as salt and pepper or gaussian noise.
    """

    def __init__(self, out_labels, out_images, transform=None):
        """ Constructor

        Args
        out_labels (iterable)           List of image labels (true digit value)
        # in_images (iterable)            List of lists representing linearized
        #                                 squared input (an have noise) images
        out_images (iterable)           List of lists representing linearized
                                        squared output (no noise) images
        transform (torch.Transform)     One or more transformations to apply to
                                        input images, such as gaussian noise

        Raise
        (ValueError)                    In case input images and labels sizes
                                        do not match
        (ValueError)                    In case given images are not squared
        """
        # Get output images shape
        n, m = out_images.shape
        # Check that number of input images and labels match
        if not len(out_images) == len(out_labels):
            # Raise Exception
            raise ValueError('given images and labels sizes do not match')

        # Case m is not a perfect square
        if not (m ** 0.5) == int(m ** 0.5):
            # Raise exception
            raise ValueError('given images are not squared')

        # Substitute m with its squared value
        m = int(m ** 0.5)

        # Store defined transformations
        self.transform = transform

        # Store output labels and images
        self.out_images = np.array(out_images, dtype=np.float).reshape((n, m, m))
        self.out_labels = np.array(out_labels, dtype=np.int)

    def __len__(self):
        """ Get length of the dataset

        Return
        (int)       Number of images in the dataset
        """
        # Return number of labels
        return len(self.out_labels)

    # Retrieve a single item in dataset
    def __getitem__(self, i):
        """ Random access images

        Args
        i (int)     Index of image accessed

        Return
        (int)           Label of the image
        (iterable)      Input image (eventually transformed)
        (iterable)      Output image (raw, not transformed)

        Raise
        (IndexError)    In case given index does not exist
        """
        # Check that given index exists
        if not i < len(self):
            # Raise exception
            raise IndexError('given index is out of bound')

        # Get output label
        out_label = self.out_labels[i]
        # Get output image
        out_image = np.copy(self.out_images[i, :, :])
        # Get input image
        in_image = np.copy(self.out_images[i, :, :])

        # Case transform is set
        if self.transform is not None:
            # Apply transform over all the triple
            out_label, in_image, out_image = self.transform(out_label, in_image, out_image)

        # Return both output label, input image and output image
        return out_label, in_image, out_image

    @classmethod
    def from_mat(cls, path, transform=None):
        """ Load dataset from matrix format

        Args
        path (str)                      Path to file containing images
        transform (torch.transform)     One or multiple transformations to
                                        apply to input images

        Return
        (MNIST)                         MNIST dataset instance
        """
        # Load dict(label: linearized image) using scipy
        loaded = scipy.io.loadmat(path)
        # Define list of labels
        out_labels = np.array(loaded.get('output_labels'), dtype=np.int)
        # Define list of images
        out_images = np.array(loaded.get('input_images'), dtype=np.float)
        # Retrun new MNIST dataset instance
        return cls(out_labels, out_images, transform=transform)

    # def to_list(self):
    #     """ Store dataset into a list
    #
    #     Return
    #     (list)      List of tuples (out_label, in_image, out_image) with given
    #                 transformation applied over the three items
    #     """
    #     # Initialize output list
    #     rtn_list = list()
    #     # Loop through each item in dataset
    #     for i in range(len(self)):
    #         # Retrieve i-th tuple
    #         out_label, in_image, out_image = self[i]
    #         # Store tuple into output list
    #         rtn_list.append(tuple(out_label, in_image, out_image))
    #     # Return retrieved list
    #     return rtn_list

    @staticmethod
    def plot_digit(label, image, ax):
        # Make image
        ax.imshow(image, cmap='gist_gray')
        # Define title and labels
        ax.set_title('Label: %d' % label)
        ax.set_xticks([])
        ax.set_yticks([])


class GaussianNoise(object):

    def __init__(self, mu=0.0, sigma=1.0):
        # Store gaussian noise parameters
        self.mu = mu
        self.sigma = sigma

    def __call__(self, out_label, in_image, out_image):
        # Update input image
        in_image += np.random.normal(self.mu, self.sigma, in_image.shape)
        # Return new triple
        return out_label, in_image, out_image


class SaltPepperNoise(object):

    def __init__(self, perc=0.1, scale=(0, 1)):
        # Save noise parameters
        self.perc = perc  # Percentage of pixel to flip
        self.scale = scale  # Range in which pixel must be scaled

    def __call__(self, out_label, in_image, out_image):
        # Get input image shape
        n, m = in_image.shape
        # Flatten input image
        in_image = in_image.flatten()
        # Chose some indices of the image, according to given percentage
        chosen = np.random.choice(n * m, int(n * m * self.perc), replace=False)
        # Compute the index for selecting half of the chosen index
        half = int(round(len(chosen)/2))
        # Set half chosen pixels to minimum, the other half to maximum
        in_image[chosen[half:]], in_image[chosen[:half]] = self.scale
        # Reset input image shape
        in_image = in_image.reshape((n, m))
        # Return triple
        return out_label, in_image, out_image


class OcclusionNoise(object):

    def __init__(self, perc=0.1, scale=(0, 1)):
        # Save noise parameters
        self.perc = perc  # Percentage of pixel to deactivate
        self.scale = scale  # Range in which pixel must be scaled

    def __call__(self, out_label, in_image, out_image):
        # Get image shape
        n, m = in_image.shape
        # Define block height (prc * n) and width (prc * m)
        h, w = int(self.perc * n), int(self.perc * m)
        # Choose at random an index on the rows
        i = np.random.choice(n - h, replace=False)
        # Choose at random an index of the columns
        j = np.random.choice(m - w, replace=False)
        # Fully deactivate the selected region (minimum value)
        in_image[i:i+h, j:j+w], _ = self.scale
        # Return triple
        return out_label, in_image, out_image


class RandomNoise(object):

    def __init__(self, transforms):
        # Store transformations
        self.transforms = transforms

    def __call__(self, out_label, in_image, out_image):
        # Choose a transformer index at random
        i = np.random.choice(len(self.transforms))
        # Retrieve chosen transformer
        transformer = self.transforms[i]
        # Apply retrieved transformer
        return transformer(out_label, in_image, out_image)


class Clip(object):

    def __init__(self, scale=(0, 1)):
        # Store clip (minimum, maximum) tuple
        self.scale = scale

    def __call__(self, out_label, in_image, out_image):
        # Clip input image
        in_image = np.clip(in_image, *self.scale)
        # Clip output image
        out_image = np.clip(out_image, *self.scale)
        # Return triple
        return out_label, in_image, out_image


class ToTensor(object):

    def __call__(self, out_label, in_image, out_image):
        # Cast output label to tensor
        out_label = torch.tensor(out_label).long()
        # Cast input image to tensor
        in_image = torch.tensor([in_image]).float()
        # Cast output image to tensor
        out_image = torch.tensor([out_image]).float()
        # # Debug
        # print('Input image shape', in_image.shape)
        # print('Output image shape', out_image.shape)
        # Return the new triple
        return out_label, in_image, out_image


class Compose(object):

    def __init__(self, transforms):
        # Store transforms list
        self.transforms = transforms

    def __call__(self, out_label, in_image, out_image):
        # Loop through each tranformation
        for i in range(len(self.transforms)):
            # Get current transformer
            transformer = self.transforms[i]
            # Apply transformer
            out_label, in_image, out_image = transformer(out_label, in_image, out_image)
        # Return triple
        return out_label, in_image, out_image


def train_test_split(dataset, train_perc=0.8, train_size=None):
    """ Split dataset into a train and a test one

    Args
    dataset (iterable)        Input dataset
    train_perc (float)      Percentage of given dataset rows to reserve for
                            training, the rest will be reserved for testing
    train_size (int)        Number of given dataset rows to reserve for
                            training, the rest will be reserved for testing.
                            This option overrides percentage one

    Return
    (iterator)              Iterator over train dataset
    (iterator)              Iterator over test dataset

    Raise
    (ValueError)            In case given percentage exceeds [0, 1] boundaries
    (ValueError)            In case train size exceeds input dataset size
    """
    # Define input dataset length
    n = len(dataset)

    # Case training dataset size has been set
    if train_size is not None:
        # Check given training size
        if train_size < 0 or n < train_size:
            # Define error message
            err = 'specified training dataset size must be'
            err += 'between 0 and dataset size'
            # Raise exception
            raise ValueError(err)

    # Check given training percentage
    elif train_perc < 0.0 or 1.0 < train_perc:
        # Define error message
        err = 'percentage of given dataset reserved for training '
        err += 'must be between 0.0 and 1.0'
        # Raise exception
        raise ValueError(err)

    # Case train size is none
    if train_size is None:
        # Compute training dataset size
        train_size = int(round(train_perc * n))

    # Define test size
    test_size = n - train_size
    # # Define train dataset indices
    # train_indices = set(np.random.choice(n, train_size, replace=False))
    # # Define test dataset indices
    # test_indices = set(range(n)) - train_indices
    #
    # # Define train iterator
    # train_iter = [(yield dataset[i]) for i in train_indices]
    # # Define test iterator
    # test_iter = [(yield dataset[i]) for i in test_indices]
    # # Return both iterators
    # return train_iter, test_iter

    # Return either train and test dataset
    return torch.utils.data.random_split(dataset, [train_size, test_size])


# Test
if __name__ == '__main__':

    # Define project root path
    ROOT_PATH = os.path.dirname(__file__) + '/..'
    # Define data folder path
    DATA_PATH = ROOT_PATH + '/data'
    # Define MNIST dataset path
    MNIST_PATH = DATA_PATH + '/MNIST.mat'

    # Define function for plotting a sample of digits
    def plot_sample(dataset, sample, title='10 randomly chosen digits'):
        # Initialize plot: show some digits at random
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        # Add main title
        fig.suptitle(title)
        # Flatten axis
        axs = axs.flatten()
        # Loop through every randomly chosen index
        for i, k in enumerate(sample):
            # Get current image and label
            label, image, _ = dataset[k]
            # Make digit plot
            MNIST.plot_digit(label, image, ax=axs[i])
        # Tighten plot layout
        plt.tight_layout()
        # Show plot
        plt.show()

    # Load MNIST dataset
    dataset = MNIST.from_mat(path=MNIST_PATH)
    # Sample 10 digits at random
    sample = np.random.choice(len(dataset), 10)
    # Plot sampled digits
    plot_sample(dataset, sample, title='10 randomly chosen digits')

    # Add gaussian noise transform
    dataset.transform = GaussianNoise(0, 0.3)
    # Plot sampled digits
    plot_sample(dataset, sample, title='10 randomly chosen digits, with gaussian noise')

    # Add salt and pepper noise
    dataset.transform = SaltPepperNoise(perc=0.3)
    # Plot sampled digits
    plot_sample(dataset, sample, title='10 randomly chosen digits, with salt and pepper noise')

    # Add occlusion noise
    dataset.transform = OcclusionNoise(perc=0.4)
    # Plot sampled digits
    plot_sample(dataset, sample, title='10 randomly chosen digits, with occlusion noise')

    # Turn dataset to list
    # data = dataset.to_list()
    # Split dataset into train and test
    train_data, test_data = train_test_split(dataset, train_perc=0.8)
    # # Paste train dataset and test dataset to list
    # train_dataset, test_dataset = list(train_iter), list(test_iter)
    # Compare lengths
    print('Input dataset has shape %d' % len(dataset))
    print('Train dataset has shape %d' % len(train_data))
    print('Test dataset has shape %d' % len(test_data))

    print(train_data.transform)
    print(test_data.transform)
