# Dependencies
import numpy as np
import torch


# Add Gaussian noise to tensor image
class AddGaussianNoise(object):

    # Constructor
    def __init__(self, mean=0.0, std=1.0):
        # Set noise distribution parameters
        self.mean = mean
        self.std = std

    # Generate random noise on a 2D tensor
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


# Add Salt and Pepper noise to tensor image
class AddSaltPepperNoise(object):

    # Constructor
    def __init__(self, percentage=0.1, scale=(0.0, 1.0)):
        # Store percentage of noised pixels
        self.prc = percentage
        # Store minimum and maximum available value
        self.min, self.max = scale

    # Generate salt and pepper noise on a 2D tesor
    def __call__(self, tensor):
        # Define rows(n) and columns (m)
        n, m = tuple(tensor.squeeze().size())
        # Get all available (i, j) indices combinations
        indices = [(i, j) for i in range(n) for j in range(m)]
        # Choose at random some cells to be noised
        chosen = np.random.choice(n * m, int(n * m * self.prc))
        # Loop thriugh <percentage> random cells
        for k in range(len(chosen)):
            # Get i (row) and j (column) indices
            i, j = indices[chosen[k]]
            # Half of the selected cells are set to minimum value
            if k < (n * m * self.prc) / 2:
                # Set cell value to minimum
                tensor[0, i, j] = self.min
            # Other half selected cells are set to maximum value
            else:
                # Set cell value to maximum
                tensor[0, i, j] = self.max
        # Return tensor with changed values
        return tensor


# Add block noide to tensor image
class AddBlockNoise(object):

    # Constructor
    def __init__(self, percentage, scale=(0.0, 1.0)):
        # Percentage of deactivated block size (for each image edge)
        self.prc = percentage
        # Get minimum and maximum scale values
        self.min, self.max = scale

    def __call__(self, tensor):
        # Define rows(n) and columns (m)
        n, m = tuple(tensor.squeeze().size())
        # Define block edge (prc * n) and base (prc * m) sizes
        e, b = int(self.prc * n), int(self.prc * m)
        # Choose at random an index on the rows
        i = np.random.choice(n - e)
        # Choose at random an index of the columns
        j = np.random.choice(m - b)
        # Fully deactivate the selected region
        tensor[0, i:i+e, j:j+b] = self.min
        # Return noised tensor
        return tensor


# Clamp input tensor values to a given interval
class ClampTensor(object):

    def __init__(self, scale=(0.0, 1.0)):
        # Save min and max values
        self.min, self.max = scale

    # Apply clamp to tensor
    def __call__(self, tensor):
        return torch.clamp(tensor, min=self.min, max=self.max)
