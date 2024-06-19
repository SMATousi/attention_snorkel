import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def gaussian_label_flip_and_zero(arr, m, n_list, zero_n_list):

    if len(n_list) != m or len(zero_n_list) != m:
        raise ValueError("The length of n_list and zero_n_list must be equal to m.")
    
    length = len(arr)
    flipped_arrays = []

    for i in range(m):
        print(i)
        n = n_list[i]
        zero_n = zero_n_list[i]
        indices_to_zero = []


        mean = np.random.uniform(0, length)
        std_dev = np.random.uniform(1, length)


        while len(indices_to_zero) < zero_n:
            gaussian_index = int(np.random.normal(mean, std_dev))
            if 0 <= gaussian_index < length and gaussian_index not in indices_to_zero:
                indices_to_zero.append(gaussian_index)


        modified_arr = arr.copy()
        for idx in indices_to_zero:
            modified_arr[idx] = 0


        untouched_indices = [i for i in range(length) if i not in indices_to_zero]
        indices_to_flip = []


        mean_zero = np.random.uniform(0, len(untouched_indices))
        std_dev_zero = np.random.uniform(1, len(untouched_indices) / 10)
        
        while len(indices_to_flip) < n and len(untouched_indices) > 0:
            gaussian_index = int(np.random.normal(mean_zero, std_dev_zero))
            if 0 <= gaussian_index < len(untouched_indices):
                zero_index = untouched_indices[gaussian_index]
                if zero_index not in indices_to_flip:
                    indices_to_flip.append(zero_index)


        for idx in indices_to_flip:
            modified_arr[idx] = -modified_arr[idx]

        flipped_arrays.append(modified_arr)

    return flipped_arrays


def random_label_flip_and_zero(arr, m, n_list, zero_n_list):

    if len(n_list) != m or len(zero_n_list) != m:
        raise ValueError("The length of n_list and zero_n_list must be equal to m.")
    
    length = len(arr)
    flipped_arrays = []

    for i in range(m):
        n = n_list[i]
        zero_n = zero_n_list[i]

        # Randomly select indices to flip
        indices_to_flip = np.random.choice(length, n, replace=False)

        # Create a copy of the array to flip the labels
        modified_arr = arr.copy()
        modified_arr[indices_to_flip] = -modified_arr[indices_to_flip]

        # Identify the untouched indices
        untouched_indices = np.setdiff1d(np.arange(length), indices_to_flip)

        # Randomly select indices from the untouched indices to set to 0
        indices_to_zero = np.random.choice(untouched_indices, zero_n, replace=False)

        # Set the chosen indices to 0
        modified_arr[indices_to_zero] = 0

        flipped_arrays.append(modified_arr)

    return flipped_arrays


# np.random.seed(42)


X, y = make_classification(n_samples=130000, n_features=3, n_informative=3, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.5, 0.5], flip_y=0.05, class_sep=1.5)
y = 2*y - 1


arr = y

m = 5  # Number of random selections
n_list = [50, 100, 150, 200, 250]  # Different number of indices to flip for each selection
zero_n_list = [20, 40, 60, 80, 100]  # Different number of indices to set to 0 for each selection
flipped_arrays = random_label_flip_and_zero(arr, m, n_list, zero_n_list)


print("Original = ", arr)
for i, modified_arr in enumerate(flipped_arrays):
    print(f"Array {i+1}:")
    print(modified_arr)



