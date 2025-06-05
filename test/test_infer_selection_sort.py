import random
import numpy as np

from bigO.bigO import track


@track(len)
def selection_sort(arr: np.ndarray) -> np.ndarray:
    """
    Perform selection sort on a numpy array and return a new sorted array.

    Parameters:
        arr (np.ndarray): The input 1D array to sort.

    Returns:
        np.ndarray: A sorted copy of the input array.
    """
    sorted_arr = arr.copy()  # Create a copy to avoid modifying the input array
    n = len(sorted_arr)

    for i in range(n):
        # Assume the element at i is the minimum.
        min_index = i

        # Find the actual minimum element in the remaining unsorted array.
        for j in range(i + 1, n):
            if sorted_arr[j] < sorted_arr[min_index]:
                min_index = j

        # Swap the found minimum element with the first unsorted element.
        sorted_arr[i], sorted_arr[min_index] = sorted_arr[min_index], sorted_arr[i]

    return sorted_arr


for i in range(200):
    # print("sort me!")
    selection_sort(np.random.rand(random.randint(1, 500)))
