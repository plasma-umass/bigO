import random
import numpy as np

from bigO.bigO import limits

import numpy as np


@limits(len, time=0.1, mem=1_000_000, length=1800)
def insertion_sort(arr: np.ndarray) -> np.ndarray:
    sorted_arr = arr.copy()
    n = sorted_arr.shape[0]

    for i in range(1, n):
        key = sorted_arr[i]
        j = i - 1
        while j >= 0 and sorted_arr[j] > key:
            sorted_arr[j + 1] = sorted_arr[j]
            j -= 1
        sorted_arr[j + 1] = key

    return sorted_arr


for i in range(100):
    # print("sort me!")
    insertion_sort(np.random.rand(random.randint(1, 2000)))
