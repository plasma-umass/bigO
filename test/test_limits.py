import bigO
import numpy as np


def fact(x: int) -> int:
    v = 1
    for i in range(x):
        v *= i
    return v


@bigO.limits(
    lambda xs: len(xs),
    time=0.07,
    mem=32_000,
    length=3000,
)
def factorialize(xs: list[int]) -> list[int]:
    new_list = [fact(x) for x in xs]
    return new_list


# Exercise the same input
for n in np.random.normal(1500, 500, 500):
    factorialize([i for i in range(max(0, int(n)))])
