import random
import bigO
import numpy as np


def square(x):
    for i in range(200):
        z = x * x
    return z


def multiply(x, n):
    z = 1
    for i in range(100):
        z += square(x)
    return x * n


def multiply_add(x, n):
    for i in range(200):
        z = multiply(x, n) + 1
    return z


i = 1


class Test:
    def __init__(self):
        # print("foo ON")
        global i
        self.foo = (
            11111111111111111111111111111111111111111111111111111111111111111111111111111111111
            + i
        )
        i += 1
        # print("foo OFF")


@bigO.track(lambda n, **kwargs: n)
def linear_function(n, multiplier=2):
    # print("linear_function")
    q = [Test() for i in range(1_000)]
    return
    return [multiply(square(x), multiplier) for x in range(n)]


@bigO.track(lambda x, y: len(x) + len(y))
def linear_function_2(x, y):
    for i in range(999):
        z = x + y
    return x + y


@bigO.track(lambda x: len(x))
def nlogn_function(x):
    return sorted(x)


@bigO.track(lambda n: n)
def quadratic_function(n):
    # print(f"quadratic function {n=}")
    x = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(100):
                q = multiply(square(x), 1.0)
                # x += 1


# Example function calls
for i in range(10):
    # print("example functions.")
    linear_function(10)
    # linear_function(random.randint(1,1_000_000))

import sys

sys.exit(0)

for i in range(10):
    # print("do something.")
    quadratic_function(10)  #  * random.randint(1, 10))

for i in range(10):
    # print("do something.")
    quadratic_function(10 * random.randint(1, 10))
for i in range(10):
    # print("combine lists")
    linear_function_2(
        list(range(random.randint(100_000, 200_000))),
        list(range(random.randint(100_000, 200_000))),
    )
for i in range(10):
    # print("sort me")
    nlogn_function(list(np.random.rand(random.randint(100_000, 1_000_000))))
