import random
import bigO


def fact(x: int) -> int:
    v = 1
    for i in range(x):
        v *= i
    return v


@bigO.bounds(lambda xs: len(xs), time="O(n*log(n))", mem="O(n)")
def factorialize(xs: list[int]) -> list[int]:
    xs = xs + xs[0 : random.randint(0, len(xs))]
    new_list = [fact(x) for x in xs]
    return new_list


# Exercise the same input
for i in range(30):
    factorialize([i for i in range(i * 20)])
