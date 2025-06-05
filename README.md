## bigO

`bigO` automatically measures empirical computational complexity (in both time and space) of functions.
To use `bigO`, you just need to add a `@bigO.track` decorator to your function together with a "length" function (the "n").
You can then run your function as usual, ideally along a wide range of inputs, and `bigO` will report the computational
complexity of that function.

`bigO` accumulates its results in a file named `bigO_data.json` in the local directory;
you can then generate a graph of time and space complexity for each tracked function by running `python3 -m bigO.graph`.

### Demonstration

#### Inferring Time and Space Bounds

The file `test/facts.py` is a small program that demonstrates `bigO`.

```python
import bigO

def fact(x: int) -> int:
  v = 1
  for i in range(x):
    v *= i
  return v

@bigO.track(lambda xs: len(xs))
def factorialize(xs: list[int]) -> list[int]:
  new_list = [fact(x) for x in xs]
  return new_list

# Exercise the function - more inputs are better!
for i in range(30):
    factorialize([i for i in range(i * 100)])
```

Now run the program as usual:

```bash
python3 test/facts.py
```

Now you can easily generate a graph of all tracked functions. Just run the following command in the same directory.

```bash
python3 -m bigO
```

This command creates the file `bigO.pdf` that contains graphs like this:

![infer](https://github.com/user-attachments/assets/9df423db-578f-4532-9277-dc08f7145797)

#### Verifying Time and Space Bounds

`bigO` will also verify declared bounds of functions.  The file `test/facts_bounds.py` declares
`factorialize` as follows:

```python
@bigO.bounds(lambda xs: len(xs),
             time="O(n*log(n))",
             mem="O(n)")
def factorialize(xs: list[int]) -> list[int]:
    ...
```

Running
```bash
python3 -m bigO
```

now creates this plot, showing that the timing data matches a worse bound than the declared bound
and that the memory data matches the declared bound:
  
![bounds](https://github.com/user-attachments/assets/a4c2e740-110f-4f04-9915-66876083b11c)

The analysis currently supports these performance models:
`O(1)`, `O(log(log(n)))`, `O(log(n))`, `O(log(n)**2)`, `O(log(n)**3)`, `O(sqrt(n))`, `O(n)`, `O(n*log(n))`, `O(n**2)`, `O(n**3)`, `O(n**k)`, and `O(2**n)`.  It is trivial to add additional forms.

#### Lightweight A/B Performance Experiments

`bigO` also lets you run light-weight A/B performance tests.  The file `tests/test_ab_sort.py` demonstrates this.
It includes two sorting functions:

```python
import random
import numpy as np
from bigO.bigO import ab_test

def insertion_sort(arr: np.ndarray) -> np.ndarray:
    ...

@ab_test(lambda x: len(x), alt=insertion_sort)
def quick_sort(arr: np.ndarray) -> np.ndarray:
    ...

for i in range(200):
    quick_sort(np.random.rand(random.randint(1, 100)))
```

The `quick_sort` function is annotated to indicate the `bigO` should 
compare the time and memory of that function to `insertion_sort`.  At run time,
`bigO` will randomly select which of the two functions to run on each call to `quick_sort`.

After running the program,

```bash
python3 -m bigO
```

compares the running times across the input size range, identifying segments where one 
function performs statistically significantly better than the other, as in the following,
which shows smoothed performance curves, shaded regions where one function is better than the other, as
well as the p-values for each segment's statistical test.

![abtest](https://github.com/user-attachments/assets/823a26cf-2bc5-4958-a47a-c6f9a253ce33)

#### Verifying Hard Limits on Time, Space, and Input Size

`bigO` also lets you verify at run time that function calls do not exceed hard limits on
time, memory, or input size.  The file `test/test_limits_insertion_sort.py` demonstrates
this:

```python
@limits(len, 
        time=0.1, 
        mem=1_000_000, 
        length=1800)
def insertion_sort(arr: np.ndarray) -> np.ndarray:
    ...
```

After running that program

```bash
python3 -m bigO
```

produces the following plots, showing the histograms of those metrics, as well as the
specified limits.

![limits](https://github.com/user-attachments/assets/6009c9a5-0e3e-449f-9a2d-88bc47ac462c)

You can also specify only one or two of the limits instead of all three.  This is perhaps
less broadly applicable than bounds checking but may be useful in some contexts.

### Technical Details

#### Curve-fitting

`bigO`'s curve-fitting based approach is directly inspired by
["Measuring Empirical Computational
Complexity"](https://theory.stanford.edu/~aiken/publications/papers/fse07.pdf)
by Goldsmith et al., FSE 2007, using log-log plots to fit a power-law distribution.

Unlike that work, `bigO` also measures space complexity by
tracking memory allocations during function execution. 

In addition, `bigO` uses a more general curve fitting approach that can handle
complexity classes that do not follow the power law, and it uses
the [AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion) to
select the best model.  Further, `bigO` measures the statistical significance of its complexity inference
results via p-values computed by the technique outlined in [An Empirical Investigation of Statistical Significance in NLP](https://aclanthology.org/D12-1091.pdf) by Berg-Kirkpatrick, Burkett, and Klein, 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pages 995–1005.

For A/B testing, `bigO` smooths the performance curves for the two functions, segments the input range by approximating crossover points for those curves, and then performs a standard permutation test to determine whether the different in performance between the function across that range is statistically significant. The test statistic is the area between the two curves, as approximated by numerical integration via the trapezoid rule.

A version of `bigO` matching the log-log approach of the paper above can be run as follows:

```bash
python3 -m bigO.graph
```

This command creates the file `bigO.pdf` that contains graphs like this:

![bigO](https://github.com/user-attachments/assets/8428180b-a454-4fc7-822c-7a130f9ba54e)

### Caveats

* `bigO` assumes all length functions are constant.  The instrumentation of nested tracked calls may add overhead 
that effects the checking of hard limits, but nested tracked calls will not impact asymptotic bounds.

* During A/B testing, no nesting tracked functions from within the two variants.  Doing so may impact the timing comparisions

* Python's threading model is not amenable to precisely measuring concurrent calls.  `bigO` is not designed to handle threads.

