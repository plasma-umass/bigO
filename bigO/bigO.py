import atexit
import hashlib
import inspect
import json
import random
import time
import marshal

from collections import defaultdict
from functools import wraps
from typing import Any, Callable, List, Literal, TypedDict

import numpy as np

from bigO.analysis import CheckBounds, CheckLimits, FunctionData, Result

system_name = "bigO"


class BigOError(Exception):
    """
    Exceptions triggered by performance checks failing against the specified bounds or limits.
    Message is a short explanation.
    Explanation is a longer explanation of the error.
    """

    def __init__(self, message, explanation):
        super().__init__(message)
        self.message = message
        self.explanation = explanation

    def __str__(self):
        return self.message


class RuntimeData(TypedDict):
    tests: dict[str, Any]
    observations: list[Any]


# Global dictionary to store performance data
_performance_data: dict[str, RuntimeData] = defaultdict(
    lambda: RuntimeData(tests={}, observations=[])
)

# Where the performance data is stored.
_performance_data_filename = f"{system_name}_data.json"

# Hashes of function implementations, used to discard outdated perf info for modified functions
_hash_function_values: dict[str, Any] = {}

# python_version = (sys.version_info[0], sys.version_info[1])
# # Disabled for now
# use_sys_monitoring = False
# # use_sys_monitoring = python_version >= (3,12)
# TOOL_ID = 1
# if use_sys_monitoring:
#     sys.monitoring.use_tool_id(TOOL_ID, system_name)


def set_performance_data_filename(fname: str) -> str:
    """Changes the file name where performance data is stored
    and loads the performance data.

    Returns the previous file name.
    """
    global _performance_data_filename
    global _performance_data
    old_performance_data_filename = _performance_data_filename
    _performance_data_filename = fname
    try:
        with open(_performance_data_filename, "r") as infile:
            _performance_data = json.load(infile)
    except FileNotFoundError:
        _performance_data = dict()
    return old_performance_data_filename


def _get_performance_data(key):
    if key not in _performance_data:
        _performance_data[key] = RuntimeData(tests={}, observations=[])
    return _performance_data[key]


def _gather_function_data(func: Callable) -> FunctionData:
    full_name = _function_full_name(func)
    hash = _hash_function_values.get(full_name, None)
    assert hash is not None, "Function hash not found"
    records = [
        r for r in _performance_data[full_name]["observations"] if r["hash"] == hash
    ]
    lengths = [r["length"] for r in records]
    times = [r["time"] for r in records]
    mems = [r["memory"] for r in records]
    function_data = FunctionData(
        function_name=_function_name(func),
        file_name=_function_file_name(func),
        lengths=np.array(lengths),
        times=np.array(times),
        mems=np.array(mems),
    )
    return function_data


def function_hash_value(func: Callable) -> str:
    """
    Returns the hash value of the function implementation.
    """
    code = marshal.dumps(func.__code__)
    return hashlib.sha256(code).hexdigest()


def _function_file_name(func: Callable) -> str:
    """
    Returns the file name of the function.
    """
    module = inspect.getmodule(func)
    return module.__file__ if module and module.__file__ else "<unknown>"


def _function_name(func: Callable) -> str:
    """
    Returns the name of the function.
    """
    return func.__name__


def _function_full_name(func: Callable) -> str:
    """
    Returns the full name of the function (file + name).
    """
    return str((_function_file_name(func), _function_name(func)))


def _is_tracked(func: Callable) -> bool:
    """
    Check if the given function or any function in its __wrapped__ chain
    has been marked as tracked.
    """
    current = func
    while current:
        if getattr(current, "__is_tracked__", False):
            return True
        current = getattr(current, "__wrapped__", None)
    return False


def track(length_function: Callable[..., int]) -> Callable:
    """
    A decorator to measure and store performance metrics of a function.

    Args:
        length_function (callable): A function that calculates the "length"
                                       of one or more arguments.

    Returns:
        callable: The decorated function.
    """

    def decorator(func: Callable) -> Callable:

        # If the function is already tracked, return it as is.
        if _is_tracked(func):
            return func

        # Store a hash of the code to enable discarding old perf data if the
        # function has changed
        hash_value = function_hash_value(func)

        # Get the full name of the function (file + name), and save the hash value.
        full_name = _function_full_name(func)
        _hash_function_values[full_name] = hash_value

        @wraps(func)
        def wrapper(*args, **kwargs):
            import customalloc

            # Calculate the length based on the provided computation
            length = length_function(*args, **kwargs)

            # Start measuring time and memory
            start_time = time.perf_counter()
            customalloc.reset_statistics()
            customalloc.enable()

            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                peak = customalloc.get_peak_allocated()
                nobjects = customalloc.get_objects_allocated()
                customalloc.disable()

                # Store the performance data. Only allow non-zero
                # lengths to avoid issues downstream when computing
                # logs of lengths.
                if length > 0:
                    perf_data = {
                        "hash": hash_value,
                        "length": length,
                        "time": elapsed_time,
                        "memory": peak,  # Peak memory usage in bytes
                        "nobjects": nobjects,
                    }
                    _get_performance_data(full_name)["observations"].append(perf_data)
            return result

        wrapper.__is_tracked__ = True  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _raise_on_failure(results: List[Result]) -> None:
    """
    Raise an exception if any of the results indicate a failure.
    The exception should have all the failed messages and details.
    """
    messages = []
    explanations = []
    for result in results:
        if not result.success:
            messages.append(result.message)
            explanations.append(result.details)

    if messages:
        raise BigOError("\n".join(messages), "\n".join(explanations))


def bounds(
    length_function: Callable[..., int],
    time: str | None = None,
    mem: str | None = None,
    interval: int | None = None,
) -> Callable:
    """
    A decorator to check the performance of a function against specified bounds.
    It will store performance data dynamically, and compare it with the given bounds
    offline, or dynamically if an interval between checks is specified.

    Args:
        length_function (callable): A function that calculates the "length"
                                       of one or more arguments.
        time (str, optional): A string representing the time complexity bound.
        mem (str, optional): A string representing the memory complexity bound.
        interval (int, optional): Number of calls between dynamic checks.

    Returns:
        callable: The decorated function.

    Raises during dynamic checking:
        BigOError: If the function's performance does not meet the specified bounds.
    """

    def decorator(func: Callable) -> Callable:
        # wrap the function with the track decorator
        full_name = _function_full_name(func)
        tracked = track(length_function)(func)

        # record what we're testing for
        tests = {}
        if time:
            tests["time_bound"] = time
        if mem:
            tests["mem_bound"] = mem

        _performance_data[full_name] = _get_performance_data(full_name)
        _performance_data[full_name]["tests"] = (
            _performance_data[full_name]["tests"] | tests
        )

        if interval is None:

            return tracked

        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                result = tracked(*args, **kwargs)

                count = len(_performance_data[full_name]["observations"])
                if count > 0 and count % interval == 0:  # type: ignore
                    _raise_on_failure(check(func))

                return result

            return wrapper

    return decorator


def limits(
    length_function: Callable[..., int] | None = None,
    length: int | None = None,
    time: float | None = None,
    mem: float | None = None,
    interval: int | None = None,
) -> Callable:
    """
    A decorator to check the performance of a function against specified hard limits.
    It will store performance data dynamically, and check it against the given limits
    offline, or dynamically if an interval between checks is specified.

    Args:
        length_function (callable, optional): A function that calculates the "length"
                                       of one or more arguments.
        length (int, optional): The maximum allowed input length.
        time (float, optional): The maximum allowed time in seconds.
        mem (float, optional): The maximum allowed memory in bytes.
        interval (int, optional): Number of calls between dynamic checks.

    Returns:
        callable: The decorated function.

    Raises during dynamic checking:
        BigOError: If the function's performance does not meet the specified limits.
    """

    if length_function is None:
        if length is not None:
            raise ValueError("length_function must be provided if length_limit is set")
        length_function = lambda *args, **kwargs: -1

    def decorator(func: Callable) -> Callable:
        tests = {}
        if time:
            tests["time_limit"] = time
        if mem:
            tests["mem_limit"] = mem
        if length:
            tests["length_limit"] = length

        full_name = _function_full_name(func)
        tracked = track(length_function)(func)

        _performance_data[full_name] = _get_performance_data(full_name)
        _performance_data[full_name]["tests"] = (
            _performance_data[full_name]["tests"] | tests
        )

        if interval is None:

            return tracked

        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                result = tracked(*args, **kwargs)

                count = len(_performance_data[full_name]["observations"])
                if count > 0 and count % interval == 0:  # type: ignore
                    _raise_on_failure(check(func))

                return result

            return wrapper

    return decorator


def check(func: Callable) -> List[Result]:
    """
    Programmatically run the checks for a given function.
    Generates an exception if the function does not pass its bounds or limits checks.
    """
    full_name = _function_full_name(func)
    tests = _performance_data[full_name]["tests"]
    function_data = _gather_function_data(func)
    results = []

    if "length_limit" in tests:
        check = CheckLimits(
            function_data,
            tests.get("time_limit", None),
            tests.get("mem_limit", None),
            tests.get("length_limit", None),
        )
        result = check.run()
        results += [result]
    if "mem_bound" in tests or "time_bound" in tests:
        check = CheckBounds(
            function_data, tests.get("time_bound", None), tests.get("mem_bound", None)
        )
        result = check.run()
        results += [result]
    return results


def ab_test(
    length_function: Callable[..., int],
    alt: Callable,
    metric: Literal["time", "mem", "both"] = "time",
) -> Callable:
    """
    A decorator to perform A/B testing between a function and an alternative implementation.
    It will store performance data for both implementations and randomly choose between them.
    The performance data will be compared offline.

    Args:
        length_function (callable): A function that calculates the "length"
                                       of one or more arguments.
        alt (callable): An alternative implementation to compare against.
        metric (str, optional): The performance metric to compare ("time", "mem", or "both").

    Returns:
        callable: The decorated function.
    """

    def decorator(func: Callable) -> Callable:
        full_name = _function_full_name(func)
        alt_full_name = _function_full_name(alt)
        tracked = track(length_function)(func)
        alt_tracked = track(length_function)(alt)

        metrics = [metric] if metric != "both" else ["time", "mem"]

        tests = {"abtest": (alt_full_name, metrics)}

        _performance_data[full_name] = _get_performance_data(full_name)
        _performance_data[full_name]["tests"] = (
            _performance_data[full_name]["tests"] | tests
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            if random.random() < 0.5:
                return alt_tracked(*args, **kwargs)
            else:
                return tracked(*args, **kwargs)

        return wrapper

    return decorator


@atexit.register
def save_performance_data() -> None:
    """
    Saves the collected performance data to a JSON file at program exit.
    """

    # Load any saved data into a dictionary.
    global _performance_data_filename
    try:
        with open(_performance_data_filename, "r") as infile:
            old_data = json.load(infile)
        # Discard any outdated entries.
        for full_name in old_data:
            if full_name in _hash_function_values:
                validated = []
                current_hash = _hash_function_values[full_name]
                for entry in old_data[full_name]["observations"]:
                    if entry["hash"] == current_hash:
                        validated.append(entry)
                old_data[full_name]["observations"] = validated

    except FileNotFoundError:
        old_data = {}
        pass

    # Merge the old with the new dictionary
    for key, function_data in old_data.items():

        if key in _performance_data:
            # Key exists in both dicts; extend the list from performance_data with the new entries
            _performance_data[key]["observations"].extend(function_data["observations"])
            # performance_data[key]["tests"] = function_data["tests"]
        else:
            # Key only exists in old_data; add it to performance_data
            _performance_data[key] = RuntimeData(
                tests=function_data["tests"], observations=function_data["observations"]
            )

    with open(_performance_data_filename, "w") as f:
        json.dump(_performance_data, f, indent=4)
