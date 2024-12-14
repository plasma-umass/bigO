import atexit
import hashlib
import inspect
import json
import time
import marshal
import sys

from collections import defaultdict
from functools import wraps
from typing import Any, Callable

# Global dictionary to store performance data
performance_data : dict[str, list[Any]] = defaultdict(list)

system_name = "bigO"

performance_data_filename = f"{system_name}_data.json"

hash_function_values : dict[str, Any] = {}

def set_performance_data_filename(fname: str) -> None:
    global performance_data_filename
    global performance_data
    performance_data_filename = fname
    try:
        with open(performance_data_filename, 'r') as infile:
            performance_data = json.load(infile)
    except FileNotFoundError:
        performance_data = dict()
        pass


TOOL_ID = 1
sys.monitoring.use_tool_id(TOOL_ID, "counter")
    
def track(length_computation: Callable[..., int]) -> Callable:
    """
    A decorator to measure and store performance metrics of a function.

    Args:
        length_computation (callable): A function that calculates the "length"
                                       of one or more arguments.

    Returns:
        callable: The decorated function.
    """
    def decorator(func: Callable) -> Callable:
        # Store a hash of the code for discarding old values if the
        # function has changed
        code = marshal.dumps(func.__code__)
        hash_value = hashlib.sha256(code).hexdigest()

        # Get the full name of the function (file + name)
        func_name = func.__name__
        module = inspect.getmodule(func)
        file_name = module.__file__ if module and module.__file__ else "<unknown>"
        full_name = str((func_name, file_name))
        hash_function_values[full_name] = hash_value
        python_version = (sys.version_info[0], sys.version_info[1])
        
        if python_version >= (3, 12):
            events = [sys.monitoring.events.INSTRUCTION]
            event_set = 0
            for event in events:
                event_set |= event
            TOOL_ID = 1
            sys.monitoring.set_local_events(TOOL_ID, func.__code__, event_set)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            instruction_count = 0
            def increment_counter(*args):
                nonlocal instruction_count
                instruction_count += 1
            def get_counter():
                nonlocal instruction_count
                return instruction_count
            def reset_counter():
                nonlocal instruction_count
                instruction_count = 0
                
            import customalloc
            # Calculate the length based on the provided computation
            length = length_computation(*args, **kwargs)

            # Start measuring time and memory
            start_time = time.perf_counter()
            customalloc.reset_statistics()
            customalloc.enable()
            
            if python_version >= (3, 12):
                # Count instructions
                sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.INSTRUCTION, increment_counter)
            try:
                result = func(*args, **kwargs)
            finally:
                if python_version >= (3, 12):
                    # Stop counting instructions
                    sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.INSTRUCTION, None)
                instructions_executed = get_counter()
                reset_counter()
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                
                peak = customalloc.get_peak_allocated()
                nobjects = customalloc.get_objects_allocated()
                customalloc.disable()
                
                # Store the performance data. Only allow non-zero
                # lengths to avoid issues downstream when computing
                # logs of lengths.
                if length:
                    new_entry = {
                        "hash" : hash_value,
                        "length": length,
                        "time": elapsed_time,
                        "instructions" : instructions_executed,
                        "memory": peak,  # Peak memory usage in bytes
                        "nobjects": nobjects,
                    }
                    performance_data[full_name].append(new_entry)
            return result
        return wrapper
    return decorator


@atexit.register
def save_performance_data() -> None:
    """
    Saves the collected performance data to a JSON file at program exit.
    """

    # Load any saved data into a dictionary.
    global performance_data_filename
    try:
        with open(performance_data_filename, 'r') as infile:
            old_data = json.load(infile)
        for full_name in old_data:
            if full_name in hash_function_values:
                validated = []
                current_hash = hash_function_values[full_name]
                for entry in old_data[full_name]:
                    if entry["hash"] == current_hash:
                        validated.append(entry)
                old_data[full_name] = validated
            
    except FileNotFoundError:
        old_data = {}
        pass

    # Merge the old with the new dictionary
    for key, value_list in old_data.items():

        if key in performance_data:
            # Key exists in both dicts; extend the list from performance_data with the new entries
            performance_data[key].extend(value_list)
        else:
            # Key only exists in old_data; add it to performance_data
            performance_data[key] = value_list
    
    with open(performance_data_filename, "w") as f:
        json.dump(performance_data, f, indent=4)
