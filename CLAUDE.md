# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bigO is a Python package that automatically measures empirical computational complexity (time and space) of functions. It uses decorators to track function performance and generates reports with fitted complexity curves.

## Build and Development Commands

```bash
# Install the package (builds the C++ extension)
pip install -e .

# Run tests (from the test/ directory)
cd test && ./run_tests.sh

# Generate complexity report (run after executing tracked functions)
python3 -m bigO                    # PDF output (bigO.pdf)
python3 -m bigO --html             # HTML output (bigO.html)
python3 -m bigO --open-report      # Open report in browser
python3 -m bigO --debug            # Show debug output
```

## Architecture

### Core Module (`bigO/bigO.py`)
Provides decorators for tracking and analyzing function performance:
- `@track(length_fn)` - Measures time and memory for inference
- `@bounds(length_fn, time="O(...)", mem="O(...)")` - Verifies declared complexity bounds
- `@limits(length_fn, time=..., mem=..., length=...)` - Enforces hard runtime limits
- `@ab_test(length_fn, alt=other_fn)` - A/B performance comparison

Unit testing utilities (no JSON persistence, isolated state):
- `assert_bounds(func, length_fn, inputs, time=..., mem=...)` - One-liner complexity assertion
- `no_persistence()` - Context manager to disable JSON file writing
- `disable_persistence()` / `enable_persistence()` - Manual persistence control
- `clear_performance_data()` - Reset global state between tests

Performance data is accumulated in `bigO_data.json` and persisted at program exit.

### Analysis Module (`bigO/analysis.py`)
Defines analysis classes that implement the `Analysis` abstract base class:
- `InferPerformance` - Fits data to complexity models
- `CheckBounds` - Validates declared bounds against observations
- `CheckLimits` - Checks hard limits on time/memory/input size
- `ABTest` - Performs segmented permutation tests between two implementations

### Models Module (`bigO/models.py`)
Complexity model definitions using `FunctionCanonicalForm(r, s, t, u)` representing `r^n * n^s * (log n)^t * (log log n)^u`. Supports: O(1), O(log(log(n))), O(log(n)), O(log(n)^2), O(log(n)^3), O(sqrt(n)), O(n), O(n*log(n)), O(n^2), O(n^3), O(n^k), O(2^n).

Model selection uses AIC (Akaike Information Criterion) with p-value computation via bootstrap resampling.

### Memory Tracking (`bigO/custom_alloc.cpp`)
C++ Python extension that intercepts Python's memory allocator to track peak memory usage during function execution. Provides `enable()`, `disable()`, `reset_statistics()`, `get_peak_allocated()`, and `get_objects_allocated()` functions.

### Report Generation (`bigO/__main__.py`)
CLI tool using Click that reads `bigO_data.json`, runs all analysis items, and generates PDF or HTML reports with matplotlib/seaborn visualizations.

## Key Implementation Details

- Function implementations are hashed to automatically discard stale performance data when code changes
- Outliers are removed using IQR method before curve fitting
- A/B testing uses LOESS/spline smoothing, segments curves by crossover points, and performs permutation tests with Holm-Bonferroni correction
- The `customalloc` extension must be built before tracking works (requires C++20 compiler)
