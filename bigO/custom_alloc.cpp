#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mutex>
#include <time.h>
#include <unordered_map>

// Track current and peak memory use statistics.
class MemoryStatistics {
public:
  MemoryStatistics()
  {
    reset();
  }
  void reset() {
    currentMemoryUsage = 0;
    peakMemoryUsage = 0;
    totalObjectsAllocated = 0;
  }
  auto getCurrentMemoryUsage() const {
    return currentMemoryUsage;
  }
  auto getPeakMemoryUsage() const {
    return peakMemoryUsage;
  }
  auto getTotalObjectsAllocated() const {
    return totalObjectsAllocated;
  }
  void allocate(size_t sz) {
    totalObjectsAllocated++;
    currentMemoryUsage += sz;
    if (currentMemoryUsage > peakMemoryUsage) {
      peakMemoryUsage = currentMemoryUsage;
    }
  }
  void deallocate(size_t sz) {
    // Defensive programming
    if (sz <= currentMemoryUsage) {
      currentMemoryUsage -= sz;
    }
  }
private:
  size_t currentMemoryUsage = 0;
  size_t peakMemoryUsage = 0;
  size_t totalObjectsAllocated = 0;
};

static MemoryStatistics stats;

// Track object sizes
class SizeTracker {
public:
  void setSize(void * ptr, size_t sz) {
    std::lock_guard<std::mutex> lk(_objSizesLock);
    _objSizes[ptr] = sz;
  }
  size_t getSize(void * ptr) {
    std::lock_guard<std::mutex> lk(_objSizesLock);
    return _objSizes[ptr];
  }
private:  
  std::unordered_map<void *, size_t> _objSizes;
  std::mutex _objSizesLock;
};

SizeTracker objSizes;

// Function declarations
void custom_free(void *ctx, void *ptr);
void *custom_malloc(void *ctx, size_t size);
void *custom_calloc(void *ctx, size_t nelems, size_t size);
void *custom_realloc(void *ctx, void *ptr, size_t size);

// The original allocator, used for actually performing allocation operations
static PyMemAllocatorEx origAlloc;

// The wrapping allocator
static PyMemAllocatorEx alloc = {
  .ctx = NULL,
  .malloc = custom_malloc,
  .calloc = custom_calloc,
  .realloc = custom_realloc,
  .free = custom_free,
};

// Custom malloc implementation
void *custom_malloc(void *ctx, size_t size) {
  auto * ptr = origAlloc.malloc(ctx, size);
  objSizes.setSize(ptr, size);
  stats.allocate(size);
  return ptr;
}

// Custom calloc implementation
void *custom_calloc(void *ctx, size_t nelem, size_t elsize) {
  auto size = nelem * elsize;
  auto * ptr = origAlloc.calloc(ctx, size, 1);
  objSizes.setSize(ptr, size);
  stats.allocate(size);
  return ptr;
}

// Custom realloc implementation
void *custom_realloc(void *ctx, void *ptr, size_t size) {
  if (ptr == NULL) {
    return custom_malloc(ctx, size);
  }
  if (size == 0) {
    custom_free(ctx, ptr);
    return NULL;
  }
  auto sz = objSizes.getSize(ptr);
  stats.deallocate(sz);
  auto * new_ptr = origAlloc.realloc(ctx, ptr, size);
  objSizes.setSize(new_ptr, size);
  stats.allocate(size);
  return new_ptr;
}

// Custom free implementation
void custom_free(void *ctx, void *ptr) {
  auto sz = objSizes.getSize(ptr);
  origAlloc.free(ctx, ptr);
  stats.deallocate(sz);
}

// Set the custom allocator
void set_custom_allocator() {
    PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &alloc);
    PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &alloc);
}

// Reset the custom allocator
void reset_custom_allocator() {
    PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &origAlloc);
    PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &origAlloc);
}

static PyObject* enable(PyObject *self, PyObject *args) {
  set_custom_allocator();
  Py_RETURN_NONE;
}

static PyObject* disable(PyObject *self, PyObject *args) {
  reset_custom_allocator();
  Py_RETURN_NONE;
}

static PyObject* reset_statistics(PyObject* self, PyObject* args) {
  stats.reset();
  Py_RETURN_NONE;
}

static PyObject* get_current_allocated(PyObject* self, PyObject* args) {
  return PyLong_FromSize_t(stats.getCurrentMemoryUsage());
}

static PyObject* get_peak_allocated(PyObject* self, PyObject* args) {
  return PyLong_FromSize_t(stats.getPeakMemoryUsage());
}

static PyObject* get_objects_allocated(PyObject* self, PyObject* args) {
  return PyLong_FromSize_t(stats.getTotalObjectsAllocated());
}

// Python methods
static PyMethodDef CustomAllocMethods[] = {
    {"reset_statistics", reset_statistics, METH_NOARGS, "Clear allocation statistics."},
    {"enable", enable, METH_NOARGS, "Enable allocation statistics collection."},
    {"disable", disable, METH_NOARGS, "Disable allocation statistics collection."},
    {"get_current_allocated", get_current_allocated, METH_NOARGS, "Get the current allocated memory size."},
    {"get_peak_allocated", get_peak_allocated, METH_NOARGS, "Get the peak allocated memory size."},
    {"get_objects_allocated", get_objects_allocated, METH_NOARGS, "Get the total number of objects allocated."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef customalloc_module = {
    PyModuleDef_HEAD_INIT,
    "customalloc",
    "Tracking memory allocator",
    -1,
    CustomAllocMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_customalloc(void) {
  //    seed_xorshift(time(NULL));  // Seed the random number generator
    PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &origAlloc);
    stats.reset();
    return PyModule_Create(&customalloc_module);
}
