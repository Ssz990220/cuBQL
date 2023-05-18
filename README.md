# cuBQL ("cubicle") - A CUDA BVH Build and Query Library

'Cubicle' (cuBQL) is a (intentionally) small library for
GPU-accelerated building and queries of Bounding Volume Hierarchies
(BVHs), with a focus on simplicity. I.e., all operations in this
library are designed to run on the GPU, run in parallel, and be
"reasonably" fast; but if in doubt this library picks simplicy (and
thus, maintainability, rubustness, and ease of use) over
speed-of-light performance.

To build this library (see below) you need CUDA 11.2 or later, and
cmake. This library uses "modern CMake".

# Supported BVH Type(s)

The main BVH type of this library is a binary BVH, where each node
contains that node's bounding box, as well as two ints, `count` and
`offset`.

```
  struct BinaryBVH {
    struct CUBQL_ALIGN(16) Node {
      box3f    bounds;
      uint64_t offset : 48;
      uint64_t count  : 16;
    };

    Node     *nodes;
    uint32_t  numNodes;
    uint32_t *primIDs;
    uint32_t  numPrims;
  };
```

The `count` value is 0 for inner node, and for leaf nodes
specifies the number of primitives in this leaf. For inner nodes, the
`offset` value points into the `BinaryBVH::nodes[]` array (its two
children are at offset, and offset+1, respectively); for leaf nodes it
points into the `BinaryBVH::primIDs` array (i.e., that leaf contains
`primID[offset+0]`, `primID[offset+1]`, etc).

A `WideBVH<N>` type (templated over BVH width) will soon be introduced
as well, but is not yet ready for release. 

Though most of the algorithms and data types in this library could
absoltely be templated over both dimensionality and underlying data
type (i.e., a BVH over `double4` data rather than `float3`), for sake
of readability in this particular implementation this has not been
done (yet?). If this is a feature you would like to have, please let
me know.

# Parallel GPU BVH Construction

The main workhorse of this library is a CUDA-acclerated and `on
device` parallel BVH builder (with spatial median splits). The primary
feature of the BVH builder is its simplicity; i.e., it is still
"reasonably fast", but it is much simpler than other variants. Though
performance will obviously vary for different data types, data
distributions, etcpp, right now this builder builds a BinaryBVH over
10 million uniformly distributed random points in under 13ms; that's
not the fastest builder I have, but I believe is still quite
reasonable.

To use that builder, you need to first create a (device-side) array of
`box3f`s---one for each primitive. Each `box3f` is simply two `float3`s:

```
  struct box3f {
    float3 lower, upper;
  };

```
Given such an array, the builder then gets invoked as follows:

```
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
...
	box3f *d_boxes;
	int numBoxes;
	userCodeForGeneratingPrims(d_boxes,numBoxes);
	...
    cuBQL::BinaryBVH bvh;
    cuBQL::gpuBuilder(bvh,d_boxes,numBoxes,maxLeafSize);
...
```
The builder will not modify the `d_boxes[]` array; after the build
is complete the `bvh.primIDs[]` array contains ints referring to indices in this 
array. This builder will properly handle "invalid prims" and "empty boxes":
Primitives that are not supposed to be included in the BVH can simply
use a box for which `lower.x > upper.x`; such primitives will be
detected during the build, and will simply get excluded from the build
process - i.e., they will simply not appear in any of the leaves, but
also not influence any of the (valid) bounding boxes. However,
behavior for NaNs, denorms, etc. is not defined. Zero-volume
primitives are considered valid primitives, and will get included in
the BVH.


A few notes:

- Optionally you can also pass a `cudaStream_t` if desired. All
  operations, synchronization, and memory allocs should happen in that
  stream.

- The builder uses `cudaMallocAsync`; this means the first build may
  be slower than subsequent ones.

- Following the same pattern as other libraries like tinyOBJ or STB,
  this library has most of its actual algorithms conditionally visible
  within the header files; by default a included header file will only
  pull in the type and function *declaration*s, but specifying
  `CUBQL_GPU_BUILDER_IMPLEMENTATION` will also pull in the
  implementation.

# Building

This library can be used/built in two ways: either standalone (with
various test cases and samples), or as a submodule for another
project.

## Use as a cmake submodule

If you are only interested in using the builder (and maybe queries)
within your own code, do the following:

- first, include this library in your own project, in a separate subdirectory
  (git submodules are a great way of doing this)
  
- within your own cmake file include this library

```
    # root CMakeLists.txt (your own)
    add_subdirectory(<pathTo>/cuBQL EXCLUDE_FROM_ALL)
```
- cmake-link to your owl project
```
    # root CMakeLists.txt (your own)
    add_executable(userExec .... myFile.cu otherFile.cpp)
	target_link_libraries(userExec cuBQL)
```
- The previous step should make sure that include paths to cuBQL etc are
  properly set for your own code. Now simply `#include "cuBQL/bvh.h" etc.
  
- In *one* of your files, `#define `CUBQL_GPU_BUILDER_IMPLEMENTATION
1` before including `cuBQL/bvh.h`.
  


## Building as a standalone project (and running some tests)

In the former case, build via cmake:

```
   mkdir build
   cd build
   cmake ..
   make
```
You can then, for example, run some simple benchmarks:

```
   # generate 1M uniform random points in [0,1]^3
   ./cuBQL_makePoints_uniform -n 1000000 -o dataPoints-1M
   # measure build perf for these points
   ./cuBQL_buildPerf dataPoints-1M
   ...
   # generate another 1M uniform random points in [0,1]^3
   ./cuBQL_makePoints_uniform -n 1000000 -o queryPoints-1M
   # run sample 'fcp' (find closest point) query
   ./cuBQL_fcp dataPoints-1M queryPoints-1M
```



