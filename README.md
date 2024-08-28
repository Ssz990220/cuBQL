# cuBQL ("cubicle") - A CUDA BVH Build and Query Library

'Cubicle' (cuBQL) is a (mostly) header-only CUDA/C++ library for the
easy and efficient GPU-construction and -traversal of bounding volume
hierarchies (BVHes), with the ultimate goal of providing the tools and
infrastructure to realize a wide range of (GPU-accelerated) spatial
queries over various geometric primitives.

CuBQL is largely inspired by two libraries: the standard template
library (`STL`), and `cub`. Like those two libraries cuBQL largely
relies on header-only CUDA/C++ code, and on the use of templates and
lambda functions to make sure that certain key operations (like
traversing a BVH) can work for different primititive, different data
type and dimensionality (e.g., `float3` vs `int2`), multiple different
but similar geometric queries (e.g., `find closest point` vs
`k-nearest neighbor (kNN)` vs `signed distance functions (SDF)`, etc).

Throughout cuBQL, the main driving goal are robustness, generality,
and ease of use: each builder for each BVH type should always work for
all input types and dimensionality, and even for numerically
challenging input data.

# cuBQL Functionality - Overview

CuBQL offers four separate layers of functionality:

- `Abstract BVH Type` layer: defines the basic (GPU friendly) type(s)
  for different kinds of BVHes. In particular, the cuBQL bvh types is
  templated over what geometric space the BVH is to be built over;
  i.e., you can realize not only BVHes over `float3` data, but also
  BVHes over `int4`, `double2`, etc (cuBQL spans the entire space of
  {int,float,double,long}x{2,3,4,N}).

- `BVH builders` layer: provides a set of primarily GPU-side (but also
  some simple host side) builder(s) for the underlying BVH
  type(s). This level offers multiple different builders with
  different speed/quality tradeoffs (though the default `gpuBuilder`
  should work well for most cases).
  
- `BVH Traversal Templates` layer: though different types of geometric
  queries are often *similar in concept*, nevertheless they often
  slightly *differ in detail*. Instead of only providing a fixed set
  of very specific geometric queries cuBQL focusses on providing a set
  of traversal *templates* that, though the use of lambda functions,
  can easily be modified in their details. E.g., both a kNN and a find
  closest point query will build on the same `shrinking radius query`,
  with just different way of processing a given candidate primitive
  encountered during traversal.
  
- `Various (specific) Geometric Queries`, realized with the underlying
   layers. cuBQL provides these queries more as *samples* than
   anything else, fully assuming that many users will have
   requirements that the existing samples will not capture---but which
   these samples's use of the traversal templates should show how to
   realize.

# Supported BVH Type(s)

The main BVH type of this library is a binary BVH, where each node
contains that node's bounding box, as well as two ints, `count` and
`offset`.

```
  template<typename /*ScalarType*/T, 
           int /*Dimensionality*/D>
  struct BinaryBVH {
    struct CUBQL_ALIGN(16) Node {
      box_t<T,D> bounds;
      uint64_t   offset : 48;
      uint64_t   count  : 16;
    };

    Node     *nodes;
    uint32_t  numNodes;
    uint32_t *primIDs;
    uint32_t  numPrims;
  };
```

The `count` value is 0 for inner nodes, and for leaf nodes specifies
the number of primitives in this leaf. For inner nodes, the `offset`
indexes into the `BinaryBVH::nodes[]` array (the current node's two
children are at `nodes[offset]`, and `nodes[offset+1]`, respectively);
for leaf nodes it points into the `BinaryBVH::primIDs[]` array (i.e.,
that leaf contains `primID[offset+0]`, `primID[offset+1]`, etc).

A `WideBVH<N>` type (templated over BVH width) is supported as
well. WideBVH'es always have a fixed branching factor of N (i.e., a
fixed number of `N` children in each inner node); however, some of
these may be 'null' (marked as not valid). Note that most builders
will only work for binary BVHes; these can then "collapsed" into
Wide-BVHes.

For readability, cuBQL offers many pre-defined types for specific type
and dimensionality of data: for the (arguably) most common use case of
`float3` type of data there are `vec3f`s, `box3f`s, and `bvh3f`s (for
`float3` points, boxes, and binary BVHes, respectively); and many
samples use these types.  However, cuBQL is absolutely *not* limited
to just float3 data, but all BVH types (and traversal routines) are
templated over both scalar type (`float`, `int`, `double`, `long`) and
dimensionality (2,3,4). In fact, `bvh3f` is simply an instantiation
(not even a specialization) of `cuBQL::BinaryBVH<float,3>`.

# (on-GPU) BVH Construction

The main workhorse of this library is a CUDA-acclerated and `on
device` parallel BVH builder (with spatial median splits). The primary
feature of the BVH builder is its simplicity; i.e., it is still
"reasonably fast", but it is much simpler than other variants. Though
performance will obviously vary for different data types, data
distributions, etcpp, right now this builder builds a BinaryBVH over
10 million uniformly distributed random points in under 13ms; that's
not the fastest builder I have, but IMHO quite reasonable for most
applications. In addition to this `cuBQL::gpuBuilder()` there are also
various other builders, including a regular morton/radix builder, a
wide GPU builder (for BVHes with branching factors greater than 2), a
surface-area-heuristic (SAH) builder, and a modified morton/radix
builder that for numerically challenging inputs is significantly more
robust than a regular morton/radix builder.

For all builders, the overall build process is always the same: Create
an array of bounding boxes (one box per primitive), and call the
builder with a pointer to this array, and the number of
primitives. For GPU-side builders this array has to live in device (or
managed) memory; for host side builds it has to be in host
memory. Obiously, device side builders will create node and primitmive
ID arrays in device memory, the host builder will create these in host
memory.

Given such an array, the builder (in this case, for `float3` data)
gets invoked as follows:

```
#include "cuBQL/bvh.h"
...
	box3f *d_boxes  = 0;
	int    numBoxes = 0;
	userCodeForGeneratingPrims(&d_boxes,&numBoxes, ...);
	...
    cuBQL::BinaryBVH<float,3> bvh;
	cuBQL::BuildConfig buildParams;
    cuBQL::gpuBuilder(bvh,d_boxes,numBoxes,buildParams);
...
```
Builds for other data types (such as, e.g., `<int,4>` or <double,2>`)
work exactly the same way (though obviously, the scalar type and dimensionality of the
boxes has to be the same as that for the BVH).

The builder will not modify the `d_boxes[]` array; after the build is
complete the `bvh.primIDs[]` array contains ints referring to indices
in this array. This builder will properly handle "invalid prims" and
"empty boxes": Primitives that are not supposed to be included in the
BVH can simply use a box for which `lower.x > upper.x`; such
primitives will be detected during the build, and will simply get
excluded from the build process - i.e., they will simply not appear in
any of the leaves, but also not influence any of the (valid) bounding
boxes. However, behavior for NaNs, denorms, etc. is not
defined. Zero-volume primitives (ie, those with `box.lower ==
box.upper`) are considered valid primitives, and will get included in
the BVH.

The `BuildConfig` class can be used to influence things like whether
the BVH should be built with a surface area heuristic (SAH) cost
metric (more expensive build, but faster queries for some types of
inputs and query operations), or how coarse vs how fine the BVH should
be built (ie, at which point to make a leaf).

A few notes:

- For GPU builders one can optionally also pass a `cudaStream_t` if
  desired. All operations, synchronization, and memory allocs should
  happen in that stream.

- By default the GPU side builder(s) will allocate device memory; but
  it is also be possible to make them use managed memory or async
  device memory by passing the appropriate `cuBQL::GpuMemoryResource`
  to the builder.

- Following the same pattern as other libraries like tinyOBJ or STB,
  this library *can* be used in a header-only form: By default a
  included header file will only pull in the type and function
  *declaration*s, but specifying `CUBQL_GPU_BUILDER_IMPLEMENTATION` to
  1 will also pull in the implementation, so using this in one of
  one's source files allows the user to compile the builders with
  exactly the cmd-line flags, CUDA architecture, etc, that he or she
  desires. *Alternatively* (and purely optionally), when using `cmake`
  one can also link to one (or more) of specific pre-defined targets
  such as, for example, `cuBQL_cuda_float3` or `cuBQL_host_int4` that
  will then build that specific device and type specific builder(s).
  
# Dependencies

To use `cuBQL`, you need:

- CUDA, version 12 and up. In theory some versions of CUDA 11 should work too, but 
  using 12.2 and upwards is highly recommended.
- `cmake`

Under linux, these can be installed (all except CUDA) via:

	sudo apt install cmake cmake-curses-gui build-essential

# Building

As all builders *can* be used in a header-only form, cuBQL can be used
from within any compiler and build system, by simply providing the proper
include paths and including the `cuBQL/bvh.h` or other header files as required.

However, we strongly suggest to use `cmake`, include cuBQL as a cmake
`add_subdirectory(...)`, and then `target_link_libraries(...)` with
the desired cuBQL cmake target.

## Building in Header-only (explicit instantiation) mode:

- in your own CUDA sources (say, `userMain.cu`):
``` 
	#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
	#include <cuBQL/bvh.h>
	...
	void foo(...) {
	   cuBQL::gpuBuilder(...)
	}
```

- in your own `CMakeLists.txt`:
```
	add_subdirectory(<pathTo>/cuBQL)
	
	add_executable(userExec ... 
	   userMain.cu ...)
	
	target_link_libraries(userExec ...
	   cuBQL)
```

In this case, the 'cuBQL' target that we link to is only a cmake
`INTERFACE` target that merely sets up the right include paths, but
does not build any actual library.

## Building with predefined target (eg, for float3 data)

- in your own CUDA sources (say, `userMain.cu`):
```
    // do NOT define CUBQL_GPU_BUILDER_IMPLEMENTATION 
	#include <cuBQL/bvh.h>
	...
	void foo(...) {
	   cuBQL::gpuBuilder(...)
	}
```

- in your own `CMakeLists.txt`:
```
	add_subdirectory(<pathTo>/cuBQL)
	
	add_executable(userExec ... 
	   userMain.cu ...)
	
	target_link_libraries(userExec ...
	   cuBQL_cuda_float3)
```

