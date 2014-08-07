#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUB_STDERR

#include "stdafx.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/pair.h>
#include <thrust/set_operations.h>

#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/count.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
