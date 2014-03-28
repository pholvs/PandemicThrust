
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define HOST 0
#define GPU 1
#define CPU_THREADED 2

#define VECTOR_MODE GPU

/*
#if VECTOR_MODE == GPU
	typedef thrust::device_vector<int> vec_t;

#elif VECTOR_MODE == CPU_THREADED
	typedef thrust::device_vector<int> vec_t;

#else //default:  MODE == HOST
	typedef thrust::host_vector<int> vec_t;
#endif
*/

typedef thrust::device_vector<int> vec_t;

typedef thrust::host_vector<int> h_vec;
typedef thrust::device_vector<int> d_vec;


typedef vec_t::iterator IntIterator;
typedef thrust::device_ptr<int> d_ptr;

typedef thrust::tuple<IntIterator,IntIterator,IntIterator> IntIteratorTriple;
typedef thrust::zip_iterator<IntIteratorTriple> ZipIntTripleIterator;

typedef thrust::tuple<IntIterator,IntIterator,IntIterator,IntIterator> IntIteratorQuad;
typedef thrust::zip_iterator<IntIteratorQuad> ZipIntQuadIterator;

typedef thrust::tuple<IntIterator,IntIterator,IntIterator,IntIterator,IntIterator> IntIteratorFiveTuple;
typedef thrust::zip_iterator<IntIteratorFiveTuple> ZipIntFiveTupleIterator;

typedef int kval_t;