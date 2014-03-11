
//define const table for location lookup
__device__ __constant__ int DATA_TABLES[3][10];


#define DATA_IDX_WEEKDAY_ERRAND_DSTN 0
#define DATA_IDX_WEEKEND_ERRAND_DSTN 1
#define DATA_IDX_PANDEMIC_INFECTIOUS_CURVE 2
#define DATA_IDX_SEASONAL_INFECTIOUS_CURVE 3


__global__ void random_pdf_selector(int *output_array, int table_index, int N, int global_rand_offset)
{
	int myPos = (2* blockIdx.x * blockDim.x + threadIdx.x);

	threefry2x32_key_t tf_k = {{rng_seed_,rng_seed_2}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;
	int myRandOffset = myPos + global_rand_offset;
	while(myPos < N)
	{
		threefry2x32_ctr_t tf_ctr = {{myRandOffset,myRandOffset+1}};
		u.c = threefry2x32(tf_ctr, tf_k);

		//fractionate
		
		float yval = (float) u.i[0] / UNSIGNED_MAX;
		
		
		int i = 0;
		do							//search out y-val
		{
			yval -= DATA_TABLES[table_index]
		}while(yval > 0.0f);		//if it's <0 then we have hit the target yval, return category
			

		output_array[myPos] 
			
		myPos += (2 * blockDim.x);
		myRandOffset = myPos + globalRandOffset;
	}
}