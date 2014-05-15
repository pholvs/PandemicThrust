#include "stdafx.h"
#include "host_functions.h"


locId_t host_recalcWorkplace(int myIdx, age_t myAge)
{
	randOffset_t myRandOffset = host_randOffsetsStruct->workplace_randOffset + (myIdx / 4);

	threefry2x64_key_t tf_k = {{(long) SEED_HOST[0], (long) SEED_HOST[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;
	threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};

	throw;

	//rand_union.c = threefry2x64(tf_ctr,tf_k);

	int rand_slot = myIdx % 4;

	locId_t workplace_val;
	host_setup_assignWorkplaceOrSchool(rand_union.i[rand_slot],&myAge,&workplace_val);

	return workplace_val;
}

void host_setup_assignWorkplaceOrSchool(unsigned int rand_val, age_t * age_ptr,locId_t * workplace_ptr)
{
	age_t myAge = *age_ptr;

	if(myAge == AGE_ADULT)
	{
		workplace_ptr[0] = host_setup_fishWorkplace(rand_val);
	}
	else
	{
		host_setup_fishSchoolAndAge(rand_val,age_ptr,workplace_ptr);
	}
}

void host_setup_fishSchoolAndAge(unsigned int rand_val, age_t * output_age_ptr, locId_t * output_school_ptr)
{
	float y = (float) rand_val / UNSIGNED_MAX;

	//fish out age group and resulting school type from CDF
	int row = 0;
	while(row < CHILD_DATA_ROWS - 1 && y > CHILD_AGE_CDF_HOST[row])
		row++;

	//int wp_type = CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[row];
	int wp_type = row + BUSINESS_TYPE_PRESCHOOL;

	//of this school type, which one will this kid be assigned to?
	float frac;
	if(row == 0)
		frac = y / (CHILD_AGE_CDF_HOST[row]);
	else
	{
		//we need to back out a PDF from the CDF
		float pdf_here = CHILD_AGE_CDF_HOST[row] - CHILD_AGE_CDF_HOST[row - 1];
		float y_here = y - CHILD_AGE_CDF_HOST[row - 1];

		frac =  y_here / pdf_here;
	}

	int type_count = WORKPLACE_TYPE_COUNT_HOST[wp_type];
	int business_num = frac * type_count;

	if(business_num >= type_count)
		business_num = type_count - 1;

	//how many other workplaces have we gone past?
	int type_offset = WORKPLACE_TYPE_OFFSET_HOST[wp_type];
	*output_school_ptr = business_num + type_offset;

	age_t myAge = (age_t) row;
	*output_age_ptr = myAge;
}

locId_t host_setup_fishWorkplace(unsigned int rand_val)
{
	float y = (float) rand_val / UNSIGNED_MAX;

	int row = 0;
	while(WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[row] < y && row < NUM_BUSINESS_TYPES - 1)
	{
		y -= WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[row];
		row++;
	}

	//of this workplace type, which number is this?
	float frac = y / WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[row];
	int type_count = WORKPLACE_TYPE_COUNT_HOST[row];
	int business_num = frac * type_count;  //truncate to int

	if(business_num >= type_count)
		business_num = type_count - 1;

	//how many other workplaces have we gone past?
	int type_offset = WORKPLACE_TYPE_OFFSET_HOST[row];

	locId_t ret = business_num + type_offset;
	return ret;
}
