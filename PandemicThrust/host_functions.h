#pragma once

#include "stdafx.h"
#include "simParameters.h"


//host methods for validation
locId_t host_recalcWorkplace(int myIdx, age_t myAge);
void host_setup_assignWorkplaceOrSchool(unsigned int rand_val, age_t * age_ptr,locId_t * workplace_ptr);
locId_t host_setup_fishWorkplace(unsigned int rand_val);
void host_setup_fishSchoolAndAge(unsigned int rand_val, age_t * output_age_ptr, locId_t * output_school_ptr);


extern float WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[NUM_BUSINESS_TYPES];
extern float WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[NUM_BUSINESS_TYPES];
extern float WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[NUM_BUSINESS_TYPES];
extern int WORKPLACE_TYPE_OFFSET_HOST[NUM_BUSINESS_TYPES];
extern int WORKPLACE_TYPE_COUNT_HOST[NUM_BUSINESS_TYPES];
extern float CHILD_AGE_CDF_HOST[CHILD_DATA_ROWS];
extern int CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[CHILD_DATA_ROWS];
extern simRandOffsetsStruct_t host_randOffsetsStruct[1];
extern SEED_T SEED_HOST[SEED_LENGTH];