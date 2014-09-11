#pragma once

#include "stdafx.h"
#include "simParameters.h"


//host methods for validation
locId_t host_recalcWorkplace(int myIdx, age_t myAge);
void host_setup_assignWorkplaceOrSchool(unsigned int rand_val, age_t * age_ptr,locId_t * workplace_ptr);
locId_t host_setup_fishWorkplace(unsigned int rand_val);
void host_setup_fishSchoolAndAge(unsigned int rand_val, age_t * output_age_ptr, locId_t * output_school_ptr);

errandContactsProfile_t host_recalc_weekdayErrandDests_assignProfile(
	personId_t myIdx, age_t myAge, 
	locId_t * output_dest1, locId_t * output_dest2);
void host_assignAfterschoolOrErrandDests_weekday(
	unsigned int rand_val1, unsigned int rand_val2,
	age_t myAge, int num_locations,
	locId_t * output_dest1, locId_t * output_dest2);
locId_t host_fishAfterschoolOrErrandDestination_weekday(
	unsigned int rand_val, age_t myAge);

void host_recalc_weekendErrandDests(personId_t myIdx, locId_t * errand_array_ptr);
void host_generateWeekendErrands(locId_t * errand_output_ptr, randOffset_t myRandOffset);
locId_t host_fishWeekendErrandDestination(unsigned int rand_val);


extern float WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[NUM_BUSINESS_TYPES];
extern float WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[NUM_BUSINESS_TYPES];
extern float WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[NUM_BUSINESS_TYPES];
extern int WORKPLACE_TYPE_OFFSET_HOST[NUM_BUSINESS_TYPES];
extern int WORKPLACE_TYPE_COUNT_HOST[NUM_BUSINESS_TYPES];
extern float CHILD_AGE_CDF_HOST[CHILD_DATA_ROWS];
extern int CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[CHILD_DATA_ROWS];
extern simRandOffsetsStruct_t host_randOffsetsStruct[1];
extern simSizeConstantsStruct_t host_simSizeStruct[1];
extern SEED_T SEED_HOST[SEED_LENGTH];