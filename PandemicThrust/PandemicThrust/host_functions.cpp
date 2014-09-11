#include "stdafx.h"
#include "host_functions.h"
#include "simParameters.h"


locId_t host_recalcWorkplace(int myIdx, age_t myAge)
{
	randOffset_t myRandOffset = host_randOffsetsStruct->workplace_randOffset + (myIdx / 4);

	threefry2x64_key_t tf_k = {{(long) SEED_HOST[0], (long) SEED_HOST[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;
	threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};

	rand_union.c = threefry2x64(tf_ctr,tf_k);

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

errandContactsProfile_t host_recalc_weekdayErrandDests_assignProfile(
	personId_t myIdx, age_t myAge, 
	locId_t * output_dest1, locId_t * output_dest2)
{
	//find the counter settings when this errand was generated
	int myGridPos = myIdx / 2;
	randOffset_t myRandOffset = host_randOffsetsStruct->errand_randOffset + myGridPos;
	int num_locations = host_simSizeStruct->number_workplaces;

	//regen the random numbers
	threefry2x64_key_t tf_k = {{SEED_HOST[0], SEED_HOST[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} u;
	threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
	u.c = threefry2x64(tf_ctr, tf_k);

	int rand_slot = 2 * (myIdx % 2); //either 0 or 2
	host_assignAfterschoolOrErrandDests_weekday(
		u.i[rand_slot],u.i[rand_slot+1],
		myAge,num_locations,
		output_dest1,output_dest2);

	//return a contacts profile for this person
	//If they're not an adult, return the afterschool contacts profile
	if(myAge != AGE_ADULT)
		return WEEKDAY_ERRAND_PROFILE_AFTERSCHOOL;
	//else they're an adult

	//for the sake of thoroughness, we'll XOR the rands so that we get a new one
	int other_rand_slot = (rand_slot + 2) % 4;
	unsigned int xor_rand = u.i[rand_slot] ^ u.i[rand_slot+1];

	//the afterschool profile is the highest number, get a profile less than that
	errandContactsProfile_t profile = xor_rand % WEEKDAY_ERRAND_PROFILE_AFTERSCHOOL;

	return profile;
}


void host_assignAfterschoolOrErrandDests_weekday(
	unsigned int rand_val1, unsigned int rand_val2,
	age_t myAge, int num_locations,
	locId_t * output_dest1, locId_t * output_dest2)
{
	//to avoid divergence, the base case will assign the same errand to both hours
	//(i.e. the norm for children)
	int dest1 = host_fishAfterschoolOrErrandDestination_weekday(rand_val1,myAge);

	int dest2 = dest1;
	if(myAge == AGE_ADULT)
		dest2 = host_fishAfterschoolOrErrandDestination_weekday(rand_val2,myAge);

	dest2 += num_locations;

	*output_dest1 = dest1;
	*output_dest2 = dest2;
}


locId_t host_fishAfterschoolOrErrandDestination_weekday(
	unsigned int rand_val, age_t myAge)
{
	int business_type = BUSINESS_TYPE_AFTERSCHOOL;
	float frac = (float) rand_val / UNSIGNED_MAX;

	//for adults, loop through the errand types and find the one this yval assigns us to
	if(myAge == AGE_ADULT)
	{
		business_type = FIRST_WEEKDAY_ERRAND_ROW;
		float row_pdf = WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[business_type];

		while(frac > row_pdf && business_type < (NUM_BUSINESS_TYPES - 1))
		{
			frac -= row_pdf;
			business_type++;
			row_pdf = WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[business_type];
		}

		frac = frac / row_pdf;
	}

	//lookup type count and offset
	int type_count = WORKPLACE_TYPE_COUNT_HOST[business_type];
	int type_offset = WORKPLACE_TYPE_OFFSET_HOST[business_type];

	//we now have a fraction between 0 and 1 representing which of this business type we are at
	unsigned int business_num = frac * type_count;

	//frac should be between 0 and 1 but we may lose a little precision here
	if(business_num >= type_count)
		business_num = type_count - 1;

	//add the offset to the first business of this type
	business_num += type_offset;

	return business_num;
}


void host_recalc_weekendErrandDests(personId_t myIdx, locId_t * errand_array_ptr)
{
	randOffset_t myRandOffset = host_randOffsetsStruct->errand_randOffset + (2*myIdx);
	host_generateWeekendErrands(errand_array_ptr,myRandOffset);
}

void host_generateWeekendErrands(locId_t * errand_output_ptr, randOffset_t myRandOffset)
{
	int num_locations = host_simSizeStruct->number_workplaces;

	threefry2x64_key_t tf_k = {{SEED_HOST[0], SEED_HOST[1]}};
	union{
		threefry2x64_ctr_t c[2];
		unsigned int i[8];
	} u;

	threefry2x64_ctr_t tf_ctr_1 = {{ myRandOffset,  myRandOffset}};
	u.c[0] = threefry2x64(tf_ctr_1, tf_k);
	threefry2x64_ctr_t tf_ctr_2 = {{ myRandOffset + 1,  myRandOffset + 1}};
	u.c[1] = threefry2x64(tf_ctr_2, tf_k);

	int hour1, hour2, hour3;

	//get first hour
	hour1 = u.i[0] % NUM_WEEKEND_ERRAND_HOURS;

	//get second hour, if it matches then increment
	hour2 = u.i[1] % NUM_WEEKEND_ERRAND_HOURS;
	if(hour2 == hour1)
		hour2 = (hour2 + 1) % NUM_WEEKEND_ERRAND_HOURS;

	//get third hour, increment until it no longer matches
	hour3 = u.i[2] % NUM_WEEKEND_ERRAND_HOURS;
	while(hour3 == hour1 || hour3 == hour2)
		hour3 = (hour3 + 1) % NUM_WEEKEND_ERRAND_HOURS;

	errand_output_ptr[0] = host_fishWeekendErrandDestination(u.i[3]) + (hour1 * num_locations);
	errand_output_ptr[1] = host_fishWeekendErrandDestination(u.i[4]) + (hour2 * num_locations);
	errand_output_ptr[2] = host_fishWeekendErrandDestination(u.i[5]) + (hour3 * num_locations);
}

locId_t host_fishWeekendErrandDestination(unsigned int rand_val)
{
	float y = (float) rand_val / UNSIGNED_MAX;

	int row = FIRST_WEEKEND_ERRAND_ROW;
	while(y > WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[row] && row < (NUM_BUSINESS_TYPES - 1))
	{
		y -= WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[row];
		row++;
	}
	float frac = y / WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[row];
	int type_count = WORKPLACE_TYPE_COUNT_HOST[row];
	int business_num = frac * type_count;

	if(business_num >= type_count)
		business_num = type_count - 1;

	int type_offset = WORKPLACE_TYPE_OFFSET_HOST[row];

	return business_num + type_offset;
}
