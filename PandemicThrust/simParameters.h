#pragma once

#include <thrust/pair.h>

typedef unsigned long randOffset_t;

typedef int personId_t;

typedef unsigned char age_t;
typedef unsigned char day_t;
typedef unsigned char gen_t;

typedef unsigned char errand_contacts_profile_t;
typedef unsigned char kval_type_t;

typedef int locId_t;
typedef int locOffset_t;

typedef int errandSchedule_t;
typedef int maxContacts_t;  

typedef int action_t;	//could be smaller, but only stored in validation mode
#define ACTION_INFECT_NONE 0
#define ACTION_INFECT_SEASONAL 1
#define ACTION_INFECT_PANDEMIC 2
#define ACTION_INFECT_BOTH 3


typedef unsigned int status_t;
#define STATUS_SUSCEPTIBLE 1
#define STATUS_INFECTED 2
#define STATUS_RECOVERED 0

//these null values will take val -1 if their type is signed, or MAX_VAL if it's unsigned
#define DAY_NOT_INFECTED (day_t) -1
#define GENERATION_NOT_INFECTED (gen_t) -1
#define NULL_PERSON_INDEX (personId_t) -1
#define AGE_NOT_SET (age_t) -1
#define LOC_ID_NOT_SET (locId_t) -1
#define NULL_ERRAND (errandSchedule_t) -1

#define INITIAL_DAY 0
#define INITIAL_GEN 0

#define BUSINESS_TYPE_PRESCHOOL 3
#define BUSINESS_TYPE_ELEMENTARYSCHOOL 4
#define BUSINESS_TYPE_MIDDLESCHOOL 5
#define BUSINESS_TYPE_HIGHSCHOOL 6
#define BUSINESS_TYPE_UNIVERSITY 7
#define BUSINESS_TYPE_AFTERSCHOOL 8

#define NUM_WEEKDAY_ERRANDS 2
#define NUM_WEEKDAY_ERRAND_HOURS 2

#define NUM_WEEKEND_ERRANDS 3
#define NUM_WEEKEND_ERRAND_HOURS 10

#define NUM_CONTACT_TYPES 6
#define CONTACT_TYPE_NONE 0
#define CONTACT_TYPE_WORKPLACE 1
#define CONTACT_TYPE_SCHOOL 2
#define CONTACT_TYPE_ERRAND 3
#define CONTACT_TYPE_AFTERSCHOOL 4
#define CONTACT_TYPE_HOME 5

#define NUM_SHEDDING_PROFILES 6
#define PROFILE_GAMMA1 0
#define PROFILE_LOGNORM1 1
#define PROFILE_WEIB1 2
#define PROFILE_GAMMA2 3
#define PROFILE_LOGNORM2 4
#define PROFILE_WEIB2 5

#define NUM_WEEKDAY_ERRAND_PROFILES 4
#define NUM_WEEKEND_ERRAND_PROFILES 6

#define WEEKDAY_ERRAND_PROFILE_AFTERSCHOOL 3

#define AGE_5 0
#define AGE_9 1
#define AGE_14 2
#define AGE_17 3
#define AGE_22 4
#define AGE_ADULT 5

//the first row of the PDF with a value > 0
#define FIRST_WEEKDAY_ERRAND_ROW 9
#define FIRST_WEEKEND_ERRAND_ROW 9