#pragma once


typedef unsigned int randOffset_t;

typedef int personId_t;
typedef int status_t;
typedef int age_t;

typedef int day_t;
typedef int gen_t;

typedef int loc_work_t;
typedef int loc_hh_t;
typedef int max_contacts_t;

typedef int errand_contacts_profile_t;

typedef int action_t;

typedef int kval_type_t;


#define ACTION_INFECT_NONE 0
#define ACTION_INFECT_SEASONAL 1
#define ACTION_INFECT_PANDEMIC 2
#define ACTION_INFECT_BOTH 3

#define STATUS_SUSCEPTIBLE -1
#define STATUS_INFECTED 0
#define STATUS_RECOVERED -2


#define DAY_NOT_INFECTED -1
#define GENERATION_NOT_INFECTED -1
#define NULL_PERSON_INDEX -1;

#define INITIAL_DAY 0
#define INITIAL_GEN 0

const int BUSINESS_TYPE_PRESCHOOL = 3;
const int BUSINESS_TYPE_ELEMENTARYSCHOOL = 4;
const int BUSINESS_TYPE_MIDDLESCHOOL = 5;
const int BUSINESS_TYPE_HIGHSCHOOL = 6;
const int BUSINESS_TYPE_UNIVERSITY = 7;
const int BUSINESS_TYPE_AFTERSCHOOL = 8;

const int NUM_WEEKDAY_ERRANDS = 2;
const int NUM_WEEKDAY_ERRAND_HOURS = 2;

const int NUM_WEEKEND_ERRANDS = 3;
const int NUM_WEEKEND_ERRAND_HOURS = 10;

const int NUM_CONTACT_TYPES = 6;
const kval_type_t CONTACT_TYPE_NONE = 0;
const kval_type_t CONTACT_TYPE_WORKPLACE = 1;
const kval_type_t CONTACT_TYPE_SCHOOL = 2;
const kval_type_t CONTACT_TYPE_ERRAND = 3;
const kval_type_t CONTACT_TYPE_AFTERSCHOOL = 4;
const kval_type_t CONTACT_TYPE_HOME = 5;

#define DEFINE_NUM_SHEDDING_PROFILES 6
const int NUM_SHEDDING_PROFILES = DEFINE_NUM_SHEDDING_PROFILES;
const int PROFILE_GAMMA1 = 0;
const int PROFILE_LOGNORM1 = 1;
const int PROFILE_WEIB1 = 2;
const int PROFILE_GAMMA2 = 3;
const int PROFILE_LOGNORM2 = 4;
const int PROFILE_WEIB2 = 5;

#define DEFINE_NUM_WEEKDAY_ERRAND_PROFILES 4
#define DEFINE_NUM_WEEKEND_ERRAND_PROFILES 6

#define DEFINE_WEEKDAY_ERRAND_PROFILE_AFTERSCHOOL 3
const int WEEKDAY_ERRAND_PROFILE_AFTERSCHOOL = DEFINE_WEEKDAY_ERRAND_PROFILE_AFTERSCHOOL;



const age_t AGE_5 = 0;
const age_t AGE_9 = 1;
const age_t AGE_14 = 2;
const age_t AGE_17 = 3;
const age_t AGE_22 = 4;
const age_t AGE_ADULT = 5;
