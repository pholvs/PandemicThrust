#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "simParameters.h"
#include "profiler.h"
#include "indirect.h"
#include "resource_logging.h"


#ifndef __max
#define __max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#define COUNTING_GRID_BLOCKS 32
#define COUNTING_GRID_THREADS 256

#define CONSOLE_OUTPUT 1
#define TIMING_BATCH_MODE 0
#define OUTPUT_FILES_IN_PARENTDIR 0
#define POLL_MEMORY_USAGE 1

#define DEBUG_SYNCHRONIZE_NEAR_KERNELS 0

#define SIM_PROFILING 1

//sim_validation must be 1 to log things
#ifndef SIM_VALIDATION
#define SIM_VALIDATION 1
#endif

#define log_contacts 0
#define log_infected_info 0
#define log_location_info 0
#define log_actions 0
#define log_actions_filtered 0
#define log_people_info 0

//low overhead
#define debug_log_function_calls 0

#define CULMINATION_PERIOD 10
#define NUM_BUSINESS_TYPES 14
#define CHILD_DATA_ROWS 5
#define HH_TABLE_ROWS 9

#define MAX_CONTACTS_WEEKDAY 8
#define MAX_CONTACTS_WEEKEND 5
#define MAX_POSSIBLE_CONTACTS_PER_DAY __max(MAX_CONTACTS_WEEKDAY, MAX_CONTACTS_WEEKEND)

#define SEED_LENGTH 4

//typedef unsigned long long randOffset_t;



class PandemicSim
{
public:
	PandemicSim(void);
	~PandemicSim(void);



	randOffset_t rand_offset;
	int MAX_DAYS;
	day_t current_day;

	void setupSim();
	void setup_loadParameters();
	void setup_loadSeed();
	void setup_loadFourSeeds();

	void setup_calculateInfectionData();
	void setup_generateHouseholds();

	void setup_pushDeviceData();
	void setup_initialInfected();
	void setup_buildFixedLocations();
	void setup_sizeGlobalArrays();
	void setup_fetchVectorPtrs();

	void setup_scaleSimulation();
	void setup_setCudaTopology();

	void setup_calcLocationOffsets(vec_t * ids_to_sort,vec_t lookup_table_copy,	vec_t * location_offsets,int num_people, int num_locs);

	CudaProfiler profiler;

	void runToCompletion();
	void daily_countAndRecover();
	void daily_writeInfectedStats();
	void daily_buildInfectedArray_global();
	void final_countReproduction();

	void logging_openOutputStreams();
	void logging_closeOutputStreams();


	void debug_nullFillDailyArrays();

	float sim_scaling_factor;
	float asymp_factor;

	int number_people;
	int number_households;
	int number_workplaces;
	int number_errand_locations;

	thrust::device_vector<status_t> people_status_pandemic;
	status_t * people_status_pandemic_ptr;
	thrust::device_vector<status_t> people_status_seasonal;
	status_t * people_status_seasonal_ptr;

	vec_t people_workplaces;
	int * people_workplaces_ptr;
	vec_t people_households;
	int * people_households_ptr;

	thrust::device_vector<age_t> people_ages;
	age_t * people_ages_ptr;

	thrust::device_vector<day_t> people_days_pandemic;
	day_t * people_days_pandemic_ptr;
	thrust::device_vector<day_t> people_days_seasonal;
	day_t * people_days_seasonal_ptr;
	thrust::device_vector<gen_t> people_gens_pandemic;
	gen_t * people_gens_pandemic_ptr;
	thrust::device_vector<gen_t> people_gens_seasonal;
	gen_t * people_gens_seasonal_ptr;

	int number_adults;
	int number_children;

	int INITIAL_INFECTED_PANDEMIC;
	int INITIAL_INFECTED_SEASONAL;

	int infected_count;	
	thrust::device_vector<personId_t> infected_indexes;
	personId_t * infected_indexes_ptr;

	thrust::device_vector<personId_t> daily_contact_infectors;
	personId_t * daily_contact_infectors_ptr;
	thrust::device_vector<personId_t> daily_contact_victims;
	personId_t * daily_contact_victims_ptr;
	thrust::device_vector<kval_type_t> daily_contact_kval_types;
	kval_type_t * daily_contact_kval_types_ptr;

	//int daily_actions;
	thrust::device_vector<action_t> daily_action_type;
	int * daily_action_type_ptr;

	FILE *fInfected, *fLocationInfo, *fContacts, *fActions, *fActionsFiltered;
	FILE * fContactsKernelSetup;

	vec_t workplace_offsets;
	int * workplace_offsets_ptr;
	thrust::device_vector<personId_t> workplace_people;
	personId_t * workplace_people_ptr;
	vec_t workplace_max_contacts;
	int * workplace_max_contacts_ptr;

	vec_t household_offsets;
	int * household_offsets_ptr;

	thrust::device_vector<personId_t> errand_people_table;		//people_array for errands
	personId_t * errand_people_table_ptr;
	thrust::device_vector<schedulingPair_t> errand_scheduling_vec;
	schedulingPair_t * errand_scheduling_vec_ptr;

	vec_t errand_infected_locations;			//the location of infected
	int * errand_infected_locations_ptr;
	vec_t errand_infected_weekendHours;				//the hours an infected person does their errands/contacts
	int * errand_infected_weekendHours_ptr;
	thrust::device_vector<errand_contacts_profile_t> errand_infected_ContactsDesired;		//how many contacts are desired on a given errand
	errand_contacts_profile_t * errand_infected_ContactsDesired_ptr;

	vec_t errand_locationOffsets_multiHour;
	int * errand_locationOffsets_multiHour_ptr;
	vec_t errand_hourOffsets_weekend;
	int * errand_hourOffsets_weekend_ptr;


	//DEBUG: these can be used to dump kernel internal data
	thrust::device_vector<float> debug_contactsToActions_float1;
	float * debug_contactsToActions_float1_ptr;
	thrust::device_vector<float> debug_contactsToActions_float2;
	float * debug_contactsToActions_float2_ptr;
	thrust::device_vector<float> debug_contactsToActions_float3;
	float * debug_contactsToActions_float3_ptr;
	thrust::device_vector<float> debug_contactsToActions_float4;
	float * debug_contactsToActions_float4_ptr;

	//new whole-day contact methods

	void doWeekday_wholeDay();
	void weekday_generateAfterschoolAndErrandDestinations();
	void weekday_doInfectedSetup_wholeDay();

	void doWeekend_wholeDay();
	void weekend_assignErrands();
	void weekend_doInfectedSetup_wholeDay();

	void validateContacts_wholeDay();
	void debug_copyFixedData();
	void debug_sizeHostArrays();
	void debug_copyErrandLookup();


	void debug_dumpInfectedErrandLocs();
	void debug_validateInfectedLocArrays();
	void debug_validateErrandSchedule();
	void debug_doublecheckContact_usingPeopleTable(int pos, int number_hours, int infector, int victim);

	void debug_dumpWeekendErrandTables(h_vec * h_sorted_people, h_vec * h_sorted_hours, h_vec * h_sorted_dests);

	void debug_validatePeopleSetup();
	void debug_freshenPeopleStatus();
	void debug_freshenErrands();
	void debug_freshenInfected();
	void debug_freshenContacts();
	void debug_freshenActions();
	void debug_validateLocationArrays();
	void debug_validateInfectionStatus();

	void debug_dump_array_toTempFile(const char * filename, const char * description, d_vec * array, int count);

	void debug_clearActionsArray();



	d_vec status_counts;
	int * status_counts_dev_ptr;
	int status_counts_today[16];

	cudaStream_t stream_secondary;

	int cuda_householdTypeAssignmentKernel_blocks;
	int cuda_householdTypeAssignmentKernel_threads;

	int cuda_peopleGenerationKernel_blocks;
	int cuda_peopleGenerationKernel_threads;

	int cuda_doWeekdayErrandAssignment_blocks;
	int cuda_doWeekdayErrandAssignment_threads;

	int cuda_makeWeekdayContactsKernel_blocks;
	int cuda_makeWeekdayContactsKernel_threads;

	int cuda_makeWeekendContactsKernel_blocks;
	int cuda_makeWeekendContactsKernel_threads;

	int cuda_contactsToActionsKernel_blocks;
	int cuda_contactsToActionsKernel_threads;

	int cuda_doInfectionActionsKernel_blocks;
	int cuda_doInfectionAtionsKernel_threads;

	//TO REMOVE:
	void debug_validateActions();
};

#define day_of_week() (current_day % 7)
#define is_weekend() (day_of_week() >= 5)
//#define is_weekend() (1)

#define errands_per_person() (is_weekend()? NUM_WEEKEND_ERRANDS : NUM_WEEKDAY_ERRANDS)
#define contacts_per_person() (is_weekend() ? MAX_CONTACTS_WEEKEND : MAX_CONTACTS_WEEKDAY)
#define num_infected_contacts_today() (is_weekend() ? infected_count * MAX_CONTACTS_WEEKEND : infected_count * MAX_CONTACTS_WEEKDAY)
#define num_infected_errands_today() (is_weekend() ? infected_count * NUM_WEEKEND_ERRANDS : infected_count * NUM_WEEKDAY_ERRANDS)


int roundHalfUp_toInt(double d);



__device__ kval_t device_makeContacts_weekday(
	personId_t myIdx, errand_contacts_profile_t errand_contacts_profile,
	int myPos,
	int * household_lookup, personId_t * household_offsets,// personId_t * household_people,
	int * workplace_max_contacts, int * workplace_lookup,
	personId_t * workplace_offsets, personId_t * workplace_people,
	int * errand_infected_locs,
	personId_t * errand_loc_offsets, personId_t * errand_people,
	int number_locations,
	personId_t * output_victim_arr, kval_type_t * output_kval_arr,
	randOffset_t myRandOffset, personId_t number_people);

__device__ kval_t device_makeContacts_weekend(personId_t myIdx, int myPos,
											  int * household_lookup, personId_t * household_offsets, // personId_t * household_people,
											  int * infected_errand_hours, int * infected_errand_destinations,
											  personId_t * errand_loc_offsets, personId_t * errand_people,
											  int * errand_populationCount_exclusiveScan,
											  int number_locations,
											  personId_t * output_victim_ptr, kval_type_t * output_kval_ptr,
											  randOffset_t myRandOffset);


__device__ personId_t device_getVictimAtIndex(personId_t index_to_fetch, personId_t * location_people, kval_type_t contact_type);
__device__ kval_t device_selectRandomPersonFromLocation(personId_t infector_idx, personId_t loc_offset, int loc_count, unsigned int rand_val, kval_type_t desired_kval, personId_t * location_people_arr, personId_t * output_victim_idx_arr, kval_type_t * output_kval_arr);
__device__ void device_lookupLocationData_singleHour(int myIdx, int * lookup_arr, int * loc_offset_arr, int * loc_offset, int * loc_count);
__device__ void device_lookupLocationData_singleHour(int myIdx, int * lookup_arr, int * loc_offset_arr, int * loc_max_contacts_arr, int * loc_offset, int * loc_count, int * loc_max_contacts);
__device__ void device_lookupLocationData_weekendErrand(int myPos, int errand_slot, int * infected_hour_val_arr, int * infected_hour_destination_arr, int * loc_offset_arr, int number_locations, int * hour_populationCount_exclusiveScan, int * output_location_offset, int * output_location_count);
__device__ void device_lookupInfectedLocation_multiHour(int myPos, int hour, int * infected_loc_arr, int * loc_offset_arr, int number_locations, int number_people, int number_hours, int * output_loc_offset, int * output_loc_count);
__device__ void device_lookupInfectedErrand_weekend(int myPos, int hour_slot,
													int * contacts_desired_arr, int * hour_arr, int * location_arr, 
													int * output_contacts_desired, int * output_hour, int * output_location);

__device__ void device_nullFillContact(personId_t * output_victim_idx, kval_type_t * output_kval);


//weekday errand assignment
__global__ void kernel_assignWeekdayAfterschoolAndErrands(age_t * people_ages_arr, int number_people, schedulingPair_t * errand_schedule_array, randOffset_t rand_offset);
__device__ void device_assignAfterschoolOrErrandDests_weekday(unsigned int rand_val1, unsigned int rand_val2,age_t myAge,  schedulingPair_t * output_dest1, schedulingPair_t * output_dest2);
__device__ unsigned int device_fishAfterschoolOrErrandDestination_weekday(unsigned int rand_val, age_t myAge);

//weekday infected setup
__global__ void kernel_doInfectedSetup_weekday_wholeDay(personId_t * infected_index_arr, int num_infected, schedulingPair_t * errand_scheduling_array, age_t * ages_lookup_arr, int num_people, int * output_infected_locs, errand_contacts_profile_t * output_infected_contacts_desired, randOffset_t rand_offset);
__device__ void device_doAllWeekdayInfectedSetup(unsigned int rand_val, int myPos, personId_t * infected_indexes_arr, schedulingPair_t * errand_scheduling_array, age_t * ages_lookup_arr, int num_people, int * output_infected_locs, errand_contacts_profile_t * output_infected_contacts_desired);
__device__ void device_assignContactsDesired_weekday_wholeDay(unsigned int rand_val, age_t myAge, errand_contacts_profile_t * output_contacts_desired);
__device__ void device_copyInfectedErrandLocs_weekday(schedulingPair_t * errand_lookup_ptr, int * output_infected_locs_ptr, int num_people);


//weekend errand assignment
__global__ void kernel_assignWeekendErrands(personId_t * people_indexes_arr, schedulingPair_t * errand_scheduling_array, int num_people, randOffset_t rand_offset);
__device__ void device_copyPeopleIndexes_weekend_wholeDay(int * id_dest_ptr, int myIdx);
__device__ void device_assignErrandHours_weekend_wholeDay(schedulingPair_t * errand_array_ptr, randOffset_t myRandOffset);
__device__ void device_assignErrandDestinations_weekend_wholeDay(schedulingPair_t * errand_array_ptr, int my_rand_offset);
__device__ int device_fishWeekendErrandDestination(unsigned int rand_val);

//weekend errand infected setup
__global__ void kernel_doInfectedSetup_weekend(personId_t * input_infected_indexes_ptr, int * input_errand_hours_ptr, int * input_errand_destinations_ptr,
											   int * output_infected_hour_ptr, int * output_infected_dest_ptr,
											   int num_infected);
__device__ void device_copyInfectedErrandLocs_weekend(int * input_hours_ptr, int * input_dests_ptr, int * output_hours_ptr, int * output_dests_ptr);


//output metrics
__global__ void kernel_countInfectedStatusAndRecover(
	status_t * pandemic_status_array, status_t * seasonal_status_array, 
	day_t * pandemic_days_array, day_t * seasonal_days_array,
	int num_people, day_t current_day,
	int * output_pandemic_counts, int * output_seasonal_counts);

//contacts_to_action
__device__ float device_calculateInfectionProbability(int profile, int day_of_infection, int strain, kval_t kval_sum);
__device__ void device_checkActionAndWrite(bool infects_pandemic, bool infects_seasonal, personId_t victim, status_t * pandemic_status_arr, status_t * seasonal_status_arr, int * dest_ptr);

//initial setup methods
__global__ void kernel_householdTypeAssignment(int * hh_type_array, int num_households, int rand_offset);
__device__ int device_setup_fishHouseholdType(unsigned int rand_val);

__global__ void kernel_generateHouseholds(
	int * hh_type_array, int * adult_exscan_arr, 
	int * child_exscan_arr, int num_households,
	int * household_offset_arr,
	age_t * people_age_arr, int * people_households_arr, int * people_workplaces_arr,
	randOffset_t rand_offset);
__device__ int device_setup_fishWorkplace(unsigned int rand_val);
__device__ void device_setup_fishSchoolAndAge(unsigned int rand_val, age_t * output_age_ptr, int * output_school_ptr);


void n_unique_numbers(h_vec * array, int n, int max);

char * action_type_to_string(action_t action);
char status_int_to_char(status_t s);
int lookup_school_typecode_from_age_code(int age_code);
char * profile_int_to_string(int p);

const char * lookup_contact_type(int contact_type);
const char * lookup_workplace_type(int workplace_type);
const char * lookup_age_type(int age_type);

void debug_print(char * message);
void debug_assert(bool condition, char * message);
void debug_assert(char *message, int expected, int actual);
void debug_assert(bool condition, char * message, int idx);

extern int WORKPLACE_TYPE_OFFSET_HOST[NUM_BUSINESS_TYPES];
extern int WORKPLACE_TYPE_COUNT_HOST[NUM_BUSINESS_TYPES];
extern int CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[CHILD_DATA_ROWS];


//sharedmem contact methods

__global__ void kernel_weekday_sharedMem(int num_infected, personId_t * infected_indexes, age_t * people_age,
										 int * household_lookup, personId_t * household_offsets,// personId_t * household_people,
										 int * workplace_max_contacts, int * workplace_lookup, 
										 personId_t * workplace_offsets, personId_t * workplace_people,
										 errand_contacts_profile_t * errand_contacts_profile_arr, int * errand_infected_locs,
										 personId_t * errand_loc_offsets, personId_t * errand_people,
										 int number_locations, 
										 personId_t * output_infector_arr, personId_t * output_victim_arr, kval_type_t * output_kval_arr,
										 action_t * output_action_arr,
										 status_t * people_status_p_arr, status_t * people_status_s_arr,
										 day_t * people_days_pandemic, day_t * people_days_seasonal,
										 gen_t * people_gens_pandemic, gen_t * people_gens_seasonal,
										 float * float_rand1, float * float_rand2,
										 float * float_rand3, float * float_rand4,
										 day_t current_day,
										 randOffset_t rand_offset, personId_t number_people);

__global__ void kernel_weekend_sharedMem(int num_infected, personId_t * infected_indexes, age_t * people_age,
										 int * household_lookup, personId_t * household_offsets,
										 int * infected_errand_hours, int * errand_infected_destinations,
										 personId_t * errand_loc_offsets, personId_t * errand_people,
										 int * errand_populationCount_exclusiveScan,
										 int number_locations, 
										 personId_t * output_infector_arr, personId_t * output_victim_arr, kval_type_t * output_kval_arr,
										 action_t * output_action_arr,
										 status_t * people_status_p_arr, status_t * people_status_s_arr,
										 day_t * people_days_pandemic, day_t * people_days_seasonal,
										 gen_t * people_gens_pandemic, gen_t * people_gens_seasonal,
										 float * float_rand1, float * float_rand2,
										 float * float_rand3, float * float_rand4,
										 day_t current_day,
										 randOffset_t rand_offset);

__device__ int device_setInfectionStatus(status_t profile_to_set, day_t day_to_set, gen_t gen_to_set,
										 status_t * output_profile, day_t * output_day, gen_t * output_gen);

__device__ action_t device_doInfectionActionImmediately(personId_t victim,day_t day_to_set,
														bool infects_pandemic, bool infects_seasonal,
														status_t profile_p_to_set, status_t profile_s_to_set,
														gen_t gen_p_to_set, gen_t gen_s_to_set,
														status_t * people_status_pandemic, status_t * people_status_seasonal,
														day_t * people_days_pandemic, day_t * people_days_seasonal,
														gen_t * people_gens_pandemic, gen_t * people_gens_seasonal);
__device__ void device_doContactsToActions_immediately(
	personId_t myIdx, kval_t kval_sum,
	personId_t * contact_victims_arr, kval_type_t *contact_type_arr, int contacts_per_infector,
	status_t * people_status_p_arr, status_t * people_status_s_arr,
	day_t * people_days_pandemic, day_t * people_days_seasonal,
	gen_t * people_gens_pandemic, gen_t * people_gens_seasonal,
	action_t * output_action_arr,
	float * rand_arr_1, float * rand_arr_2, float * rand_arr_3, float * rand_arr_4,
	day_t current_day,
	randOffset_t myRandOffset);


//currently unused
struct personStatusStruct_t{
	unsigned char status_pandemic;
	day_t day_pandemic;
	gen_t gen_pandemic;
	unsigned char status_seasonal;
	day_t day_seasonal;
	gen_t gen_seasonal;
	age_t age;
};

typedef unsigned long long personStatusStruct_word_t;
union personStatusUnion{
	struct personStatusStruct_t s;
	personStatusStruct_word_t w;
};

