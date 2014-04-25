#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "profiler.h"
#include "indirect.h"

#define CULMINATION_PERIOD 10
#define NUM_BUSINESS_TYPES 14

const int MAX_CONTACTS_WEEKDAY = 8;
const int MAX_CONTACTS_WEEKEND = 5;

#define MAX_CONTACTS_PER_DAY __max(MAX_CONTACTS_WEEKDAY, MAX_CONTACTS_WEEKEND)

#define SEED_LENGTH 4

class PandemicSim
{
public:
	PandemicSim(void);
	~PandemicSim(void);



	unsigned int rand_offset;
	int MAX_DAYS;
	int current_hour;
	int current_day;

	void setupSim();
	void setup_loadParameters();
	void setup_generateHouseholds();

	void setup_pushDeviceData();
	void setup_initialInfected();
	void setup_buildFixedLocations();
	void setup_sizeGlobalArrays();

	void setup_scaleSimulation();
	//void setup_setCudaTopology();

	void calcLocationOffsets(vec_t * ids_to_sort,vec_t lookup_table_copy,	vec_t * location_offsets,int num_people, int num_locs);

	CudaProfiler profiler;

	void runToCompletion();

	void logging_openOutputStreams();
	void logging_closeOutputStreams();

	void dailyUpdate();


	void debug_nullFillDailyArrays();

	float sim_scaling_factor;
	float asymp_factor;

	int number_people;
	int number_households;
	int number_workplaces;
	int number_errand_locations;

	vec_t people_status_pandemic;
	vec_t people_status_seasonal;
	vec_t people_workplaces;
	vec_t people_households;

	int number_adults;
	vec_t people_adult_indexes;

	int number_children;
	vec_t people_child_indexes;

	int INITIAL_INFECTED_PANDEMIC;
	int INITIAL_INFECTED_SEASONAL;

	int infected_count;	
	vec_t infected_indexes;
	thrust::device_vector<kval_t> infected_daily_kval_sum;

	int daily_contacts;
	vec_t daily_contact_infectors;
	vec_t daily_contact_victims;
	vec_t daily_contact_kvals;

	int daily_actions;
	vec_t daily_action_type;




	FILE *fInfected, *fLocationInfo, *fContacts, *fActions, *fActionsFiltered;
	FILE * fContactsKernelSetup;

	vec_t workplace_offsets;
	vec_t workplace_people;
	vec_t workplace_max_contacts;

	vec_t household_offsets;
	vec_t household_people;

	vec_t errand_people_table;		//people_array for errands
	vec_t errand_people_weekendHours;		//which hours people will do errands on weekends
	vec_t errand_people_destinations;		//where each person is going on their errand

	vec_t errand_infected_locations;			//the location of infected
	vec_t errand_infected_weekendHours;				//the hours an infected person does their errands/contacts
	vec_t errand_infected_ContactsDesired;		//how many contacts are desired on a given errand

	vec_t errand_locationOffsets_multiHour;
	vec_t errand_hourOffsets_weekend;

	vec_t people_ages;

	//DEBUG: these can be used to dump kernel internal data
	thrust::device_vector<float> debug_float1;
	thrust::device_vector<float> debug_float2;
	thrust::device_vector<float> debug_float3;
	thrust::device_vector<float> debug_float4;

	//new whole-day contact methods

	void doWeekday_wholeDay();
	void weekday_scatterAfterschoolLocations_wholeDay(d_vec * people_locs);
	void weekday_scatterErrandDestinations_wholeDay(d_vec * people_locs);
	void weekday_doInfectedSetup_wholeDay(vec_t * lookup_array, vec_t * inf_locs, vec_t * inf_contacts_desired);

	void doWeekend_wholeDay();
	void weekend_assignErrands(vec_t * errand_people, vec_t * errand_hours, vec_t * errand_destinations);
	void weekend_doInfectedSetup_wholeDay(vec_t * errand_hours, vec_t * errand_destinations, vec_t * infected_hours, vec_t * infected_destinations, vec_t * infected_contacts_desired);

	void validateContacts_wholeDay();
	void debug_copyFixedData();
	void debug_sizeHostArrays();
	void debug_copyErrandLookup();


	void debug_dumpInfectedErrandLocs();
	void debug_validateInfectedLocArrays();
	void debug_validateErrandSchedule();
	void debug_doublecheckContact_usingPeopleTable(int pos, int number_hours, int infector, int victim);

	void debug_dumpWeekendErrandTables(h_vec * h_sorted_people, h_vec * h_sorted_hours, h_vec * h_sorted_dests);
	void debug_validateLocationArrays();

	void debug_dumpPeopleInfo();

	void debug_dump_array_toTempFile(const char * filename, const char * description, d_vec * array, int count);

	void daily_buildInfectedArray_global();
	void daily_contactsToActions_new();
	void daily_filterActions_new();
	void daily_doInfectionActions();
	void daily_recoverInfected_new();
	void final_countReproduction();
	vec_t people_days_pandemic;
	vec_t people_days_seasonal;
	vec_t people_gens_pandemic;
	vec_t people_gens_seasonal;

	void daily_countInfectedStats();
	void daily_writeInfectedStats();
	int status_counts_today[16];
	cudaEvent_t event_statusCountsReadyToDump;
	cudaStream_t stream_countInfectedStatus;

	cudaEvent_t event_actionArrayNulled;
	cudaStream_t stream_nullActionsArray;


	//TO REMOVE:
};

#define day_of_week() (current_day % 7)
//#define is_weekend() (day_of_week() >= 5)
#define is_weekend() (0)


int roundHalfUp_toInt(double d);



__global__ void makeContactsKernel_weekday(int num_infected, int * infected_indexes, int * people_age,
										   int * household_lookup, int * household_offsets, int * household_people,
										   int * workplace_max_contacts, int * workplace_lookup, 
										   int * workplace_offsets, int * workplace_people,
										   int * errand_contacts_desired, int * errand_infected_locs,
										   int * errand_loc_offsets, int * errand_people,
										   int number_locations, 
										   int * output_infector_arr, int * output_victim_arr, int * output_kval_arr,
										   kval_t * output_kval_sum_arr,int rand_offset, int number_people);

__global__ void makeContactsKernel_weekend(int num_infected, int * infected_indexes,
										   int * household_lookup, int * household_offsets, int * household_people,
										   int * infected_errand_hours, int * infected_errand_destinations,
										   int * infected_errand_contacts_profile,
										   int * errand_loc_offsets, int * errand_people,
										   int * errand_populationCount_exclusiveScan,
										   int number_locations, 
										   int * output_infector_arr, int * output_victim_arr, int * output_kval_arr,
										   kval_t * output_kval_sum_arr, int rand_offset);

__device__ void device_selectRandomPersonFromLocation(int infector_idx, int loc_offset, int loc_count, unsigned int rand_val, int desired_kval, int * location_people_arr, int * output_infector_idx_arr, int * output_victim_idx_arr, int * output_kval_arr, kval_t * output_kval_sum);
__device__ void device_lookupLocationData_singleHour(int myIdx, int * lookup_arr, int * loc_offset_arr, int * loc_offset, int * loc_count);
__device__ void device_lookupLocationData_singleHour(int myIdx, int * lookup_arr, int * loc_offset_arr, int * loc_max_contacts_arr, int * loc_offset, int * loc_count, int * loc_max_contacts);
__device__ void device_lookupLocationData_weekendErrand(int myPos, int errand_slot, int * infected_hour_val_arr, int * infected_hour_destination_arr, int * loc_offset_arr, int number_locations, int * hour_populationCount_exclusiveScan, int * output_location_offset, int * output_location_count);
__device__ void device_lookupInfectedLocation_multiHour(int myPos, int hour, int * infected_loc_arr, int * loc_offset_arr, int number_locations, int number_people, int * contacts_desired_lookup, int number_hours, int * output_loc_offset, int * output_loc_count, int * output_contacts_desired);
__device__ void device_lookupInfectedErrand_weekend(int myPos, int hour_slot,
													int * contacts_desired_arr, int * hour_arr, int * location_arr, 
													int * output_contacts_desired, int * output_hour, int * output_location);

__device__ void device_nullFillContact(int myIdx, int * output_infector_idx, int * output_victim_idx, int * output_kval);


//weekday errand assignment
__global__ void kernel_assignErrandLocations_weekday_wholeDay(int * adult_indexes_arr, int number_adults, int number_people, int * output_arr, int rand_offset);
__device__ void device_fishWeekdayErrand(unsigned int * rand_val, int * output_destination);
__global__ void kernel_assignAfterschoolLocations_wholeDay(int * child_indexes_arr, int * output_array, int number_children, int number_people, int rand_offset);
__device__ void device_fishAfterschoolLocation(unsigned int * rand_val, int number_people, int afterschool_count, int afterschool_offset, int * output_schedule);

//weekday infected setup
__global__ void kernel_doInfectedSetup_weekday_wholeDay(int * infected_index_arr, int num_infected, int * loc_lookup_arr, int * ages_lookup_arr, int num_people, int * output_infected_locs, int * output_infected_contacts_desired, int rand_offset);
__device__ void device_doAllWeekdayInfectedSetup(unsigned int * rand_val, int myPos, int * infected_indexes_arr, int * loc_lookup_arr, int * ages_lookup_arr, int num_people, int * output_infected_locs, int * output_infected_contacts_desired);
__device__ void device_assignContactsDesired_weekday_wholeDay(unsigned int rand_val, int myAge, int * output_contacts_desired);
__device__ void device_copyInfectedErrandLocs_weekday(int * loc_lookup_ptr, int * output_infected_locs_ptr, int num_people);


//weekend errand assignment
__global__ void kernel_assignErrands_weekend(int * people_indexes_arr, int * errand_hours_arr, int * errand_destination_arr, int num_people, int rand_offset);
__device__ void device_copyPeopleIndexes_weekend_wholeDay(int * id_dest_ptr, int myIdx);
__device__ void device_assignErrandHours_weekend_wholeDay(int * hours_dest_ptr, int my_rand_offset);
__device__ void device_assignErrandDestinations_weekend_wholeDay(int * errand_destination_arr, int my_rand_offset);
__device__ void device_fishWeekendErrandDestination(unsigned int * rand_val, int * output_ptr);

//weekend errand infected setup
__global__ void kernel_doInfectedSetup_weekend(int * input_infected_indexes_ptr, int * input_errand_hours_ptr, int * input_errand_destinations_ptr,
											   int * output_infected_hour_ptr, int * output_infected_dest_ptr, int * output_contacts_desired_ptr,
											   int num_infected, int rand_offset);
__device__ void device_doAllInfectedSetup_weekend(unsigned int * rand_val, int myPos, int * infected_indexes_arr, int * input_hours_arr, int * input_dests_arr, int * output_hours_arr, int * output_dests_arr, int * output_contacts_desired_arr);
__device__ void device_copyInfectedErrandLocs_weekend(int * input_hours_ptr, int * input_dests_ptr, int * output_hours_ptr, int * output_dests_ptr);


//output metrics
__global__ void kernel_countInfectedStatus(int * pandemic_status_array, int * seasonal_status_array, int num_people, int * output_pandemic_counts, int * output_seasonal_counts);

//contacts_to_action
__global__ void kernel_contactsToActions(int * infected_idx_arr, kval_t * infected_kval_sum_arr, int infected_count,
										 int * contact_victims_arr, int *contact_type_arr, int contacts_per_infector,
										 int * people_day_pandemic_arr, int * people_day_seasonal_arr,
										 int * people_status_p_arr, int * people_status_s_arr,
										 int * output_action_arr,
										 int current_day, int rand_offset);
__device__ float device_calculateInfectionProbability(int profile, int day_of_infection, int strain, kval_t kval_sum);
__device__ void device_checkActionAndWrite(bool infects_pandemic, bool infects_seasonal, int * pandemic_status_arr, int * seasonal_status_arr, int * dest_ptr);

//initial setup methods
__global__ void kernel_householdTypeAssignment(int * hh_type_array, int num_households, int rand_offset);
__device__ int device_setup_fishHouseholdType(unsigned int rand_val);

__global__ void kernel_generateHouseholds(
	int * hh_type_array, int * adult_exscan_arr, int * child_exscan_arr, int num_households,
	int * adult_index_arr, int * child_index_arr, 
	int * people_age_arr, int * people_households_arr, int * people_workplaces_arr,
	int rand_offset);
__device__ int device_setup_fishWorkplace(unsigned int rand_val);
__device__ void device_setup_fishSchoolAndAge(unsigned int rand_val, int * output_age_ptr, int * output_school_ptr);

//do_action methods
__global__ void kernel_doInfectionActions(
int * contact_action_arr, int * contact_victim_arr, int * contact_infector_arr,
	int action_count,
	int * people_status_p_arr, int * people_status_s_arr,
	int * people_gen_p_arr, int * people_gen_s_arr,
	int * people_day_p_arr, int * people_day_s_arr,
	int day_tomorrow, int rand_offset);
__device__ void device_doInfectionAction(
	unsigned int rand_val1, unsigned int rand_val2,
	int day_tomorrow,
	int action_type, int infector, int victim,
	int * people_status_p_arr, int * people_status_s_arr,
	int * people_gen_p_arr, int * people_gen_s_arr,
	int * people_day_p_arr, int * people_day_s_arr);
__device__ void device_assignProfile(unsigned int rand_val, int * output_status_ptr);

void n_unique_numbers(h_vec * array, int n, int max);
inline char * action_type_to_string(int action);
inline char status_int_to_char(int s);
inline int lookup_school_typecode_from_age_code(int age_code);
inline char * profile_int_to_string(int p);

extern inline const char * lookup_contact_type(int contact_type);
extern inline const char * lookup_workplace_type(int workplace_type);
extern inline const char * lookup_age_type(int age_type);

extern inline void debug_print(char * message);
extern inline void debug_assert(bool condition, char * message);
extern inline void debug_assert(char *message, int expected, int actual);
extern inline void debug_assert(bool condition, char * message, int idx);


extern const int FIRST_WEEKDAY_ERRAND_ROW;
extern const int FIRST_WEEKEND_ERRAND_ROW;
extern const int PROFILE_SIMULATION;
extern int WORKPLACE_TYPE_OFFSET_HOST[NUM_BUSINESS_TYPES];
