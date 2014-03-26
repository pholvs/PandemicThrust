#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "profiler.h"
#include "indirect.h"

#define CULMINATION_PERIOD 10
#define NUM_BUSINESS_TYPES 14

#define MAX_CONTACTS_PER_DAY 8

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
	void setup_configureLogging();
	void setup_loadParameters();
	void setup_generateHouseholds();
	int setup_assignWorkplace();
	int setup_assignSchool();
	void setup_pushDeviceData();
	void setup_initialInfected();
	void setup_buildFixedLocations();
	void setup_sizeGlobalArrays();

	void setup_scaleSimulation();
	void setup_setCudaTopology();

	void calcLocationOffsets(vec_t * ids_to_sort,vec_t lookup_table_copy,	vec_t * location_offsets,int num_people, int num_locs);

	CudaProfiler profiler;


	void runToCompletion();
	void dump_people_info();
	void debug_validate_infected();
	void dump_infected_info();
	void test_locs();

	void logging_openOutputStreams();
	void logging_closeOutputStreams();

	void doWeekday();
	void makeContacts_byLocationMax(const char * contact_type,
		vec_t *infected_list, int infected_list_count,
		vec_t *loc_people, vec_t *loc_max_contacts,
		vec_t *loc_offsets, int num_locs,
		vec_t *people_lookup);

	void makeContacts_byContactsDesiredArray(const char * hour_string,
		vec_t *infected_list, int infected_list_count,
		vec_t *loc_people, vec_t * contacts_desired,
		vec_t * loc_offsets, int num_locs, 
		vec_t * people_lookup);

	void weekday_scatterAfterschoolLocations(vec_t * child_locs);

	int filterInfectedByPopulationGroup(const char * hour_string, vec_t * population_group, vec_t * infected_present);
	void assign_weekday_errand_contacts(d_vec * contacts_desired, int num_infected_adults);
	void doWeekdayErrands();
	void weekday_scatterErrandLocations(d_vec * locations_array);

	void doWeekend();
	void doWeekendErrands();
	void weekend_copyPeopleIndexes(vec_t * errand_people);
	void weekend_generateThreeUniqueHours(vec_t * errand_hours);
	void weekend_generateErrandDestinations(vec_t * errand_locations);
	void dump_weekend_errands(vec_t errand_people, vec_t errand_hours, vec_t errand_locations, int num_to_print, int num_people);
	void weekendErrand_doInfectedSetup(vec_t * errand_hours, vec_t * errand_destinations, vec_t * infected_present, vec_t * infected_locations, vec_t * infected_contacts_desired, vec_t * infected_hour_offsets);


	void buildContactsDesired_byLocationMax(
		vec_t *infected_locations, int num_infected,
		vec_t *loc_offsets, vec_t *loc_max_contacts,
		vec_t *contacts_desired);

	void clipContactsDesired_byLocationCount(d_ptr infected_locations_begin, int num_infected, vec_t *loc_offsets, d_ptr contacts_desired_begin);

	void dump_contact_kernel_setup(
		const char * hour_string,
		d_ptr infected_indexes_present_begin, d_ptr infected_locations_begin,
		d_ptr infected_contacts_desired_begin, d_vec * output_offsets,
		int * location_people_ptr, d_vec *location_offsets,
		int num_infected);

	void launchContactsKernel(
		const char * hour_string,
		vec_t *infected_indexes_present, vec_t *infected_locations, 
		vec_t *infected_contacts_desired, int infected_present,
		int * loc_people_ptr, vec_t *location_offsets, int num_locs);

	void launchContactsKernel(
		const char * hour_string,
		d_ptr infected_indexes_present_begin, d_ptr infected_locations_begin, 
		d_ptr infected_contacts_desired_begin, int infected_present_count,
		int * loc_people_ptr, vec_t *location_offsets, int num_locs);

	void validate_contacts(const char * hour_string, d_vec *d_people, d_vec *d_lookup, d_vec *d_offsets, int N);
	void validate_contacts(const char * hour_string, h_vec * h_people, h_vec *h_lookup, h_vec *h_offsets, h_vec *contact_infectors, h_vec *contact_victims, int N);
	void validate_weekend_errand_contacts(const char * hour_string, d_ptr people_begin, d_ptr destinations_begin, int people_present, d_ptr loc_offsets_ptr, int number_locations, int num_contacts_to_dump);

	void dailyUpdate();
	void deprected_daily_assignVictimGenerations();
	void daily_contactsToActions();
	void dump_actions();
	void daily_filterActions();
	void do_infection_actions(int action);
	int daily_recoverInfected();
	void daily_countReproductionNumbers(int action_type);
	void dump_actions_filtered();
	void daily_rebuildInfectedArray();

	void calculateFinalReproduction();

	void debug_dump_array(const char * description, d_vec * gens_array, int array_count);
	void debug_dump_array_toTempFile(const char * description, d_vec * array, int count);
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
	vec_t infected_days_pandemic;
	vec_t infected_days_seasonal;
	vec_t infected_generation_pandemic;
	vec_t infected_generation_seasonal;
	vec_t infected_k_sum;

	int daily_contacts;
	vec_t daily_contact_infectors;
	vec_t daily_contact_victims;
	vec_t daily_contact_kvals;

	int daily_actions;
	vec_t daily_action_type;
	vec_t daily_action_victim_index;
	vec_t daily_action_victim_gen_p;
	vec_t daily_action_victim_gen_s;

	//stores number of infections by generation
	vec_t reproduction_count_pandemic;
	vec_t reproduction_count_seasonal;

	FILE *fInfected, *fLocationInfo, *fContacts, *fActions, *fActionsFiltered;
	FILE * fContactsKernelSetup;

	vec_t workplace_counts;
	vec_t workplace_offsets;
	vec_t workplace_people;
	vec_t workplace_max_contacts;

	vec_t household_counts;
	vec_t household_offsets;
	vec_t household_people;
	vec_t household_max_contacts;

	vec_t errand_people_table;		//people_array for errands
	vec_t errand_people_weekendHours;		//which hours people will do errands on weekends
	vec_t errand_people_destinations;		//lookup table (may not contain all entries on weekends)
	vec_t errand_locationOffsets;		//location offset table

	vec_t errand_infected_indexesPresent;		//list of infected present during an errand
	vec_t errand_infected_locations;			//the location of infected
	vec_t errand_infected_weekendHours;				//the hours an infected person does their errands/contacts
	vec_t errand_infected_ContactsDesired;		//how many contacts are desired on a given errand

	vec_t infected_output_offsets;

	//DEBUG: these can be used to dump kernel internal data
	thrust::device_vector<float> debug_float1;
	thrust::device_vector<float> debug_float2;
	thrust::device_vector<float> debug_float3;
	thrust::device_vector<float> debug_float4;
};

#define day_of_week() (current_day % 7)
//#define is_weekend() (day_of_week() >= 5)
#define is_weekend() (0)

void n_unique_numbers(h_vec * array, int n, int max);
inline char * action_type_to_char(int action);
inline char status_int_to_char(int s);
inline void debug_print(char * message);
inline void debug_assert(bool condition, char * message);
inline void debug_assert(char *message, int expected, int actual);
inline void debug_assert(bool condition, char * message, int idx);

int roundHalfUp_toInt(double d);


struct weekend_getter;
__global__ void weekend_errand_hours_kernel(int * hours_array, int N, int rand_offset);