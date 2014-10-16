#pragma once
#include "cuda_includes.h"

#include "simParameters.h"
#include "profiler.h"
#include "indirect.h"
#include "resource_logging.h"

//include only one
#include "device_K20.h"
//#include "device_GT640.h"
//#include "device_Q880M.h"

//set to 1 for NVIDIA Graphical Profiler
#define CUDA_PROFILER_ENABLE 1

//G++ needs an inline max defined
#ifndef __max
#define __max(a,b) (((a) > (b)) ? (a) : (b))
#endif

//logged into output_resource_log and some other output files
#define NAME_OF_SIM_TYPE "gpu_cub"

//1 for some status messages during run, 0 for totally silent
#define CONSOLE_OUTPUT 1

//if 1, output files go in ".." otherwise they go into "."
#define OUTPUT_FILES_IN_PARENTDIR 0

//if 0, memory usage is polled once per simulation
//if 1, memory usage is polled each simulated day
#define POLL_MEMORY_USAGE 0

//force explicit synchronization after kernel calls to prevent any async issues
#define DEBUG_SYNCHRONIZE_NEAR_KERNELS 0

//use recursive stack profiler to log function calls?   Low overhead
#define SIM_PROFILING 1

//Write a file which contains the percentage of population with active infections at the peak of the simulation
#define LOG_INFECTED_PROPORTION 1

//logs the fraction of the population with active infections at the peak of the outbreak
#define LOG_INFECTED_PROPORTION 1

//Enables testing and debug-logging during run
//preprocessor if's allow makefile to compile separate validation binary
#ifndef SIM_VALIDATION
#define SIM_VALIDATION 0
#endif

//SIM_VALIDATION must be 1 or these do nothing
#define FLUSH_VALIDATION_IMMEDIATELY 0	//force flush to disk immediately, prevents disappearing logs if program crashes
#define log_contacts 0		//logs all contacts selected
#define log_infected_info 0		//logs a list of infected agents and associated data
#define log_location_info 0		//logs number of people at each location each hour
#define log_actions 0			//logs what happened for each contact during each hour

//if CALC_NUM_PEOPLE_FIRST==1, the household types will be generated twice.  
//The first time will find the total population of the sim using a transform-reduce.
//Then memory will be allocated for the global arrays, and the households will actually be generated
//This means we use more processing but avoid potential fragmentation
//If ==0, then global memory must be allocated to store the household types and the exclusive-scans
//This space will be allocated before the global arrays can be sized, and it can cause fragmentation
#define CALC_NUM_PEOPLE_FIRST 1	

class PandemicSim
{
public:
	PandemicSim(void);
	~PandemicSim(void);

	randOffset_t rand_offset;
	int MAX_DAYS;
	day_t current_day;

	int core_seed;

	void setupSim();
	void setup_loadParameters();
	void setup_loadSeed();
	void setup_loadFourSeeds();

	void setup_calculateInfectionData();
	void setup_generateHouseholds();
	void setup_assignWorkplaces();
	int setup_calcPopulationSize();
	int setup_calcPopulationSize_thrust();
	void setup_initializeStatusArrays();

	void setup_pushDeviceData();
	void setup_initialInfected();
	void setup_buildFixedLocations();
	void setup_sizeGlobalArrays();
	void setup_fetchVectorPtrs();
	void setup_configCubBuffers();
	void setup_sizeCubTempArray();

	void setup_scaleSimulation();
	void setup_setCudaTopology();

	void setup_calcLocationOffsets(vec_t * ids_to_sort,vec_t lookup_table_copy,	vec_t * location_offsets,int num_people, int num_locs);

	CudaProfiler profiler;

	void runToCompletion();
	void daily_countAndRecover();
	void daily_writeInfectedStats();
	void daily_buildInfectedArray_global();
	void final_releaseMemory();
	void final_countReproduction();

	void logging_openOutputStreams();
	void logging_closeOutputStreams();


	void debug_nullFillDailyArrays();

	float people_scaling_factor;
	float location_scaling_factor;
	float asymp_factor;

	int number_people;
	int number_households;
	int number_workplaces;

	thrust::device_vector<status_t> people_status_pandemic;
	status_t * people_status_pandemic_ptr;
	thrust::device_vector<status_t> people_status_seasonal;
	status_t * people_status_seasonal_ptr;

//	thrust::device_vector<locId_t> people_workplaces;
//	locId_t * people_workplaces_ptr;
	thrust::device_vector<locId_t> people_households;
	locId_t * people_households_ptr;

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

	float max_infected_proportion;
	int expected_max_infected;

	thrust::device_vector<personId_t> infected_indexes;
	personId_t * infected_indexes_ptr;

	thrust::device_vector<personId_t> daily_contact_infectors;
	personId_t * daily_contact_infectors_ptr;
	thrust::device_vector<personId_t> daily_contact_victims;
	personId_t * daily_contact_victims_ptr;
	thrust::device_vector<kval_type_t> daily_contact_kval_types;
	kval_type_t * daily_contact_kval_types_ptr;
	thrust::device_vector<locId_t> daily_contact_locations;
	locId_t * daily_contact_locations_ptr;
	thrust::device_vector<action_t> daily_action_type;
	action_t * daily_action_type_ptr;

	FILE *fInfected, *fLocationInfo, *fContacts, *fActions, *fActionsFiltered;
	FILE * fContactsKernelSetup;

	thrust::device_vector<locOffset_t> workplace_offsets;
	locOffset_t * workplace_offsets_ptr;
	thrust::device_vector<personId_t> workplace_people;
	personId_t * workplace_people_ptr;
//	thrust::device_vector<maxContacts_t> workplace_max_contacts;
//	maxContacts_t * workplace_max_contacts_ptr;

	thrust::device_vector<locOffset_t> household_offsets;
	locOffset_t * household_offsets_ptr;

	thrust::device_vector<personId_t> errand_people_table_a;		//people_array for errands
	thrust::device_vector<personId_t> errand_people_table_b;
	cub::DoubleBuffer<personId_t> errand_people_doubleBuffer;
	//personId_t * errand_people_table_ptr;
	thrust::device_vector<locId_t> people_errands_a;
	thrust::device_vector<locId_t> people_errands_b;
	cub::DoubleBuffer<locId_t> people_errands_doubleBuffer;
	void * errand_sorting_tempStorage;
	size_t errand_sorting_tempStorage_size;
	//locId_t * people_errands_ptr;

//	thrust::device_vector<locId_t> infected_errands;
//	locId_t * infected_errands_ptr;

	thrust::device_vector<locOffset_t> errand_locationOffsets;
	locOffset_t * errand_locationOffsets_ptr;


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
	void debug_generateErrandLookup();


	void debug_validateErrandSchedule();

	void debug_dumpWeekendErrandTables(h_vec * h_sorted_people, h_vec * h_sorted_hours, h_vec * h_sorted_dests);

	void debug_validatePeopleSetup();
	void debug_freshenPeopleStatus();
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

	void debug_validateActions();
	void doInitialInfections_statusWord();

	void debug_testErrandRegen_weekday();
	void debug_testErrandRegen_weekend();

	void debug_testWorkplaceAssignmentFunctor();
};

#define day_of_week() (current_day % 7)
#define is_weekend() (day_of_week() >= 5)
//#define is_weekend() (1)
//#define is_weekend() (current_day % 1 == 1)

#define errand_hours_today() (is_weekend() ? NUM_WEEKEND_ERRAND_HOURS : NUM_WEEKDAY_ERRAND_HOURS)

#define errands_per_person_today() (is_weekend()? NUM_WEEKEND_ERRANDS : NUM_WEEKDAY_ERRANDS)
#define num_infected_errands_today() (is_weekend() ? infected_count * NUM_WEEKEND_ERRANDS : infected_count * NUM_WEEKDAY_ERRANDS)
#define num_all_errands_today() (is_weekend() ? number_people * NUM_WEEKEND_ERRANDS : number_people * NUM_WEEKDAY_ERRANDS)

#define contacts_per_person() (is_weekend() ? MAX_CONTACTS_WEEKEND : MAX_CONTACTS_WEEKDAY)
#define num_infected_contacts_today() (is_weekend() ? infected_count * MAX_CONTACTS_WEEKEND : infected_count * MAX_CONTACTS_WEEKDAY)

#define status_is_infected(status) (status >= STATUS_INFECTED)
#define person_is_infected(status_p, status_s) (status_is_infected(status_p) || status_is_infected(status_s))
#define get_profile_from_status(status) (status - STATUS_INFECTED)

#define get_hour_from_locId_t(errand) (errand / number_workplaces)
#define get_location_from_locId_t(errand) (errand % number_workplaces)

int roundHalfUp_toInt(double d);


__device__ personId_t device_getVictimAtIndex(personId_t index_to_fetch, personId_t * location_people, kval_type_t contact_type);
__device__ kval_t device_selectRandomPersonFromLocation(personId_t infector_idx, personId_t loc_offset, int loc_count, unsigned int rand_val, kval_type_t desired_kval, personId_t * location_people_arr, personId_t * output_victim_idx_arr, kval_type_t * output_kval_arr);
__device__ void device_lookupLocationData_singleHour(personId_t myIdx, locId_t * lookup_arr, locOffset_t * loc_offset_arr, locOffset_t * loc_offset, int * loc_count);
__device__ void device_lookupWorkplaceData_singleHour(locId_t myLoc, locOffset_t * loc_offset_arr, locOffset_t * loc_offset, int * loc_count, maxContacts_t * loc_max_contacts);
__device__ void device_lookupErrandLocationData(locId_t myLoc, locOffset_t * loc_offset_arr, locOffset_t * output_loc_offset, int * output_loc_count);
__device__ void device_nullFillContact(personId_t * output_victim_idx, kval_type_t * output_kval);


//weekday errand assignment
__global__ void kernel_assignWeekdayAfterschoolAndErrands(age_t * people_ages_arr, int number_people, int num_locations, locId_t * errand_schedule_array, personId_t * errand_people_array, randOffset_t rand_offset);
__device__ void device_assignAfterschoolOrErrandDests_weekday(unsigned int rand_val1, unsigned int rand_val2,age_t myAge,  int num_locations, locId_t * output_dest1, locId_t * output_dest2);
__device__ locId_t device_fishAfterschoolOrErrandDestination_weekday(unsigned int rand_val, age_t myAge);

//weekday contacts kernel support method
__device__ errandContactsProfile_t device_assignContactsDesired_weekday_wholeDay(unsigned int rand_val, age_t myAge);

//weekend errand assignment
__global__ void kernel_assignWeekendErrands(personId_t * people_indexes_arr, locId_t * errand_scheduling_array, int num_people, int num_locations, randOffset_t rand_offset);
__device__ void device_copyPeopleIndexes_weekend_wholeDay(personId_t * id_dest_ptr, personId_t myIdx);
__device__ void device_generateWeekendErrands(locId_t * errand_array_ptr, randOffset_t myRandOffset);
__device__ locId_t device_fishWeekendErrandDestination(unsigned int rand_val);

//output metrics
__global__ void kernel_countInfectedStatusAndRecover(
	status_t * pandemic_status_array, status_t * seasonal_status_array, 
	day_t * pandemic_days_array, day_t * seasonal_days_array,
	int num_people, day_t current_day,
	int * output_pandemic_counts, int * output_seasonal_counts);

//contacts_to_action
__device__ float device_calculateInfectionProbability(int profile, int day_of_infection, int strain, kval_t kval_sum);

//initial setup methods
__global__ void kernel_householdTypeAssignment(householdType_t * hh_type_array, int num_households, randOffset_t rand_offset);
__device__ householdType_t device_setup_fishHouseholdType(unsigned int rand_val);

__global__ void kernel_generateHouseholds(
	householdType_t * hh_type_array, 
	int * adult_exscan_arr, int * child_exscan_arr, int num_households,
	locOffset_t * household_offset_arr,
	age_t * people_age_arr, locId_t * people_households_arr,
	randOffset_t rand_offset);
__device__ int device_setup_fishWorkplace(unsigned int rand_val);
__device__ void device_setup_fishSchoolAndAge(unsigned int rand_val, age_t * output_age_ptr, int * output_school_ptr);


void n_unique_numbers(h_vec * array, int n, int max);

char * action_type_to_string(action_t action);
char status_int_to_char(status_t s);
int lookup_school_typecode_from_age_code(int age_code);
char * status_profile_code_to_string(int p);

const char * lookup_contact_type(int contact_type);
const char * lookup_workplace_type(int workplace_type);
const char * lookup_age_type(age_t age_type);

void debug_print(char * message);
void debug_assert(bool condition, char * message);
void debug_assert(char *message, int expected, int actual);
void debug_assert(bool condition, char * message, int idx);


//sharedmem contact methods

__global__ void kernel_doWeekday(int num_infected, personId_t * infected_indexes, 
										 locOffset_t * errand_loc_offsets, personId_t * errand_people,
#if SIM_VALIDATION == 1
										 personId_t * output_infector_arr, personId_t * output_victim_arr,
										 kval_type_t * output_kval_arr,  action_t * output_action_arr, 
										 locId_t * output_contact_loc_arr,
										 float * float_rand1, float * float_rand2,
										 float * float_rand3, float * float_rand4,
#endif
										 day_t current_day,randOffset_t rand_offset);

__device__ kval_t device_makeContacts_weekday(
	personId_t myIdx, age_t myAge,
	personId_t * errand_loc_offsets, personId_t * errand_people,
	personId_t * output_victim_arr, kval_type_t * output_kval_arr,
#if SIM_VALIDATION == 1
	locId_t * output_contact_location,
#endif
	randOffset_t myRandOffset);

__global__ void kernel_doWeekend(int num_infected, personId_t * infected_indexes,
										 locOffset_t * errand_loc_offsets, personId_t * errand_people,
#if SIM_VALIDATION == 1
										 personId_t * output_infector_arr, personId_t * output_victim_arr, 
										 kval_type_t * output_kval_arr, action_t * output_action_arr,
										 locId_t * output_contact_location_arr,
										 float * float_rand1, float * float_rand2,
										 float * float_rand3, float * float_rand4,
#endif
										 day_t current_day,  randOffset_t rand_offset);

__device__ kval_t device_makeContacts_weekend(personId_t myIdx,
											  locOffset_t * errand_loc_offsets, personId_t * errand_people,
											  personId_t * output_victim_ptr, kval_type_t * output_kval_ptr,
#if SIM_VALIDATION == 1
											  locId_t * output_contact_loc_ptr,
#endif
											  randOffset_t myRandOffset);

__device__ int device_setInfectionStatus(status_t profile_to_set, day_t day_to_set, gen_t gen_to_set,
										 status_t * output_profile, day_t * output_day, gen_t * output_gen);

__device__ action_t device_doInfectionActionImmediately(personId_t victim,day_t day_to_set,
														bool infects_pandemic, bool infects_seasonal,
														status_t profile_p_to_set, status_t profile_s_to_set,
														gen_t gen_p_to_set, gen_t gen_s_to_set);
__device__ void device_processContacts(
	personId_t myIdx, kval_t kval_sum,
	personId_t * contact_victims_arr, kval_type_t *contact_type_arr, int contacts_per_infector,
#if SIM_VALIDATION == 1
	action_t * output_action_arr,
	float * rand_arr_1, float * rand_arr_2, float * rand_arr_3, float * rand_arr_4,
#endif
	day_t current_day,randOffset_t myRandOffset);
__device__ status_t device_getInfectionProfile(unsigned int rand_val);

__device__ maxContacts_t device_getWorkplaceMaxContacts(locId_t errand);


__device__ locId_t device_recalcWorkplace(personId_t myIdx, age_t myAge);
__device__ errandContactsProfile_t device_recalc_weekdayErrandDests_assignProfile(
	personId_t myIdx, age_t myAge, locId_t * output_dest1, locId_t * output_dest2);
__device__ void device_recalc_weekendErrandDests(personId_t myIdx, locId_t * errand_array_ptr);

__device__ void device_setup_assignWorkplaceOrSchool(unsigned int rand_val, age_t * age_ptr,locId_t * workplace_ptr);
__global__ void kernel_assignWorkplaces(age_t * people_ages_arr, locId_t * people_workplaces_arr, int number_people, randOffset_t rand_offset);


__device__ void device_getCalibrationValues(day_t day_of_infection, status_t profile,dailyFloatSum_t * output_expected_infections, dailyFloatSum_t * output_kval_sum);

__device__ void device_getMyKvalSum(personId_t personIdx);

//currently unused
struct personStatusStruct_t{
	unsigned char status_pandemic;
	day_t day_pandemic;
	gen_t gen_pandemic;
	unsigned char status_seasonal;
	day_t day_seasonal;
	gen_t gen_seasonal;
	age_t age;
	//unsigned char padding;
};

typedef unsigned long long personStatusStruct_word_t;

union personStatusUnion{
	struct personStatusStruct_t s;
	personStatusStruct_word_t w;
};


