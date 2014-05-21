#include "stdafx.h"

#include "simParameters.h"
#include "profiler.h"

#include "PandemicSim.h"
#include "thrust_functors.h"

#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <stdexcept>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>


int cuda_blocks = DEVICE_GRID_BLOCKS;
int cuda_threads = DEVICE_GRID_THREADS;


FILE * f_outputInfectedStats;

FILE * fDebug;
float max_proportion_infected = 0;
int max_proportion_infected_day = 0;

__device__ __constant__ SEED_T SEED_DEVICE[SEED_LENGTH];
SEED_T SEED_HOST[SEED_LENGTH];


__device__ __constant__ int WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[6][2];
int WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[6][2];
__device__ __constant__ int WEEKDAY_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[4][2];
int WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[4][2];

#define STRAIN_COUNT 2
#define STRAIN_PANDEMIC 0
#define STRAIN_SEASONAL 1
//__device__ __constant__ float BASE_REPRODUCTION_DEVICE[STRAIN_COUNT];
float BASE_REPRODUCTION_HOST[STRAIN_COUNT];

#define BASE_R_PANDEMIC_HOST BASE_REPRODUCTION_HOST[0]
#define BASE_R_SEASONAL_HOST BASE_REPRODUCTION_HOST[1]

__device__ __constant__ float INFECTIOUSNESS_FACTOR_DEVICE[STRAIN_COUNT];
float INFECTIOUSNESS_FACTOR_HOST[STRAIN_COUNT];

__device__ __constant__ float PERCENT_SYMPTOMATIC_DEVICE[1];
float PERCENT_SYMPTOMATIC_HOST[1];

__device__ __constant__ kval_t KVAL_LOOKUP_DEVICE[NUM_CONTACT_TYPES];
kval_t KVAL_LOOKUP_HOST[NUM_CONTACT_TYPES];


float WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[NUM_BUSINESS_TYPES];
__device__ float WORKPLACE_TYPE_ASSIGNMENT_PDF_DEVICE[NUM_BUSINESS_TYPES];

__device__ __constant__ float WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_DEVICE[NUM_BUSINESS_TYPES];				//stores PDF for weekday errand destinations
float WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[NUM_BUSINESS_TYPES];
__device__ __constant__ float WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_DEVICE[NUM_BUSINESS_TYPES];				//stores PDF for weekend errand destinations
float WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[NUM_BUSINESS_TYPES];

int WORKPLACE_TYPE_OFFSET_HOST[NUM_BUSINESS_TYPES];
__device__ __constant__ int WORKPLACE_TYPE_OFFSET_DEVICE[NUM_BUSINESS_TYPES];			//stores location number of first business of this type
int WORKPLACE_TYPE_COUNT_HOST[NUM_BUSINESS_TYPES];
__device__ __constant__ int WORKPLACE_TYPE_COUNT_DEVICE[NUM_BUSINESS_TYPES];				//stores number of each type of business
int WORKPLACE_TYPE_MAX_CONTACTS_HOST[NUM_BUSINESS_TYPES];
__device__ __constant__ int WORKPLACE_TYPE_MAX_CONTACTS_DEVICE[NUM_BUSINESS_TYPES];

//future experiment: const vs global memory
//__device__ float VIRAL_SHEDDING_PROFILES_GLOBALMEM[NUM_PROFILES][CULMINATION_PERIOD];
//__constant__ float VIRAL_SHEDDING_PROFILES_CONSTMEM[NUM_PROFILES][CULMINATION_PERIOD];

__device__ __constant__ float VIRAL_SHEDDING_PROFILES_DEVICE[NUM_SHEDDING_PROFILES][CULMINATION_PERIOD];
float VIRAL_SHEDDING_PROFILES_HOST[NUM_SHEDDING_PROFILES][CULMINATION_PERIOD];


float CHILD_AGE_CDF_HOST[CHILD_DATA_ROWS];
__device__ float CHILD_AGE_CDF_DEVICE[CHILD_DATA_ROWS];
int CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[CHILD_DATA_ROWS];
__device__ int CHILD_AGE_SCHOOLTYPE_LOOKUP_DEVICE[CHILD_DATA_ROWS];

float HOUSEHOLD_TYPE_CDF_HOST[HH_TABLE_ROWS];
__device__ __constant__ float HOUSEHOLD_TYPE_CDF_DEVICE[HH_TABLE_ROWS];
int HOUSEHOLD_TYPE_ADULT_COUNT_HOST[HH_TABLE_ROWS];
__device__ __constant__ int HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[HH_TABLE_ROWS];
int HOUSEHOLD_TYPE_CHILD_COUNT_HOST[HH_TABLE_ROWS];
__device__ __constant__ int HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[HH_TABLE_ROWS];

__device__ __constant__ simRandOffsetsStruct_t device_randOffsetsStruct[1];
simRandOffsetsStruct_t host_randOffsetsStruct[1];

__device__ __constant__ simSizeConstantsStruct_t device_simSizeStruct[1];
simSizeConstantsStruct_t host_simSizeStruct[1];

__device__ __constant__ simArrayPtrStruct_t device_arrayPtrStruct[1];
__device__ __constant__ simDebugArrayPtrStruct_t device_debugArrayPtrStruct[1];

PandemicSim::PandemicSim() 
{
	//age adult must not be zero, since we use that as a flag for being a child to generate households
	if(AGE_ADULT == 0)
		throw;


	logging_openOutputStreams();

	if(SIM_PROFILING)
	{
		const char * profile_filename = OUTPUT_FILES_IN_PARENTDIR ? "../profile_log.csv" : "profile_log.csv";
		const char * function_log_filename = NULL;

		if(debug_log_function_calls)
			function_log_filename = OUTPUT_FILES_IN_PARENTDIR ? "../function_log.csv" : "function_log.csv";

		profiler.initStack(profile_filename,function_log_filename);
	}

	cudaStreamCreate(&stream_secondary);

	setup_loadParameters();
	setup_scaleSimulation();
	setup_calculateInfectionData();

	logging_setSimData(people_scaling_factor,location_scaling_factor,NAME_OF_SIM_TYPE, NAME_OF_SIM_DEVICE,core_seed);

	//copy everything down to the GPU
	setup_pushDeviceData();

	if(TIMING_BATCH_MODE == 0)
	{
		setup_setCudaTopology();
	}

	if(SIM_VALIDATION && debug_log_function_calls)
		debug_print("parameters loaded");

}


PandemicSim::~PandemicSim(void)
{
	cudaStreamDestroy(stream_secondary);

	if(SIM_PROFILING)
		profiler.done();
	logging_closeOutputStreams();
}

void PandemicSim::setupSim()
{
	if(SIM_PROFILING)
	{
		profiler.beginFunction(-1,"setupSim");
	}

	//moved to constructor for batching
	//	open_debug_streams();
	//	setupLoadParameters();

	rand_offset = 0;				//set global rand counter to 0

	current_day = 0;
	
	if(SIM_VALIDATION && debug_log_function_calls)
		debug_print("setting up households");

	//finish copydown of __constant__ sim data
	cudaDeviceSynchronize();

	//must be done before generating households
	number_people = setup_calcPopulationSize_thrust();
	setup_sizeGlobalArrays();

	host_simSizeStruct[0].number_people = number_people;
	host_simSizeStruct[0].number_households = number_households;
	host_simSizeStruct[0].number_workplaces = number_workplaces;
	cudaMemcpyToSymbolAsync(device_simSizeStruct,host_simSizeStruct,sizeof(simSizeConstantsStruct_t),0,cudaMemcpyHostToDevice);
	
	//setup households
	setup_generateHouseholds();	//generates according to PDFs
	setup_assignWorkplaces();
	setup_initializeStatusArrays();

	if(CONSOLE_OUTPUT)
		printf("%d people, %d households, %d workplaces\n",number_people, number_households, number_workplaces);

	setup_initialInfected();

	if(SIM_VALIDATION)
	{
		cudaDeviceSynchronize();

		debug_sizeHostArrays();
		debug_copyFixedData();
		debug_validatePeopleSetup();
	}

	//must be done every simulation, even if we're not doing a daily log
	logging_pollMemoryUsage_takeSample(current_day);

	if(SIM_PROFILING)
	{
		profiler.endFunction(-1, number_people);
	}

	if(SIM_VALIDATION && debug_log_function_calls)
		debug_print("simulation setup complete");

	if(SIM_VALIDATION)
		fflush(fDebug);
}


void PandemicSim::logging_openOutputStreams()
{
	if(SIM_VALIDATION && log_infected_info)
	{
		if(OUTPUT_FILES_IN_PARENTDIR)
			fInfected = fopen("../debug_infected.csv", "w");
		else
			fInfected = fopen("debug_infected.csv", "w");

		fprintf(fInfected, "current_day, i, idx, status_p, day_p, gen_p, status_s, day_s, gen_s\n");
	}

/*	if(log_location_info)
	{
		fLocationInfo = fopen("../debug_location_info.csv","w");
		fprintf(fLocationInfo, "current_day, hour_index, i, offset, count, max_contacts\n");
	}*/

	if(SIM_VALIDATION && log_contacts)
	{
		if(OUTPUT_FILES_IN_PARENTDIR)
			fContacts = fopen("../debug_contacts.csv", "w");
		else
			fContacts = fopen("debug_contacts.csv", "w");
		
		fprintf(fContacts, "current_day, i, infector_idx, victim_idx, contact_type, contact_loc, infector_loc, victim_loc, locs_matched\n");
	}


	if(SIM_VALIDATION && log_actions)
	{
		if(OUTPUT_FILES_IN_PARENTDIR)
			fActions = fopen("../debug_actions.csv", "w");
		else
			fActions = fopen("debug_actions.csv", "w");
		fprintf(fActions, "current_day, i, infector, victim, action_type, action_type_string\n");
	}

	if(SIM_VALIDATION && log_actions_filtered)
	{
		if(OUTPUT_FILES_IN_PARENTDIR)
			fActionsFiltered = fopen("../debug_filtered_actions.csv", "w");
		else
			fActionsFiltered = fopen("debug_filtered_actions.csv", "w");
		fprintf(fActionsFiltered, "current_day, i, type, victim, victim_status_p, victim_gen_p, victim_status_s, victim_gen_s\n");
	}
	

	if(SIM_VALIDATION)
	{
		if(OUTPUT_FILES_IN_PARENTDIR)
			fDebug = fopen("../debug.txt", "w");
		else
			fDebug = fopen("debug.txt", "w");
	}

	if(OUTPUT_FILES_IN_PARENTDIR)
		f_outputInfectedStats=fopen("../output_infected_stats.csv","w");
	else
		f_outputInfectedStats=fopen("output_infected_stats.csv","w");
	fprintf(f_outputInfectedStats, "day,pandemic_susceptible,pandemic_infectious,pandemic_symptomatic,pandemic_asymptomatic,pandemic_recovered,seasonal_susceptible,seasonal_infectious,seasonal_symptomatic,seasonal_asymptomatic,seasonal_recovered\n");

}

void PandemicSim::setup_loadParameters()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_loadParameters");

	setup_loadSeed();

	//if printing seeds is desired for debug, etc
	if(CONSOLE_OUTPUT)
	{
		printf("seeds:\t");
		for(int i = 0; i < SEED_LENGTH; i++)
			if(i < SEED_LENGTH - 1)
				printf("%d\t",SEED_HOST[i]);
			else
				printf("%d\n",SEED_HOST[i]);
	}

	//read constants file 
	FILE * fConstants = fopen("constants.csv","r");	//open file
	if(fConstants == NULL)
	{
		debug_print("failed to open constants file");
		perror("Error opening constants file");
		exit(1);
	}

	//get a line buffer
#define LINEBUFF_SIZE 512
	char line[LINEBUFF_SIZE];	

	fgets(line, LINEBUFF_SIZE, fConstants);	//read the first line into the buffer to skip it
	fscanf(fConstants,"%*[^,]%*c");	//skip the first column of the table
	fscanf(fConstants, "%d%*c", &MAX_DAYS);
	fscanf(fConstants, "%f%*c", &BASE_R_PANDEMIC_HOST);
	fscanf(fConstants, "%f%*c", &BASE_R_SEASONAL_HOST);
	fscanf(fConstants, "%d%*c", &INITIAL_INFECTED_PANDEMIC);
	fscanf(fConstants, "%d%*c", &INITIAL_INFECTED_SEASONAL);
	fscanf(fConstants, "%f%*c", &people_scaling_factor);
	fscanf(fConstants, "%f%*c", &location_scaling_factor);
	fscanf(fConstants, "%f%*c", PERCENT_SYMPTOMATIC_HOST);
	fscanf(fConstants, "%f", &asymp_factor);
	fclose(fConstants);

	number_households = 1000000;
	number_workplaces = 12800;

	if(CONSOLE_OUTPUT)
		printf("max days: %d\nr_p: %f\nr_s: %f\ninitial_pandemic: %d\ninitial_seasonal: %d\nnumber_households: %d\n",
		MAX_DAYS,
		BASE_R_PANDEMIC_HOST,
		BASE_R_SEASONAL_HOST,
		INITIAL_INFECTED_PANDEMIC,
		INITIAL_INFECTED_SEASONAL,
		number_households);

	//read other parameter sets
	//hard coded these for time currently since we have no other sets

	//cdf for child age
	CHILD_AGE_CDF_HOST[0] = 0.24f;
	CHILD_AGE_CDF_HOST[1] = 0.47f;
	CHILD_AGE_CDF_HOST[2] = 0.72f;
	CHILD_AGE_CDF_HOST[3] = 0.85f;
	CHILD_AGE_CDF_HOST[4] = 1.0f;

	//what workplace type children get for this age
	CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[0] = 3;
	CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[1] = 4;
	CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[2] = 5;
	CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[3] = 6;
	CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[4] = 7;

	//workplace PDF for adults
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[0] = 0.06586f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[1] = 0.05802f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[2] = 0.30227f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[3] = 0.0048f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[4] = 0.00997f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[5] = 0.203f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[6] = 0.09736f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[7] = 0.10598f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[8] = 0.00681f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[9] = 0.02599f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[10] = 0.f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[11] = 0.08749f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[12] = 0.03181f;
	WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[13] = 0.00064f;

	//number of each type of workplace
	WORKPLACE_TYPE_COUNT_HOST[0] = 1000;
	WORKPLACE_TYPE_COUNT_HOST[1] = 7000;
	WORKPLACE_TYPE_COUNT_HOST[2] = 2400;
	WORKPLACE_TYPE_COUNT_HOST[3] = 300;
	WORKPLACE_TYPE_COUNT_HOST[4] = 100;
	WORKPLACE_TYPE_COUNT_HOST[5] = 200;
	WORKPLACE_TYPE_COUNT_HOST[6] = 100;
	WORKPLACE_TYPE_COUNT_HOST[7] = 100;
	WORKPLACE_TYPE_COUNT_HOST[8] = 300;
	WORKPLACE_TYPE_COUNT_HOST[9] = 500;
	WORKPLACE_TYPE_COUNT_HOST[10] = 0;
	WORKPLACE_TYPE_COUNT_HOST[11] = 300;
	WORKPLACE_TYPE_COUNT_HOST[12] = 400;
	WORKPLACE_TYPE_COUNT_HOST[13] = 100;

	//maximum number of contacts made at each workplace type
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[0] = 3;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[1] = 3;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[2] = 3;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[3] = 2;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[4] = 2;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[5] = 3;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[6] = 3;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[7] = 2;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[8] = 2;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[9] = 2;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[10] = 0;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[11] = 2;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[12] = 2;
	WORKPLACE_TYPE_MAX_CONTACTS_HOST[13] = 2;

	//pdf for weekday errand location generation
	//most entries are 0.0
	for(int type = 0; type < NUM_BUSINESS_TYPES; type++)
		WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[type] = 0.0f;
	WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[9] = 0.61919f;
	WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[11] = 0.27812f;
	WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[12] = 0.06601f;
	WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[13] = 0.03668f;

	//pdf for weekend errand location generation
	//most entries are 0.0
	for(int type = 0; type < NUM_BUSINESS_TYPES; type++)
		WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[type] = 0.0f;
	WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[9] = 0.51493f;
	WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[11] = 0.25586f;
	WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[12] = 0.1162f;
	WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[13] = 0.113f;


	//how many adults in each household type
	HOUSEHOLD_TYPE_ADULT_COUNT_HOST[0] = 1;
	HOUSEHOLD_TYPE_ADULT_COUNT_HOST[1] = 1;
	HOUSEHOLD_TYPE_ADULT_COUNT_HOST[2] = 2;
	HOUSEHOLD_TYPE_ADULT_COUNT_HOST[3] = 1;
	HOUSEHOLD_TYPE_ADULT_COUNT_HOST[4] = 2;
	HOUSEHOLD_TYPE_ADULT_COUNT_HOST[5] = 1;
	HOUSEHOLD_TYPE_ADULT_COUNT_HOST[6] = 2;
	HOUSEHOLD_TYPE_ADULT_COUNT_HOST[7] = 1;
	HOUSEHOLD_TYPE_ADULT_COUNT_HOST[8] = 2;

	//how many children in each household type
	HOUSEHOLD_TYPE_CHILD_COUNT_HOST[0] = 0;
	HOUSEHOLD_TYPE_CHILD_COUNT_HOST[1] = 1;
	HOUSEHOLD_TYPE_CHILD_COUNT_HOST[2] = 0;
	HOUSEHOLD_TYPE_CHILD_COUNT_HOST[3] = 2;
	HOUSEHOLD_TYPE_CHILD_COUNT_HOST[4] = 1;
	HOUSEHOLD_TYPE_CHILD_COUNT_HOST[5] = 3;
	HOUSEHOLD_TYPE_CHILD_COUNT_HOST[6] = 2;
	HOUSEHOLD_TYPE_CHILD_COUNT_HOST[7] = 4;
	HOUSEHOLD_TYPE_CHILD_COUNT_HOST[8] = 3;

	//the PDF of each household type
	HOUSEHOLD_TYPE_CDF_HOST[0] = 0.279f;
	HOUSEHOLD_TYPE_CDF_HOST[1] = 0.319f;
	HOUSEHOLD_TYPE_CDF_HOST[2] = 0.628f;
	HOUSEHOLD_TYPE_CDF_HOST[3] = 0.671f;
	HOUSEHOLD_TYPE_CDF_HOST[4] = 0.8f;
	HOUSEHOLD_TYPE_CDF_HOST[5] = 0.812f;
	HOUSEHOLD_TYPE_CDF_HOST[6] = 0.939f;
	HOUSEHOLD_TYPE_CDF_HOST[7] = 0.944f;
	HOUSEHOLD_TYPE_CDF_HOST[8] = 1.0f;

	//store all permutations of contact assignments

	//number of contacts made in each hour
	WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[0][0] = 2;
	WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[0][1] = 0;

	WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[1][0] = 0;
	WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[1][1] = 2;

	WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[2][0] = 1;
	WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[2][1] = 1;

	//afterschool errands
	WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[3][0] = WORKPLACE_TYPE_MAX_CONTACTS_HOST[BUSINESS_TYPE_AFTERSCHOOL];
	WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[3][1] = 0;

	//DIFFERENT FORMAT: hours each of the 2 contacts are made in
	//2 contacts in errand  0
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[0][0] = 0;
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[0][1] = 0;

	//2 contacts in errand 1
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[1][0] = 1;
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[1][1] = 1;

	//2 contacts in errand 2
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[2][0] = 2;
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[2][1] = 2;

	//contact in errand 0 and errand 1
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[3][0] = 0;
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[3][1] = 1;

	//contact in errand 0 and errand 2
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[4][0] = 0;
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[4][1] = 2;

	//contact in errand 1 and 2
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[5][0] = 1;
	WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[5][1] = 2;


#pragma region profiles
	//gamma1
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][0] = 0.007339835f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][1] = 0.332600216f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][2] = 0.501192066f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][3] = 0.142183447f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][4] = 0.015675154f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][5] = 0.000967407f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][6] = 4.055E-05f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][7] = 1.29105E-06f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][8] = 3.34836E-08f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA1][9] = 7.41011E-10f;

	//lognorm1
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][0] = 0.002533572f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][1] = 0.348252834f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][2] = 0.498210218f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][3] = 0.130145145f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][4] = 0.018421298f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][5] = 0.002158374f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][6] = 0.000245489f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][7] = 2.88922E-05f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][8] = 3.61113E-06f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM1][9] = 4.83901E-07f;

	//weib1
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][0] = 0.05927385f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][1] = 0.314171688f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][2] = 0.411588802f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][3] = 0.187010054f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][4] = 0.026934715f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][5] = 0.001013098f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][6] = 7.78449E-06f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][7] = 9.29441E-09f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][8] = 1.29796E-12f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB1][9] = 0;

	//gamma2
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][0] = 0.04687299f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][1] = 0.248505983f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][2] = 0.30307952f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][3] = 0.211008627f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][4] = 0.11087006f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][5] = 0.049241932f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][6] = 0.019562658f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][7] = 0.007179076f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][8] = 0.002482875f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_GAMMA2][9] = 0.000820094f;

	//lognorm2
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][0] = 0.028667712f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][1] = 0.283445338f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][2] = 0.319240133f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][3] = 0.190123057f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][4] = 0.093989959f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][5] = 0.044155659f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][6] = 0.020682822f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][7] = 0.009841839f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][8] = 0.00479234f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_LOGNORM2][9] = 0.002393665f;

	//weib2
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][0] = 0.087866042f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][1] = 0.223005225f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][2] = 0.258992749f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][3] = 0.208637267f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][4] = 0.127489076f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][5] = 0.061148649f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][6] = 0.023406737f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][7] = 0.007216643f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][8] = 0.001802145f;
	VIRAL_SHEDDING_PROFILES_HOST[PROFILE_WEIB2][9] = 0.00036581f;

#pragma endregion profiles

	//store kvals - all 1 except for no-contact
	KVAL_LOOKUP_HOST[CONTACT_TYPE_NONE] = 0;
	for(int i = CONTACT_TYPE_NONE + 1; i < NUM_CONTACT_TYPES;i++)
		KVAL_LOOKUP_HOST[i] = 1;

	if(SIM_PROFILING)
		profiler.endFunction(-1,1);
}

//push various things to device constant memory
void PandemicSim::setup_pushDeviceData()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_pushDeviceData");

	//data for generating households
	cudaMemcpyToSymbolAsync(
		HOUSEHOLD_TYPE_CDF_DEVICE,
		HOUSEHOLD_TYPE_CDF_HOST,
		sizeof(float) * HH_TABLE_ROWS,
		0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(
		HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE,
		HOUSEHOLD_TYPE_ADULT_COUNT_HOST,
		sizeof(int) * HH_TABLE_ROWS,
		0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(
		HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE,
		HOUSEHOLD_TYPE_CHILD_COUNT_HOST,
		sizeof(int) * HH_TABLE_ROWS,
		0,cudaMemcpyHostToDevice);

	//data for assigning children age and school
	cudaMemcpyToSymbolAsync(
		CHILD_AGE_CDF_DEVICE,
		CHILD_AGE_CDF_HOST,
		sizeof(float) * CHILD_DATA_ROWS,
		0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(
		CHILD_AGE_SCHOOLTYPE_LOOKUP_DEVICE,
		CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST,
		sizeof(int) * CHILD_DATA_ROWS,
		0,cudaMemcpyHostToDevice);

	//data for assigning workplaces
	cudaMemcpyToSymbolAsync(
		WORKPLACE_TYPE_ASSIGNMENT_PDF_DEVICE,
		WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST,
		sizeof(float) * NUM_BUSINESS_TYPES,
		0,cudaMemcpyHostToDevice);

	//workplace location data
	cudaMemcpyToSymbolAsync(
		WORKPLACE_TYPE_COUNT_DEVICE,
		WORKPLACE_TYPE_COUNT_HOST,
		sizeof(int) * NUM_BUSINESS_TYPES,
		0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(
		WORKPLACE_TYPE_OFFSET_DEVICE,
		WORKPLACE_TYPE_OFFSET_HOST,
		sizeof(int) * NUM_BUSINESS_TYPES,
		0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(
		WORKPLACE_TYPE_MAX_CONTACTS_DEVICE,
		WORKPLACE_TYPE_MAX_CONTACTS_HOST,
		sizeof(int) * NUM_BUSINESS_TYPES,
		0,cudaMemcpyHostToDevice);

	//weekday+weekend errand PDFs
	cudaMemcpyToSymbolAsync(
		WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_DEVICE,
		WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST,
		sizeof(float) * NUM_BUSINESS_TYPES,
		0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(
		WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_DEVICE,
		WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST,
		sizeof(float) * NUM_BUSINESS_TYPES,
		0,cudaMemcpyHostToDevice);


	//alternate weekend contacts_desired assignment mode
	cudaMemcpyToSymbolAsync(
		WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_DEVICE,
		WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST,
		sizeof(int) * 6 * 2,
		0,cudaMemcpyHostToDevice);

	cudaMemcpyToSymbolAsync(
		WEEKDAY_ERRAND_CONTACT_ASSIGNMENTS_DEVICE,
		WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST,
		sizeof(int) * 4 * 2,
		0,cudaMemcpyHostToDevice);

	//seeds
	cudaMemcpyToSymbolAsync(
		SEED_DEVICE,
		SEED_HOST,
		sizeof(SEED_T) * SEED_LENGTH,
		0,cudaMemcpyHostToDevice);

	//kvals
	cudaMemcpyToSymbolAsync(
		KVAL_LOOKUP_DEVICE,
		KVAL_LOOKUP_HOST,
		sizeof(kval_t) * NUM_CONTACT_TYPES);

	//copy adjusted reproduction numbers
	cudaMemcpyToSymbolAsync(
		INFECTIOUSNESS_FACTOR_DEVICE,
		INFECTIOUSNESS_FACTOR_HOST,
		sizeof(float) * STRAIN_COUNT,
		0,cudaMemcpyHostToDevice);

	//copy viral shedding profiles
	cudaMemcpyToSymbolAsync(
		VIRAL_SHEDDING_PROFILES_DEVICE,
		VIRAL_SHEDDING_PROFILES_HOST,
		sizeof(float) * NUM_SHEDDING_PROFILES * CULMINATION_PERIOD,
		0,cudaMemcpyHostToDevice);

	cudaMemcpyToSymbolAsync(
		PERCENT_SYMPTOMATIC_DEVICE,
		PERCENT_SYMPTOMATIC_HOST,
		sizeof(float) * 1,
		0,cudaMemcpyHostToDevice);

	//future experiment: constant vs global memory
/*	cudaMemcpyToSymbolAsync(
		VIRAL_SHEDDING_PROFILES_GLOBALMEM,
		VIRAL_SHEDDING_PROFILES_HOST,
		sizeof(float) * NUM_PROFILES * CULMINATION_PERIOD,
		0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(
		VIRAL_SHEDDING_PROFILES_CONSTMEM,
		VIRAL_SHEDDING_PROFILES_HOST,
		sizeof(float) * NUM_PROFILES * CULMINATION_PERIOD,
		0, cudaMemcpyHostToDevice);*/

	//must synchronize later!
	if(SIM_PROFILING)
		profiler.endFunction(-1,1);
}



//Sets up the initial infection at the beginning of the simulation
//BEWARE: you must not generate dual infections with this code, or you will end up with duplicate infected indexes
void PandemicSim::setup_initialInfected()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"setup_initialInfected");

	//fill infected array with null info (not infected)
	int initial_infected = INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL;

	//get N unique indexes - they should not be sorted
	h_vec h_init_indexes(initial_infected);
	n_unique_numbers(&h_init_indexes, initial_infected, number_people);
	thrust::copy_n(h_init_indexes.begin(), initial_infected, infected_indexes.begin());

	///// INFECTED PANDEMIC:
	//infect first INITIAL_INFECTED_PANDEMIC people with pandemic
	//set status to infected
	thrust::fill(
		thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes.begin()),	//begin at infected 0
		thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes.begin() + INITIAL_INFECTED_PANDEMIC),	//end at index INITIAL_INFECTED_PANDEMIC
		STATUS_INFECTED);

	//set day/generation pandemic to 0 (initial)
	thrust::fill(
		thrust::make_permutation_iterator(people_days_pandemic.begin(), infected_indexes.begin()),	//begin at infected 0
		thrust::make_permutation_iterator(people_days_pandemic.begin(), infected_indexes.begin() + INITIAL_INFECTED_PANDEMIC),	//end at index INITIAL_INFECTED_PANDEMIC
		INITIAL_DAY);//val
	thrust::fill(
		thrust::make_permutation_iterator(people_gens_pandemic.begin(), infected_indexes.begin()),	//begin at infected 0
		thrust::make_permutation_iterator(people_gens_pandemic.begin(), infected_indexes.begin() + INITIAL_INFECTED_PANDEMIC),	//end at index INITIAL_INFECTED_PANDEMIC
		INITIAL_GEN);	//fill infected with gen 0

	///// INFECTED SEASONAL:
	//set status to infected
	thrust::fill(
		thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin()+ INITIAL_INFECTED_PANDEMIC), //begin at index INITIAL_INFECTED_PANDEMIC
		thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin() + INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL),	//end INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL
		STATUS_INFECTED);

	//set day/generation seasonal to 0
	thrust::fill(
		thrust::make_permutation_iterator(people_days_seasonal.begin(), infected_indexes.begin()+ INITIAL_INFECTED_PANDEMIC), //begin at index INITIAL_INFECTED_PANDEMIC
		thrust::make_permutation_iterator(people_days_seasonal.begin(), infected_indexes.begin() + INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL),	//end INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL
		INITIAL_DAY);		//day: 0
	thrust::fill(
		thrust::make_permutation_iterator(people_gens_seasonal.begin(), infected_indexes.begin()+ INITIAL_INFECTED_PANDEMIC), //begin at index INITIAL_INFECTED_PANDEMIC
		thrust::make_permutation_iterator(people_gens_seasonal.begin(), infected_indexes.begin() + INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL),	//end INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL
		INITIAL_GEN);	//first generation

	if(SIM_PROFILING)
		profiler.endFunction(current_day,initial_infected);
}

//sets up the locations which are the same every day and do not change
//i.e. workplace and household
void PandemicSim::setup_buildFixedLocations()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_buildFixedLocations");
	///////////////////////////////////////
	//home/////////////////////////////////

	//disabled: now done in generateHouseholds() function
	/*thrust::sequence(household_people.begin(), household_people.begin() + number_people);	//fill array with IDs to sort
	calcLocationOffsets(
		&household_people,
		people_households,
		&household_offsets,
		number_people, number_households);*/

	///////////////////////////////////////
	//work/////////////////////////////////
	/*
	thrust::sequence(workplace_people.begin(), workplace_people.begin() + number_people);	//fill array with IDs to sort

	setup_calcLocationOffsets(
		&workplace_people,
		people_workplaces,
		&workplace_offsets,
		number_people, number_workplaces);*/

	if(SIM_PROFILING)
		profiler.endFunction(-1,number_people);
}


//given an array of people's ID numbers and locations
//sort them by location, and then build the location offset/count tables
//ids_to_sort will be sorted by workplace
void PandemicSim::setup_calcLocationOffsets(
	vec_t * ids_to_sort,
	vec_t lookup_table_copy,
	vec_t * location_offsets,
	int num_people, int num_locs)
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1, "calcLocationOffsets");

	//sort people by workplace
	thrust::sort_by_key(
		lookup_table_copy.begin(),
		lookup_table_copy.begin() + num_people,
		ids_to_sort->begin());

	//build count/offset table
	thrust::counting_iterator<int> count_iterator(0);
	thrust::lower_bound(		//find lower bound of each location
		lookup_table_copy.begin(),
		lookup_table_copy.begin() + num_people,
		count_iterator,
		count_iterator + num_locs,
		location_offsets->begin());

	//originally, we calculated the count by using an upper bound and then subtracting the lower bound
	//instead, we can calculate the count by the following formula:
	//loc_count = loc_offset[i+1] - loc_offset[i]
	//i.e. people = {1, 1, 2, 2, 3}
	//location_numbers = {1, 2, 3}
	//loc_offsets = {0, 2, 4}
	//We need to add one extra offset so the last location doesn't go out of bounds - this is equal to
	//the number of people in the array
	//so loc_offsets = {0, 2, 4, 5}
	(*location_offsets)[num_locs] = num_people;

	if(SIM_PROFILING)
		profiler.endFunction(-1,number_people);
}


void PandemicSim::logging_closeOutputStreams()
{
	if(SIM_VALIDATION && log_infected_info)
	{
		fclose(fInfected);
	}

	/*if(log_location_info)
	{
		fclose(fLocationInfo);
	}*/

	if(SIM_VALIDATION && log_contacts)
	{
		fclose(fContacts);
	}

	if(SIM_VALIDATION && log_actions)
	{
		fclose(fActions);
	}

	if(SIM_VALIDATION && log_actions_filtered)
	{
		fclose(fActionsFiltered);
	}

	if(LOG_INFECTED_PROPORTION)
	{
		FILE * fInfectedMaxPolling;
		if(OUTPUT_FILES_IN_PARENTDIR)
			fInfectedMaxPolling = fopen("../infected_max.txt","w");
		else
			fInfectedMaxPolling = fopen("infected_max.txt","w");

		fprintf(fInfectedMaxPolling,"max_proportion,day\n%f,%d\n",max_proportion_infected,max_proportion_infected_day);

		fclose(fInfectedMaxPolling);
	}

	if(SIM_VALIDATION)
		fclose(fDebug);

	fclose(f_outputInfectedStats);
} 



void PandemicSim::runToCompletion()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1, "runToCompletion");

	for(current_day = 0; current_day < MAX_DAYS; current_day++)
	{
		if(SIM_VALIDATION && debug_log_function_calls)
			debug_print("beginning day...");

		if(SIM_VALIDATION)
			debug_nullFillDailyArrays();

		//recover anyone who's culminated, and count the number of each status type as we go
		daily_countAndRecover();

		//build infected index array
		daily_buildInfectedArray_global();
		
		if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
			cudaDeviceSynchronize();

		if(infected_count == 0)
			break;

		if(SIM_VALIDATION)
		{
			debug_clearActionsArray(); //must occur AFTER we have counted infected

			debug_validateInfectionStatus();

			fprintf(fDebug, "\n\n---------------------\nday %d\ninfected: %d\n---------------------\n\n", current_day, infected_count);
			fflush(fDebug);
		}

		if(CONSOLE_OUTPUT)
		{
			printf("Day %3d:\tinfected: %5d\n", current_day + 1, infected_count);
		}

		if(POLL_MEMORY_USAGE)
			logging_pollMemoryUsage_takeSample(current_day);
		if(LOG_INFECTED_PROPORTION)
		{
			float proportion_infected = (float) infected_count / number_people;
			if(proportion_infected > max_proportion_infected)
			{
				max_proportion_infected = proportion_infected;
				max_proportion_infected_day = current_day;
			}
		}

		//MAKE CONTACTS DEPENDING ON TYPE OF DAY
		if(is_weekend())
			doWeekend_wholeDay();
		else
			doWeekday_wholeDay();

		if(SIM_VALIDATION)
		{
			validateContacts_wholeDay();
			debug_validateActions();
			fflush(fDebug);
		}

		cudaDeviceSynchronize();

		//if we're using the profiler, flush each day in case of crash
		if(SIM_PROFILING)
		{
			profiler.dailyFlush();
		}
	}

	cudaDeviceSynchronize();
	final_releaseMemory();
	final_countReproduction();

	if(SIM_PROFILING)
		profiler.endFunction(-1, number_people);


	//moved to destructor for batching
	//close_output_streams();
}


//copies indexes 3 times into array, i.e. for IDS 1-3 produces array:
// 1 2 3 1 2 3 1 2 3
__device__ void device_copyPeopleIndexes_weekend_wholeDay(personId_t * id_dest_ptr, personId_t myIdx)
{
	id_dest_ptr[0] = myIdx;
	id_dest_ptr[1] = myIdx;
	id_dest_ptr[2] = myIdx;
}

//gets three UNIQUE errand hours 
__device__ void device_generateWeekendErrands(locId_t * errand_output_ptr, randOffset_t myRandOffset)
{
	int num_locations = device_simSizeStruct->number_workplaces;

	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
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

	errand_output_ptr[0] = device_fishWeekendErrandDestination(u.i[3]) + (hour1 * num_locations);
	errand_output_ptr[1] = device_fishWeekendErrandDestination(u.i[4]) + (hour2 * num_locations);
	errand_output_ptr[2] = device_fishWeekendErrandDestination(u.i[5]) + (hour3 * num_locations);
}

__device__ locId_t device_fishWeekendErrandDestination(unsigned int rand_val)
{
	float y = (float) rand_val / UNSIGNED_MAX;

	int row = FIRST_WEEKEND_ERRAND_ROW;
	while(y > WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_DEVICE[row] && row < (NUM_BUSINESS_TYPES - 1))
	{
		y -= WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_DEVICE[row];
		row++;
	}
	float frac = y / WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_DEVICE[row];
	int type_count = WORKPLACE_TYPE_COUNT_DEVICE[row];
	int business_num = frac * type_count;

	if(business_num >= type_count)
		business_num = type_count - 1;

	int type_offset = WORKPLACE_TYPE_OFFSET_DEVICE[row];

	return business_num + type_offset;
}

//will resize the infected, contact, and action arrays to fit the entire population
void PandemicSim::setup_sizeGlobalArrays()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_sizeGlobalArrays");

	people_status_pandemic.resize(number_people);
	people_status_seasonal.resize(number_people);

	people_days_pandemic.resize(number_people);
	people_days_seasonal.resize(number_people);

	people_gens_pandemic.resize(number_people);
	people_gens_seasonal.resize(number_people);

	people_ages.resize(number_people);
	people_households.resize(number_people);
//	people_workplaces.resize(number_people);

	household_offsets.resize(number_households + 1);

	workplace_offsets.resize(number_workplaces + 1);
	workplace_people.resize(number_people);
//	workplace_max_contacts.resize(number_workplaces);

	expected_max_infected = number_people;
	infected_indexes.resize(expected_max_infected);

	//weekend errands arrays tend to be very large, so pre-allocate them
	int num_weekend_errands = number_people * NUM_WEEKEND_ERRANDS;
	errand_people_table_a.resize(num_weekend_errands);
	errand_people_table_b.resize(num_weekend_errands);
	people_errands_a.resize(num_weekend_errands);
	people_errands_b.resize(num_weekend_errands);
	setup_configCubBuffers();
	setup_sizeCubTempArray();

//	infected_errands.resize(num_weekend_errands);

	errand_locationOffsets.resize((number_workplaces * NUM_WEEKEND_ERRAND_HOURS) + 1);

	status_counts.resize(16);

	if(SIM_VALIDATION)
	{
		int expected_max_contacts = expected_max_infected * MAX_POSSIBLE_CONTACTS_PER_DAY;
		daily_contact_infectors.resize(expected_max_contacts);
		daily_contact_victims.resize(expected_max_contacts);
		daily_contact_kval_types.resize(expected_max_contacts);
		daily_action_type.resize(expected_max_contacts);
		daily_contact_locations.resize(expected_max_contacts);

		debug_contactsToActions_float1.resize(expected_max_contacts);
		debug_contactsToActions_float2.resize(expected_max_contacts);
		debug_contactsToActions_float3.resize(expected_max_contacts);
		debug_contactsToActions_float4.resize(expected_max_contacts);
	}

	setup_fetchVectorPtrs(); //get the raw int * pointers

	if(SIM_PROFILING)
	{
		profiler.endFunction(-1,number_people);
	}
}



void PandemicSim::debug_nullFillDailyArrays()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"debug_nullFillDailyArrays");

	thrust::fill(daily_contact_infectors.begin(), daily_contact_infectors.end(), NULL_PERSON_INDEX);
	thrust::fill(daily_contact_victims.begin(), daily_contact_victims.end(), NULL_PERSON_INDEX);
	thrust::fill(daily_contact_kval_types.begin(), daily_contact_kval_types.end(), CONTACT_TYPE_NONE);

	thrust::fill(daily_action_type.begin(), daily_action_type.end(), ACTION_INFECT_NONE);

//	thrust::fill(infected_errands.begin(), infected_errands.end(), NULL_ERRAND);

	if(SIM_PROFILING)
		profiler.endFunction(current_day, number_people);
}

void PandemicSim::setup_scaleSimulation()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_scaleSimulation");

	number_households = roundHalfUp_toInt(people_scaling_factor * (double) number_households);

	int sum = 0;
	for(int business_type = 0; business_type < NUM_BUSINESS_TYPES; business_type++)
	{
		//for each type of business, scale by overall simulation scalar
		int original_type_count = roundHalfUp_toInt(WORKPLACE_TYPE_COUNT_HOST[business_type]);
		int new_type_count = roundHalfUp_toInt(location_scaling_factor * original_type_count);

		//if at least one business of this type existed in the original data, make sure at least one exists in the new data
		if(new_type_count == 0 && original_type_count > 0)
			new_type_count = 1;

		WORKPLACE_TYPE_COUNT_HOST[business_type] = new_type_count;
		sum += new_type_count;
	}

	number_workplaces = sum;

	//calculate the offset of each workplace type
	thrust::exclusive_scan(
		WORKPLACE_TYPE_COUNT_HOST,
		WORKPLACE_TYPE_COUNT_HOST + NUM_BUSINESS_TYPES,
		WORKPLACE_TYPE_OFFSET_HOST);

	if(SIM_PROFILING)
		profiler.endFunction(-1,NUM_BUSINESS_TYPES);
}

void PandemicSim::debug_dump_array_toTempFile(const char * filename, const char * description, d_vec * target_array, int array_count)
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "debug_dumpArray_toTempFile");

	h_vec host_array(array_count);
	thrust::copy_n(target_array->begin(), array_count, host_array.begin());

	FILE * fTemp = fopen(filename,"w");
	fprintf(fTemp,"i,%s\n",description);
	for(int i = 0; i < array_count; i++)
	{
		fprintf(fTemp,"%d,%d\n",i,host_array[i]);
	}
	fclose(fTemp);

	if(SIM_PROFILING)
		profiler.endFunction(current_day,array_count);
}


void PandemicSim::doWeekday_wholeDay()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "doWeekday_wholeDay");

	//generate errands and afterschool locations
	weekday_generateAfterschoolAndErrandDestinations();

	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
		cudaDeviceSynchronize();

//	debug_dump_array_toTempFile("../unsorted_dests.txt","errand dest", &errand_people_destinations, number_people * NUM_WEEKDAY_ERRAND_HOURS);

	//fish out the locations of the infected people
//	weekday_doInfectedSetup_wholeDay();

//	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
//		cudaDeviceSynchronize();
	if(SIM_VALIDATION)
	{
		//debug_testErrandRegen_weekday();
	}

	//generate location arrays for each hour
	/*for(int hour = 0; hour < NUM_WEEKDAY_ERRAND_HOURS; hour++)
	{
		int people_offset_start = hour * number_people;
		int people_offset_end = (hour+1) * number_people;

		//int * errand_people_ptr = (int *) errand_people_doubleBuffer

		//write sequential blocks of indexes, i.e. 0 1 2 0 1 2
		thrust::sequence(
			errand_people_doubleBuffer.Current() + people_offset_start,
			errand_people_doubleBuffer.Current() + people_offset_end);
	}*/

	int num_errands =  (2 * number_people);
	cub::DeviceRadixSort::SortPairs(
		errand_sorting_tempStorage, errand_sorting_tempStorage_size, //temp buffer
		people_errands_doubleBuffer, errand_people_doubleBuffer,	//key, val
		num_errands);	//N


	thrust::counting_iterator<locId_t> count_it(0);
	thrust::device_vector<locId_t>::iterator errands_iterator;

	if(people_errands_doubleBuffer.selector == 0)
	{
		errands_iterator = people_errands_a.begin();
	}
	else
	{
		errands_iterator = people_errands_b.begin();
	}

	thrust::lower_bound(
			errands_iterator,
			errands_iterator + num_errands,
			count_it,
			count_it + (NUM_WEEKDAY_ERRAND_HOURS * number_workplaces),
			errand_locationOffsets.begin());

	errand_locationOffsets[NUM_WEEKDAY_ERRAND_HOURS * number_workplaces] = (NUM_WEEKDAY_ERRANDS * number_people);

//	debug_dump_array_toTempFile("../sorted_dests.txt", "errand_dest", &errand_people_destinations, number_people * NUM_WEEKDAY_ERRAND_HOURS);
//	debug_dump_array_toTempFile("../loc_offsets.txt", "loc_offset", &errand_locationOffsets_multiHour, NUM_WEEKDAY_ERRAND_HOURS * number_workplaces);
//	debug_dump_array_toTempFile("../inf_locs.txt", "loc", &errand_infected_locations, infected_count * NUM_WEEKDAY_ERRAND_HOURS);

//	debug_dumpInfectedErrandLocs();

	int blocks = cuda_makeWeekdayContactsKernel_blocks;
	int threads = cuda_makeWeekdayContactsKernel_threads;

	//get the amount of shared memory needed for each block
	size_t smem_size = sizeof(personId_t) + sizeof(kval_type_t);
	smem_size *= MAX_CONTACTS_WEEKDAY;
	smem_size *= threads;

	//size_t smem_size = 0;

	kernel_weekday_sharedMem<<<blocks,threads,smem_size>>> (infected_count,
		infected_indexes_ptr,people_ages_ptr,
		people_households_ptr,household_offsets_ptr,
		workplace_offsets_ptr,workplace_people_ptr,
		errand_locationOffsets_ptr, errand_people_doubleBuffer.Current(),
		people_status_pandemic_ptr,people_status_seasonal_ptr,
		people_days_pandemic_ptr,people_days_seasonal_ptr,
		people_gens_pandemic_ptr,people_gens_seasonal_ptr,
#if SIM_VALIDATION == 1
		daily_contact_infectors_ptr, daily_contact_victims_ptr,
		daily_contact_kval_types_ptr, daily_action_type_ptr,
		daily_contact_locations_ptr,
		debug_contactsToActions_float1_ptr, debug_contactsToActions_float2_ptr,
		debug_contactsToActions_float3_ptr, debug_contactsToActions_float4_ptr,
#endif
		current_day, rand_offset);

	if(TIMING_BATCH_MODE == 0)
	{
		const int rand_counts_consumed = 6;
		rand_offset += (rand_counts_consumed * infected_count);
	}

	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
		cudaDeviceSynchronize();

//	debug_dump_array_toTempFile("../infected_kvals.txt","kval",&infected_daily_kval_sum, infected_count);

	if(SIM_PROFILING)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::doWeekend_wholeDay()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "doWeekend_wholeDay");

	//assign all weekend errands
	weekend_assignErrands();
//	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
//		cudaDeviceSynchronize();

	//fish the infected errands out
//	weekend_doInfectedSetup_wholeDay();
	if(SIM_VALIDATION)
	{
	//	debug_testErrandRegen_weekend();
	}

	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
		cudaDeviceSynchronize();

	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "doWeekend_wholeDay_setup_sort");
	
	//each person gets 3 errands
	const int num_weekend_errands_total = NUM_WEEKEND_ERRANDS * number_people;

	//now sort the errand_people array into a large multi-hour location table
	cub::DeviceRadixSort::SortPairs(
		errand_sorting_tempStorage, errand_sorting_tempStorage_size,
		people_errands_doubleBuffer,errand_people_doubleBuffer,
		num_weekend_errands_total);

	if(SIM_PROFILING)
		profiler.endFunction(current_day, number_people);

	//people_hour_offsets[NUM_WEEKEND_ERRAND_HOURS] = num_weekend_errands_total;	//moved to size_global_array method

//	debug_dump_array_toTempFile("../weekend_hour_offsets.txt","hour offset",&errand_hourOffsets_weekend,NUM_WEEKEND_ERRAND_HOURS + 1);
	
	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"doWeekend_wholeDay_setup_locationSearch");


	//find how many people are going on errands during each hour
	thrust::counting_iterator<locId_t> count_it(0);
	thrust::device_vector<locId_t>::iterator errands_iterator;

	if(people_errands_doubleBuffer.selector == 0)
	{
		errands_iterator = people_errands_a.begin();
	}
	else
	{
		errands_iterator = people_errands_b.begin();
	}

	thrust::lower_bound(
		errands_iterator,
		errands_iterator + num_weekend_errands_total,
		count_it,
		count_it + (number_workplaces * NUM_WEEKEND_ERRAND_HOURS),
		errand_locationOffsets.begin());

	errand_locationOffsets[number_workplaces * NUM_WEEKEND_ERRAND_HOURS] = (NUM_WEEKEND_ERRANDS * number_people);

	if(SIM_PROFILING)
		profiler.endFunction(current_day,number_people);

	if(SIM_VALIDATION)
		debug_validateLocationArrays();

//	debug_dump_array_toTempFile("../weekend_loc_offsets.csv","loc offset",&errand_locationOffsets_multiHour, (NUM_WEEKEND_ERRAND_HOURS * number_workplaces));


	int blocks = cuda_makeWeekendContactsKernel_blocks;
	int threads = cuda_makeWeekendContactsKernel_threads;

	//get the amount of shared memory needed for each block
	size_t smem_size = sizeof(personId_t) + sizeof(kval_type_t);
	smem_size *= MAX_CONTACTS_WEEKEND;
	smem_size *= threads;

	//launch kernel
	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
		cudaDeviceSynchronize();

	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"doWeekend_wholeDay_kernel");

	kernel_weekend_sharedMem<<<blocks,threads,smem_size>>>(
		infected_count,	infected_indexes_ptr,
		people_households_ptr,household_offsets_ptr,
		errand_locationOffsets_ptr, errand_people_doubleBuffer.Current(),
		people_status_pandemic_ptr,people_status_seasonal_ptr,
		people_days_pandemic_ptr,people_days_seasonal_ptr,
		people_gens_pandemic_ptr,people_gens_seasonal_ptr,
#if SIM_VALIDATION == 1
		daily_contact_infectors_ptr,daily_contact_victims_ptr, 
		daily_contact_kval_types_ptr, daily_action_type_ptr,
		daily_contact_locations_ptr,
		debug_contactsToActions_float1_ptr,debug_contactsToActions_float2_ptr,
		debug_contactsToActions_float3_ptr,debug_contactsToActions_float4_ptr,
#endif
		current_day,rand_offset);

	if(SIM_PROFILING)
		profiler.endFunction(current_day,infected_count);

	if(TIMING_BATCH_MODE == 0)
	{
		int rand_counts_consumed = 6;
		rand_offset += (infected_count * rand_counts_consumed);
	}
	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
		cudaDeviceSynchronize();

	if(SIM_PROFILING)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::weekday_doInfectedSetup_wholeDay()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "weekday_doInfectedSetup_wholeDay");

//	kernel_doInfectedSetup_weekday_wholeDay<<<cuda_blocks, cuda_threads>>>(
//		infected_indexes_ptr,infected_count,
//		people_errands_ptr,	infected_errands_ptr);

//	const int rand_counts_used = infected_count / 4;
//	rand_offset += rand_counts_used;

	if(SIM_PROFILING)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::weekend_doInfectedSetup_wholeDay()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "weekend_doInfectedSetup");

//	kernel_doInfectedSetup_weekend<<<cuda_blocks,cuda_threads>>>(
//		infected_indexes_ptr,people_errands_ptr, infected_errands_ptr,
//		infected_count);

	if(SIM_PROFILING)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::weekend_assignErrands()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "weekend_assignErrands");

	host_randOffsetsStruct->errand_randOffset = rand_offset;
	cudaMemcpyToSymbolAsync(device_randOffsetsStruct,host_randOffsetsStruct,sizeof(simRandOffsetsStruct_t),0,cudaMemcpyHostToDevice);

	kernel_assignWeekendErrands<<<cuda_blocks,cuda_threads>>>(errand_people_doubleBuffer.Current() , people_errands_doubleBuffer.Current(), number_people, number_workplaces, rand_offset);

	int rand_counts_consumed = 2 * number_people;
	rand_offset += rand_counts_consumed;

	if(SIM_PROFILING)
		profiler.endFunction(current_day,number_people);
}

__device__ errandContactsProfile_t device_assignContactsDesired_weekday_wholeDay(unsigned int rand_val, age_t myAge)
{
	errandContactsProfile_t myProfile = WEEKDAY_ERRAND_PROFILE_AFTERSCHOOL;

	if(myAge == AGE_ADULT)
		myProfile = rand_val % (NUM_WEEKDAY_ERRAND_PROFILES - 1);

	return myProfile;
}
__device__ void device_copyInfectedErrandLocs_weekday(locId_t * errand_lookup_ptr, locId_t * infected_errand_ptr)
{
	int num_people = device_simSizeStruct->number_people;
	infected_errand_ptr[0] = errand_lookup_ptr[0];
	infected_errand_ptr[1] = errand_lookup_ptr[num_people];
}

__device__ void device_doAllWeekdayInfectedSetup(int myPos, personId_t * infected_indexes_arr, locId_t * errand_scheduling_array, locId_t * output_infected_locs)
{
	int myIdx = infected_indexes_arr[myPos];

	int output_offset = 2 * myPos;
	device_copyInfectedErrandLocs_weekday(errand_scheduling_array + myIdx, output_infected_locs + output_offset);
}
__global__ void kernel_doInfectedSetup_weekday_wholeDay(personId_t * infected_index_arr, int num_infected, locId_t * errand_scheduling_array, locId_t * infected_errands_array)
{
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected ; myPos += gridDim.x * blockDim.x)
	{
			device_doAllWeekdayInfectedSetup(myPos, infected_index_arr, errand_scheduling_array,infected_errands_array);
	}
}

#pragma region debug_printing_funcs

void debug_print(char * message)
{
	fprintf(fDebug, "%s\n", message);

	if(FLUSH_VALIDATION_IMMEDIATELY)
		fflush(fDebug);
} 



void debug_assert(bool condition, char * message)
{
	if(!condition)
	{
		fprintf(fDebug, "ERROR: ");
		debug_print(message);

		if(FLUSH_VALIDATION_IMMEDIATELY)
			fflush(fDebug);
	}
}

void debug_assert(char *message, int expected, int actual)
{
	if(expected != actual)
	{
		fprintf(fDebug, "ERROR: %s expected: %d actual: %d\n", message, expected, actual);

		if(FLUSH_VALIDATION_IMMEDIATELY)
			fflush(fDebug);
	}
}

void debug_assert(bool condition, char * message, int idx)
{
	if(!condition)
	{
		fprintf(fDebug, "ERROR: %s index: %d\n", message, idx);

		if(FLUSH_VALIDATION_IMMEDIATELY)
			fflush(fDebug);
	}
}
#pragma endregion debug_printing_funcs

#pragma region debug_lookup_funcs

char status_int_to_char(status_t s)
{
	if(s == STATUS_SUSCEPTIBLE)
		return 'S';
	else if (s == STATUS_RECOVERED)
		return 'R';
	else if(status_is_infected(s) && get_profile_from_status(s) < NUM_SHEDDING_PROFILES)
		return 'I';
	else
		return '?';
}

char * action_type_to_string(action_t action)
{
	switch(action)
	{
	case ACTION_INFECT_NONE:
		return "NONE";
	case ACTION_INFECT_PANDEMIC:
		return "PAND";
	case ACTION_INFECT_SEASONAL:
		return "SEAS";
	case ACTION_INFECT_BOTH:
		return "BOTH";
	default:
		return "????";
	}
}

int lookup_school_typecode_from_age_code(int age_code)
{
	switch(age_code)
	{
	case AGE_5:
		return BUSINESS_TYPE_PRESCHOOL;
	case AGE_9:
		return BUSINESS_TYPE_ELEMENTARYSCHOOL;
	case AGE_14:
		return BUSINESS_TYPE_MIDDLESCHOOL;
	case AGE_17:
		return BUSINESS_TYPE_HIGHSCHOOL;
	case AGE_22:
		return BUSINESS_TYPE_UNIVERSITY;
	default:
		throw std::runtime_error("invalid school typecode");
	}
}

char * profile_int_to_string(int p)
{
	switch(p)
	{
	case PROFILE_GAMMA1:
		return "GAMMA1";
	case PROFILE_GAMMA2:
		return "GAMMA2";
	case PROFILE_LOGNORM1:
		return "LOGNORM1";
	case PROFILE_LOGNORM2:
		return "LOGNORM2";
	case PROFILE_WEIB1:
		return "WEIB1";
	case PROFILE_WEIB2:
		return "WEIB2";
	default:
		return "ERR_BAD_PROFILE_NUM";
	}
}
#pragma endregion debug_lookup_funcs

//generates N unique numbers between 0 and max, exclusive
//assumes max is big enough that this won't be pathological
void n_unique_numbers(h_vec *array, int n, int max)
{
	for(int i = 0; i < n; i++)
	{
		do
		{
			(*array)[i] = rand() % max;
			for(int j =0; j < i; j++)
			{
				if((*array)[j] == (*array)[i])
				{
					(*array)[i] = -1;
					break;
				}
			}
		}while((*array)[i] == -1);
	}
}


int roundHalfUp_toInt(double d)
{
	return floor(d + 0.5);
}


__device__ kval_t device_makeContacts_weekday(
	personId_t myIdx, age_t myAge,
	locId_t * household_lookup, locOffset_t * household_offsets,// personId_t * household_people,
	locOffset_t * workplace_offsets, personId_t * workplace_people,
	personId_t * errand_loc_offsets, personId_t * errand_people,
	personId_t * output_victim_arr, kval_type_t * output_kval_arr,
#if SIM_VALIDATION == 1
	locId_t * output_contact_location,
#endif
	randOffset_t myRandOffset)

{

	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c[2];
		unsigned int i[8];
	} rand_union;
	//generate first set of random numbers

	threefry2x64_ctr_t tf_ctr_1 = {{myRandOffset, myRandOffset}};
	rand_union.c[0] = threefry2x64(tf_ctr_1, tf_k);

	kval_t household_kval_sum = 0;
	{
		locOffset_t loc_offset;
		int loc_count;

		//household: make three contacts
		device_lookupLocationData_singleHour(myIdx, household_lookup, household_offsets, &loc_offset, &loc_count);  //lookup location data for household
		
		household_kval_sum += device_selectRandomPersonFromLocation(
			myIdx, loc_offset, loc_count,rand_union.i[0], CONTACT_TYPE_HOME,
			NULL,	//workplace_people
			output_victim_arr + 0,
			output_kval_arr + 0);

		household_kval_sum += device_selectRandomPersonFromLocation(
			myIdx, loc_offset, loc_count,rand_union.i[1], CONTACT_TYPE_HOME,
			NULL,	//workplace_people
			output_victim_arr + 1,
			output_kval_arr + 1);

		household_kval_sum += device_selectRandomPersonFromLocation(
			myIdx, loc_offset, loc_count,rand_union.i[2], CONTACT_TYPE_HOME,
			NULL,	//workplace_people
			output_victim_arr + 2,
			output_kval_arr + 2);			
	}

#if SIM_VALIDATION == 1
	locId_t hh = household_lookup[myIdx];
	output_contact_location[0] = hh;
	output_contact_location[1] = hh;
	output_contact_location[2] = hh;
#endif

	//generate the second set of random numbers
	threefry2x64_ctr_t tf_ctr_2 = {{myRandOffset + 1, myRandOffset + 1}};
	rand_union.c[1] = threefry2x64(tf_ctr_2, tf_k);

	//now the number of contacts made will diverge, so we need to count it
	int contacts_made = 3;
	kval_t workplace_kval_sum = 0;
	{
		locId_t loc_wp = device_recalcWorkplace(myIdx,myAge);

		int loc_count;
		locOffset_t loc_offset;
		maxContacts_t contacts_desired;

		device_lookupWorkplaceData_singleHour(loc_wp,workplace_offsets, &loc_offset, &loc_count, &contacts_desired);

		//look up max_contacts into contacts_desired
		int local_contacts_made = contacts_made;			//this will let both loops interleave
		contacts_made += contacts_desired;
		
#if SIM_VALIDATION == 1
		for(int c = 0; c < contacts_desired; c++)
			output_contact_location[3 + c] = loc_wp;
#endif

		kval_type_t kval_type = CONTACT_TYPE_WORKPLACE;
		if(myAge != AGE_ADULT)
			kval_type = CONTACT_TYPE_SCHOOL;

		while(contacts_desired > 0) // && local_contacts_made < MAX_CONTACTS_WEEKDAY
		{
			workplace_kval_sum += device_selectRandomPersonFromLocation(
				myIdx,loc_offset, loc_count, rand_union.i[local_contacts_made], kval_type,
				workplace_people,
				output_victim_arr + local_contacts_made,
				output_kval_arr + local_contacts_made);

			contacts_desired--;
			local_contacts_made++;
		}
	}

	//do errands
	kval_t errand_kval_sum = 0;
	{
		//set kval for the errands
		kval_type_t kval_type = CONTACT_TYPE_ERRAND;
		if(myAge != AGE_ADULT)
			kval_type = CONTACT_TYPE_AFTERSCHOOL;

		//look up our errands and get a contact profile
		locId_t errand_dests[NUM_WEEKDAY_ERRANDS];
		errandContactsProfile_t errand_contacts_profile = device_recalc_weekdayErrandDests_assignProfile(myIdx,myAge,errand_dests, errand_dests+1);

		for(int hour = 0; hour < NUM_WEEKDAY_ERRAND_HOURS; hour++)
		{
			int contacts_desired = WEEKDAY_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[errand_contacts_profile][hour];

#if SIM_VALIDATION == 1
			for(int c = 0; c < contacts_desired; c++)
				output_contact_location[contacts_made + c] = errand_dests[hour];
#endif
			int loc_count;
			locOffset_t loc_offset;
			//fish out location offset, count
			device_lookupErrandLocationData(errand_dests[hour], errand_loc_offsets,	&loc_offset, &loc_count);

			//make contacts
			while(contacts_desired > 0 && contacts_made < MAX_CONTACTS_WEEKDAY)
			{
				errand_kval_sum += device_selectRandomPersonFromLocation(
					myIdx, loc_offset, loc_count, rand_union.i[contacts_made], kval_type,
					errand_people, 
					output_victim_arr + contacts_made,
					output_kval_arr + contacts_made);

				contacts_desired--;
				contacts_made++;
			}
		}

		//if person has made less than max contacts, fill the end with null contacts
		while(contacts_made < MAX_CONTACTS_WEEKDAY)
		{
			device_nullFillContact(	output_victim_arr + contacts_made,output_kval_arr + contacts_made);
			
#if SIM_VALIDATION == 1
			output_contact_location[contacts_made] = NULL_ERRAND;
#endif
			contacts_made++;
		}
	}
	kval_t kval_sum = household_kval_sum + workplace_kval_sum + errand_kval_sum;

	return kval_sum;
}

__device__ kval_t device_makeContacts_weekend(personId_t myIdx,
											  locId_t * household_lookup, locOffset_t * household_offsets, // personId_t * household_people,
											  locOffset_t * errand_loc_offsets, personId_t * errand_people,
											  personId_t * output_victim_ptr, kval_type_t * output_kval_ptr,
#if SIM_VALIDATION == 1
											  locId_t * output_contact_loc_ptr,
#endif
											  randOffset_t myRandOffset)
{

	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;
	//generate first set of random numbers

	threefry2x64_ctr_t tf_ctr_1 = {{myRandOffset, myRandOffset}};
	rand_union.c = threefry2x64(tf_ctr_1, tf_k);

	//household: make three contacts
	kval_t household_kval_sum = 0;
	{
		int loc_count;
		locOffset_t loc_offset;

		device_lookupLocationData_singleHour(myIdx, household_lookup, household_offsets, &loc_offset, &loc_count);  //lookup location data for household

		household_kval_sum += device_selectRandomPersonFromLocation(
			myIdx, loc_offset, loc_count,rand_union.i[0], CONTACT_TYPE_HOME,
			NULL, //household_people
			output_victim_ptr + 0,
			output_kval_ptr + 0);

		household_kval_sum += device_selectRandomPersonFromLocation(
			myIdx, loc_offset, loc_count,rand_union.i[1], CONTACT_TYPE_HOME,
			NULL, //household_people
			output_victim_ptr + 1,
			output_kval_ptr + 1);

		household_kval_sum += device_selectRandomPersonFromLocation(
			myIdx, loc_offset, loc_count,rand_union.i[2], CONTACT_TYPE_HOME,
			NULL, //household_people
			output_victim_ptr + 2,
			output_kval_ptr + 2);
	}

#if SIM_VALIDATION == 1
		locId_t hh = household_lookup[myIdx];
		output_contact_loc_ptr[0] = hh;
		output_contact_loc_ptr[1] = hh;
		output_contact_loc_ptr[2] = hh;
#endif

	//get an errand profile between 0 and 5
	errandContactsProfile_t myContactsProfile = rand_union.i[3] % NUM_WEEKEND_ERRAND_PROFILES;

	//we need two more random numbers for the errands
	threefry2x32_key_t tf_k_32 = {{ SEED_DEVICE[0], SEED_DEVICE[1]}};
	threefry2x32_ctr_t tf_ctr_32 = {{myRandOffset + 1, myRandOffset + 1}};		
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} rand_union_32;
	rand_union_32.c = threefry2x32(tf_ctr_32, tf_k_32);


	locId_t errand_dests[NUM_WEEKEND_ERRANDS];
	device_recalc_weekendErrandDests(myIdx,errand_dests);

	kval_t errand_kval_sum = 0;
	{
		int errand_slot = WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[myContactsProfile][0]; //the errand number the contact will be made in
		locId_t errand_loc = errand_dests[errand_slot];

#if SIM_VALIDATION == 1
			output_contact_loc_ptr[3] = errand_loc;
#endif

		locOffset_t loc_offset;
		int loc_count;
		device_lookupErrandLocationData(errand_loc,errand_loc_offsets,&loc_offset, &loc_count);

		errand_kval_sum += device_selectRandomPersonFromLocation(			//select a random person at the location
			myIdx, loc_offset, loc_count, rand_union_32.i[0], CONTACT_TYPE_ERRAND,
			errand_people,
			output_victim_ptr + 3,
			output_kval_ptr + 3);
	}
	{
		//do it again for the second errand contact
		int errand_slot = WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[myContactsProfile][1];
		locId_t errand_loc = errand_dests[errand_slot];

#if SIM_VALIDATION == 1
			output_contact_loc_ptr[4] = errand_loc;
#endif

		locOffset_t loc_offset;
		int loc_count;
		device_lookupErrandLocationData(errand_loc,errand_loc_offsets,&loc_offset, &loc_count);

		errand_kval_sum += device_selectRandomPersonFromLocation(			//select a random person at the location
			myIdx, loc_offset, loc_count, rand_union_32.i[1], CONTACT_TYPE_ERRAND,
			errand_people,
			output_victim_ptr + 4,
			output_kval_ptr + 4);
	}

	kval_t kval_sum = household_kval_sum + errand_kval_sum;
	return kval_sum;
}


/// <summary> given an index, look up the location and fetch the offset/count data from the memory array </summary>
/// <param name="myIdx">Input: Index of the infector to look up</param>
/// <param name="lookup_arr">Input: Pointer to an array containing all infector locations</param>
/// <param name="loc_offset_arr">Input: Pointer to an array containing location offsets</param>
/// <param name="loc_offset">Output value: offset to first person in infector's location</param>
/// <param name="loc_count">Output value: number of people at infector's location</param>
__device__ void device_lookupLocationData_singleHour(personId_t myIdx, locId_t * lookup_arr, locOffset_t * loc_offset_arr, locOffset_t * loc_offset, int * loc_count)
{
	locId_t myLoc = lookup_arr[myIdx];

	locOffset_t loc_o = loc_offset_arr[myLoc];
	locOffset_t next_loc_o = loc_offset_arr[myLoc + 1];

	//NOTE: these arrays have the final number_locs+1 value set, so we do not need to do the trick for the last location
	*loc_offset = loc_o;
	*loc_count = next_loc_o - loc_o;
}

/// <summary> given an index, look up the location and fetch the offset/count/max_contacts values from the memory array </summary>
/// <param name="myIdx">Input: Index of the infector to look up</param>
/// <param name="lookup_arr">Input: Pointer to an array containing all infector locations</param>
/// <param name="loc_offset_arr">Input: Pointer to an array containing a location offsets</param>
/// <param name="loc_max_contacts_arr">Input: pointer to an array containing max_contact values</param>
/// <param name="loc_offset">Output: offset to first person in infector's location</param>
/// <param name="loc_count">Output: number of people at infector's location</param>
/// <param name="loc_max_contacts">Output: max_contacts for infector's location</param>
__device__ void device_lookupWorkplaceData_singleHour(
	locId_t myLoc, locOffset_t * loc_offset_arr,
	locOffset_t * loc_offset, int * loc_count, maxContacts_t * loc_max_contacts)
{
	locOffset_t loc_o = loc_offset_arr[myLoc];
	locOffset_t next_loc_o = loc_offset_arr[myLoc + 1];

	*loc_offset = loc_o;
	*loc_count = next_loc_o - loc_o;
	*loc_max_contacts = device_getWorkplaceMaxContacts(myLoc);
}

/// <summary>Gets location data and number of contacts desired from a multi-hour errand array</summary>
/// <param name="myPos">Input: Which of the N infected individuals we are working with, 0 <= myPos <= infected_count</param>
/// <param name="hour">Input: Which hour we are looking up information for, 0 < hour <= <paramref name="number_hours" /></param>
/// <param name="infected_loc_arr">Input: Pointer to an array containing the errand destinations of infected in packed arrangement</param>
/// <param name="loc_offset_arr>Input: pointer to an array containing location offsets in collated arrangement</param>
/// <param name="number_locations>Input: the number of locations (excluding households) in the simulation</param>
/// <param name="number_people">Input: number of people present (must be same all hours)</param>
/// <param name="contacts_desired_lookup">Input: pointer to an array containing the number of contacts desired for each hour, in packed form</param>
/// <param name="number_hours">Input: The number of hours stored in the multi-hour array, probably NUM_WEEKEND_ERRAND_HOURS or NUM_WEEKDAY_ERRAND_HOURS</param>
/// <param name="output_location_offset">Output: the offset from the start of the array to the first person at this location for this hour</param>
/// <param name="output_location_count">Output: the number of people at this location for this hour</param>
/// <param name="output_contacts_desired">Output: the number of contacts we will make this hour</param>
__device__ void device_lookupErrandLocationData(locId_t myLoc, locOffset_t * loc_offset_arr, locOffset_t * output_loc_offset, int * output_loc_count)
{
	//location offsets are stored in collated format, eg for locations 1 2 3
	// 1 2 3 1 2 3

	locOffset_t loc_o = loc_offset_arr[myLoc];
	locOffset_t next_loc_o = loc_offset_arr[myLoc + 1];

	*output_loc_count = next_loc_o - loc_o;	//calculate the number of people at this location
	*output_loc_offset = loc_o;
}

__device__ personId_t device_getVictimAtIndex(personId_t index_to_fetch, personId_t * location_people, kval_type_t contact_type)
{
	//required: loc_count > 1 (checked in device_selectRandomPersonFromLocation)

	//special case: the value at household_people[i] is always i
	if(contact_type == CONTACT_TYPE_HOME)
		return index_to_fetch;

	return location_people[index_to_fetch];
}

__device__ kval_t device_selectRandomPersonFromLocation(
	personId_t infector_idx, 
	personId_t loc_offset, int loc_count, 
	unsigned int rand_val, 
	kval_type_t desired_contact_type, 
	personId_t * location_people_arr,
	personId_t * output_victim_idx_ptr, kval_type_t * output_kval_ptr)
{
	//start with null data
	//int victim_idx = NULL_PERSON_INDEX;
	personId_t victim_idx = loc_count;
	kval_type_t contact_type = CONTACT_TYPE_NONE;

	//if there is only one person, keep the null data, else select one other person who is not our infector
	if(loc_count > 1)
	{
		int victim_offset = rand_val % loc_count;	//select a random person between 0 and loc_count

		//victim_idx = location_people_arr[loc_offset + victim_offset];	//get the index
		victim_idx = device_getVictimAtIndex(loc_offset + victim_offset, location_people_arr, desired_contact_type);

		//if we have selected the infector, we need to get a different person
		if(victim_idx == infector_idx)
		{
			//increase the offset by 1 and wrap around to start if necessary
			victim_offset = (victim_offset + 1) % loc_count;

			//victim_idx = location_people_arr[loc_offset + victim_offset];
			victim_idx = device_getVictimAtIndex(loc_offset + victim_offset, location_people_arr, desired_contact_type);
		}

		contact_type = desired_contact_type;
	}

	//write data into output memory locations
//	(*output_infector_idx_ptr) = infector_idx;
	(*output_victim_idx_ptr) = victim_idx;
	(*output_kval_ptr) = contact_type;

	//increment the kval sum by the kval of this contact type

	kval_t contact_type_kval = KVAL_LOOKUP_DEVICE[contact_type];
	return contact_type_kval;
}

//write a null contact to the memory locations
__device__ void device_nullFillContact(personId_t * output_victim_idx, kval_type_t * output_kval)
{
	(*output_victim_idx) = NULL_PERSON_INDEX;
	(*output_kval) = CONTACT_TYPE_NONE;
}


const char * lookup_contact_type(int contact_type)
{
	switch(contact_type)
	{
	case CONTACT_TYPE_NONE:
		return "CONTACT_TYPE_NONE";
	case CONTACT_TYPE_WORKPLACE:
		return "CONTACT_TYPE_WORKPLACE";
	case CONTACT_TYPE_SCHOOL:
		return "CONTACT_TYPE_SCHOOL";
	case CONTACT_TYPE_ERRAND:
		return "CONTACT_TYPE_ERRAND";
	case CONTACT_TYPE_AFTERSCHOOL:
		return "CONTACT_TYPE_AFTERSCHOOL";
	case CONTACT_TYPE_HOME:
		return "CONTACT_TYPE_HOME";
	default:
		return "BAD_CONTACT_TYPE_NUM";
	}
}

inline const char * lookup_workplace_type(int workplace_type)
{
	switch(workplace_type)
	{
	case 0:
		return "home";
	case 1:
		return "factory";
	case 2:
		return "office";
	case 3:
		return "preschool";
	case 4:
		return "elementary school";
	case 5:
		return "middle school";
	case 6:
		return "highschool";
	case 7:
		return "university";
	case 8:
		return "afterschool center";
	case 9:
		return "grocery store";
	case 10:
		return "other store";
	case 11:
		return "restaurant";
	case 12:
		return "entertainment";
	case 13:
		return "church";
	default:
		return "INVALID WORKPLACE TYPE";
	}
}

const char * lookup_age_type(age_t age_type)
{
	switch(age_type)
	{
	case AGE_5:
		return "AGE_5";
	case AGE_9:
		return "AGE_9";
	case AGE_14:
		return "AGE_14";
	case AGE_17:
		return "AGE_17";
	case AGE_22:
		return "AGE_22";
	case AGE_ADULT:
		return "AGE_ADULT";
	default:
		return "INVALID AGE CODE";
	}
}

__global__ void kernel_assignWeekendErrands(personId_t * people_indexes_arr, locId_t * errand_scheduling_array, int num_people, int num_locations, randOffset_t rand_offset)
{
	const int RAND_COUNTS_CONSUMED = 2;	//one for hours, one for destinations

	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_people; myPos += gridDim.x * blockDim.x)
	{
		int offset = myPos * NUM_WEEKEND_ERRANDS;
		randOffset_t myRandOffset = rand_offset + (myPos * RAND_COUNTS_CONSUMED);
		
		device_copyPeopleIndexes_weekend_wholeDay(people_indexes_arr + offset, myPos);
		device_generateWeekendErrands(errand_scheduling_array + offset, myRandOffset);
	}
}
__global__ void kernel_doInfectedSetup_weekend(personId_t * infected_indexes_array, locId_t * errand_scheduling_array,  locId_t * infected_errands_array, int num_infected)
{
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos <num_infected; myPos += gridDim.x * blockDim.x)
	{

		int myIdx = infected_indexes_array[myPos];
		int input_offset = NUM_WEEKEND_ERRANDS * myIdx;
		int output_offset = NUM_WEEKEND_ERRANDS * myPos;

		device_copyInfectedErrandLocs_weekend(
			errand_scheduling_array + input_offset,
			infected_errands_array + output_offset);
	}
}

__device__ void device_copyInfectedErrandLocs_weekend(locId_t * errand_array_ptr, locId_t * infected_errand_ptr)
{
	infected_errand_ptr[0] = errand_array_ptr[0];
	infected_errand_ptr[1] = errand_array_ptr[1];
	infected_errand_ptr[2] = errand_array_ptr[2];
}

__global__ void kernel_countInfectedStatusAndRecover(
	status_t * pandemic_status_array, status_t * seasonal_status_array, 
	day_t * pandemic_days_array, day_t * seasonal_days_array,
	int num_people, day_t current_day,
	int * output_pandemic_counts, int * output_seasonal_counts)
{
	int tid = threadIdx.x;

	__shared__ int pandemic_reduction_array[COUNTING_GRID_THREADS][8];
	__shared__ int seasonal_reduction_array[COUNTING_GRID_THREADS][8];

	//zero out the counters
	pandemic_reduction_array[tid][0] = 0;
	pandemic_reduction_array[tid][1] = 0;
	pandemic_reduction_array[tid][2] = 0;
	pandemic_reduction_array[tid][3] = 0;
	pandemic_reduction_array[tid][4] = 0;
	pandemic_reduction_array[tid][5] = 0;
	pandemic_reduction_array[tid][6] = 0;
	pandemic_reduction_array[tid][7] = 0;

	seasonal_reduction_array[tid][0] = 0;
	seasonal_reduction_array[tid][1] = 0;
	seasonal_reduction_array[tid][2] = 0;
	seasonal_reduction_array[tid][3] = 0;
	seasonal_reduction_array[tid][4] = 0;
	seasonal_reduction_array[tid][5] = 0;
	seasonal_reduction_array[tid][6] = 0;
	seasonal_reduction_array[tid][7] = 0;
	
	int day_to_recover = current_day - CULMINATION_PERIOD;

	//count all statuses
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_people; myPos += gridDim.x * blockDim.x)
	{
		//get pandemic status
		int status_pandemic = pandemic_status_array[myPos];
		if(day_to_recover >= 0 && status_is_infected(status_pandemic))//if we're culminating a valid day and the person is infected
		{
			day_t day_p = pandemic_days_array[myPos];		//get their day of infection
			if(day_p <= day_to_recover)						//if they've reached culmination, set their status to recovered
			{
				status_pandemic = STATUS_RECOVERED;
				pandemic_status_array[myPos] = STATUS_RECOVERED;
			}
		}

		int status_seasonal = seasonal_status_array[myPos];
		if(day_to_recover >= 0 && status_is_infected(status_seasonal))
		{
			day_t day_s = seasonal_days_array[myPos];
			if(day_s <= day_to_recover)
			{
				status_seasonal = STATUS_RECOVERED;
				seasonal_status_array[myPos] = STATUS_RECOVERED;
			}
		}

		pandemic_reduction_array[tid][status_pandemic]++;
		seasonal_reduction_array[tid][status_seasonal]++;
	}

	//do reduction
	__syncthreads();   //wait for all threads to finish, or reduction will hit a race condition
	for(int offset = blockDim.x / 2; offset > 0;  offset /= 2)
	{
		if(tid < offset)
		{
			pandemic_reduction_array[tid][0] += pandemic_reduction_array[tid+offset][0];
			pandemic_reduction_array[tid][1] += pandemic_reduction_array[tid+offset][1];
			pandemic_reduction_array[tid][2] += pandemic_reduction_array[tid+offset][2];
			pandemic_reduction_array[tid][3] += pandemic_reduction_array[tid+offset][3];
			pandemic_reduction_array[tid][4] += pandemic_reduction_array[tid+offset][4];
			pandemic_reduction_array[tid][5] += pandemic_reduction_array[tid+offset][5];
			pandemic_reduction_array[tid][6] += pandemic_reduction_array[tid+offset][6];
			pandemic_reduction_array[tid][7] += pandemic_reduction_array[tid+offset][7];

			seasonal_reduction_array[tid][0] += seasonal_reduction_array[tid+offset][0];
			seasonal_reduction_array[tid][1] += seasonal_reduction_array[tid+offset][1];
			seasonal_reduction_array[tid][2] += seasonal_reduction_array[tid+offset][2];
			seasonal_reduction_array[tid][3] += seasonal_reduction_array[tid+offset][3];
			seasonal_reduction_array[tid][4] += seasonal_reduction_array[tid+offset][4];
			seasonal_reduction_array[tid][5] += seasonal_reduction_array[tid+offset][5];
			seasonal_reduction_array[tid][6] += seasonal_reduction_array[tid+offset][6];
			seasonal_reduction_array[tid][7] += seasonal_reduction_array[tid+offset][7];
		}
		__syncthreads();
	}

	//threads 0-7 store sums, which are in the spot for tid 0
	if(tid < 8)
	{
		atomicAdd(output_pandemic_counts + tid, pandemic_reduction_array[0][tid]);
		atomicAdd(output_seasonal_counts + tid, seasonal_reduction_array[0][tid]);
	}
}

struct isInfectedPred
{
	__device__ bool operator() (thrust::tuple<status_t, status_t> status_tuple)
	{
		status_t status_seasonal = thrust::get<0>(status_tuple);
		status_t status_pandemic = thrust::get<1>(status_tuple);

		return person_is_infected(status_pandemic, status_seasonal);
	}
};

void PandemicSim::daily_buildInfectedArray_global()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "daily_buildInfectedArray_global");

	thrust::counting_iterator<int> count_it(0);
	IntIterator infected_indexes_end = thrust::copy_if(
		count_it, count_it + number_people,
		thrust::make_zip_iterator(thrust::make_tuple(
			people_status_pandemic.begin(), people_status_seasonal.begin())),
		infected_indexes.begin(),
		isInfectedPred());

	infected_count = infected_indexes_end - infected_indexes.begin();

	if(SIM_PROFILING)
		profiler.endFunction(current_day, infected_count);
}

void PandemicSim::final_countReproduction()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"final_countReproduction");

	if(SIM_PROFILING)
		profiler.beginFunction(-1,"final_countReproduction_sort");

	thrust::sort(people_gens_pandemic.begin(), people_gens_pandemic.end());
	thrust::sort(people_gens_seasonal.begin(), people_gens_seasonal.end());
		
	if(SIM_PROFILING)
		profiler.endFunction(-1,number_people);
	
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"final_countReproduction_genSearch");

	thrust::counting_iterator<int> count_it(0);

	vec_t pandemic_gen_counts(MAX_DAYS + 1);
	pandemic_gen_counts[MAX_DAYS] = number_people;
	thrust::lower_bound(
		people_gens_pandemic.begin(), people_gens_pandemic.end(),
		count_it, count_it + MAX_DAYS,
		pandemic_gen_counts.begin());

	vec_t seasonal_gen_counts(MAX_DAYS + 1);
	seasonal_gen_counts[MAX_DAYS] = number_people;
	thrust::lower_bound(
		people_gens_seasonal.begin(), people_gens_seasonal.end(),
		count_it, count_it + MAX_DAYS,
		seasonal_gen_counts.begin());

	if(SIM_PROFILING)
		profiler.endFunction(-1,number_people);

	if(SIM_PROFILING)
		profiler.beginFunction(-1,"final_countReproduction_copyToHost");

	//copy to host
	h_vec h_pandemic_gens = pandemic_gen_counts;
	h_vec h_seasonal_gens = seasonal_gen_counts;
	
	if(SIM_PROFILING){
		profiler.endFunction(-1,MAX_DAYS);
		profiler.beginFunction(-1,"final_countReproduction_hostCode");
	}
	FILE * fReproduction;
	if(OUTPUT_FILES_IN_PARENTDIR)
		fReproduction = fopen("../output_rn.csv","w");
	else
		fReproduction = fopen("output_rn.csv","w");

	fprintf(fReproduction, "gen,gen_size_p,rn_p,gen_size_s,rn_s\n");
	//calculate reproduction numbers
	for(int gen = 0; gen < MAX_DAYS-1; gen++)
	{
		int gen_size_p = h_pandemic_gens[gen+1] - h_pandemic_gens[gen];
		int next_gen_size_p = h_pandemic_gens[gen+2] - h_pandemic_gens[gen+1];
		float rn_p = (float) next_gen_size_p / gen_size_p;

		int gen_size_s = h_seasonal_gens[gen+1] - h_seasonal_gens[gen];
		int next_gen_size_s = h_seasonal_gens[gen+2] - h_seasonal_gens[gen+1];
		float rn_s = (float) next_gen_size_s / gen_size_s;

		fprintf(fReproduction, "%d,%d,%f,%d,%f\n",
			gen, gen_size_p, rn_p, gen_size_s, rn_s);
	}
	fclose(fReproduction);


	if(SIM_PROFILING){
		profiler.endFunction(-1,MAX_DAYS);
		profiler.endFunction(-1,number_people);
	}
}

__device__ void device_checkActionAndWrite(bool infects_pandemic, bool infects_seasonal, personId_t victim, status_t * pandemic_status_arr, status_t * seasonal_status_arr, int * dest_ptr)
{
	if(infects_pandemic)
	{
		int victim_status_p = pandemic_status_arr[victim];
		if(victim_status_p != STATUS_SUSCEPTIBLE)
			infects_pandemic = false;
	}
	if(infects_seasonal)
	{
		int victim_status_s = seasonal_status_arr[victim];
		if(victim_status_s != STATUS_SUSCEPTIBLE)
			infects_seasonal = false;
	}

	if(infects_pandemic && infects_seasonal)
		*dest_ptr = ACTION_INFECT_BOTH;
	else if(infects_pandemic)
		*dest_ptr = ACTION_INFECT_PANDEMIC;
	else if(infects_seasonal)
		*dest_ptr = ACTION_INFECT_SEASONAL;
}

__device__ float device_calculateInfectionProbability(int profile, int day_of_infection, int strain, kval_t kval_sum)
{
	if(kval_sum == 0)
		kval_sum = 1;

	//alpha: fraction of infectiousness that will occur on this day of infection for this profile
	float alpha = VIRAL_SHEDDING_PROFILES_DEVICE[profile][day_of_infection];

	//strain_adjustment_factor: equals rn_base / ((1.0-asymp) * pct_symptomatic)
	float strain_adjustment_factor = INFECTIOUSNESS_FACTOR_DEVICE[strain];

	//the average number of infections this person will generate today
	float average_infections_today = alpha * strain_adjustment_factor;

	//the average chance for infection for a contact with kappa = 1.0
	float normalized_infection_prob = (float) average_infections_today / kval_sum;

	return normalized_infection_prob;
}


__global__ void kernel_householdTypeAssignment(householdType_t * hh_type_array, int num_households, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= num_households / 4; myGridPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = rand_offset + myGridPos;
		threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr,tf_k);

		int myPos = myGridPos * 4;

		if(myPos < num_households)
			hh_type_array[myPos+0] = device_setup_fishHouseholdType(rand_union.i[0]);

		if(myPos + 1 < num_households)
			hh_type_array[myPos+1] = device_setup_fishHouseholdType(rand_union.i[1]);

		if(myPos + 2 < num_households)
			hh_type_array[myPos+2] = device_setup_fishHouseholdType(rand_union.i[2]);

		if(myPos + 3 < num_households)
			hh_type_array[myPos+3] = device_setup_fishHouseholdType(rand_union.i[3]);
	}
}

__device__ householdType_t device_setup_fishHouseholdType(unsigned int rand_val)
{
	float y = (float) rand_val / UNSIGNED_MAX;

	int row = 0;
	while(y > HOUSEHOLD_TYPE_CDF_DEVICE[row] && row < HH_TABLE_ROWS - 1)
		row++;

	householdType_t ret = row;
	return ret;
}

__device__ locId_t device_setup_fishWorkplace(unsigned int rand_val)
{
	float y = (float) rand_val / UNSIGNED_MAX;

	int row = 0;
	while(WORKPLACE_TYPE_ASSIGNMENT_PDF_DEVICE[row] < y && row < NUM_BUSINESS_TYPES - 1)
	{
		y -= WORKPLACE_TYPE_ASSIGNMENT_PDF_DEVICE[row];
		row++;
	}

	//of this workplace type, which number is this?
	float frac = y / WORKPLACE_TYPE_ASSIGNMENT_PDF_DEVICE[row];
	int type_count = WORKPLACE_TYPE_COUNT_DEVICE[row];
	int business_num = frac * type_count;  //truncate to int

	if(business_num >= type_count)
		business_num = type_count - 1;

	//how many other workplaces have we gone past?
	int type_offset = WORKPLACE_TYPE_OFFSET_DEVICE[row];

	locId_t ret = business_num + type_offset;
	return ret;
}

__device__ void device_setup_fishSchoolAndAge(unsigned int rand_val, age_t * output_age_ptr, locId_t * output_school_ptr)
{
	float y = (float) rand_val / UNSIGNED_MAX;

	//fish out age group and resulting school type from CDF
	int row = 0;
	while(row < CHILD_DATA_ROWS - 1 && y > CHILD_AGE_CDF_DEVICE[row])
		row++;

	//int wp_type = CHILD_AGE_SCHOOLTYPE_LOOKUP_DEVICE[row];
	int wp_type = row + BUSINESS_TYPE_PRESCHOOL;

	//of this school type, which one will this kid be assigned to?
	float frac;
	if(row == 0)
		frac = y / (CHILD_AGE_CDF_DEVICE[row]);
	else
	{
		//we need to back out a PDF from the CDF
		float pdf_here = CHILD_AGE_CDF_DEVICE[row] - CHILD_AGE_CDF_DEVICE[row - 1];
		float y_here = y - CHILD_AGE_CDF_DEVICE[row - 1];

		frac =  y_here / pdf_here;
	}

	int type_count = WORKPLACE_TYPE_COUNT_DEVICE[wp_type];
	int business_num = frac * type_count;

	if(business_num >= type_count)
		business_num = type_count - 1;

	//how many other workplaces have we gone past?
	int type_offset = WORKPLACE_TYPE_OFFSET_DEVICE[wp_type];
	*output_school_ptr = business_num + type_offset;

	age_t myAge = (age_t) row;
	*output_age_ptr = myAge;
}

__device__ void device_setup_assignWorkplaceOrSchool(unsigned int rand_val, age_t * age_ptr,locId_t * workplace_ptr)
{
	age_t myAge = *age_ptr;

	if(myAge == AGE_ADULT)
	{
		workplace_ptr[0] = device_setup_fishWorkplace(rand_val);
	}
	else
	{
		device_setup_fishSchoolAndAge(rand_val,age_ptr,workplace_ptr);
	}
}


__global__ void kernel_assignWorkplaces(age_t * people_ages_arr, locId_t * people_workplaces_arr, int number_people, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= number_people/4; myGridPos += gridDim.x * blockDim.x)
	{
		//get random numbers
		randOffset_t myRandOffset = rand_offset + myGridPos;
		threefry2x64_ctr_t tf_ctr_1 = {{myRandOffset, myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr_1, tf_k);

		int myPos = myGridPos * 4;

		if(myPos < number_people)
			device_setup_assignWorkplaceOrSchool(rand_union.i[0],people_ages_arr + myPos + 0, people_workplaces_arr + myPos + 0);
		if(myPos + 1 < number_people)
			device_setup_assignWorkplaceOrSchool(rand_union.i[1],people_ages_arr + myPos + 1, people_workplaces_arr + myPos + 1);
		if(myPos + 2 < number_people)
			device_setup_assignWorkplaceOrSchool(rand_union.i[2],people_ages_arr + myPos + 2, people_workplaces_arr + myPos + 2);
		if(myPos + 3 < number_people)
			device_setup_assignWorkplaceOrSchool(rand_union.i[3],people_ages_arr + myPos + 3, people_workplaces_arr + myPos + 3);
	}
}

__global__ void kernel_generateHouseholds(
	householdType_t * hh_type_array, 
	int * adult_exscan_arr, int * child_exscan_arr, int num_households, 
	locOffset_t * household_offset_arr,
	age_t * people_age_arr, locId_t * people_households_arr)
{

	for(int hh = blockIdx.x * blockDim.x + threadIdx.x;  hh < num_households ; hh += gridDim.x * blockDim.x)
	{
		int adults_offset = adult_exscan_arr[hh];
		int children_offset = child_exscan_arr[hh];

		int hh_type = hh_type_array[hh];
		int adults_count = HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[hh_type];
		int children_count = HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[hh_type];

		int hh_offset = adults_offset + children_offset;
		household_offset_arr[hh] = hh_offset;

		for(int people_generated = 0; people_generated < adults_count; people_generated++)
		{
			int person_id = hh_offset + people_generated;
			people_households_arr[person_id] = hh;				//store the household number

			people_age_arr[person_id] = AGE_ADULT;					//mark as an adult
		}

		//increment the base ID number by the adults we just added
		hh_offset += adults_count;

		for(int people_generated = 0; people_generated < children_count; people_generated++)
		{
			int person_id = hh_offset + people_generated;
			people_households_arr[person_id] = hh;		//store the household number

			people_age_arr[person_id] = AGE_5;
		}
	}
}


struct hh_adult_count_functor : public thrust::unary_function<householdType_t,int>
{
	__device__ int operator () (householdType_t hh_type) const
	{
		return HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[hh_type];
	}
};

struct hh_child_count_functor : public thrust::unary_function<householdType_t,int>
{
	__device__ int operator () (householdType_t hh_type) const
	{
		return HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[hh_type];
	}
};


//Sets up people's households and workplaces according to the probability functions
void PandemicSim::setup_generateHouseholds()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_generateHouseholds");

//	thrust::fill_n(people_ages.begin(), number_people, (age_t) 0);


	thrust::device_vector<householdType_t> hh_types_array(number_households+1);
	householdType_t * hh_types_array_ptr = thrust::raw_pointer_cast(hh_types_array.data());

	//finish copydown of __constant__ sim data
	cudaDeviceSynchronize();

	//assign household types
	kernel_householdTypeAssignment<<<cuda_householdTypeAssignmentKernel_blocks,cuda_householdTypeAssignmentKernel_threads>>>(hh_types_array_ptr, number_households,rand_offset);
	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
		cudaDeviceSynchronize();

	if(TIMING_BATCH_MODE == 0)
	{
		int rand_counts_consumed_1 = number_households / 4;
		rand_offset += rand_counts_consumed_1;
	}

	thrust::device_vector<int> adult_count_exclScan(number_households+1);
	thrust::device_vector<int> child_count_exclScan(number_households+1);

	//these count_functors convert household types into the number of children/adults in that type
	//use a transform-functor to convert the HH types and take an exclusive_scan of each
	//this will let us build the adult_index and child_index arrays
	thrust::exclusive_scan(
		thrust::make_transform_iterator(hh_types_array.begin(), hh_adult_count_functor()),
		thrust::make_transform_iterator(hh_types_array.begin() + number_households + 1, hh_adult_count_functor()),
		adult_count_exclScan.begin());
	thrust::exclusive_scan(
		thrust::make_transform_iterator(hh_types_array.begin(), hh_child_count_functor()),
		thrust::make_transform_iterator(hh_types_array.begin() + number_households + 1, hh_child_count_functor()),
		child_count_exclScan.begin());
	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
		cudaDeviceSynchronize();

	//the exclusive_scan of number_households+1 holds the total number of adult and children in the sim
	//(go one past the end to find the totals)
	number_adults = adult_count_exclScan[number_households];
	number_children = child_count_exclScan[number_households];

	if(SIM_VALIDATION)
	{
//		thrust::constant_iterator<age_t> const_it(0);
//		bool ages_set_to_adult_val = thrust::equal(people_ages.begin(), people_ages.begin() + number_people,const_it);

		thrust::fill_n(people_households.begin(), number_people, NULL_LOC_ID);
//		thrust::fill_n(people_workplaces.begin(), number_people, NULL_LOC_ID);

		thrust::fill_n(household_offsets.begin(), number_households, -1);
	}

	int * adult_exscan_ptr = thrust::raw_pointer_cast(adult_count_exclScan.data());
	int * child_exscan_ptr = thrust::raw_pointer_cast(child_count_exclScan.data());

	int blocks = cuda_peopleGenerationKernel_blocks;
	int threads = cuda_peopleGenerationKernel_threads;

	//and then do the rest of the setup
	kernel_generateHouseholds<<<blocks,threads>>>(
		hh_types_array_ptr, adult_exscan_ptr, child_exscan_ptr, number_households,
		household_offsets_ptr,
		people_ages_ptr, people_households_ptr);
	household_offsets[number_households] = number_people;  //put the last household_offset in position

	if(DEBUG_SYNCHRONIZE_NEAR_KERNELS)
		cudaDeviceSynchronize();

	if(SIM_PROFILING)
	{
		profiler.endFunction(-1,number_people);
	}
}

void PandemicSim::setup_fetchVectorPtrs()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_fetchVectorPtrs");

	people_status_pandemic_ptr = thrust::raw_pointer_cast(people_status_pandemic.data());
	people_status_seasonal_ptr = thrust::raw_pointer_cast(people_status_seasonal.data());

	people_days_pandemic_ptr = thrust::raw_pointer_cast(people_days_pandemic.data());
	people_days_seasonal_ptr = thrust::raw_pointer_cast(people_days_seasonal.data());
	people_gens_pandemic_ptr = thrust::raw_pointer_cast(people_gens_pandemic.data());
	people_gens_seasonal_ptr = thrust::raw_pointer_cast(people_gens_seasonal.data());

	people_households_ptr = thrust::raw_pointer_cast(people_households.data());
//	people_workplaces_ptr = thrust::raw_pointer_cast(people_workplaces.data());
	people_ages_ptr = thrust::raw_pointer_cast(people_ages.data());

	infected_indexes_ptr = thrust::raw_pointer_cast(infected_indexes.data());

	workplace_offsets_ptr = thrust::raw_pointer_cast(workplace_offsets.data());
	workplace_people_ptr = thrust::raw_pointer_cast(workplace_people.data());
//	workplace_max_contacts_ptr = thrust::raw_pointer_cast(workplace_max_contacts.data());

	household_offsets_ptr = thrust::raw_pointer_cast(household_offsets.data());

	//errand_people_table_ptr = thrust::raw_pointer_cast(errand_people_table_a.data());
	
	//people_errands_ptr = thrust::raw_pointer_cast(people_errands_a.data());

//	infected_errands_ptr = thrust::raw_pointer_cast(infected_errands.data());

	errand_locationOffsets_ptr = thrust::raw_pointer_cast(errand_locationOffsets.data());

	status_counts_dev_ptr = thrust::raw_pointer_cast(status_counts.data());


	simArrayPtrStruct_t host_arrayPtrStruct[1];
	host_arrayPtrStruct->people_status_pandemic = thrust::raw_pointer_cast(people_status_pandemic.data());
	host_arrayPtrStruct->people_status_seasonal = thrust::raw_pointer_cast(people_status_seasonal.data());
	host_arrayPtrStruct->people_days_pandemic = thrust::raw_pointer_cast(people_days_pandemic.data());
	host_arrayPtrStruct->people_days_seasonal = thrust::raw_pointer_cast(people_days_seasonal.data());
	host_arrayPtrStruct->people_gens_pandemic = thrust::raw_pointer_cast(people_gens_pandemic.data());
	host_arrayPtrStruct->people_gens_seasonal = thrust::raw_pointer_cast(people_gens_seasonal.data());

	host_arrayPtrStruct->people_ages = thrust::raw_pointer_cast(people_ages.data());
	host_arrayPtrStruct->people_households = thrust::raw_pointer_cast(people_households.data());
//	host_arrayPtrStruct->people_workplaces = thrust::raw_pointer_cast(people_workplaces.data());
	host_arrayPtrStruct->people_errands = thrust::raw_pointer_cast(people_errands_a.data());

	host_arrayPtrStruct->household_locOffsets = thrust::raw_pointer_cast(household_offsets.data());
	host_arrayPtrStruct->workplace_locOffsets = thrust::raw_pointer_cast(workplace_offsets.data());
	host_arrayPtrStruct->errand_locOffsets = thrust::raw_pointer_cast(errand_locationOffsets.data());

//	host_arrayPtrStruct->workplace_people = thrust::raw_pointer_cast(workplace_people.data());
//	host_arrayPtrStruct->errand_people = thrust::raw_pointer_cast(errand_people_table_a.data());

//	host_arrayPtrStruct->workplace_max_contacts = thrust::raw_pointer_cast(workplace_max_contacts.data());

	cudaMemcpyToSymbolAsync(device_arrayPtrStruct,host_arrayPtrStruct,sizeof(simArrayPtrStruct_t),0,cudaMemcpyHostToDevice);


	if(SIM_VALIDATION)
	{
		daily_contact_infectors_ptr = thrust::raw_pointer_cast(daily_contact_infectors.data());
		daily_contact_victims_ptr = thrust::raw_pointer_cast(daily_contact_victims.data());
		daily_contact_kval_types_ptr = thrust::raw_pointer_cast(daily_contact_kval_types.data());
		daily_action_type_ptr = thrust::raw_pointer_cast(daily_action_type.data());
		daily_contact_locations_ptr = thrust::raw_pointer_cast(daily_contact_locations.data());

		debug_contactsToActions_float1_ptr = thrust::raw_pointer_cast(debug_contactsToActions_float1.data());
		debug_contactsToActions_float2_ptr = thrust::raw_pointer_cast(debug_contactsToActions_float2.data());
		debug_contactsToActions_float3_ptr = thrust::raw_pointer_cast(debug_contactsToActions_float3.data());
		debug_contactsToActions_float4_ptr = thrust::raw_pointer_cast(debug_contactsToActions_float4.data());

		simDebugArrayPtrStruct_t host_debugArrayPtrStruct[1];
		host_debugArrayPtrStruct->contact_infectors = thrust::raw_pointer_cast(daily_contact_infectors.data());
		host_debugArrayPtrStruct->contact_victims = thrust::raw_pointer_cast(daily_contact_victims.data());
		host_debugArrayPtrStruct->contact_kval_types = thrust::raw_pointer_cast(daily_contact_kval_types.data());
		host_debugArrayPtrStruct->contact_actions = thrust::raw_pointer_cast(daily_action_type.data());
		host_debugArrayPtrStruct->contact_locations = thrust::raw_pointer_cast(daily_contact_locations.data());

		host_debugArrayPtrStruct->float1 = thrust::raw_pointer_cast(debug_contactsToActions_float1.data());
		host_debugArrayPtrStruct->float2 = thrust::raw_pointer_cast(debug_contactsToActions_float2.data());
		host_debugArrayPtrStruct->float3 = thrust::raw_pointer_cast(debug_contactsToActions_float3.data());
		host_debugArrayPtrStruct->float4 = thrust::raw_pointer_cast(debug_contactsToActions_float4.data());
		cudaMemcpyToSymbolAsync(host_debugArrayPtrStruct,host_debugArrayPtrStruct,sizeof(simDebugArrayPtrStruct_t),0,cudaMemcpyHostToDevice);
	}
	else
	{
		daily_contact_infectors_ptr = NULL;
		daily_contact_victims_ptr = NULL;
		daily_contact_kval_types_ptr = NULL;
		daily_action_type_ptr = NULL;
		daily_contact_locations_ptr = NULL;

		debug_contactsToActions_float1_ptr = NULL;
		debug_contactsToActions_float2_ptr = NULL;
		debug_contactsToActions_float3_ptr = NULL;
		debug_contactsToActions_float4_ptr = NULL;
	}

	if(SIM_PROFILING)
	{
		profiler.endFunction(-1,1);
	}
}

void PandemicSim::debug_clearActionsArray()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"debug_clearActionsArray");

	int size_to_clear = is_weekend() ? MAX_CONTACTS_WEEKEND * infected_count : MAX_CONTACTS_WEEKDAY * infected_count;
	cudaMemset(daily_action_type_ptr, 0, sizeof(action_t) * size_to_clear);

	if(SIM_PROFILING)
		profiler.endFunction(current_day,size_to_clear);
}


void PandemicSim::daily_countAndRecover()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "daily_countAndRecover");

	//get pointers
	int * pandemic_counts_ptr = status_counts_dev_ptr;
	int * seasonal_counts_ptr = pandemic_counts_ptr + 8;

	//memset to 0
	cudaMemsetAsync(pandemic_counts_ptr, 0, sizeof(int) * 16);

	int blocks = COUNTING_GRID_BLOCKS;
	int threads = COUNTING_GRID_THREADS;
	size_t dynamic_smemsize = 0;

	kernel_countInfectedStatusAndRecover<<<blocks,threads,dynamic_smemsize>>>(
		people_status_pandemic_ptr, people_status_seasonal_ptr, 
		people_days_pandemic_ptr, people_days_seasonal_ptr,
		number_people, current_day,
		pandemic_counts_ptr, seasonal_counts_ptr);

	cudaMemcpyAsync(&status_counts_today, pandemic_counts_ptr,sizeof(int) * 16,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	daily_writeInfectedStats();

	if(SIM_PROFILING)
		profiler.endFunction(current_day,number_people);
}

void PandemicSim::daily_writeInfectedStats()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"daily_writeInfectedStats");

	int pandemic_recovered = status_counts_today[0];
	int pandemic_susceptible = status_counts_today[1];

	int pandemic_symptomatic = status_counts_today[2] + status_counts_today[3] + status_counts_today[4];
	int pandemic_asymptomatic = status_counts_today[5] + status_counts_today[6] + status_counts_today[7];
	int pandemic_infected = pandemic_symptomatic + pandemic_asymptomatic;

	int seasonal_recovered = status_counts_today[8];
	int seasonal_susceptible = status_counts_today[9];

	int seasonal_symptomatic = status_counts_today[10] + status_counts_today[11] + status_counts_today[12];
	int seasonal_asymptomatic = status_counts_today[13] + status_counts_today[14] + status_counts_today[15];
	int seasonal_infected = seasonal_symptomatic + seasonal_asymptomatic;


	if(SIM_VALIDATION)
	{
		int pandemic_total = pandemic_susceptible + pandemic_infected + pandemic_recovered;
		int seasonal_total = seasonal_susceptible + seasonal_infected + seasonal_recovered;

		debug_assert("pandemic_total does not equal number_people in infected_status func", number_people, pandemic_total);
		debug_assert("seasonal_total does not equal number_people in infected_status func", number_people, seasonal_total);

		if(current_day == 0)
		{
			debug_assert("initial_infected_pandemic does not match the observed count on first day",INITIAL_INFECTED_PANDEMIC, pandemic_infected);
			debug_assert("initial_infected_seasonal does not match the observed count on first day",INITIAL_INFECTED_SEASONAL, seasonal_infected);
		}
	}

	fprintf(f_outputInfectedStats,
		"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
		current_day,
		pandemic_susceptible,
		pandemic_infected,
		pandemic_symptomatic,
		pandemic_asymptomatic,
		pandemic_recovered,
		seasonal_susceptible,
		seasonal_infected,
		seasonal_symptomatic,
		seasonal_asymptomatic,
		seasonal_recovered);

	if(SIM_VALIDATION)
		fflush(f_outputInfectedStats);

	if(SIM_PROFILING)
		profiler.endFunction(current_day,1);
}

void PandemicSim::setup_calculateInfectionData()
{
	//adjust the asymptomatic profiles downwards
	for(int i = 3; i < NUM_SHEDDING_PROFILES; i++)
		for(int j = 0; j < CULMINATION_PERIOD; j++)
			VIRAL_SHEDDING_PROFILES_HOST[i][j] *= asymp_factor;

	//calculate reproduction factors
	for(int i = 0; i < STRAIN_COUNT; i++)
	{
		INFECTIOUSNESS_FACTOR_HOST[i] = BASE_REPRODUCTION_HOST[i] / ((1.0f - asymp_factor) * PERCENT_SYMPTOMATIC_HOST[0]);
	}
}


void PandemicSim::setup_loadSeed()
{
	FILE * fSeed = fopen("seed.txt","r");
	if(fSeed == NULL)
	{
		debug_print("failed to open seed file");
		perror("Error opening seed file");
		exit(1);
	}

	fscanf(fSeed, "%d", &core_seed);
	fclose(fSeed);

	srand(core_seed);
	for(int i = 0; i < SEED_LENGTH; i++)
	{
		int generated_seed = rand();
		SEED_HOST[i] = generated_seed;
	}
}

void PandemicSim::setup_loadFourSeeds()
{
	//load 4 seeds from file
	FILE * fSeed = fopen("seed.txt","r");
	if(fSeed == NULL)
	{
		debug_print("failed to open seed file");
		perror("Error opening seed file");
		exit(1);
	}

	for(int i = 0; i < SEED_LENGTH; i++)
	{
		fscanf(fSeed, "%d", &(SEED_HOST[i]));
	}
	fclose(fSeed);
}

void PandemicSim::setup_setCudaTopology()
{
	cuda_householdTypeAssignmentKernel_blocks = cuda_blocks;
	cuda_householdTypeAssignmentKernel_threads = cuda_threads;

	cuda_peopleGenerationKernel_blocks = cuda_blocks;
	cuda_peopleGenerationKernel_threads = cuda_threads;

	cuda_doWeekdayErrandAssignment_blocks = cuda_blocks;
	cuda_doWeekdayErrandAssignment_threads = cuda_threads;

	cuda_makeWeekdayContactsKernel_blocks = cuda_blocks;
	cuda_makeWeekdayContactsKernel_threads = cuda_threads;

	cuda_makeWeekendContactsKernel_blocks = cuda_blocks;
	cuda_makeWeekendContactsKernel_threads = cuda_threads;

	cuda_contactsToActionsKernel_blocks = cuda_blocks;
	cuda_contactsToActionsKernel_threads = cuda_threads;

	cuda_doInfectionActionsKernel_blocks = cuda_blocks;
	cuda_doInfectionAtionsKernel_threads = cuda_threads;
}

__device__ locId_t device_fishAfterschoolOrErrandDestination_weekday(
	unsigned int rand_val, age_t myAge)
{
	int business_type = BUSINESS_TYPE_AFTERSCHOOL;
	float frac = (float) rand_val / UNSIGNED_MAX;

	//for adults, loop through the errand types and find the one this yval assigns us to
	if(myAge == AGE_ADULT)
	{
		business_type = FIRST_WEEKDAY_ERRAND_ROW;
		float row_pdf = WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_DEVICE[business_type];

		while(frac > row_pdf && business_type < (NUM_BUSINESS_TYPES - 1))
		{
			frac -= row_pdf;
			business_type++;
			row_pdf = WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_DEVICE[business_type];
		}

		frac = frac / row_pdf;
	}

	//lookup type count and offset
	int type_count = WORKPLACE_TYPE_COUNT_DEVICE[business_type];
	int type_offset = WORKPLACE_TYPE_OFFSET_DEVICE[business_type];

	//we now have a fraction between 0 and 1 representing which of this business type we are at
	unsigned int business_num = frac * type_count;

	//frac should be between 0 and 1 but we may lose a little precision here
	if(business_num >= type_count)
		business_num = type_count - 1;

	//add the offset to the first business of this type
	business_num += type_offset;

	return business_num;
}

__device__ void device_assignAfterschoolOrErrandDests_weekday(
	unsigned int rand_val1, unsigned int rand_val2,
	age_t myAge, int num_locations,
	locId_t * output_dest1, locId_t * output_dest2)
{
	//to avoid divergence, the base case will assign the same errand to both hours
	//(i.e. the norm for children)
	int dest1 = device_fishAfterschoolOrErrandDestination_weekday(rand_val1,myAge);

	int dest2 = dest1;
	if(myAge == AGE_ADULT)
		dest2 = device_fishAfterschoolOrErrandDestination_weekday(rand_val2,myAge);

	dest2 += num_locations;

	*output_dest1 = dest1;
	*output_dest2 = dest2;
}

__global__ void kernel_assignWeekdayAfterschoolAndErrands(
	age_t * people_ages_arr, int number_people, int num_locations,
	locId_t * errand_schedule_array, personId_t * errand_people_array,
	randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} u;

	locId_t * errand_schedule_array_hour2 = errand_schedule_array + number_people;
	personId_t * errand_people_array_hour2 = errand_people_array + number_people;

	//for each adult
	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= number_people / 2; myGridPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = myGridPos + rand_offset;
		threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
		u.c = threefry2x64(tf_ctr, tf_k);

		personId_t myIdx_1 = myGridPos * 2;
		if(myIdx_1 < number_people)
		{
			age_t myAge = people_ages_arr[myIdx_1];
			device_assignAfterschoolOrErrandDests_weekday(
				u.i[0],u.i[1],
				myAge, num_locations,
				errand_schedule_array + myIdx_1,
				errand_schedule_array_hour2 + myIdx_1);

			errand_people_array[myIdx_1] = myIdx_1;
			errand_people_array_hour2[myIdx_1] = myIdx_1;
		}

		personId_t myIdx_2 = myIdx_1 + 1;
		if(myIdx_2 < number_people)
		{
			age_t myAge = people_ages_arr[myIdx_2];
			device_assignAfterschoolOrErrandDests_weekday(
				u.i[2],u.i[3],
				myAge, num_locations,
				errand_schedule_array + myIdx_2,
				errand_schedule_array_hour2 + myIdx_2);

			errand_people_array[myIdx_2] = myIdx_2;
			errand_people_array_hour2[myIdx_2] = myIdx_2;
		}
	}
}

void PandemicSim::weekday_generateAfterschoolAndErrandDestinations()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"weekday_generateAfterschoolAndErrandDestinations");

	host_randOffsetsStruct->errand_randOffset = rand_offset;
	cudaMemcpyToSymbolAsync(device_randOffsetsStruct,host_randOffsetsStruct,sizeof(simRandOffsetsStruct_t),0,cudaMemcpyHostToDevice);

	int blocks = cuda_doWeekdayErrandAssignment_blocks;
	int threads = cuda_doWeekdayErrandAssignment_threads;

	kernel_assignWeekdayAfterschoolAndErrands<<<blocks,threads>>>
		(people_ages_ptr, number_people, number_workplaces, people_errands_doubleBuffer.Current(), errand_people_doubleBuffer.Current(), rand_offset);

	if(TIMING_BATCH_MODE == 0)
	{
		const int rand_counts_consumed = number_people / 2;
		rand_offset += rand_counts_consumed;
	}

	if(SIM_PROFILING)
		profiler.endFunction(current_day,number_people);
}

__device__ int device_setInfectionStatus(status_t profile_to_set, day_t day_to_set, gen_t gen_to_set,
										  status_t * output_profile, day_t * output_day, gen_t * output_gen)
{
	status_t val_in_mem = *output_profile;
	if(val_in_mem != STATUS_SUSCEPTIBLE)
		return 0;

	val_in_mem = atomicCAS(output_profile, STATUS_SUSCEPTIBLE, profile_to_set);

	if(val_in_mem != STATUS_SUSCEPTIBLE)
		return 0;

	*output_day = day_to_set;
	*output_gen = gen_to_set;

	return 1;
}


__device__ action_t device_doInfectionActionImmediately(personId_t victim,day_t day_to_set,
												bool infects_pandemic, bool infects_seasonal,
												status_t profile_p_to_set, status_t profile_s_to_set,
												gen_t gen_p_to_set, gen_t gen_s_to_set,
												status_t * people_status_pandemic, status_t * people_status_seasonal,
												day_t * people_days_pandemic, day_t * people_days_seasonal,
												gen_t * people_gens_pandemic, gen_t * people_gens_seasonal)
{
	int success_pandemic = ACTION_INFECT_NONE;
	if(infects_pandemic)
	{
		//returns 1 if action was successful, 0 otherwise
		success_pandemic = device_setInfectionStatus(profile_p_to_set, day_to_set, gen_p_to_set,
			people_status_pandemic + victim, people_days_pandemic + victim, people_gens_pandemic + victim);
		success_pandemic *= ACTION_INFECT_PANDEMIC;  //if successful, success_pandemic contains ACTION_INFECT_SEASONAL, otherwise ACTION_INFECT_NONE
	}

	int success_seasonal = ACTION_INFECT_NONE;
	if(infects_seasonal)
	{
		success_seasonal = device_setInfectionStatus(profile_s_to_set,day_to_set, gen_s_to_set,
			people_status_seasonal + victim, people_days_seasonal + victim, people_gens_seasonal + victim);
		success_seasonal *= ACTION_INFECT_SEASONAL;
	}

	action_t result = success_pandemic + success_seasonal;

	return result;
}


__device__ void device_doContactsToActions_immediately(
	personId_t myIdx, kval_t kval_sum,
	personId_t * contact_victims_arr, kval_type_t *contact_type_arr, int contacts_per_infector,
	status_t * people_status_p_arr, status_t * people_status_s_arr,
	day_t * people_days_pandemic, day_t * people_days_seasonal,
	gen_t * people_gens_pandemic, gen_t * people_gens_seasonal,
#if SIM_VALIDATION == 1
	action_t * output_action_arr,
	float * rand_arr_1, float * rand_arr_2, float * rand_arr_3, float * rand_arr_4,
#endif
	day_t current_day,randOffset_t myRandOffset)
{
	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c[4];
		unsigned int i[16];
	} rand_union;

	//		if(kval_sum == 0)
	//			continue;
	status_t status_p = people_status_p_arr[myIdx];
	status_t status_s = people_status_s_arr[myIdx];

	float inf_prob_p = -1.f;
	float inf_prob_s = -1.f;

	gen_t gen_p_to_set = GENERATION_NOT_INFECTED;
	gen_t gen_s_to_set = GENERATION_NOT_INFECTED;

	//int profile_day_p = -1;
	if(status_is_infected(status_p))
	{
		int profile_day_p = current_day - people_days_pandemic[myIdx];

		//refinement: when doing contacts_to_actions live from shared memory, status_p may be changed out from
		//underneath us.  In this case, day_p may be in an inconsistent state.  Check that it is within bounds
		if(profile_day_p >= 0 && profile_day_p < CULMINATION_PERIOD)
		{
			//to get the profile, subtract the offset to the first profile
			int profile_p = get_profile_from_status(status_p);

			inf_prob_p = device_calculateInfectionProbability(profile_p,profile_day_p, STRAIN_PANDEMIC,kval_sum);
			gen_p_to_set = people_gens_pandemic[myIdx] + 1;
		}
	}

	//int profile_day_s = -1;
	if(status_is_infected(status_s))
	{
		int profile_day_s = current_day - people_days_seasonal[myIdx];

		if(profile_day_s >= 0 && profile_day_s < CULMINATION_PERIOD)
		{
			int profile_s = get_profile_from_status(status_s);

			inf_prob_s = device_calculateInfectionProbability(profile_s ,profile_day_s, STRAIN_SEASONAL,kval_sum);
			gen_s_to_set = people_gens_seasonal[myIdx] + 1;
		}
	}


	threefry2x64_ctr_t tf_ctr_1 = {{myRandOffset, myRandOffset}};
	rand_union.c[0] = threefry2x64(tf_ctr_1, tf_k);
	//myRandOffset += gridDim.x * blockDim.x;

	threefry2x64_ctr_t tf_ctr_2 = {{myRandOffset + 1, myRandOffset + 1}};
	rand_union.c[1] = threefry2x64(tf_ctr_2, tf_k);
	//myRandOffset += gridDim.x * blockDim.x;

	threefry2x64_ctr_t tf_ctr_3 = {{myRandOffset + 2, myRandOffset + 2}};
	rand_union.c[2] = threefry2x64(tf_ctr_3, tf_k);
	//myRandOffset += gridDim.x * blockDim.x;

	threefry2x64_ctr_t tf_ctr_4 = {{myRandOffset + 3, myRandOffset + 3}};
	rand_union.c[3] = threefry2x64(tf_ctr_4, tf_k);

	int rand_vals_used = 0;
	for(int contacts_processed = 0; contacts_processed < contacts_per_infector; contacts_processed++)
	{
		int contact_victim = contact_victims_arr[contacts_processed];
		int contact_type = contact_type_arr[contacts_processed];

		kval_t contact_kval = KVAL_LOOKUP_DEVICE[contact_type];

		float y_p = (float) rand_union.i[rand_vals_used++] / UNSIGNED_MAX;
		bool infects_p = y_p < (float) (inf_prob_p * contact_kval);

		float y_s = (float) rand_union.i[rand_vals_used++] / UNSIGNED_MAX;
		bool infects_s = y_s < (float) (inf_prob_s * contact_kval);

		//function handles parsing bools into an action and checking that victim is susceptible
		action_t result = device_doInfectionActionImmediately(
			contact_victim, current_day + 1,
			infects_p,infects_s,
			STATUS_INFECTED, STATUS_INFECTED,
			gen_p_to_set, gen_s_to_set,
			people_status_p_arr,people_status_s_arr,
			people_days_pandemic,people_days_seasonal,
			people_gens_pandemic,people_gens_seasonal);

#if SIM_VALIDATION == 1
			//if result was successful, copy out the action that resulted
			if(result != ACTION_INFECT_NONE)
				output_action_arr[contacts_processed] = result;

			rand_arr_1[contacts_processed] = y_p;
			rand_arr_2[contacts_processed] = (float) (inf_prob_p * contact_kval);
			rand_arr_3[contacts_processed] = y_s;
			rand_arr_4[contacts_processed] = (float) (inf_prob_s * contact_kval);
#endif
	}

}



__global__ void kernel_weekday_sharedMem(int num_infected, personId_t * infected_indexes, age_t * people_age,
										   locId_t * household_lookup, locOffset_t * household_offsets,// personId_t * household_people,
										   locOffset_t * workplace_offsets, personId_t * workplace_people,
										   locOffset_t * errand_loc_offsets, personId_t * errand_people,
										   status_t * people_status_p_arr, status_t * people_status_s_arr,
										   day_t * people_days_pandemic, day_t * people_days_seasonal,
										   gen_t * people_gens_pandemic, gen_t * people_gens_seasonal,
#if SIM_VALIDATION == 1
										   personId_t * output_infector_arr, personId_t * output_victim_arr,
										   kval_type_t * output_kval_arr,  action_t * output_action_arr, 
										   locId_t * output_contact_loc_arr,
										   float * float_rand1, float * float_rand2,
										   float * float_rand3, float * float_rand4,
#endif
										   day_t current_day,randOffset_t rand_offset)

{
	int contactsPerBlock = blockDim.x * MAX_CONTACTS_WEEKDAY;
	
	extern __shared__ int sharedMem[];
	personId_t * victim_array = (personId_t *) sharedMem;
	kval_type_t * contact_kval_array = (kval_type_t *) &victim_array[contactsPerBlock];
//	threefry2x64_ctr_t * shared_rand_ctrs = (threefry2x64_ctr_t *) &contact_kval_array[contactsPerBlock];

	personId_t * myVictimArray = victim_array + (threadIdx.x * MAX_CONTACTS_WEEKDAY);
	kval_type_t * myKvalArray = contact_kval_array + (threadIdx.x * MAX_CONTACTS_WEEKDAY);

//	threefry2x64_ctr_t * mySharedRandCtr = shared_rand_ctrs + (threadIdx.x / 4);
//	int * mySharedInt = ((int *) mySharedRandCtr) + (threadIdx.x % 4);

	const int rand_counts_consumed = 6;

	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = rand_offset + (myPos * rand_counts_consumed);

		//starter code for in-kernel errand profile assignment

		/*if(threadIdx.x % 4 == 0)
		{
			*mySharedRandCtr = threefry2x64(tf_ctr, tf_k);
		}
		__syncthreads();*/


		//myRandOffset++;

		int output_offset_base = MAX_CONTACTS_WEEKDAY * myPos;

		personId_t myIdx = infected_indexes[myPos];
		age_t myAge = people_age[myIdx];

		kval_t kval_sum = device_makeContacts_weekday(
			myIdx, myAge,
			household_lookup, household_offsets,// household_people,
			workplace_offsets, workplace_people,
			errand_loc_offsets, errand_people,
			myVictimArray, myKvalArray,
#if SIM_VALIDATION == 1
			output_contact_loc_arr + output_offset_base,
#endif
			myRandOffset);

		//makeContacts consumes 2 counts
		myRandOffset += 2;

		//convert the contacts to actions immediately
		device_doContactsToActions_immediately(
			myIdx, kval_sum,
			myVictimArray,myKvalArray, MAX_CONTACTS_WEEKDAY,
			people_status_p_arr,people_status_s_arr,
			people_days_pandemic,people_days_seasonal,
			people_gens_pandemic,people_gens_seasonal,
#if SIM_VALIDATION == 1
			output_action_arr + output_offset_base,
			float_rand1 + output_offset_base, float_rand2 + output_offset_base,
			float_rand3 + output_offset_base, float_rand4 + output_offset_base,
#endif
			current_day, myRandOffset);
		//consumes 4 counts

#if SIM_VALIDATION == 1
		for(int c = 0; c < MAX_CONTACTS_WEEKDAY; c++)
		{
			int output_offset = output_offset_base + c;

			output_infector_arr[output_offset] = myIdx;
			output_victim_arr[output_offset] = myVictimArray[c];
			output_kval_arr[output_offset] = myKvalArray[c];
		}
#endif
	}
}

__global__ void kernel_weekend_sharedMem(int num_infected, personId_t * infected_indexes,
										 locId_t * household_lookup, locOffset_t * household_offsets,
										 locOffset_t * errand_loc_offsets, personId_t * errand_people,
										 status_t * people_status_p_arr, status_t * people_status_s_arr,
										 day_t * people_days_pandemic, day_t * people_days_seasonal,
										 gen_t * people_gens_pandemic, gen_t * people_gens_seasonal,
#if SIM_VALIDATION == 1
										 personId_t * output_infector_arr, personId_t * output_victim_arr, 
										 kval_type_t * output_kval_arr, action_t * output_action_arr,
										 locId_t * output_contact_location_arr,
										 float * float_rand1, float * float_rand2,
										 float * float_rand3, float * float_rand4,
#endif
										 day_t current_day,  randOffset_t rand_offset)

{
	int contactsPerBlock = blockDim.x * MAX_CONTACTS_WEEKEND;
	
	extern __shared__ int sharedMem[];
	personId_t * victim_array = (personId_t *) sharedMem;
	kval_type_t * contact_kval_array = (kval_type_t *) &victim_array[contactsPerBlock];

	personId_t * myVictimArray = victim_array + (threadIdx.x * MAX_CONTACTS_WEEKEND);
	kval_type_t * myKvalArray = contact_kval_array + (threadIdx.x * MAX_CONTACTS_WEEKEND);

	const int rand_counts_consumed = 6;

	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = rand_offset + (myPos * rand_counts_consumed);

		int output_offset_base = MAX_CONTACTS_WEEKEND * myPos;

		personId_t myIdx = infected_indexes[myPos];

		kval_t kval_sum = device_makeContacts_weekend(
			myIdx,
			household_lookup,household_offsets,
			errand_loc_offsets,errand_people,
			myVictimArray,myKvalArray,
#if SIM_VALIDATION == 1
			output_contact_location_arr + output_offset_base,
#endif
			myRandOffset);

		//makeContacts consumes 2 counts
		myRandOffset += 2;

		//convert the contacts to actions immediately
		device_doContactsToActions_immediately(
			myIdx, kval_sum,
			myVictimArray,myKvalArray, MAX_CONTACTS_WEEKEND,
			people_status_p_arr,people_status_s_arr,
			people_days_pandemic,people_days_seasonal,
			people_gens_pandemic,people_gens_seasonal,
#if SIM_VALIDATION == 1
			output_action_arr + output_offset_base,
			float_rand1 + output_offset_base, float_rand2 + output_offset_base,
			float_rand3 + output_offset_base, float_rand4 + output_offset_base,
#endif
			current_day, myRandOffset);
		//consumes 4 counts

#if SIM_VALIDATION == 1
		for(int c = 0; c < MAX_CONTACTS_WEEKEND; c++)
		{
			int output_offset = output_offset_base + c;

			output_infector_arr[output_offset] = myIdx;
			output_victim_arr[output_offset] = myVictimArray[c];
			output_kval_arr[output_offset] = myKvalArray[c];
		}
#endif
	}
}


__device__ action_t device_doInfectionActionImmediately_statusWord(personId_t victim,day_t day_to_set,
														bool infects_pandemic, bool infects_seasonal,
														status_t profile_p_to_set, status_t profile_s_to_set,
														gen_t gen_p_to_set, gen_t gen_s_to_set,
														personStatusStruct_word_t * status_struct_array)
{
	action_t result;

	while(true)
	{
		//get status word from memory and unpack
		personStatusStruct_word_t packed_word = status_struct_array[victim];
		union personStatusUnion u;	//get union
		u.w = packed_word;		//store packed word in union

		//attempt to do pandemic changes
		int success_pandemic = ACTION_INFECT_NONE;
		if(infects_pandemic && u.s.status_pandemic == STATUS_SUSCEPTIBLE)
		{
			u.s.status_pandemic = profile_p_to_set;
			u.s.day_pandemic = day_to_set;
			u.s.gen_pandemic = gen_p_to_set;

			success_pandemic = ACTION_INFECT_PANDEMIC;
		}

		//attempt to do seasonal changes
		int success_seasonal = ACTION_INFECT_NONE;
		if(infects_seasonal && u.s.status_seasonal == STATUS_SUSCEPTIBLE)
		{
			u.s.status_seasonal = profile_s_to_set;
			u.s.day_seasonal = day_to_set;
			u.s.gen_seasonal = gen_s_to_set;

			success_seasonal = ACTION_INFECT_SEASONAL;
		}

		result = success_pandemic + success_seasonal;

		if(result == ACTION_INFECT_NONE)
			return result;

		personStatusStruct_word_t packedword_in_mem = atomicCAS(
			status_struct_array + victim, //target
			packed_word,   //expected: the storage word we got from memory
			u.w); //new val: the modified storage word from the memory

		if(packedword_in_mem == packed_word)
			return result;
		//else, loop and try again
	}
}

//NOTE: does not consider inconsistent state, so it should not be used until all infections have completed
struct isInfectedPred_statusWord
{
	__device__ bool operator() (personStatusStruct_word_t statusWord)
	{
		union personStatusUnion u;
		u.w = statusWord;

		bool is_infected = person_is_infected(u.s.status_pandemic, u.s.status_seasonal);

		return is_infected;
	}
};

struct initalInfection_pandemic_functor : public thrust::unary_function<personStatusStruct_word_t, personStatusStruct_word_t>
{
	__device__ personStatusStruct_word_t operator () (personStatusStruct_word_t initial_status) const
	{
		//read it into a union
		personStatusUnion u;
		u.w = initial_status;

		//set status
		u.s.status_pandemic = STATUS_INFECTED;
		u.s.day_pandemic = INITIAL_DAY;
		u.s.gen_pandemic = INITIAL_GEN;

		//return the new word
		return u.w;
	}
};

struct initalInfection_seasonal_functor : public thrust::unary_function<personStatusStruct_word_t, personStatusStruct_word_t>
{
	__device__ personStatusStruct_word_t operator () (personStatusStruct_word_t initial_status) const
	{
		//read it into a union
		personStatusUnion u;
		u.w = initial_status;

		//set status
		u.s.status_seasonal = STATUS_INFECTED;
		u.s.day_seasonal = INITIAL_DAY;
		u.s.gen_seasonal = INITIAL_GEN;

		//return the new word
		return u.w;
	}
};

__device__ maxContacts_t device_getWorkplaceMaxContacts(locId_t errand)
{
	//strip the "hour" encoding away to get the location ID
	locId_t loc_id = errand % device_simSizeStruct->number_workplaces;

	int location_type = 0;
	int type_offset;
	int type_count;
	maxContacts_t type_max;

	do 
	{
		type_offset = WORKPLACE_TYPE_OFFSET_DEVICE[location_type];
		type_count = WORKPLACE_TYPE_COUNT_DEVICE[location_type];
		type_max = WORKPLACE_TYPE_MAX_CONTACTS_DEVICE[location_type];
		location_type++;
	} while (loc_id >= type_offset + type_count);

	return type_max;
}


#define SETUP_COUNTING_GRID_BLOCKS 32
#define SETUP_COUNTING_GRID_THREADS 256

__global__ void kernel_calcPopulationSize(int * sum_ptr, int number_households)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	//__shared__ int reduction_array[SETUP_COUNTING_GRID_THREADS];
	extern __shared__ int reduction_array[];

	int tid = threadIdx.x;
	int * myLocalSum = &reduction_array[tid];
	myLocalSum[0] = 0;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= number_households / 4; myGridPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = myGridPos;
		threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr,tf_k);

		int myPos = myGridPos * 4;

		if(myPos < number_households)
		{
			householdType_t hh_type = device_setup_fishHouseholdType(rand_union.i[0]);
			myLocalSum[0] += HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[hh_type];
			myLocalSum[0] += HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[hh_type];
		}
		if(myPos + 1 < number_households)
		{
			householdType_t hh_type = device_setup_fishHouseholdType(rand_union.i[1]);
			myLocalSum[0] += HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[hh_type];
			myLocalSum[0] += HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[hh_type];
		}
		if(myPos + 2 < number_households)
		{
			householdType_t hh_type = device_setup_fishHouseholdType(rand_union.i[2]);
			myLocalSum[0] += HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[hh_type];
			myLocalSum[0] += HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[hh_type];
		}
		if(myPos + 3 < number_households)
		{
			householdType_t hh_type = device_setup_fishHouseholdType(rand_union.i[3]);
			myLocalSum[0] += HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[hh_type];
			myLocalSum[0] += HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[hh_type];
		}
	}

	__syncthreads();
	for(int offset = blockDim.x / 2; offset > 0;  offset /= 2)
	{
		if(tid < offset)
			reduction_array[tid] += reduction_array[tid+offset];
		__syncthreads();
	}
	if(tid == 0)
		atomicAdd(sum_ptr,reduction_array[0]);
}


struct hh_peopleCount_functor : public thrust::unary_function<randOffset_t,int>
{
	int number_households;

	__device__ int operator () (randOffset_t myRandOffset) const
	{
		threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
		union{
			threefry2x64_ctr_t c;
			unsigned int i[4];
		} rand_union;
		threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr,tf_k);

		int ret_val = 0;

		int myPos = myRandOffset * 4;
		for(int i = 0; i < 4 && myPos + i < number_households; i++)
		{
			householdType_t hh_type = device_setup_fishHouseholdType(rand_union.i[i]);
			ret_val += HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[hh_type];
			ret_val += HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[hh_type];
		}

		return ret_val;
	}
};


int PandemicSim::setup_calcPopulationSize()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_calcPopulationSize");

	thrust::device_vector<int> device_popSize(1);
	device_popSize[0] = 0;
	int * device_popSize_ptr = thrust::raw_pointer_cast(device_popSize.data());

	int blocks = SETUP_COUNTING_GRID_BLOCKS;
	int threads = SETUP_COUNTING_GRID_THREADS;
	size_t smem_size = threads * sizeof(int);
	//size_t smem_size = 0;

	kernel_calcPopulationSize<<<blocks,threads,smem_size>>>(device_popSize_ptr,number_households);
	cudaDeviceSynchronize();

	int host_popSize = device_popSize[0];

	if(SIM_PROFILING)
		profiler.endFunction(-1,number_households);

	return host_popSize;
}

int PandemicSim::setup_calcPopulationSize_thrust()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_calcPopulationSize_thrust");

	//get a counting iterator
	thrust::counting_iterator<randOffset_t> count_it(0);

	hh_peopleCount_functor peoplecount_functor;
	peoplecount_functor.number_households = number_households;

	int p = thrust::transform_reduce(count_it, count_it + (number_households/4)+1,peoplecount_functor,0,thrust::plus<int>());

	if(SIM_PROFILING)
		profiler.endFunction(-1,number_households);

	return p;
}

void PandemicSim::setup_initializeStatusArrays()
{
	thrust::fill(people_status_pandemic.begin(), people_status_pandemic.end(), STATUS_SUSCEPTIBLE);
	thrust::fill(people_status_seasonal.begin(), people_status_seasonal.end(), STATUS_SUSCEPTIBLE);

	thrust::fill(people_days_pandemic.begin(), people_days_pandemic.end(), DAY_NOT_INFECTED);
	thrust::fill(people_days_seasonal.begin(), people_days_seasonal.end(), DAY_NOT_INFECTED);

	thrust::fill(people_gens_pandemic.begin(), people_gens_pandemic.end(), GENERATION_NOT_INFECTED);
	thrust::fill(people_gens_seasonal.begin(), people_gens_seasonal.end(), GENERATION_NOT_INFECTED);
}

__device__ errandContactsProfile_t device_recalc_weekdayErrandDests_assignProfile(
	personId_t myIdx, age_t myAge, 
	locId_t * output_dest1, locId_t * output_dest2)
{
	//find the counter settings when this errand was generated
	int myGridPos = myIdx / 2;
	randOffset_t myRandOffset = device_randOffsetsStruct->errand_randOffset + myGridPos;
	int num_locations = device_simSizeStruct->number_workplaces;

	//regen the random numbers
	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} u;
	threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
	u.c = threefry2x64(tf_ctr, tf_k);

	int rand_slot = 2 * (myIdx % 2); //either 0 or 2
	device_assignAfterschoolOrErrandDests_weekday(
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

__global__ void kernel_testWeekdayRecalc(personId_t * infected_indexes, int num_infected, age_t * people_ages, locId_t * output_infected_errands)
{
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		personId_t myIdx = infected_indexes[myPos];
		age_t myAge = people_ages[myIdx];

		int output_offset = 2 * myPos;

		device_recalc_weekdayErrandDests_assignProfile(myIdx,myAge, output_infected_errands + output_offset, output_infected_errands + output_offset + 1);
	}

}
void PandemicSim::debug_testErrandRegen_weekday()
{
	//assumes this has been called after errands have been copied into location array
	randOffset_t old_rand_val = rand_offset - (number_people / 2) - (infected_count / 4);

	int blocks = cuda_doWeekdayErrandAssignment_blocks;
	int threads = cuda_doWeekdayErrandAssignment_threads;

	thrust::device_vector<locId_t> d_regen_dests(2 * infected_count);
	locId_t * d_regen_dests_ptr = thrust::raw_pointer_cast(d_regen_dests.data());

	kernel_testWeekdayRecalc<<<blocks,threads>>>(infected_indexes_ptr,infected_count,people_ages_ptr,d_regen_dests_ptr);
	cudaDeviceSynchronize();

//	bool ranges_equal = thrust::equal(d_regen_dests.begin(), d_regen_dests.begin() + (2*infected_count),infected_errands.begin());
//	debug_assert(ranges_equal, "weekday regenerated errands do not match expectation");
}

//__device__ void device_generateWeekendErrands(locId_t * errand_array_ptr, int num_locations,randOffset_t myRandOffset)

__device__ void device_recalc_weekendErrandDests(personId_t myIdx, locId_t * errand_array_ptr)
{
	randOffset_t myRandOffset = device_randOffsetsStruct->errand_randOffset + (2*myIdx);
	device_generateWeekendErrands(errand_array_ptr,myRandOffset);
}



struct assignWorkplaceFunctor : public thrust::unary_function<int,void>
{
	int number_people;
	randOffset_t functor_rand_offset;
	locId_t * people_workplaces_arr;
	age_t * people_ages_arr;

	__device__ void operator () (int myGridPos) const
	{
		//	age_t * people_ages_arr = const_people_ages[0];

		randOffset_t myRandOffset = functor_rand_offset + myGridPos;

		threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
		union{
			threefry2x64_ctr_t c;
			unsigned int i[4];
		} rand_union;
		threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr,tf_k);

		int myPos = myGridPos * 4;
		for(int i = 0; i < 4 && myPos + i < number_people; i++)
		{
			int myIdx = myPos + i;
			device_setup_assignWorkplaceOrSchool(rand_union.i[i],people_ages_arr + myIdx,people_workplaces_arr + myIdx);
		}
	}
};


void PandemicSim::debug_testWorkplaceAssignmentFunctor()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"debug_testWorkplaceAssignmentFunctor");

	thrust::device_vector<locId_t> workplaces_copy(number_people);
	locId_t * people_wp_ptr = thrust::raw_pointer_cast(workplaces_copy.data());

	thrust::device_vector<age_t> ages_copy(number_people);
	thrust::copy_n(people_ages.begin(), number_people,ages_copy.begin());
	age_t * people_a_ptr = thrust::raw_pointer_cast(ages_copy.data());

	assignWorkplaceFunctor wp_functor;
	wp_functor.functor_rand_offset = rand_offset;
	wp_functor.number_people = number_people;
	wp_functor.people_workplaces_arr = people_wp_ptr;
	wp_functor.people_ages_arr = people_a_ptr;

	thrust::counting_iterator<int> count_it(0);
	thrust::for_each_n(thrust::device,count_it,(number_people/4)+1,wp_functor);

//	bool workplaces_match = thrust::equal(workplaces_copy.begin(), workplaces_copy.end(), people_workplaces.begin());
	bool ages_match = thrust::equal(ages_copy.begin(), ages_copy.end(), people_ages.begin());

	//debug_dump_array_toTempFile("functor_wps.txt","wp",&workplaces_copy,number_people);

	if(SIM_PROFILING)
		profiler.endFunction(-1,number_people);
}

void PandemicSim::setup_assignWorkplaces()
{
	if(SIM_PROFILING)
		profiler.beginFunction(-1,"setup_assignWorkplaces");

	host_randOffsetsStruct->workplace_randOffset = rand_offset;

	assignWorkplaceFunctor wp_functor;
	wp_functor.functor_rand_offset = rand_offset;
	wp_functor.number_people = number_people;
	wp_functor.people_workplaces_arr = thrust::raw_pointer_cast(people_errands_a.data());			//generate into errands array temporarily
	wp_functor.people_ages_arr = people_ages_ptr;

	thrust::counting_iterator<int> count_it(0);
	thrust::for_each_n(thrust::device,count_it,(number_people/4)+1,wp_functor);

	if(TIMING_BATCH_MODE == 0)
	{
		const int rand_counts_consumed_2 = number_people / 4;
		rand_offset += rand_counts_consumed_2;
	}

	//IDEA: we want the sorted people IDs to end up in workplace_people, so we will write them into the errand
	//array and then set the output buffer as workplace_people

	errand_people_doubleBuffer.selector = 0;
	errand_people_doubleBuffer.d_buffers[1] = thrust::raw_pointer_cast(workplace_people.data());
	thrust::sequence(errand_people_table_a.begin(), errand_people_table_a.begin() + number_people);
	people_errands_doubleBuffer.selector = 0;	//select array A

	cub::DeviceRadixSort::SortPairs(
		errand_sorting_tempStorage, errand_sorting_tempStorage_size, //temp buffer
		people_errands_doubleBuffer, errand_people_doubleBuffer,	//key, val
		number_people);	//N

	thrust::device_vector<locId_t>::iterator loc_iterator;
	if(people_errands_doubleBuffer.selector == 0)
	{
		loc_iterator = people_errands_a.begin();
	}
	else
	{
		loc_iterator = people_errands_b.begin();
	}

	//find lower bound of each location
	thrust::lower_bound(		
		loc_iterator,
		loc_iterator + number_people,
		count_it,
		count_it + number_workplaces,
		workplace_offsets.begin());
	workplace_offsets[number_workplaces] = number_people;

	//now set the buffers back up properly
	setup_configCubBuffers();

	if(SIM_PROFILING)
		profiler.endFunction(-1,number_people);
}

__device__ locId_t device_recalcWorkplace(personId_t myIdx, age_t myAge)
{
	randOffset_t myRandOffset = device_randOffsetsStruct->workplace_randOffset + (myIdx / 4);

	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;
	threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
	rand_union.c = threefry2x64(tf_ctr,tf_k);

	int rand_slot = myIdx % 4;

	locId_t workplace_val;
	device_setup_assignWorkplaceOrSchool(rand_union.i[rand_slot],&myAge,&workplace_val);

	return workplace_val;
}


__global__ void kernel_testConstMem(int * simSize_arr, randOffset_t * randOffset_arr)
{
//	if(threadIdx.x == 0)
	{
		simSize_arr[0] = device_simSizeStruct->number_people;
		simSize_arr[1] = device_simSizeStruct->number_households;
		simSize_arr[2] = device_simSizeStruct->number_workplaces;

		randOffset_arr[0] = device_randOffsetsStruct->workplace_randOffset;
		randOffset_arr[1] = device_randOffsetsStruct->errand_randOffset;
	}
}

void PandemicSim::debug_testConstsMem()
{
	cudaDeviceSynchronize();

	thrust::device_vector<int> d_sizeArr(3);
	int * sizearr_ptr = thrust::raw_pointer_cast(d_sizeArr.data());
	thrust::device_vector<randOffset_t> d_randArr(2);
	randOffset_t * randarr_ptr = thrust::raw_pointer_cast(d_randArr.data());

	kernel_testConstMem<<<1,1>>>(sizearr_ptr,randarr_ptr);
	cudaDeviceSynchronize();

//	int h_sizeArr[3];
//	thrust::copy_n(d_sizeArr.begin(), 3, h_sizeArr);
	thrust::host_vector<int> h_sizeArr = d_sizeArr;
	thrust::host_vector<randOffset_t> h_randArr = d_randArr;

	int num_p = h_sizeArr[0];
	int num_hh = h_sizeArr[1];
	int num_wp = h_sizeArr[2];

	for(int i = 0; i < 3; i++)
	{
		printf("%d:\t%d\n", i,h_sizeArr[i]);
	}

	for(int i = 0; i < 2; i++)
	{
		printf("%d:\t%lu\n",i,h_randArr[i]);
	}
}

struct regenWeekendErrand_test_functor : public thrust::unary_function<int,void>
{
	locId_t * output_errand_arr;
	personId_t * infected_idx_arr;

	__device__ void operator () (int myPos) const
	{
		personId_t myIdx = infected_idx_arr[myPos];

		int output_offset = NUM_WEEKEND_ERRANDS * myPos;

		device_recalc_weekendErrandDests(myIdx,output_errand_arr + output_offset);
	}
};

void PandemicSim::debug_testErrandRegen_weekend()
{
	thrust::device_vector<locId_t> inf_locs_copy(NUM_WEEKEND_ERRANDS * infected_count);
	thrust::counting_iterator<int> count_it(0);

	regenWeekendErrand_test_functor weekendErrandFunctor;
	weekendErrandFunctor.output_errand_arr = thrust::raw_pointer_cast(inf_locs_copy.data());
	weekendErrandFunctor.infected_idx_arr = infected_indexes_ptr;

	thrust::for_each_n(thrust::device,count_it,infected_count,weekendErrandFunctor);

//	bool errands_match = thrust::equal(infected_errands.begin(), infected_errands.begin() + (NUM_WEEKEND_ERRANDS * infected_count),inf_locs_copy.begin());
//	debug_assert(errands_match, "errend regen method does not match infected locs array");
}

void PandemicSim::setup_sizeCubTempArray()
{
	errand_sorting_tempStorage = NULL;
	errand_sorting_tempStorage_size = 0;

	int num_errands = NUM_WEEKEND_ERRANDS * number_people;

	cub::DeviceRadixSort::SortPairs(errand_sorting_tempStorage, errand_sorting_tempStorage_size, people_errands_doubleBuffer,errand_people_doubleBuffer,num_errands);

	//printf("cub needs %Iu megabytes to sort\n",temp_storage_bytes >> 20);

	cudaError_t result = cudaMalloc(&errand_sorting_tempStorage,errand_sorting_tempStorage_size);
	if(result != cudaSuccess)
	{
		fprintf(stderr,"cudaMalloc failed to allocate temp space for cub, error: %s\n",cudaGetErrorString(result));
		exit(result);
	}
}

void PandemicSim::setup_configCubBuffers()
{
	errand_people_doubleBuffer.d_buffers[0] = thrust::raw_pointer_cast(errand_people_table_a.data());
	errand_people_doubleBuffer.d_buffers[1] = thrust::raw_pointer_cast(errand_people_table_b.data());
	errand_people_doubleBuffer.selector = 0;
	people_errands_doubleBuffer.d_buffers[0] = thrust::raw_pointer_cast(people_errands_a.data());
	people_errands_doubleBuffer.d_buffers[1] = thrust::raw_pointer_cast(people_errands_b.data());
	errand_people_doubleBuffer.selector = 0;
}


//free some memory up for final processing to guarantee we won't overflow here
void PandemicSim::final_releaseMemory()
{
	if(SIM_PROFILING)
		profiler.beginFunction(DAY_NOT_INFECTED,"final_releaseMemory");

	errand_people_table_a.clear();
	errand_people_table_a.shrink_to_fit();

	if(POLL_MEMORY_USAGE)
		logging_pollMemoryUsage_takeSample(DAY_NOT_INFECTED);

	if(SIM_PROFILING)
		profiler.endFunction(DAY_NOT_INFECTED,1);
}