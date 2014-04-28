#include "stdafx.h"

#include "simParameters.h"
#include "profiler.h"

#include "PandemicSim.h"
#include "thrust_functors.h"

#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

//output status messages to console?  Slows things down

//Simulation profiling master control - low performance overhead
const int PROFILE_SIMULATION = 1;


int cuda_blocks = 32;
int cuda_threads = 256;


FILE * f_outputInfectedStats;

FILE * fDebug;

__device__ __constant__ int SEED_DEVICE[SEED_LENGTH];
int SEED_HOST[SEED_LENGTH];

__device__ __constant__ float WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_DEVICE[NUM_BUSINESS_TYPES];				//stores PDF for weekday errand destinations
float WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_HOST[NUM_BUSINESS_TYPES];
__device__ __constant__ float WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_DEVICE[NUM_BUSINESS_TYPES];				//stores PDF for weekend errand destinations
float WORKPLACE_TYPE_WEEKEND_ERRAND_PDF_HOST[NUM_BUSINESS_TYPES];


__device__ __constant__ int WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[6][2];
int WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_HOST[6][2];
__device__ __constant__ int WEEKDAY_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[3][2];
int WEEKDAY_ERRAND_CONTACT_ASSIGNMENT_HOST[3][2];

#define STRAIN_COUNT 2
#define STRAIN_PANDEMIC 0
#define STRAIN_SEASONAL 1
//__device__ __constant__ float BASE_REPRODUCTION_DEVICE[STRAIN_COUNT];
float BASE_REPRODUCTION_HOST[STRAIN_COUNT];

#define BASE_R_PANDEMIC_HOST BASE_REPRODUCTION_HOST[0]
#define BASE_R_SEASONAL_HOST BASE_REPRODUCTION_HOST[1]


__device__ __constant__ float INFECTIOUSNESS_FACTOR_DEVICE[STRAIN_COUNT];
float INFECTIOUSNESS_FACTOR_HOST[STRAIN_COUNT];

__device__ __constant__ float PERCENT_SYMPTOMATIC_DEVICE;
float PERCENT_SYMPTOMATIC_HOST;

__device__ __constant__ kval_t KVAL_LOOKUP_DEVICE[NUM_CONTACT_TYPES];
kval_t KVAL_LOOKUP_HOST[NUM_CONTACT_TYPES];

#define UNSIGNED_MAX (unsigned int) -1

float WORKPLACE_TYPE_ASSIGNMENT_PDF_HOST[NUM_BUSINESS_TYPES];
__device__ float WORKPLACE_TYPE_ASSIGNMENT_PDF_DEVICE[NUM_BUSINESS_TYPES];

int WORKPLACE_TYPE_OFFSET_HOST[NUM_BUSINESS_TYPES];
__device__ __constant__ int WORKPLACE_TYPE_OFFSET_DEVICE[NUM_BUSINESS_TYPES];			//stores location number of first business of this type
int WORKPLACE_TYPE_COUNT_HOST[NUM_BUSINESS_TYPES];
__device__ __constant__ int WORKPLACE_TYPE_COUNT_DEVICE[NUM_BUSINESS_TYPES];				//stores number of each type of business
int WORKPLACE_TYPE_MAX_CONTACTS_HOST[NUM_BUSINESS_TYPES];
__device__ __constant__ int WORKPLACE_TYPE_MAX_CONTACTS_DEVICE[NUM_BUSINESS_TYPES];


__device__ __constant__ float VIRAL_SHEDDING_PROFILES_DEVICE[NUM_PROFILES][CULMINATION_PERIOD];
float VIRAL_SHEDDING_PROFILES_HOST[NUM_PROFILES][CULMINATION_PERIOD];


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



//the first row of the PDF with a value > 0
const int FIRST_WEEKDAY_ERRAND_ROW = 9;
const int FIRST_WEEKEND_ERRAND_ROW = 9;


PandemicSim::PandemicSim() 
{
	logging_openOutputStreams();

	if(PROFILE_SIMULATION)
		profiler.initStack();

	cudaStreamCreate(&stream_secondary);

	setup_loadParameters();
	setup_scaleSimulation();
	setup_calculateInfectionData();

	//copy everything down to the GPU
	setup_pushDeviceData();

	if(TIMING_BATCH_MODE == 0)
	{
		setup_setCudaTopology();
	}

	if(debug_log_function_calls)
		debug_print("parameters loaded");

}


PandemicSim::~PandemicSim(void)
{
	cudaStreamDestroy(stream_secondary);

	if(PROFILE_SIMULATION)
		profiler.done();
	logging_closeOutputStreams();
}

void PandemicSim::setupSim()
{
	if(PROFILE_SIMULATION)
	{
		profiler.beginFunction(-1,"setupSim");
	}

	//moved to constructor for batching
	//	open_debug_streams();
	//	setupLoadParameters();

	rand_offset = 0;				//set global rand counter to 0

	current_day = -1;
	
	if(debug_log_function_calls)
		debug_print("setting up households");
	
	//setup households
	setup_generateHouseholds();	//generates according to PDFs

	if(CONSOLE_OUTPUT)
		printf("%d people, %d households, %d workplaces\n",number_people, number_households, number_workplaces);

	setup_buildFixedLocations();	//household and workplace
	setup_initialInfected();

	if(SIM_VALIDATION)
	{
		cudaDeviceSynchronize();

		debug_sizeHostArrays();
		debug_copyFixedData();
		debug_validatePeopleSetup();
	}

	if(POLL_MEMORY_USAGE)
		logging_pollMemoryUsage_takeSample(current_day);

	if(PROFILE_SIMULATION)
	{
		profiler.endFunction(-1, number_people);
	}

	if(debug_log_function_calls)
		debug_print("simulation setup complete");
}


void PandemicSim::logging_openOutputStreams()
{
	if(log_infected_info)
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

	if(log_contacts)
	{
		if(OUTPUT_FILES_IN_PARENTDIR)
			fContacts = fopen("../debug_contacts.csv", "w");
		else
			fContacts = fopen("debug_contacts.csv", "w");
		
		fprintf(fContacts, "current_day, i, infector_idx, victim_idx, contact_type, infector_loc, victim_loc, locs_matched\n");
	}


	if(log_actions)
	{
		if(OUTPUT_FILES_IN_PARENTDIR)
			fActions = fopen("../debug_actions.csv", "w");
		else
			fActions = fopen("debug_actions.csv", "w");
		fprintf(fActions, "current_day, i, infector, victim, action_type, action_type_string\n");
	}

	if(log_actions_filtered)
	{
		if(OUTPUT_FILES_IN_PARENTDIR)
			fActionsFiltered = fopen("../debug_filtered_actions.csv", "w");
		else
			fActionsFiltered = fopen("debug_filtered_actions.csv", "w");
		fprintf(fActionsFiltered, "current_day, i, type, victim, victim_status_p, victim_gen_p, victim_status_s, victim_gen_s\n");
	}
	

	if(SIM_VALIDATION || debug_log_function_calls)
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
	if(PROFILE_SIMULATION)
		profiler.beginFunction(-1,"setup_loadParameters");

	setup_loadSeed();

	//if printing seeds is desired for debug, etc
	if(1)
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
	fscanf(fConstants, "%f%*c", &sim_scaling_factor);
	fscanf(fConstants, "%f%*c", &PERCENT_SYMPTOMATIC_HOST);
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

	if(PROFILE_SIMULATION)
		profiler.endFunction(-1,1);
}

//push various things to device constant memory
void PandemicSim::setup_pushDeviceData()
{
	if(PROFILE_SIMULATION)
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
		sizeof(int) * 3 * 2,
		0,cudaMemcpyHostToDevice);

	//seeds
	cudaMemcpyToSymbolAsync(
		SEED_DEVICE,
		SEED_HOST,
		sizeof(int) * SEED_LENGTH,
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
		sizeof(float) * NUM_PROFILES * CULMINATION_PERIOD,
		0,cudaMemcpyHostToDevice);

	cudaMemcpyToSymbolAsync(
		&PERCENT_SYMPTOMATIC_DEVICE,
		&PERCENT_SYMPTOMATIC_HOST,
		sizeof(float) * 1,
		0,cudaMemcpyHostToDevice);

	//must synchronize later!

	if(PROFILE_SIMULATION)
		profiler.endFunction(-1,1);
}



//Sets up the initial infection at the beginning of the simulation
//BEWARE: you must not generate dual infections with this code, or you will end up with duplicate infected indexes
void PandemicSim::setup_initialInfected()
{
	if(PROFILE_SIMULATION)
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

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,initial_infected);
}

//sets up the locations which are the same every day and do not change
//i.e. workplace and household
void PandemicSim::setup_buildFixedLocations()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(-1,"setup_buildFixedLocations");
	///////////////////////////////////////
	//home/////////////////////////////////

	//moved to size_global_arrays func
	//household_offsets.resize(number_households + 1);
	//household_people.resize(number_people);

	/*thrust::sequence(household_people.begin(), household_people.begin() + number_people);	//fill array with IDs to sort
	calcLocationOffsets(
		&household_people,
		people_households,
		&household_offsets,
		number_people, number_households);*/

	///////////////////////////////////////
	//work/////////////////////////////////
	//workplace_offsets.resize(number_workplaces + 1);	//size arrays
	//workplace_people.resize(number_people);

	thrust::sequence(workplace_people.begin(), workplace_people.begin() + number_people);	//fill array with IDs to sort

	setup_calcLocationOffsets(
		&workplace_people,
		people_workplaces,
		&workplace_offsets,
		number_people, number_workplaces);

	//set up workplace max contacts
	workplace_max_contacts.resize(number_workplaces);		//size the array

	//copy the number of contacts per location type to device
	vec_t workplace_type_max_contacts(NUM_BUSINESS_TYPES);		
	thrust::copy_n(WORKPLACE_TYPE_MAX_CONTACTS_HOST, NUM_BUSINESS_TYPES, workplace_type_max_contacts.begin());

	//TODO:  make this work right with device constant memory.  For now, just make a copy in global memory
	vec_t business_type_count_vec(NUM_BUSINESS_TYPES);
	thrust::copy_n(WORKPLACE_TYPE_COUNT_HOST,NUM_BUSINESS_TYPES,business_type_count_vec.begin());
	vec_t business_type_count_offset_vec(NUM_BUSINESS_TYPES);
	thrust::exclusive_scan(business_type_count_vec.begin(), business_type_count_vec.end(), business_type_count_offset_vec.begin());

	//scatter code is based on Thrust example: expand.cu
	//first, scatter the indexes of the type of business into the array mapped by the output offset
	thrust::counting_iterator<int> count_iterator(0);
	thrust::scatter_if(
		count_iterator,							//value to scatter - begin - index of the type to load
		count_iterator + NUM_BUSINESS_TYPES,		//value to scatter - end
		business_type_count_offset_vec.begin(),				//map of scatter destinations
		business_type_count_vec.begin(),			//stencil: no predicate given means scatter if the count for a type is >0
		workplace_max_contacts.begin());

	//next, use a max_scan to fill in the holes, so all entries in max_contacts hold the index of their business type
	thrust::inclusive_scan(
		workplace_max_contacts.begin(),
		workplace_max_contacts.end(),
		workplace_max_contacts.begin(),
		thrust::maximum<int>());

	//now use a gather to pull the max_contacts into position
	thrust::gather(
		workplace_max_contacts.begin(),
		workplace_max_contacts.end(),
		workplace_type_max_contacts.begin(),
		workplace_max_contacts.begin());

	if(PROFILE_SIMULATION)
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
	if(PROFILE_SIMULATION)
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

	if(PROFILE_SIMULATION)
		profiler.endFunction(-1,number_people);
}


void PandemicSim::logging_closeOutputStreams()
{
	if(log_infected_info)
	{
		fclose(fInfected);
	}

	/*if(log_location_info)
	{
		fclose(fLocationInfo);
	}*/

	if(log_contacts)
	{
		fclose(fContacts);
	}

	if(log_actions)
	{
		fclose(fActions);
	}

	if(log_actions_filtered)
	{
		fclose(fActionsFiltered);
	}

	if(SIM_VALIDATION || debug_log_function_calls)
		fclose(fDebug);

	fclose(f_outputInfectedStats);
} 



void PandemicSim::runToCompletion()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(-1, "runToCompletion");

	for(current_day = 0; current_day < MAX_DAYS; current_day++)
	{
		if(debug_log_function_calls)
			debug_print("beginning day...");

		if(debug_null_fill_daily_arrays)
			debug_nullFillDailyArrays();

		daily_actions = 0;

		//begin asynchronous count of the infected stats
		daily_countInfectedStats();			

		//build infected index array
		daily_buildInfectedArray_global();
		cudaDeviceSynchronize();

		if(infected_count == 0)
			break;

		daily_clearActionsArray(); //must occur AFTER we have counted infected

		if(SIM_VALIDATION)
		{
			debug_validateInfectionStatus();

			fprintf(fDebug, "\n\n---------------------\nday %d\ninfected: %d\n---------------------\n\n", current_day, infected_count);
			fflush(fDebug);
		}

		if(CONSOLE_OUTPUT)
		{
			printf("Day %d:\tinfected: %5d\n", current_day + 1, infected_count);
		}

		if(POLL_MEMORY_USAGE)
			logging_pollMemoryUsage_takeSample(current_day);

		//MAKE CONTACTS DEPENDING ON TYPE OF DAY
		if(is_weekend())
		{
			doWeekend_wholeDay();
		}
		else
		{
			doWeekday_wholeDay();
		}

		//PROCESS CONTACTS AND UPDATE INFECTED
		dailyUpdate();

		if(1)
			fflush(f_outputInfectedStats);

		//if we're using the profiler, flush each day in case of crash
		if(PROFILE_SIMULATION)
		{
			profiler.dailyFlush();
		}
	}

	final_countReproduction();

	if(PROFILE_SIMULATION)
		profiler.endFunction(-1, number_people);


	//moved to destructor for batching
	//close_output_streams();
}


//copies indexes 3 times into array, i.e. for IDS 1-3 produces array:
// 1 2 3 1 2 3 1 2 3
__device__ void device_copyPeopleIndexes_weekend_wholeDay(int * id_dest_ptr, int myIdx)
{
	id_dest_ptr[0] = myIdx;
	id_dest_ptr[1] = myIdx;
	id_dest_ptr[2] = myIdx;
}

//gets three UNIQUE errand hours 
__device__ void device_assignErrandHours_weekend_wholeDay(int * hours_dest_ptr, randOffset_t myRandOffset)
{
	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} u;
	
	threefry2x64_ctr_t tf_ctr = {{ myRandOffset,  myRandOffset}};
	u.c = threefry2x64(tf_ctr, tf_k);

	int first, second, third;

	//get first hour
	first = u.i[0] % NUM_WEEKEND_ERRAND_HOURS;

	//get second hour, if it matches then increment
	second = u.i[1] % NUM_WEEKEND_ERRAND_HOURS;
	if(second == first)
		second = (second + 1) % NUM_WEEKEND_ERRAND_HOURS;

	//get third hour, increment until it no longer matches
	third = u.i[2] % NUM_WEEKEND_ERRAND_HOURS;
	while(third == first || third == second)
		third = (third + 1 ) % NUM_WEEKEND_ERRAND_HOURS;

	//store in output array
	hours_dest_ptr[0] = first;
	hours_dest_ptr[1] = second;
	hours_dest_ptr[2] = third;
}

__device__ void device_fishWeekendErrandDestination(unsigned int * rand_val, int * output_ptr)
{
	float y = (float) *rand_val / UNSIGNED_MAX;

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

	*output_ptr = business_num + type_offset;
}

//This method consumes the accumulated contacts, and causes infections and recovery to occur
void PandemicSim::dailyUpdate()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "dailyUpdate");

	if(debug_log_function_calls)
		debug_print("beginning daily update");

	//synchronize the secondary stream - this ensures that the countInfected kernel has finished
	//and sent its data, and that the actions array has been nulled
	cudaStreamSynchronize(stream_secondary);
	daily_writeInfectedStats();

	//process contacts into actions
	daily_contactsToActions_new();

	//filter invalid actions - not susceptible, duplicate, etc
	daily_filterActions_new();

	//do infection actions
	daily_doInfectionActions();

	//recover infected who have reached culmination
	daily_recoverInfected_new();

	if(debug_log_function_calls)
		debug_print("daily update complete");

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}



//will resize the infected, contact, and action arrays to fit the entire population
void PandemicSim::setup_sizeGlobalArrays()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(-1,"setup_sizeGlobalArrays");
	//setup people status:
	people_status_pandemic.resize(number_people);
	people_status_seasonal.resize(number_people);
	thrust::fill(people_status_pandemic.begin(), people_status_pandemic.end(), STATUS_SUSCEPTIBLE);
	thrust::fill(people_status_seasonal.begin(), people_status_seasonal.end(), STATUS_SUSCEPTIBLE);

	people_days_pandemic.resize(number_people);
	people_days_seasonal.resize(number_people);
	thrust::fill(people_days_pandemic.begin(), people_days_pandemic.end(), DAY_NOT_INFECTED);
	thrust::fill(people_days_seasonal.begin(), people_days_seasonal.end(), DAY_NOT_INFECTED);

	people_gens_pandemic.resize(number_people);
	people_gens_seasonal.resize(number_people);
	thrust::fill(people_gens_pandemic.begin(), people_gens_pandemic.end(), GENERATION_NOT_INFECTED);
	thrust::fill(people_gens_seasonal.begin(), people_gens_seasonal.end(), GENERATION_NOT_INFECTED);

	people_ages.resize(number_people);
	people_households.resize(number_people);
	people_workplaces.resize(number_people);
	people_child_indexes.resize(number_children);
	people_adult_indexes.resize(number_adults);

	household_offsets.resize(number_households + 1);
	household_people.resize(number_people);

	workplace_offsets.resize(number_workplaces + 1);
	workplace_people.resize(number_people);
	workplace_max_contacts.resize(number_workplaces);

	//assume that worst-case everyone gets infected
	infected_indexes.resize(number_people);
	infected_daily_kval_sum.resize(number_people);

	int expected_max_contacts = number_people * MAX_CONTACTS_PER_DAY;

	//resize contact arrays
	daily_contact_infectors.resize(expected_max_contacts);
	daily_contact_victims.resize(expected_max_contacts);
	daily_contact_kval_types.resize(expected_max_contacts);
	daily_action_type.resize(expected_max_contacts);

	//weekend errands arrays tend to be very large, so pre-allocate them
	int num_weekend_errands = number_people * NUM_WEEKEND_ERRANDS;
	errand_people_table.resize(num_weekend_errands);
	errand_people_weekendHours.resize(num_weekend_errands);
	errand_people_destinations.resize(num_weekend_errands);

	errand_infected_locations.resize(num_weekend_errands);
	errand_infected_weekendHours.resize(num_weekend_errands);
	errand_infected_ContactsDesired.resize(number_people);

	errand_locationOffsets_multiHour.resize((number_workplaces * NUM_WEEKEND_ERRAND_HOURS) + 1);
	errand_hourOffsets_weekend.resize(NUM_WEEKEND_ERRAND_HOURS + 1);
	errand_hourOffsets_weekend[NUM_WEEKEND_ERRAND_HOURS] = NUM_WEEKEND_ERRANDS * number_people;

	status_counts.resize(16);

	if(SIM_VALIDATION)
	{
		debug_contactsToActions_float1.resize(expected_max_contacts);
		debug_contactsToActions_float2.resize(expected_max_contacts);
		debug_contactsToActions_float3.resize(expected_max_contacts);
		debug_contactsToActions_float4.resize(expected_max_contacts);
	}

	setup_fetchVectorPtrs(); //get the raw int * pointers

	if(PROFILE_SIMULATION)
	{
		profiler.endFunction(-1,number_people);
	}
}



void PandemicSim::debug_nullFillDailyArrays()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day,"debug_nullFillDailyArrays");

	thrust::fill(daily_contact_infectors.begin(), daily_contact_infectors.end(), -1);
	thrust::fill(daily_contact_victims.begin(), daily_contact_victims.end(), -1);
	thrust::fill(daily_contact_kval_types.begin(), daily_contact_kval_types.end(), CONTACT_TYPE_NONE);
	thrust::fill(infected_daily_kval_sum.begin(), infected_daily_kval_sum.end(), 0);

	thrust::fill(daily_action_type.begin(), daily_action_type.end(), ACTION_INFECT_NONE);

	thrust::fill(errand_infected_locations.begin(), errand_infected_locations.end(), -1);
	thrust::fill(errand_infected_weekendHours.begin(), errand_infected_weekendHours.end(), -1);
	thrust::fill(errand_infected_ContactsDesired.begin(), errand_infected_ContactsDesired.end(), -1);

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, number_people);
}

void PandemicSim::setup_scaleSimulation()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(-1,"setup_scaleSimulation");

	number_households = roundHalfUp_toInt(sim_scaling_factor * (double) number_households);

	int sum = 0;
	for(int business_type = 0; business_type < NUM_BUSINESS_TYPES; business_type++)
	{
		//for each type of business, scale by overall simulation scalar
		int original_type_count = roundHalfUp_toInt(WORKPLACE_TYPE_COUNT_HOST[business_type]);
		int new_type_count = roundHalfUp_toInt(sim_scaling_factor * original_type_count);

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

	if(PROFILE_SIMULATION)
		profiler.endFunction(-1,NUM_BUSINESS_TYPES);
}

void PandemicSim::debug_dump_array_toTempFile(const char * filename, const char * description, d_vec * target_array, int array_count)
{
	if(PROFILE_SIMULATION)
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

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,array_count);
}


void PandemicSim::doWeekday_wholeDay()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "doWeekday_wholeDay");

	//generate errands and afterschool locations
	weekday_scatterAfterschoolLocations_wholeDay(&errand_people_destinations);
	weekday_scatterErrandDestinations_wholeDay(&errand_people_destinations);
	cudaDeviceSynchronize();

//	debug_dump_array_toTempFile("../unsorted_dests.txt","errand dest", &errand_people_destinations, number_people * NUM_WEEKDAY_ERRAND_HOURS);

	//fish out the locations of the infected people
	weekday_doInfectedSetup_wholeDay(&errand_people_destinations, &errand_infected_locations, &errand_infected_ContactsDesired);
	if(SIM_VALIDATION)
		debug_copyErrandLookup();	//debug: copy the lookup tables to host memory before they are sorted
	cudaDeviceSynchronize();


	//generate location arrays for each hour
	for(int hour = 0; hour < NUM_WEEKDAY_ERRAND_HOURS; hour++)
	{
		int people_offset_start = hour * number_people;
		int people_offset_end = (hour+1) * number_people;

		//write sequential blocks of indexes, i.e. 0 1 2 0 1 2
		thrust::sequence(
			errand_people_table.begin() + people_offset_start,
			errand_people_table.begin() + people_offset_end);

		//sort the indexes by destination
		thrust::sort_by_key(
			errand_people_destinations.begin() + people_offset_start,	//key.begin
			errand_people_destinations.begin() + people_offset_end,		//key.end
			errand_people_table.begin() + people_offset_start);			//vals.begin

		int location_offset_start = hour * number_workplaces;
//		int location_offset_end = location_offset_start + number_workplaces;
		thrust::counting_iterator<int> count_it(0);

		//binary search the location offsets
		thrust::lower_bound(
			errand_people_destinations.begin() + people_offset_start,	//vals.begin: search workplace 0 to N for this hour
			errand_people_destinations.begin() + people_offset_end,			//vals.end
			count_it,
			count_it + number_workplaces,
			errand_locationOffsets_multiHour.begin() + location_offset_start);		//output.begin
	}

//	debug_dump_array_toTempFile("../sorted_dests.txt", "errand_dest", &errand_people_destinations, number_people * NUM_WEEKDAY_ERRAND_HOURS);
//	debug_dump_array_toTempFile("../loc_offsets.txt", "loc_offset", &errand_locationOffsets_multiHour, NUM_WEEKDAY_ERRAND_HOURS * number_workplaces);
//	debug_dump_array_toTempFile("../inf_locs.txt", "loc", &errand_infected_locations, infected_count * NUM_WEEKDAY_ERRAND_HOURS);

//	debug_dumpInfectedErrandLocs();

	makeContactsKernel_weekday<<<cuda_makeWeekdayContactsKernel_blocks,cuda_makeWeekdayContactsKernel_threads>>>(
		infected_count, infected_indexes_ptr, people_ages_ptr,
		people_households_ptr, household_offsets_ptr, household_people_ptr,
		workplace_max_contacts_ptr, people_workplaces_ptr, 
		workplace_offsets_ptr, workplace_people_ptr,
		errand_infected_ContactsDesired_ptr, errand_infected_locations_ptr,
		errand_locationOffsets_multiHour_ptr,errand_people_table_ptr,
		number_workplaces,
		daily_contact_infectors_ptr, daily_contact_victims_ptr, daily_contact_kval_types_ptr,
		infected_daily_kval_sum_ptr, rand_offset, number_people);

	if(TIMING_BATCH_MODE == 0)
	{
		const int rand_counts_consumed = 2;
		rand_offset += (rand_counts_consumed * infected_count);
	}
	cudaDeviceSynchronize();

	if(SIM_VALIDATION)
		validateContacts_wholeDay();

//	debug_dump_array_toTempFile("../infected_kvals.txt","kval",&infected_daily_kval_sum, infected_count);

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::weekday_scatterAfterschoolLocations_wholeDay(d_vec * people_locs)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "weekday_scatterAfterschoolLocations_wholeDay");

	int * output_arr_ptr = thrust::raw_pointer_cast(people_locs->data());

	kernel_assignAfterschoolLocations_wholeDay<<<cuda_blocks,cuda_threads>>>(people_child_indexes_ptr,output_arr_ptr, number_children,number_people,rand_offset);
	rand_offset += number_children / 4;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,number_children);
}

void PandemicSim::weekday_scatterErrandDestinations_wholeDay(d_vec * people_locs)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "weekday_scatterAfterschoolLocations_wholeDay");

	int * output_arr_ptr = thrust::raw_pointer_cast(people_locs->data());

	kernel_assignErrandLocations_weekday_wholeDay<<<cuda_blocks,cuda_threads>>>(people_adults_indexes_ptr, number_adults, number_people ,output_arr_ptr, rand_offset);
	rand_offset += number_adults / 2;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,number_adults);
}

void PandemicSim::doWeekend_wholeDay()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "doWeekend_wholeDay");

	//assign all weekend errands
	weekend_assignErrands(&errand_people_table, &errand_people_weekendHours, &errand_people_destinations);
	cudaDeviceSynchronize();

	//fish the infected errands out
	weekend_doInfectedSetup_wholeDay(&errand_people_weekendHours,&errand_people_destinations, &errand_infected_weekendHours, &errand_infected_locations, &errand_infected_ContactsDesired);
	if(SIM_VALIDATION)
		debug_copyErrandLookup();
	cudaDeviceSynchronize();

	//each person gets 3 errands
	const int num_weekend_errands_total = NUM_WEEKEND_ERRANDS * number_people;

	//now sort the errand_people array into a large multi-hour location table
	thrust::sort_by_key(
		thrust::make_zip_iterator(thrust::make_tuple(
			errand_people_weekendHours.begin(), 
			errand_people_destinations.begin())),	//key.begin
		thrust::make_zip_iterator(thrust::make_tuple(
			errand_people_weekendHours.begin() + num_weekend_errands_total, 
			errand_people_destinations.begin() + num_weekend_errands_total)),		//key.end
		errand_people_table.begin(),
		Pair_SortByFirstThenSecond_struct());									//data

	//find how many people are going on errands during each hour
	thrust::counting_iterator<int> count_it(0);
	thrust::lower_bound(
		errand_people_weekendHours.begin(),
		errand_people_weekendHours.begin() + num_weekend_errands_total,
		count_it,
		count_it + NUM_WEEKEND_ERRAND_HOURS,
		errand_hourOffsets_weekend.begin());
	//people_hour_offsets[NUM_WEEKEND_ERRAND_HOURS] = num_weekend_errands_total;	//moved to size_global_array method


//	debug_dump_array_toTempFile("../weekend_hour_offsets.txt","hour offset",&errand_hourOffsets_weekend,NUM_WEEKEND_ERRAND_HOURS + 1);

	for(int hour = 0; hour < NUM_WEEKEND_ERRAND_HOURS; hour++)
	{
		int location_offset_start = hour * number_workplaces;

		//search for the locations within this errand hour
		thrust::lower_bound(
			errand_people_destinations.begin() + errand_hourOffsets_weekend[hour],
			errand_people_destinations.begin() + errand_hourOffsets_weekend[hour+1],
			count_it,
			count_it + number_workplaces,
			errand_locationOffsets_multiHour.begin() + location_offset_start);
	}

	debug_validateLocationArrays();
//	debug_dump_array_toTempFile("../weekend_loc_offsets.csv","loc offset",&errand_locationOffsets_multiHour, (NUM_WEEKEND_ERRAND_HOURS * number_workplaces));


	//launch kernel
	cudaDeviceSynchronize();

	makeContactsKernel_weekend<<<cuda_makeWeekendContactsKernel_blocks,cuda_makeWeekendContactsKernel_threads>>>(
		infected_count, infected_indexes_ptr,
		people_households_ptr, household_offsets_ptr, household_people_ptr,
		errand_infected_weekendHours_ptr, errand_infected_locations_ptr, errand_infected_ContactsDesired_ptr,
		errand_locationOffsets_multiHour_ptr ,errand_people_table_ptr, errand_hourOffsets_weekend_ptr,
		number_workplaces,
		daily_contact_infectors_ptr, daily_contact_victims_ptr, daily_contact_kval_types_ptr,
		infected_daily_kval_sum_ptr, rand_offset);

	if(TIMING_BATCH_MODE == 0)
	{
		int rand_counts_used = 2 * infected_count;
		rand_offset += rand_counts_used;
	}
	cudaDeviceSynchronize();

	if(log_contacts)
		validateContacts_wholeDay();

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::weekday_doInfectedSetup_wholeDay(vec_t * lookup_array, vec_t * inf_locs, vec_t * inf_contacts_desired)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "weekday_doInfectedSetup_wholeDay");

	int * loc_lookup_ptr = thrust::raw_pointer_cast(lookup_array->data());
	int * inf_locs_ptr = thrust::raw_pointer_cast(inf_locs->data());
	int * inf_contacts_desired_ptr = thrust::raw_pointer_cast(inf_contacts_desired->data());

	kernel_doInfectedSetup_weekday_wholeDay<<<cuda_blocks, cuda_threads>>>(
		infected_indexes_ptr,infected_count,
		loc_lookup_ptr,people_ages_ptr,number_people,
		inf_locs_ptr,inf_contacts_desired_ptr, rand_offset);

	const int rand_counts_used = infected_count / 4;
	rand_offset += rand_counts_used;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::weekend_doInfectedSetup_wholeDay(vec_t * errand_hours, vec_t * errand_destinations, vec_t * infected_hours, vec_t * infected_destinations, vec_t * infected_contacts_desired)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "weekend_doInfectedSetup");

	//second input: collated lookup tables for hours and destinations
	int * errand_hour_ptr = thrust::raw_pointer_cast(errand_hours->data());
	int * errand_dest_ptr = thrust::raw_pointer_cast(errand_destinations->data());

	//outputs: the hour of the errands and the destinations
	int * infected_hour_ptr = thrust::raw_pointer_cast(infected_hours->data());
	int * infected_destinations_ptr = thrust::raw_pointer_cast(infected_destinations->data());
	int * infected_contacts_desired_ptr = thrust::raw_pointer_cast(infected_contacts_desired->data());

	kernel_doInfectedSetup_weekend<<<cuda_blocks,cuda_threads>>>(
		infected_indexes_ptr,errand_hour_ptr,errand_dest_ptr,
		infected_hour_ptr, infected_destinations_ptr, infected_contacts_desired_ptr,
		infected_count, rand_offset);

	int rand_counts_consumed = infected_count / 4;
	rand_offset += rand_counts_consumed;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::weekend_assignErrands(vec_t * errand_people, vec_t * errand_hours, vec_t * errand_destinations)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "weekend_assignErrands");

	int * errand_people_ptr = thrust::raw_pointer_cast(errand_people->data());
	int * errand_hours_ptr = thrust::raw_pointer_cast(errand_hours->data());
	int * errand_dests_ptr=  thrust::raw_pointer_cast(errand_destinations->data());

	kernel_assignErrands_weekend<<<cuda_blocks,cuda_threads>>>(errand_people_ptr,errand_hours_ptr,errand_dests_ptr, number_people,rand_offset);

	int rand_counts_consumed = 2 * number_people;
	rand_offset += rand_counts_consumed;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,number_people);
}

__device__ void device_assignContactsDesired_weekday_wholeDay(unsigned int rand_val, int myAge, int * output_contacts_desired)
{
	int contacts_hour[2];
	if(myAge == AGE_ADULT)
	{
		//get a profile between 0 and 2
		int contacts_profile = rand_val % 3;

		contacts_hour[0] = WEEKDAY_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[contacts_profile][0];
		contacts_hour[1] = WEEKDAY_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[contacts_profile][1];
	}
	else
	{
		contacts_hour[0] = WORKPLACE_TYPE_MAX_CONTACTS_DEVICE[BUSINESS_TYPE_AFTERSCHOOL];
		contacts_hour[1] = 0;
	}

	*(output_contacts_desired) = contacts_hour[0];
	*(output_contacts_desired) = contacts_hour[1];
}

__global__ void kernel_assignContactsDesired_weekday_wholeDay(int * infected_indexes_arr, int num_infected, int * age_lookup_arr, int * contacts_desired_arr, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos < num_infected / 4; myGridPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = myGridPos + rand_offset;
		threefry2x64_ctr_t tf_ctr = {{myRandOffset,	myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr,tf_k);

		int myPos = num_infected * 4;
		int myIdx[4];
		int myAge[4];

		if(myPos < num_infected)
		{
			myIdx[0] = infected_indexes_arr[myPos];
			myAge[0] = age_lookup_arr[myIdx[0]];
			device_assignContactsDesired_weekday_wholeDay(rand_union.i[0], myAge[0], contacts_desired_arr + (myPos * 2));
		}

		if(myPos + 1 < num_infected)
		{
			myIdx[1] = infected_indexes_arr[myPos+1];
			myAge[1] = age_lookup_arr[myIdx[1]];
			device_assignContactsDesired_weekday_wholeDay(rand_union.i[1], myAge[1], contacts_desired_arr + ((myPos+1) * 2));
		}

		if(myPos + 2 < num_infected)
		{
			myIdx[2] = infected_indexes_arr[myPos+2];
			myAge[2] = age_lookup_arr[myIdx[2]];
			device_assignContactsDesired_weekday_wholeDay(rand_union.i[2], myAge[2], contacts_desired_arr + ((myPos+2) * 2));
		}

		if(myPos + 3 < num_infected)
		{
			myIdx[3] = infected_indexes_arr[myPos+3];
			myAge[3] = age_lookup_arr[myIdx[3]];
			device_assignContactsDesired_weekday_wholeDay(rand_union.i[3], myAge[3], contacts_desired_arr + ((myPos+3) * 2));
		}
	}
}

__device__ void device_assignContactsDesired_weekday_wholeDay(unsigned int * rand_val, int age, int * contactsDesiredHour1, int * contactsDesiredHour2)
{
	int hour1, hour2;
	if(age == AGE_ADULT)
	{
		//assign 2 contacts between the two hours
		hour1 = (*rand_val) % 3;
		hour2 = 2 - hour1;
	}
	else
	{
		//look up max contacts for afterschool type
		hour1 = WORKPLACE_TYPE_MAX_CONTACTS_DEVICE[BUSINESS_TYPE_AFTERSCHOOL];
		hour2 = 0;
	}
	*contactsDesiredHour1 = hour1;
	*contactsDesiredHour2 = hour2;
}
__device__ void device_copyInfectedErrandLocs_weekday(int * loc_lookup_ptr, int * output_infected_locs_ptr, int num_people)
{
	*(output_infected_locs_ptr) = *loc_lookup_ptr;
	*(output_infected_locs_ptr+1) = *(loc_lookup_ptr + num_people);
}

__device__ void device_doAllWeekdayInfectedSetup(unsigned int * rand_val, int myPos, int * infected_indexes_arr, int * loc_lookup_arr, int * ages_lookup_arr, int num_people, int * output_infected_locs, int * output_infected_contacts_desired)
{
	int myIdx = infected_indexes_arr[myPos];
	int myAge = ages_lookup_arr[myIdx];
	int output_offset = 2 * myPos;
	device_copyInfectedErrandLocs_weekday(loc_lookup_arr + myIdx, output_infected_locs + output_offset, num_people);
	device_assignContactsDesired_weekday_wholeDay(rand_val, myAge,
		output_infected_contacts_desired + output_offset,
		output_infected_contacts_desired + output_offset + 1);
}
__global__ void kernel_doInfectedSetup_weekday_wholeDay(int * infected_index_arr, int num_infected, int * loc_lookup_arr, int * ages_lookup_arr, int num_people, int * output_infected_locs, int * output_infected_contacts_desired, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= num_infected / 4; myGridPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = myGridPos + rand_offset;
		//get 4 random numbers
		threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr,tf_k);

		//select a block of up to 4 infected people
		int myPos = myGridPos * 4;

		if(myPos < num_infected)
		{
			device_doAllWeekdayInfectedSetup(&(rand_union.i[0]),myPos, 
				infected_index_arr, loc_lookup_arr, ages_lookup_arr,
				num_people, output_infected_locs, output_infected_contacts_desired);
		}

		if(myPos + 1 < num_infected)
		{
			device_doAllWeekdayInfectedSetup(&(rand_union.i[1]),myPos + 1, 
				infected_index_arr, loc_lookup_arr, ages_lookup_arr,
				num_people, output_infected_locs, output_infected_contacts_desired);
		}

		if(myPos + 2 < num_infected)
		{
			device_doAllWeekdayInfectedSetup(&(rand_union.i[2]),myPos + 2, 
				infected_index_arr, loc_lookup_arr, ages_lookup_arr,
				num_people, output_infected_locs, output_infected_contacts_desired);
		}

		if(myPos + 3 < num_infected)
		{
			device_doAllWeekdayInfectedSetup(&(rand_union.i[3]),myPos + 3, 
				infected_index_arr, loc_lookup_arr, ages_lookup_arr,
				num_people, output_infected_locs, output_infected_contacts_desired);
		}
	}
}

#pragma region debug_printing_funcs

inline void debug_print(char * message)
{
	fprintf(fDebug, "%s\n", message);
	fflush(fDebug);
} 



inline void debug_assert(bool condition, char * message)
{
	if(!condition)
	{
		fprintf(fDebug, "ERROR: ");
		debug_print(message);
	}
}

inline void debug_assert(char *message, int expected, int actual)
{
	if(expected != actual)
	{
		fprintf(fDebug, "ERROR: %s expected: %d actual: %d\n", message, expected, actual);
		fflush(fDebug);
	}
}

inline void debug_assert(bool condition, char * message, int idx)
{
	if(!condition)
	{
		fprintf(fDebug, "ERROR: %s index: %d\n", message, idx);
	}
}
#pragma endregion debug_printing_funcs

#pragma region debug_lookup_funcs

inline char status_int_to_char(int s)
{
	switch(s)
	{
	case STATUS_SUSCEPTIBLE:
		return 'S';
	case STATUS_INFECTED:
		return 'I';
	case STATUS_RECOVERED:
		return 'R';
	default:
		return '?';
	}
}

inline char * action_type_to_string(int action)
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

inline int lookup_school_typecode_from_age_code(int age_code)
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
//assumes array is big enough that this won't be pathological
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



__global__ void makeContactsKernel_weekday(int num_infected, int * infected_indexes, int * people_age,
										   int * household_lookup, int * household_offsets, int * household_people,
										   int * workplace_max_contacts, int * workplace_lookup, 
										   int * workplace_offsets, int * workplace_people,
										   int * errand_contacts_desired, int * errand_infected_locs,
										   int * errand_loc_offsets, int * errand_people,
										   int number_locations, 
										   int * output_infector_arr, int * output_victim_arr, int * output_kval_arr,
										   kval_t * output_kval_sum_arr, int rand_offset, int number_people)

{
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		int output_offset_base = MAX_CONTACTS_WEEKDAY * myPos;

		int myIdx = infected_indexes[myPos];
		int myAge = people_age[myIdx];

		threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
		union{
			threefry2x64_ctr_t c[2];
			unsigned int i[8];
		} rand_union;
		//generate first set of random numbers

		threefry2x64_ctr_t tf_ctr_1 = {{(long) ((myPos * 2) + rand_offset), (long) ((myPos * 2) + rand_offset)}};
		rand_union.c[0] = threefry2x64(tf_ctr_1, tf_k);

		kval_t household_kval_sum = 0;
		{
			int loc_offset, loc_count;

			//household: make three contacts
			device_lookupLocationData_singleHour(myIdx, household_lookup, household_offsets, &loc_offset, &loc_count);  //lookup location data for household
			device_selectRandomPersonFromLocation(
				myIdx, loc_offset, loc_count,rand_union.i[0], CONTACT_TYPE_HOME,
				household_people,
				output_infector_arr + output_offset_base + 0,
				output_victim_arr + output_offset_base + 0,
				output_kval_arr + output_offset_base + 0,
				&household_kval_sum);
			device_selectRandomPersonFromLocation(
				myIdx, loc_offset, loc_count,rand_union.i[1], CONTACT_TYPE_HOME,
				household_people,
				output_infector_arr + output_offset_base + 1,
				output_victim_arr + output_offset_base + 1,
				output_kval_arr + output_offset_base + 1,
				&household_kval_sum);
			device_selectRandomPersonFromLocation(
				myIdx, loc_offset, loc_count,rand_union.i[2], CONTACT_TYPE_HOME,
				household_people,
				output_infector_arr + output_offset_base + 2,
				output_victim_arr + output_offset_base + 2,
				output_kval_arr + output_offset_base + 2,
				&household_kval_sum);			
		}

		//generate the second set of random numbers
		threefry2x64_ctr_t tf_ctr_2 = {{(long) ((myPos * 2) + rand_offset + 1), (long) ((myPos * 2) + rand_offset + 1)}};
		rand_union.c[1] = threefry2x64(tf_ctr_2, tf_k);

		//now the number of contacts made will diverge, so we need to count it
		int contacts_made = 3;
		kval_t workplace_kval_sum = 0;
		{
			int contacts_desired, loc_offset, loc_count, kval_type;
			int local_contacts_made = contacts_made;			//this will let both loops interleave

			//look up max_contacts into contacts_desired
			device_lookupLocationData_singleHour(
				myIdx, workplace_lookup,workplace_offsets, workplace_max_contacts,	//input
				&loc_offset, &loc_count, &contacts_desired);	
			contacts_made += contacts_desired;

			if(myAge == AGE_ADULT)
				kval_type = CONTACT_TYPE_WORKPLACE;
			else
				kval_type = CONTACT_TYPE_SCHOOL;

			while(contacts_desired > 0 && local_contacts_made < MAX_CONTACTS_WEEKDAY)
			{
				int output_offset = output_offset_base + local_contacts_made;
				device_selectRandomPersonFromLocation(
					myIdx,loc_offset, loc_count, rand_union.i[local_contacts_made], kval_type,
					workplace_people,
					output_infector_arr + output_offset,
					output_victim_arr + output_offset,
					output_kval_arr + output_offset,
					&workplace_kval_sum);

				contacts_desired--;
				local_contacts_made++;
			}
		}
		
		//do errands
		kval_t errand_kval_sum = 0;
		{
			
			int kval_type;

			//set kval for the errands
			if(myAge == AGE_ADULT)
				kval_type = CONTACT_TYPE_ERRAND;
			else
				kval_type = CONTACT_TYPE_AFTERSCHOOL;

			for(int hour = 0; hour < NUM_WEEKDAY_ERRAND_HOURS; hour++)
			{
				int contacts_desired, loc_offset, loc_count;

				//fish out location offset, count, and contacts desired
				device_lookupInfectedLocation_multiHour(
					myPos, hour, 
					errand_infected_locs, errand_loc_offsets, number_locations, number_people,
					errand_contacts_desired, NUM_WEEKDAY_ERRAND_HOURS,
					&loc_offset, &loc_count, &contacts_desired);
				
				//make contacts
				while(contacts_desired > 0 && contacts_made < MAX_CONTACTS_WEEKDAY)
				{
					int output_offset = output_offset_base + contacts_made;
					device_selectRandomPersonFromLocation(
						myIdx, loc_offset, loc_count, rand_union.i[contacts_made], kval_type,
						errand_people, 
						output_infector_arr + output_offset,
						output_victim_arr + output_offset,
						output_kval_arr + output_offset,
						&errand_kval_sum);

					contacts_desired--;
					contacts_made++;
				}
			}

			//if person has made less than max contacts, fill the end with null contacts
			while(contacts_made < MAX_CONTACTS_WEEKDAY)
			{
				int output_offset = output_offset_base + contacts_made;
				device_nullFillContact(myIdx,
					output_infector_arr + output_offset,
					output_victim_arr + output_offset,
					output_kval_arr + output_offset);
				contacts_made++;
			}

			output_kval_sum_arr[myPos] = household_kval_sum + workplace_kval_sum + errand_kval_sum;
		}
	}
}


__global__ void makeContactsKernel_weekend(int num_infected, int * infected_indexes,
										   int * household_lookup, int * household_offsets, int * household_people,
										   int * infected_errand_hours, int * infected_errand_destinations,
										   int * infected_errand_contacts_profile,
										   int * errand_loc_offsets, int * errand_people,
										   int * errand_populationCount_exclusiveScan,
										   int number_locations, 
										   int * output_infector_arr, int * output_victim_arr, int * output_kval_arr,
										   kval_t * output_kval_sum_arr, int rand_offset)
{
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		int output_offset_base = MAX_CONTACTS_WEEKEND * myPos;

		int myIdx = infected_indexes[myPos];


		threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
		union{
			threefry2x64_ctr_t c;
			unsigned int i[4];
		} rand_union;
		//generate first set of random numbers

		threefry2x64_ctr_t tf_ctr_1 = {{(long) ((myPos * 2) + rand_offset), (long) ((myPos * 2) + rand_offset)}};
		rand_union.c = threefry2x64(tf_ctr_1, tf_k);

		//household: make three contacts
		kval_t household_kval_sum = 0;
		{
			int loc_offset, loc_count;
			device_lookupLocationData_singleHour(myIdx, household_lookup, household_offsets, &loc_offset, &loc_count);  //lookup location data for household
			device_selectRandomPersonFromLocation(
				myIdx, loc_offset, loc_count,rand_union.i[0], CONTACT_TYPE_HOME,
				household_people,
				output_infector_arr + output_offset_base + 0,
				output_victim_arr + output_offset_base + 0,
				output_kval_arr + output_offset_base + 0, 
				&household_kval_sum);
			device_selectRandomPersonFromLocation(
				myIdx, loc_offset, loc_count,rand_union.i[1], CONTACT_TYPE_HOME,
				household_people,
				output_infector_arr + output_offset_base + 1,
				output_victim_arr + output_offset_base + 1,
				output_kval_arr + output_offset_base + 1, 
				&household_kval_sum);
			device_selectRandomPersonFromLocation(
				myIdx, loc_offset, loc_count,rand_union.i[2], CONTACT_TYPE_HOME,
				household_people,
				output_infector_arr + output_offset_base + 2,
				output_victim_arr + output_offset_base + 2,
				output_kval_arr + output_offset_base + 2, 
				&household_kval_sum);
		}

		//we need two more random numbers for the errands
		threefry2x32_key_t tf_k_32 = {{ SEED_DEVICE[0], SEED_DEVICE[1]}};
		threefry2x32_ctr_t tf_ctr_32 = {{((myPos * 2) + rand_offset + 1),((myPos * 2) + rand_offset + 1)}};		
		union{
			threefry2x32_ctr_t c;
			unsigned int i[2];
		} rand_union_32;
		rand_union_32.c = threefry2x32(tf_ctr_32, tf_k_32);

		kval_t errand_kval_sum = 0;
		int contacts_profile = infected_errand_contacts_profile[myPos];

		{
			int loc_offset, loc_count;
			int errand_slot = WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[contacts_profile][0]; //the errand number the contact will be made in

			device_lookupLocationData_weekendErrand(		//lookup the location data for this errand: we just need the offset and count
				myPos, errand_slot, 
				infected_errand_hours, infected_errand_destinations, 
				errand_loc_offsets, number_locations, 
				errand_populationCount_exclusiveScan, 
				&loc_offset, &loc_count);
			device_selectRandomPersonFromLocation(			//select a random person at the location
				myIdx, loc_offset, loc_count, rand_union_32.i[0], CONTACT_TYPE_ERRAND,
				errand_people,
				output_infector_arr + output_offset_base + 3,
				output_victim_arr + output_offset_base + 3,
				output_kval_arr + output_offset_base + 3,
				&errand_kval_sum);
		}
		{
			//do it again for the second errand contact
			int loc_offset, loc_count;
			int errand_slot = WEEKEND_ERRAND_CONTACT_ASSIGNMENTS_DEVICE[contacts_profile][1];		
			device_lookupLocationData_weekendErrand(			//lookup the location data for this errand
				myPos, errand_slot, 
				infected_errand_hours, infected_errand_destinations, 
				errand_loc_offsets, number_locations, 
				errand_populationCount_exclusiveScan, 
				&loc_offset, &loc_count);
			device_selectRandomPersonFromLocation(			//select a random person at the location
				myIdx, loc_offset, loc_count, rand_union_32.i[1], CONTACT_TYPE_ERRAND,
				errand_people,
				output_infector_arr + output_offset_base + 4,
				output_victim_arr + output_offset_base + 4,
				output_kval_arr + output_offset_base + 4,
				&errand_kval_sum);
		}

		output_kval_sum_arr[myPos] = household_kval_sum + errand_kval_sum;
	}
}

/// <summary> given an index, look up the location and fetch the offset/count data from the memory array </summary>
/// <param name="myIdx">Input: Index of the infector to look up</param>
/// <param name="lookup_arr">Input: Pointer to an array containing all infector locations</param>
/// <param name="loc_offset_arr">Input: Pointer to an array containing location offsets</param>
/// <param name="loc_offset">Output value: offset to first person in infector's location</param>
/// <param name="loc_count">Output value: number of people at infector's location</param>
__device__ void device_lookupLocationData_singleHour(int myIdx, int * lookup_arr, int * loc_offset_arr, int * loc_offset, int * loc_count)
{
	int myLoc = lookup_arr[myIdx];

	//NOTE: these arrays have the final number_locs+1 value set, so we do not need to do the trick for the last location
	(*loc_offset) = loc_offset_arr[myLoc];
	(*loc_count) = loc_offset_arr[myLoc + 1] - loc_offset_arr[myLoc];
}

/// <summary> given an index, look up the location and fetch the offset/count/max_contacts values from the memory array </summary>
/// <param name="myIdx">Input: Index of the infector to look up</param>
/// <param name="lookup_arr">Input: Pointer to an array containing all infector locations</param>
/// <param name="loc_offset_arr">Input: Pointer to an array containing a location offsets</param>
/// <param name="loc_max_contacts_arr">Input: pointer to an array containing max_contact values</param>
/// <param name="loc_offset">Output: offset to first person in infector's location</param>
/// <param name="loc_count">Output: number of people at infector's location</param>
/// <param name="loc_max_contacts">Output: max_contacts for infector's location</param>
__device__ void device_lookupLocationData_singleHour(int myIdx, int * lookup_arr, int * loc_offset_arr, int * loc_max_contacts_arr, int * loc_offset, int * loc_count, int * loc_max_contacts)
{
	int myLoc = lookup_arr[myIdx];

	//NOTE: these arrays have the final number_locs+1 value set, so we do not need to do the trick for the last location
	(*loc_offset) = loc_offset_arr[myLoc];
	(*loc_count) = loc_offset_arr[myLoc + 1] - loc_offset_arr[myLoc];
	(*loc_max_contacts) = loc_max_contacts_arr[myLoc];
}

/// <summary> Look up the location information for an infected person for weekend errands </summary>
/// <param name="myPos">Input: Which of the N infected individuals we are working with, 0 <= myPos <= infected_count</param>
/// <param name="errand_slot">Input: Infected go on three errands, this is which of the three the contact is for </param>
/// <param name="infected_hour_val_arr">Input:Array containing hour numbers that infected will go on errands in</param>
/// <param name="infected_hour_destination_arr">Input: Array containing the location number the errands are to</param>
__device__ void device_lookupLocationData_weekendErrand(int myPos, int errand_slot, int * infected_hour_val_arr, int * infected_hour_destination_arr, int * loc_offset_arr, int number_locations, int * hour_populationCount_exclusiveScan, int * output_location_offset, int * output_location_count)
{
	//this code is overall very similar to the multi-hour code for weekday, but modified to handle variable numbers
	//of people per hour (since errands are randomly generated between 10 hours)

	int hour_data_position = (myPos * NUM_WEEKEND_ERRANDS) + errand_slot;

	int hour = infected_hour_val_arr[hour_data_position];			//which hour the errand will be made on
	int myLoc = infected_hour_destination_arr[hour_data_position];	//destination of the errand

	//location offsets are stored in collated format, eg for 3 locations and 2 hours:
	// 1 2 3 1 2 3
	int location_offset_position = (hour * number_locations) + myLoc;

	int loc_offset = loc_offset_arr[location_offset_position];
	int next_loc_offset;
	
	//next_loc_offset is normally loc_offset_arr[loc_offset_pos + 1] but the last location is a special case
	if(myLoc == number_locations - 1)
	{
		//next_loc_offset = number of people present this hour
		int number_people_thisHour = hour_populationCount_exclusiveScan[hour + 1] - hour_populationCount_exclusiveScan[hour];
		next_loc_offset = number_people_thisHour;
	}
	else
		next_loc_offset = loc_offset_arr[location_offset_position + 1];

	(*output_location_count) = next_loc_offset - loc_offset;

	//the hourly binary searches are only the offset within the hour, so we need to add the offset to the first person for this hour
	loc_offset += hour_populationCount_exclusiveScan[hour];
	(*output_location_offset) = loc_offset;
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
__device__ void device_lookupInfectedLocation_multiHour(int myPos, int hour, int * infected_loc_arr, int * loc_offset_arr, int number_locations, int number_people, int * contacts_desired_lookup, int number_hours, int * output_loc_offset, int * output_loc_count, int * output_contacts_desired)
{
	//infected locations and contacts_desired are stored packed, eg for infected_idx 1,2
	// 1 1 2 2

	int infected_loc_offset = (number_hours * myPos) + hour;	//position of this person's location within the infected_location array
	int myLoc = infected_loc_arr[infected_loc_offset];		//which of the 1300 locations this person is at for this hour

	*output_contacts_desired = contacts_desired_lookup[infected_loc_offset];	//output the number of contacts this person will make this hour


	//location offsets are stored in collated format, eg for locations 1 2 3
	// 1 2 3 1 2 3

	int loc_offset_position = (hour * number_locations) + myLoc;	//position of the location's offset within the multi-hour offset array

	int loc_o = loc_offset_arr[loc_offset_position];
	int next_loc_o;		//stores loc_offset_arr[loc_offset_position + 1]

	//hack: next_loc_o normally gets loc_offset_arr[loc_offset_pos + 1], but this array is not set up with an extra slot for the last location
	//therefore, if we are at the last location, we need to fudge this value
	if(myLoc == number_locations - 1)
		next_loc_o = number_people;
	else
		next_loc_o = loc_offset_arr[loc_offset_position + 1];

	*output_loc_count = next_loc_o - loc_o;	//calculate the number of people at this location

	//hack: the binary search is done on a per-hour basis, so we need to offset to the first person of this hour
	loc_o += (hour * number_people);
	*output_loc_offset = loc_o;
}


__device__ void device_selectRandomPersonFromLocation(int infector_idx, int loc_offset, int loc_count, unsigned int rand_val, int desired_kval, int * location_people_arr, int * output_infector_idx_arr, int * output_victim_idx_arr, int * output_kval_arr, kval_t * output_kval_sum)
{
	//start with null data
	int victim_idx = NULL_PERSON_INDEX;
	int contact_type = CONTACT_TYPE_NONE;

	//if there is only one person, keep the null data, else select one other person who is not our infector
	if(loc_count > 1)
	{
		int victim_offset = rand_val % loc_count;	//select a random person between 0 and loc_count
		victim_idx = location_people_arr[loc_offset + victim_offset];	//get the index

		//if we have selected the infector, we need to get a different person
		if(victim_idx == infector_idx)
		{
			//get the next person
			victim_offset = victim_offset + 1;
			if(victim_offset == loc_count)		//wrap around if needed
				victim_offset = 0;
			victim_idx = location_people_arr[loc_offset + victim_offset];
		}

		contact_type = desired_kval;
	}

	//write data into output memory locations
	(*output_infector_idx_arr) = infector_idx;
	(*output_victim_idx_arr) = victim_idx;
	(*output_kval_arr) = contact_type;

	//increment the kval sum by the kval of this contact type
	if(contact_type != CONTACT_TYPE_NONE)
		*output_kval_sum += KVAL_LOOKUP_DEVICE[contact_type];
}

//write a null contact to the memory locations
__device__ void device_nullFillContact(int myIdx, int * output_infector_idx, int * output_victim_idx, int * output_kval)
{
	(*output_infector_idx) = myIdx;
	(*output_victim_idx) = NULL_PERSON_INDEX;
	(*output_kval) = CONTACT_TYPE_NONE;
}

__device__ void device_lookupInfectedErrand_weekend(int myPos, int hour_slot,
													int * inf_hour_arr, int * inf_location_arr, 
													int * output_hour, int * output_location)
{
	int offset = (myPos * NUM_WEEKEND_ERRANDS) + hour_slot;

	*output_hour = inf_hour_arr[offset];
	*output_location = inf_location_arr[offset];
}


__global__ void kernel_assignAfterschoolLocations_wholeDay(int * child_indexes_arr, int * output_array, int number_children, int number_people, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} u;

	//get the number of afterschool locations and their offset in the business array
	int afterschool_count = WORKPLACE_TYPE_COUNT_DEVICE[BUSINESS_TYPE_AFTERSCHOOL];
	int afterschool_offset = WORKPLACE_TYPE_OFFSET_DEVICE[BUSINESS_TYPE_AFTERSCHOOL];

	//for each child
	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= number_children / 4; myGridPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = myGridPos + rand_offset;
		threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
		u.c = threefry2x64(tf_ctr, tf_k);

		int myPos = myGridPos * 4;
		if(myPos < number_children)
		{
			int myIdx = child_indexes_arr[myPos];
			device_fishAfterschoolLocation(&u.i[0], number_people, afterschool_count, afterschool_offset, output_array + myIdx);
		}

		if(myPos + 1 < number_children)
		{
			int myIdx_1 = child_indexes_arr[myPos + 1];
			device_fishAfterschoolLocation(&u.i[1],number_people, afterschool_count, afterschool_offset, output_array + myIdx_1);
		}
		if(myPos + 2 < number_children)
		{
			int myIdx_2 = child_indexes_arr[myPos + 2];
			device_fishAfterschoolLocation(&u.i[2],number_people, afterschool_count, afterschool_offset,output_array + myIdx_2);
		}
		if(myPos + 3 < number_children)
		{
			int myIdx_3 = child_indexes_arr[myPos + 3];
			device_fishAfterschoolLocation(&u.i[3], number_people, afterschool_count, afterschool_offset, output_array + myIdx_3);
		}
	}
}

__device__ void device_fishAfterschoolLocation(unsigned int * rand_val, int number_people, int afterschool_count, int afterschool_offset, int * output_schedule)
{
	//turn random number into fraction between 0 and 1
	float frac = (float) *rand_val / UNSIGNED_MAX;

	int business_num = frac * afterschool_count;		//find which afterschool location they're at, between 0 <= X < count
	
	if(business_num >= afterschool_count)
		business_num = afterschool_count - 1;

	business_num = business_num + afterschool_offset;		//add the offset to the first afterschool location

	*output_schedule = business_num;					//store in the indicated output location
	*(output_schedule + number_people) = business_num;	//children go to the same location for both hours, so put it in their second errand slot
}


__global__ void kernel_assignErrandLocations_weekday_wholeDay(int * adult_indexes_arr, int number_adults, int number_people, int * output_arr, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} u;

	//for each adult
	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= number_adults / 2; myGridPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = myGridPos + rand_offset;
		threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
		u.c = threefry2x64(tf_ctr, tf_k);

		int myPos = myGridPos * 2;

		//fish out a destination
		if(myPos < number_adults)
		{
			int myAdultIdx_1 = adult_indexes_arr[myPos];
			device_fishWeekdayErrandDestination(&u.i[0], &output_arr[myAdultIdx_1]);	//for adult index i, output the destination to arr[i]
			device_fishWeekdayErrandDestination(&u.i[1], &output_arr[myAdultIdx_1 + number_people]);	//output a second destination to arr[i] for the second hour
		}
		//if still in bounds, assign another person
		if(myPos + 1 < number_adults)
		{
			int myAdultIdx_2 = adult_indexes_arr[myPos + 1];
			device_fishWeekdayErrandDestination(&u.i[2], &output_arr[myAdultIdx_2]);
			device_fishWeekdayErrandDestination(&u.i[3], &output_arr[myAdultIdx_2 + number_people]);
		}
	}
}


__device__ void device_fishWeekdayErrandDestination(unsigned int * rand_val, int * output_destination)
{
	float yval = (float) *rand_val / UNSIGNED_MAX;

	int row = FIRST_WEEKDAY_ERRAND_ROW; //which business type

	while(yval > WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_DEVICE[row] && row < (NUM_BUSINESS_TYPES - 1))
	{
		yval -= WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_DEVICE[row];
		row++;
	}

	//figure out which business of this type we're at
	float frac = yval / WORKPLACE_TYPE_WEEKDAY_ERRAND_PDF_DEVICE[row];
	int type_count = WORKPLACE_TYPE_COUNT_DEVICE[row];
	int business_num = frac * type_count;

	if(business_num >= type_count)
		business_num = type_count - 1;

	//add the offset to the first business of this type 
	int type_offset = WORKPLACE_TYPE_OFFSET_DEVICE[row];

	*output_destination = business_num + type_offset;
}


inline const char * lookup_contact_type(int contact_type)
{
	switch(contact_type)
	{
	case 0:
		return "CONTACT_TYPE_NONE";
	case 1:
		return "CONTACT_TYPE_WORKPLACE";
	case 2:
		return "CONTACT_TYPE_SCHOOL";
	case 3:
		return "CONTACT_TYPE_ERRAND";
	case 4:
		return "CONTACT_TYPE_AFTERSCHOOL";
	case 5:
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

const char * lookup_age_type(int age_type)
{
	switch(age_type)
	{
	case 0:
		return "AGE_5";
	case 1:
		return "AGE_9";
	case 2:
		return "AGE_14";
	case 3:
		return "AGE_17";
	case 4:
		return "AGE_22";
	case 5:
		return "AGE_ADULT";
	default:
		return "INVALID AGE CODE";
	}
}

__global__ void kernel_assignErrands_weekend(int * people_indexes_arr, int * errand_hours_arr, int * errand_destination_arr, int num_people, randOffset_t rand_offset)
{
	const int RAND_COUNTS_CONSUMED = 2;	//one for hours, one for destinations

	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_people; myPos += gridDim.x * blockDim.x)
	{
		int offset = myPos * NUM_WEEKEND_ERRANDS;
		randOffset_t myRandOffset = rand_offset + (myPos * RAND_COUNTS_CONSUMED);
		
		device_copyPeopleIndexes_weekend_wholeDay(people_indexes_arr + offset, myPos);
		device_assignErrandHours_weekend_wholeDay(errand_hours_arr + offset, myRandOffset);
		device_assignErrandDestinations_weekend_wholeDay(errand_destination_arr + offset, myRandOffset + 1);
	}
}

__device__ void device_assignErrandDestinations_weekend_wholeDay(int * errand_destination_ptr, int my_rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	threefry2x64_ctr_t tf_ctr = {{((long)my_rand_offset), ((long) my_rand_offset)}};
	rand_union.c = threefry2x64(tf_ctr, tf_k);

	device_fishWeekendErrandDestination(&rand_union.i[0], errand_destination_ptr);
	device_fishWeekendErrandDestination(&rand_union.i[1], errand_destination_ptr+1);
	device_fishWeekendErrandDestination(&rand_union.i[2], errand_destination_ptr+2);
}

__global__ void kernel_doInfectedSetup_weekend(int * input_infected_indexes_ptr, int * input_errand_hours_ptr, int * input_errand_destinations_ptr,
											   int * output_infected_hour_ptr, int * output_infected_dest_ptr, int * output_contacts_desired_ptr,
											   int num_infected, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= num_infected / 4; myGridPos += gridDim.x * blockDim.x)
	{
		randOffset_t myRandOffset = rand_offset + myGridPos;
		threefry2x64_ctr_t tf_ctr = {{myRandOffset, myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr, tf_k);

		int myPos = myGridPos * 4;
		if(myPos < num_infected)
		{
			device_doAllInfectedSetup_weekend(&rand_union.i[0], 
				myPos, input_infected_indexes_ptr, 
				input_errand_hours_ptr, input_errand_destinations_ptr,
				output_infected_hour_ptr, output_infected_dest_ptr, output_contacts_desired_ptr);
		}

		if(myPos + 1 < num_infected)
		{
			device_doAllInfectedSetup_weekend(&rand_union.i[1], 
				myPos+1, input_infected_indexes_ptr, 
				input_errand_hours_ptr, input_errand_destinations_ptr,
				output_infected_hour_ptr, output_infected_dest_ptr, output_contacts_desired_ptr);
		}

		if(myPos + 2  < num_infected)
		{
			device_doAllInfectedSetup_weekend(&rand_union.i[2], 
				myPos+2, input_infected_indexes_ptr, 
				input_errand_hours_ptr, input_errand_destinations_ptr,
				output_infected_hour_ptr, output_infected_dest_ptr, output_contacts_desired_ptr);
		}

		if(myPos + 3 < num_infected)
		{
			device_doAllInfectedSetup_weekend(&rand_union.i[3], 
				myPos+3, input_infected_indexes_ptr, 
				input_errand_hours_ptr, input_errand_destinations_ptr,
				output_infected_hour_ptr, output_infected_dest_ptr, output_contacts_desired_ptr);
		}
	}
}

__device__ void device_copyInfectedErrandLocs_weekend(int * input_hours_ptr, int * input_dests_ptr, int * output_hours_ptr, int * output_dests_ptr)
{
	output_hours_ptr[0] = input_hours_ptr[0];
	output_hours_ptr[1] = input_hours_ptr[1];
	output_hours_ptr[2] = input_hours_ptr[2];

	output_dests_ptr[0] = input_dests_ptr[0];
	output_dests_ptr[1] = input_dests_ptr[1];
	output_dests_ptr[2] = input_dests_ptr[2];
}

__device__ void device_doAllInfectedSetup_weekend(unsigned int * rand_val, int myPos, int * infected_indexes_arr, int * input_hours_arr, int * input_dests_arr, int * output_hours_arr, int * output_dests_arr, int * output_contacts_desired_arr)
{
	int myIdx = infected_indexes_arr[myPos];
	int input_offset = NUM_WEEKEND_ERRANDS * myIdx;
	int output_offset = NUM_WEEKEND_ERRANDS * myPos;

	device_copyInfectedErrandLocs_weekend(
		input_hours_arr + input_offset,
		input_dests_arr + input_offset,
		output_hours_arr + output_offset,
		output_dests_arr + output_offset);

	const int NUM_POSSIBLE_CONTACT_ASSIGNMENTS = 6;
	int profile = *rand_val % NUM_POSSIBLE_CONTACT_ASSIGNMENTS;
	output_contacts_desired_arr[myPos] = profile;
}


__global__ void kernel_countInfectedStatus(int * pandemic_status_array, int * seasonal_status_array, int num_people, int * output_pandemic_counts, int * output_seasonal_counts)
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

	//valid status condition codes are between -2 and 5 inclusive, get a pointer to where status 0 should go
	int * pandemic_pointer = &pandemic_reduction_array[tid][2];
	int * seasonal_pointer = &seasonal_reduction_array[tid][2];

	//count all statuses
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_people; myPos += gridDim.x * blockDim.x)
	{
		int status_pandemic = pandemic_status_array[myPos];
		pandemic_pointer[status_pandemic]++;
		int status_seasonal = seasonal_status_array[myPos];
		seasonal_pointer[status_seasonal]++;
	}
	__syncthreads();   //wait for all threads to finish, or reduction will hit a race condition
	

	//do reduction
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

	//thread 0 stores results
	if(tid == 0)
	{
		atomicAdd(output_pandemic_counts + 0, pandemic_reduction_array[0][0]);
		atomicAdd(output_pandemic_counts + 1, pandemic_reduction_array[0][1]);
		atomicAdd(output_pandemic_counts + 2, pandemic_reduction_array[0][2]);
		atomicAdd(output_pandemic_counts + 3, pandemic_reduction_array[0][3]);
		atomicAdd(output_pandemic_counts + 4, pandemic_reduction_array[0][4]);
		atomicAdd(output_pandemic_counts + 5, pandemic_reduction_array[0][5]);
		atomicAdd(output_pandemic_counts + 6, pandemic_reduction_array[0][6]);
		atomicAdd(output_pandemic_counts + 7, pandemic_reduction_array[0][7]);

		atomicAdd(output_seasonal_counts + 0, seasonal_reduction_array[0][0]);
		atomicAdd(output_seasonal_counts + 1, seasonal_reduction_array[0][1]);
		atomicAdd(output_seasonal_counts + 2, seasonal_reduction_array[0][2]);
		atomicAdd(output_seasonal_counts + 3, seasonal_reduction_array[0][3]);
		atomicAdd(output_seasonal_counts + 4, seasonal_reduction_array[0][4]);
		atomicAdd(output_seasonal_counts + 5, seasonal_reduction_array[0][5]);
		atomicAdd(output_seasonal_counts + 6, seasonal_reduction_array[0][6]);
		atomicAdd(output_seasonal_counts + 7, seasonal_reduction_array[0][7]);
	}
}

struct isInfectedPred
{
	__device__ bool operator() (thrust::tuple<int,int> status_tuple)
	{
		int status_seasonal = thrust::get<0>(status_tuple);
		int status_pandemic = thrust::get<1>(status_tuple);

		return status_pandemic >= 0 || status_seasonal >= 0;
	}
};

void PandemicSim::daily_buildInfectedArray_global()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "daily_buildInfectedArray_global");

	thrust::counting_iterator<int> count_it(0);
	IntIterator infected_indexes_end = thrust::copy_if(
		count_it, count_it + number_people,
		thrust::make_zip_iterator(thrust::make_tuple(
			people_status_pandemic.begin(), people_status_seasonal.begin())),
		infected_indexes.begin(),
		isInfectedPred());

	infected_count = infected_indexes_end - infected_indexes.begin();

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}


struct recoverInfected_pred
{
	int recover_infections_from_day;
	__device__ bool operator() (thrust::tuple<int,int> status_obj)
	{
		int status_type = thrust::get<0>(status_obj);

		//if there is no active infection, do not try to set recovered status
		if(status_type < 0)
			return false;

		//get the day this infection began
		int day_infection_began = thrust::get<1>(status_obj);
			
		//return true if it matches the day we're looking for, otherwise false
		return recover_infections_from_day == day_infection_began;
	}
};

void PandemicSim::daily_recoverInfected_new()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "daily_recoverInfected");

	int recover_day = (current_day + 1) - CULMINATION_PERIOD;
//	if(recover_day >= 0)
	if(1)
	{
		recoverInfected_pred recover_obj;
		recover_obj.recover_infections_from_day = recover_day;

			thrust::replace_if(
			thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes.begin()),
			thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes.begin() + infected_count),
			thrust::make_zip_iterator(thrust::make_tuple(
				thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes.begin()),
				thrust::make_permutation_iterator(people_days_pandemic.begin(), infected_indexes.begin()))),
			recover_obj,
			STATUS_RECOVERED);

		thrust::replace_if(
			thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin()),
			thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin() + infected_count),
			thrust::make_zip_iterator(thrust::make_tuple(
				thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin()),
				thrust::make_permutation_iterator(people_days_seasonal.begin(), infected_indexes.begin()))),
			recover_obj,
			STATUS_RECOVERED);
	}

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

void PandemicSim::final_countReproduction()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(-1,"final_countReproduction");

	thrust::sort(people_gens_pandemic.begin(), people_gens_pandemic.end());
	thrust::sort(people_gens_seasonal.begin(), people_gens_seasonal.end());

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

	//copy to host
	h_vec h_pandemic_gens = pandemic_gen_counts;
	h_vec h_seasonal_gens = seasonal_gen_counts;

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

	if(PROFILE_SIMULATION)
		profiler.endFunction(-1,number_people);
}

__device__ void device_checkActionAndWrite(bool infects_pandemic, bool infects_seasonal, int victim, int * pandemic_status_arr, int * seasonal_status_arr, int * dest_ptr)
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


__global__ void kernel_householdTypeAssignment(int * hh_type_array, int num_households, randOffset_t rand_offset)
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

__device__ int device_setup_fishHouseholdType(unsigned int rand_val)
{
	float y = (float) rand_val / UNSIGNED_MAX;

	int row = 0;
	while(y > HOUSEHOLD_TYPE_CDF_DEVICE[row] && row < HH_TABLE_ROWS - 1)
		row++;

	return row;
}



__device__ int device_setup_fishWorkplace(unsigned int rand_val)
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

	return business_num + type_offset;
}

__device__ void device_setup_fishSchoolAndAge(unsigned int rand_val, int * output_age_ptr, int * output_school_ptr)
{
	float y = (float) rand_val / RAND_MAX;

	//fish out age group and resulting school type from CDF
	int row = 0;
	while(row < CHILD_DATA_ROWS - 1 && y > CHILD_AGE_CDF_DEVICE[row])
		row++;

	int wp_type = CHILD_AGE_SCHOOLTYPE_LOOKUP_DEVICE[row];

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
	*output_age_ptr = row;
}


__global__ void kernel_generateHouseholds(
	int * hh_type_array, int * adult_exscan_arr, 
	int * child_exscan_arr, int num_households, 
	int * adult_index_arr, int * child_index_arr, 
	int * household_offset_arr,
	int * people_age_arr, int * people_households_arr, int * people_workplaces_arr, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	const int rand_counts_consumed = 2;

	for(int hh = blockIdx.x * blockDim.x + threadIdx.x;  hh < num_households ; hh += gridDim.x * blockDim.x)
	{
		int adults_offset = adult_exscan_arr[hh];
		int children_offset = child_exscan_arr[hh];

		int hh_type = hh_type_array[hh];
		int adults_count = HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[hh_type];
		int children_count = HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[hh_type];

		int hh_offset = adults_offset + children_offset;
		household_offset_arr[hh] = hh_offset;

		//get random numbers
		randOffset_t myRandOffset = rand_offset + (hh * rand_counts_consumed);
		threefry2x64_ctr_t tf_ctr_1 = {{myRandOffset, myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr_1, tf_k);

		for(int people_generated = 0; people_generated < adults_count; people_generated++)
		{
			int person_id = hh_offset + people_generated;
			people_households_arr[person_id] = hh;				//store the household number

			people_age_arr[person_id] = AGE_ADULT;					//mark as an adult
			people_workplaces_arr[person_id] = device_setup_fishWorkplace(rand_union.i[people_generated]);

			adult_index_arr[adults_offset + people_generated] = person_id; //store this ID in the adults index array
		}

		//get more random numbers
		threefry2x64_ctr_t tf_ctr_2 = {{myRandOffset + 1, myRandOffset + 1}};
		rand_union.c = threefry2x64(tf_ctr_2, tf_k);

		//increment the base ID number by the adults we just added
		hh_offset += adults_count;

		for(int people_generated = 0; people_generated < children_count; people_generated++)
		{
			int person_id = hh_offset + people_generated;
			people_households_arr[person_id] = hh;		//store the household number

			device_setup_fishSchoolAndAge(
				rand_union.i[people_generated],	
				people_age_arr + person_id,			//ptr into age_array
				people_workplaces_arr + person_id);		//ptr into workplace array

			child_index_arr[children_offset + people_generated] = person_id;	//store as a child
		}
	}
}


struct hh_adult_count_functor : public thrust::unary_function<int,int>
{
	__device__ int operator () (int hh_type) const
	{
		return HOUSEHOLD_TYPE_ADULT_COUNT_DEVICE[hh_type];
	}
};

struct hh_child_count_functor : public thrust::unary_function<int,int>
{
	__device__ int operator () (int hh_type) const
	{
		return HOUSEHOLD_TYPE_CHILD_COUNT_DEVICE[hh_type];
	}
};


//Sets up people's households and workplaces according to the probability functions
void PandemicSim::setup_generateHouseholds()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(-1,"setup_generateHouseholds");

	d_vec hh_types_array(number_households+1);
	int * hh_types_array_ptr = thrust::raw_pointer_cast(hh_types_array.data());

	//finish copydown of __constant__ sim data
	cudaDeviceSynchronize();

	//assign household types
	kernel_householdTypeAssignment<<<cuda_householdTypeAssignmentKernel_blocks,cuda_householdTypeAssignmentKernel_threads>>>(hh_types_array_ptr, number_households,rand_offset);
	cudaDeviceSynchronize();


	if(TIMING_BATCH_MODE == 0)
	{
		int rand_counts_consumed_1 = number_households / 4;
		rand_offset += rand_counts_consumed_1;
	}

	d_vec adult_count_exclScan(number_households+1);
	d_vec child_count_exclScan(number_households+1);

	//these count_functors convert household types into the number of children/adults in that type
	//use a transform-functor to convert the HH types and take an exclusive_scan of each
	//this will let us build the adult_index and child_index arrays
	thrust::exclusive_scan(
		thrust::make_transform_iterator(hh_types_array.begin(), hh_adult_count_functor()),
		thrust::make_transform_iterator(hh_types_array.end(), hh_adult_count_functor()),
		adult_count_exclScan.begin());
	thrust::exclusive_scan(
		thrust::make_transform_iterator(hh_types_array.begin(), hh_child_count_functor()),
		thrust::make_transform_iterator(hh_types_array.end(), hh_child_count_functor()),
		child_count_exclScan.begin());
	cudaDeviceSynchronize();
	
	/*
	h_vec h_hh_types = hh_types_array;
	h_vec h_child_count_exscan = child_count_exclScan;
	h_vec h_adult_count_exscan = adult_count_exclScan;
	FILE * ftemp = fopen("../households.txt","w");
	fprintf(ftemp,"i,hh_type,adult_exscan,child_exscan\n");
	for(int i = 0; i < number_households + 1; i++)
		fprintf(ftemp, "%d,%d,%d,%d\n", i, h_hh_types[i], h_adult_count_exscan[i],h_child_count_exscan[i]);
	fclose(ftemp);*/

	//the exclusive_scan of number_households+1 holds the total number of adult and children in the sim
	//(go one past the end to find the totals)
	number_adults = adult_count_exclScan[number_households];
	number_children = child_count_exclScan[number_households];
	number_people = number_adults + number_children;
	
	//now we can allocate the rest of our memory
	setup_sizeGlobalArrays();

	if(SIM_VALIDATION)
	{
		thrust::fill_n(people_ages.begin(), number_people, -1);
		thrust::fill_n(people_households.begin(), number_people, -1);
		thrust::fill_n(people_workplaces.begin(), number_people, -1);

		thrust::fill_n(household_offsets.begin(), number_people, -1);
	}

	int * adult_exscan_ptr = thrust::raw_pointer_cast(adult_count_exclScan.data());
	int * child_exscan_ptr = thrust::raw_pointer_cast(child_count_exclScan.data());

	//and then do the rest of the setup
	kernel_generateHouseholds<<<cuda_peopleGenerationKernel_blocks,cuda_peopleGenerationKernel_threads>>>(
		hh_types_array_ptr, adult_exscan_ptr, child_exscan_ptr, number_households,
		people_adults_indexes_ptr, people_child_indexes_ptr,
		household_offsets_ptr,
		people_ages_ptr, people_households_ptr, people_workplaces_ptr,
		rand_offset);
	if(TIMING_BATCH_MODE == 0)
	{
		const int rand_counts_consumed_2 = 2 * number_households;
		rand_offset += rand_counts_consumed_2;
	}

	thrust::sequence(household_people.begin(), household_people.begin() + number_people); //copy the ID numbers into the household_people table
	household_offsets[number_households] = number_people;  //put the last household_offset in position

	cudaDeviceSynchronize();

	if(PROFILE_SIMULATION)
	{
		profiler.endFunction(-1,number_people);
	}
}


struct filterContacts_pred
{
	__device__ bool operator() (thrust::tuple<int,int,int> action_tuple)
	{
		int action_type = thrust::get<0>(action_tuple);

		if(action_type == ACTION_INFECT_NONE)
			return true;

		return false;
	}
};

struct actionSortOp_new
{
	__device__
		bool operator () (thrust::tuple<int,int,int> a, thrust::tuple<int,int,int> b)
	{

		int victim_a = thrust::get<2>(a);
		int victim_b = thrust::get<2>(b);

		if(victim_a != victim_b)
		{
			return victim_a < victim_b;
		}

		int action_a = thrust::get<0>(a);
		int action_b = thrust::get<0>(b);

		return action_a > action_b;
	}
};


void PandemicSim::daily_filterActions_new()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day,"daily_filterActions");

	int num_possible_contacts = is_weekend() ? MAX_CONTACTS_WEEKEND * infected_count : MAX_CONTACTS_WEEKDAY * infected_count;

	ZipIntTripleIterator actions_begin = 
		thrust::make_zip_iterator(thrust::make_tuple(
			daily_action_type.begin(), 
			daily_contact_infectors.begin(), 
			daily_contact_victims.begin()));

	//compact - filter out null contacts
	filterContacts_pred contact_filter_obj;
	ZipIntTripleIterator actions_end = thrust::remove_if(
		actions_begin,
		thrust::make_zip_iterator(thrust::make_tuple(
			daily_action_type.begin() + num_possible_contacts, 
			daily_contact_infectors.begin() + num_possible_contacts, 
			daily_contact_victims.begin() + num_possible_contacts)),
		contact_filter_obj);

//	int size_a = actions_end - actions_begin;

	//sort - by victim_id ascending, then by action code descending
	thrust::sort(actions_begin, actions_end,actionSortOp_new());
	
	//unique - remove duplicate infection actions
	actions_end = thrust::unique(actions_begin,actions_end,uniqueActionOp());
	daily_actions = actions_end - actions_begin;

	if(CONSOLE_OUTPUT)
		printf("after filtering: %d actions remaining\n", daily_actions);

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

__global__ void kernel_contactsToActions(int * infected_idx_arr, kval_t * infected_kval_sum_arr, int infected_count,
										 int * contact_victims_arr, int *contact_type_arr, int contacts_per_infector,
										 int * people_day_pandemic_arr, int * people_day_seasonal_arr,
										 int * people_status_p_arr, int * people_status_s_arr,
										 int * output_action_arr,
										 float * rand_arr_1, float * rand_arr_2, float * rand_arr_3, float * rand_arr_4,
										 int current_day, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c[4];
		unsigned int i[16];
	} rand_union;

	const int rand_counts_consumed = 4;

	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < infected_count ; myPos += gridDim.x * blockDim.x)
	{
		int myIdx = infected_idx_arr[myPos];
		kval_t kval_sum = infected_kval_sum_arr[myPos];

//		if(kval_sum == 0)
//			continue;

		int status_p = people_status_p_arr[myIdx];
		int status_s = people_status_s_arr[myIdx];

		float inf_prob_p = -1.f;
		float inf_prob_s = -1.f;

		//int profile_day_p = -1;
		if(status_p >= 0)
		{
	//		int day_of_pandemic_infection = people_day_pandemic_arr[myIdx];
			int profile_day_p = current_day - people_day_pandemic_arr[myIdx];
			inf_prob_p = device_calculateInfectionProbability(status_p,profile_day_p, STRAIN_PANDEMIC,kval_sum);
		}
		//int profile_day_s = -1;
		if(status_s >= 0)
		{
			//int day_of_seasonal_infection = people_day_seasonal_arr[myIdx];
			int profile_day_s = current_day - people_day_seasonal_arr[myIdx];
			inf_prob_s = device_calculateInfectionProbability(status_s,profile_day_s, STRAIN_SEASONAL,kval_sum);
		}

		randOffset_t myRandOffset = rand_offset + (myPos * rand_counts_consumed);
		threefry2x64_ctr_t tf_ctr_1 = {{myRandOffset, myRandOffset}};
		rand_union.c[0] = threefry2x64(tf_ctr_1, tf_k);
		threefry2x64_ctr_t tf_ctr_2 = {{myRandOffset + 1, myRandOffset + 1}};
		rand_union.c[1] = threefry2x64(tf_ctr_2, tf_k);
		threefry2x64_ctr_t tf_ctr_3 = {{myRandOffset + 2, myRandOffset + 2}};
		rand_union.c[2] = threefry2x64(tf_ctr_3, tf_k);
		threefry2x64_ctr_t tf_ctr_4 = {{myRandOffset + 3, myRandOffset + 3}};
		rand_union.c[3] = threefry2x64(tf_ctr_4, tf_k);

		int contact_offset_base = contacts_per_infector * myPos;
		int rand_vals_used = 0;
		for(int contacts_processed = 0; contacts_processed < contacts_per_infector; contacts_processed++)
		{
			int contact_victim = contact_victims_arr[contact_offset_base + contacts_processed];
			int contact_type = contact_type_arr[contact_offset_base + contacts_processed];

			kval_t contact_kval = KVAL_LOOKUP_DEVICE[contact_type];

			float y_p = (float) rand_union.i[rand_vals_used++] / UNSIGNED_MAX;
			bool infects_p = y_p < (float) (inf_prob_p * contact_kval);

			float y_s = (float) rand_union.i[rand_vals_used++] / UNSIGNED_MAX;
			bool infects_s = y_s < (float) (inf_prob_s * contact_kval);

			//function handles parsing bools into an action and checking that victim is susceptible
			device_checkActionAndWrite(
				infects_p, infects_s, 
				contact_victim, 
				people_status_p_arr, people_status_s_arr,
				output_action_arr + contact_offset_base + contacts_processed);

			if(SIM_VALIDATION)
			{
				rand_arr_1[contact_offset_base + contacts_processed] = y_p;
				rand_arr_2[contact_offset_base + contacts_processed] = (float) (inf_prob_p * contact_kval);
				rand_arr_3[contact_offset_base + contacts_processed] = y_s;
				rand_arr_4[contact_offset_base + contacts_processed] = (float) (inf_prob_s * contact_kval);
			}
		}
	}
}

void PandemicSim::daily_contactsToActions_new()
{
	if(ACTION_INFECT_NONE != 0)
		throw new std::runtime_error(std::string("ACTION_INFECT_NONE must be zero for memset!"));

	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day,"daily_contactsToActions");

	int contacts_per_infector = is_weekend() ? MAX_CONTACTS_WEEKEND : MAX_CONTACTS_WEEKDAY;
	int total_contacts = contacts_per_infector * infected_count;

	kernel_contactsToActions<<<cuda_contactsToActionsKernel_blocks,cuda_contactsToActionsKernel_threads>>>(
		infected_indexes_ptr, infected_daily_kval_sum_ptr, infected_count,
		daily_contact_victims_ptr, daily_contact_kval_types_ptr, contacts_per_infector,
		people_days_pandemic_ptr, people_days_seasonal_ptr,
		people_status_pandemic_ptr, people_status_seasonal_ptr,
		daily_action_type_ptr,
		debug_contactsToActions_float1_ptr, debug_contactsToActions_float2_ptr,
		debug_contactsToActions_float3_ptr, debug_contactsToActions_float4_ptr,
		current_day, rand_offset);
	if(TIMING_BATCH_MODE == 0)
	{
		int rand_counts_consumed = 4 * infected_count;
		rand_offset += rand_counts_consumed;
	}
	cudaDeviceSynchronize();

	if(SIM_VALIDATION)
	{
		debug_validateActions();
	}

	if(CONSOLE_OUTPUT)
	{
		int successful_actions = thrust::count_if(daily_action_type.begin(), daily_action_type.begin() + total_contacts, actionIsSuccessful_pred());
		printf("before filtering: %d successful infection attempts\n",successful_actions);
	}

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

__device__ void device_assignProfile(unsigned int rand_val, int * output_status_ptr)
{
	/*
	//assign a profile between 0 and 2 inclusive
	int profile = rand_val % 3;

	//convert the rand to a float between 0 and 1
	float y = (float) rand_val / UNSIGNED_MAX;

	//if the symptomatic threshold is exceeded, make the profile asymptomatic
	if(y > PERCENT_SYMPTOMATIC_DEVICE)
		profile += 3;*/

	//*output_status_ptr = profile;
	*output_status_ptr = STATUS_INFECTED;
}

__device__ void device_doInfectionAction(
	unsigned int rand_val1, unsigned int rand_val2,
	int day_tomorrow,
	int action_type, int infector, int victim,
	int * people_status_p_arr, int * people_status_s_arr,
	int * people_gen_p_arr, int * people_gen_s_arr,
	int * people_day_p_arr, int * people_day_s_arr)
{
	if(action_type == ACTION_INFECT_BOTH || action_type == ACTION_INFECT_PANDEMIC)
	{
		//get infector's generation and increment for the victim
		int inf_gen_p = people_gen_p_arr[infector];
		people_gen_p_arr[victim] = inf_gen_p + 1;

		//mark tomorrow as their first day of infection
		people_day_p_arr[victim] = day_tomorrow;

		//assign them a profile
		device_assignProfile(rand_val1, people_status_p_arr + victim);
	}
	if(action_type == ACTION_INFECT_BOTH || action_type == ACTION_INFECT_SEASONAL)
	{
		//get infector's generation and increment for the victim
		int inf_gen_s = people_gen_s_arr[infector];
		people_gen_s_arr[victim] = inf_gen_s + 1;

		//mark tomorrow as their first day of infection
		people_day_s_arr[victim] = day_tomorrow;

		//assign them a profile
		device_assignProfile(rand_val2, people_status_s_arr + victim);
	}
}

__global__ void kernel_doInfectionActions(
	int * contact_action_arr, int * contact_victim_arr, int * contact_infector_arr,
	int action_count,
	int * people_status_p_arr, int * people_status_s_arr,
	int * people_gen_p_arr, int * people_gen_s_arr,
	int * people_day_p_arr, int * people_day_s_arr,
	int day_tomorrow, randOffset_t rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= action_count ; myGridPos += gridDim.x * blockDim.x)
	{
		int myPos = myGridPos * 2;

		//get random numbers
		randOffset_t myRandOffset = rand_offset + myGridPos;
		threefry2x64_ctr_t tf_ctr_1 = {{myRandOffset, myRandOffset}};
		rand_union.c = threefry2x64(tf_ctr_1, tf_k);

		if(myPos < action_count)
		{
			int action_type = contact_action_arr[myPos];
			int victim = contact_victim_arr[myPos];
			int infector = contact_infector_arr[myPos];

			device_doInfectionAction(
				rand_union.i[0],rand_union.i[1], 
				day_tomorrow,
				action_type, infector, victim,
				people_status_p_arr, people_status_s_arr,
				people_gen_p_arr,people_gen_s_arr,
				people_day_p_arr, people_day_s_arr);
		}
		if(myPos + 1 < action_count)
		{
			int action_type = contact_action_arr[myPos+1];
			int victim = contact_victim_arr[myPos+1];
			int infector = contact_infector_arr[myPos+1];

			device_doInfectionAction(
				rand_union.i[2],rand_union.i[3], 
				day_tomorrow,
				action_type, infector, victim,
				people_status_p_arr, people_status_s_arr,
				people_gen_p_arr,people_gen_s_arr,
				people_day_p_arr, people_day_s_arr);
		}
	}

}


void PandemicSim::daily_doInfectionActions()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "daily_doInfectionActions");

	kernel_doInfectionActions<<<cuda_doInfectionActionsKernel_blocks, cuda_doInfectionAtionsKernel_threads>>>(
		daily_action_type_ptr, daily_contact_victims_ptr, daily_contact_infectors_ptr,
		daily_actions,
		people_status_pandemic_ptr, people_status_seasonal_ptr,
		people_gens_pandemic_ptr, people_gens_seasonal_ptr,
		people_days_pandemic_ptr, people_days_seasonal_ptr,
		current_day + 1, rand_offset);

	if(TIMING_BATCH_MODE == 0)
	{
		int rand_counts_consumed = daily_actions / 2;
		rand_offset += rand_counts_consumed;
	}

	cudaDeviceSynchronize();

	if(PROFILE_SIMULATION)
	{
		profiler.endFunction(current_day, daily_actions);
	}
}


void PandemicSim::setup_fetchVectorPtrs()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(-1,"setup_fetchVectorPtrs");

	people_status_pandemic_ptr = thrust::raw_pointer_cast(people_status_pandemic.data());
	people_status_seasonal_ptr = thrust::raw_pointer_cast(people_status_seasonal.data());
	people_households_ptr = thrust::raw_pointer_cast(people_households.data());
	people_workplaces_ptr = thrust::raw_pointer_cast(people_workplaces.data());
	people_ages_ptr = thrust::raw_pointer_cast(people_ages.data());

	people_days_pandemic_ptr = thrust::raw_pointer_cast(people_days_pandemic.data());
	people_days_seasonal_ptr = thrust::raw_pointer_cast(people_days_seasonal.data());
	people_gens_pandemic_ptr = thrust::raw_pointer_cast(people_gens_pandemic.data());
	people_gens_seasonal_ptr = thrust::raw_pointer_cast(people_gens_seasonal.data());

	people_adults_indexes_ptr = thrust::raw_pointer_cast(people_adult_indexes.data());
	people_child_indexes_ptr = thrust::raw_pointer_cast(people_child_indexes.data());

	infected_indexes_ptr = thrust::raw_pointer_cast(infected_indexes.data());
	infected_daily_kval_sum_ptr = thrust::raw_pointer_cast(infected_daily_kval_sum.data());

	daily_contact_infectors_ptr = thrust::raw_pointer_cast(daily_contact_infectors.data());
	daily_contact_victims_ptr = thrust::raw_pointer_cast(daily_contact_victims.data());
	daily_contact_kval_types_ptr = thrust::raw_pointer_cast(daily_contact_kval_types.data());
	daily_action_type_ptr = thrust::raw_pointer_cast(daily_action_type.data());

	workplace_offsets_ptr = thrust::raw_pointer_cast(workplace_offsets.data());
	workplace_people_ptr = thrust::raw_pointer_cast(workplace_people.data());
	workplace_max_contacts_ptr = thrust::raw_pointer_cast(workplace_max_contacts.data());

	household_offsets_ptr = thrust::raw_pointer_cast(household_offsets.data());
	household_people_ptr = thrust::raw_pointer_cast(household_people.data());

	errand_people_table_ptr = thrust::raw_pointer_cast(errand_people_table.data());
	errand_people_weekendHours_ptr = thrust::raw_pointer_cast(errand_people_weekendHours.data());
	errand_people_destinations_ptr = thrust::raw_pointer_cast(errand_people_destinations.data());

	errand_infected_locations_ptr = thrust::raw_pointer_cast(errand_infected_locations.data());
	errand_infected_weekendHours_ptr = thrust::raw_pointer_cast(errand_infected_weekendHours.data());
	errand_infected_ContactsDesired_ptr = thrust::raw_pointer_cast(errand_infected_ContactsDesired.data());

	errand_locationOffsets_multiHour_ptr = thrust::raw_pointer_cast(errand_locationOffsets_multiHour.data());
	errand_hourOffsets_weekend_ptr = thrust::raw_pointer_cast(errand_hourOffsets_weekend.data());

	status_counts_dev_ptr = thrust::raw_pointer_cast(status_counts.data());

	if(SIM_VALIDATION)
	{
		debug_contactsToActions_float1_ptr = thrust::raw_pointer_cast(debug_contactsToActions_float1.data());
		debug_contactsToActions_float2_ptr = thrust::raw_pointer_cast(debug_contactsToActions_float2.data());
		debug_contactsToActions_float3_ptr = thrust::raw_pointer_cast(debug_contactsToActions_float3.data());
		debug_contactsToActions_float4_ptr = thrust::raw_pointer_cast(debug_contactsToActions_float4.data());
	}

	if(PROFILE_SIMULATION)
	{
		cudaDeviceSynchronize();
		profiler.endFunction(-1,1);
	}
}

void PandemicSim::daily_clearActionsArray()
{
	int size_to_clear = is_weekend() ? MAX_CONTACTS_WEEKEND * infected_count : MAX_CONTACTS_WEEKDAY * infected_count;
	cudaMemsetAsync(daily_action_type_ptr, 0, sizeof(int) * size_to_clear,stream_secondary);
}


void PandemicSim::daily_countInfectedStats()
{
	//get pointers
	int * pandemic_counts_ptr = status_counts_dev_ptr;
	int * seasonal_counts_ptr = pandemic_counts_ptr + 8;

	//memset to 0
	cudaMemsetAsync(pandemic_counts_ptr, 0, sizeof(int) * 16,stream_secondary);

	size_t dynamic_smemsize = 0;
	///	kernel_countInfectedStatus<<<COUNTING_GRID_BLOCKS, COUNTING_GRID_THREADS,smemsize, stream_countInfectedStatus>>>(
	kernel_countInfectedStatus<<<COUNTING_GRID_BLOCKS, COUNTING_GRID_THREADS, dynamic_smemsize, stream_secondary>>>(
		people_status_pandemic_ptr, people_status_seasonal_ptr, 
		number_people, 
		pandemic_counts_ptr, seasonal_counts_ptr);

	cudaMemcpyAsync(&status_counts_today, pandemic_counts_ptr,sizeof(int) * 16,cudaMemcpyDeviceToHost,stream_secondary);
}

void PandemicSim::daily_writeInfectedStats()
{
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
}

void PandemicSim::setup_calculateInfectionData()
{
	//adjust the asymptomatic profiles downwards
	for(int i = 3; i < NUM_PROFILES; i++)
		for(int j = 0; j < CULMINATION_PERIOD; j++)
			VIRAL_SHEDDING_PROFILES_HOST[i][j] *= asymp_factor;

	//calculate reproduction factors
	for(int i = 0; i < STRAIN_COUNT; i++)
	{
		INFECTIOUSNESS_FACTOR_HOST[i] = BASE_REPRODUCTION_HOST[i] / ((1.0f - asymp_factor) * PERCENT_SYMPTOMATIC_HOST);
	}
}
/*
struct memReadFunctor_int
{
	int * memPtr;
	__device__ int operator() (int offset)
	{
		return memPtr[offset];
	}
};*/

struct memReadFunctor_float
{
	__device__ float operator () (int offset1, int offset2)
	{
		return VIRAL_SHEDDING_PROFILES_DEVICE[offset1][offset2];
	}
};


void PandemicSim::debug_helper()
{
	int elements = NUM_PROFILES * CULMINATION_PERIOD;
	thrust::device_vector<float> d_profiles(elements);
//	thrust::copy_n(VIRAL_SHEDDING_PROFILES_DEVICE,elements,d_profiles.begin());

	int profile =2;
	thrust::counting_iterator<int> count_it(0);
	thrust::constant_iterator<int> const_it(profile);
	memReadFunctor_float memrdObj;
	thrust::transform(const_it, const_it+10, count_it, d_profiles.begin(), memrdObj);

	thrust::host_vector<float> h_profiles = d_profiles;

	FILE * fprofiledata = fopen("../profile_data.csv","w");
	fprintf(fprofiledata,"profile,day,val\n");
		for(int day = 0; day < CULMINATION_PERIOD; day++)
		{
			int idx = (profile * CULMINATION_PERIOD) + day;
			fprintf(fprofiledata,"%d,%d,%f\n",profile,day, h_profiles[idx]);
		}
	
	fclose(fprofiledata);
}


void PandemicSim::setup_loadSeed()
{
	int core_seed;

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

	cuda_makeWeekdayContactsKernel_blocks = cuda_blocks;
	cuda_makeWeekdayContactsKernel_threads = cuda_threads;

	cuda_makeWeekendContactsKernel_blocks = cuda_blocks;
	cuda_makeWeekendContactsKernel_threads = cuda_threads;

	cuda_contactsToActionsKernel_blocks = cuda_blocks;
	cuda_contactsToActionsKernel_threads = cuda_threads;

	cuda_doInfectionActionsKernel_blocks = cuda_blocks;
	cuda_doInfectionAtionsKernel_threads = cuda_threads;
}
