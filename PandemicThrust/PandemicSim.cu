#include "stdafx.h"

#include "simParameters.h"
#include "profiler.h"

#include "PandemicSim.h"
#include "thrust_functors.h"



#define USING_ERRAND_ID_ARRAYS 1

#pragma region settings

#define debug_null_fill_daily_arrays 1

//Simulation profiling master control - low performance overhead
const int PROFILE_SIMULATION = 1;

//output status messages to console?  Slows things down
#define CONSOLE_OUTPUT 1

//controls master logging - everything except for profiler
#define GLOBAL_LOGGING 1
#define SANITY_CHECK 1

#define print_infected_info 0
#define log_infected_info GLOBAL_LOGGING

#define print_location_info 0
#define log_location_info 0

//#define print_contact_kernel_setup 0
//#define log_contact_kernel_setup GLOBAL_LOGGING

#define dump_contact_kernel_random_data 0

#define print_contacts 0
#define log_contacts GLOBAL_LOGGING

#define print_actions 0
#define log_actions GLOBAL_LOGGING

#define print_actions_filtered 0
#define log_actions_filtered GLOBAL_LOGGING

#define log_people_info GLOBAL_LOGGING

//low overhead
#define debug_log_function_calls 1


#pragma endregion settings

int cuda_blocks = 32;
int cuda_threads = 256;



FILE * fDebug;

int SEED_HOST[SEED_LENGTH];
__device__ __constant__ int SEED_DEVICE[SEED_LENGTH];

__device__ __constant__ int business_type_count[NUM_BUSINESS_TYPES];				//stores number of each type of business
__device__ __constant__ int business_type_count_offset[NUM_BUSINESS_TYPES];			//stores location number of first business of this type
__device__ __constant__ float weekday_errand_pdf[NUM_BUSINESS_TYPES];				//stores PDF for weekday errand destinations
__device__ __constant__ float weekend_errand_pdf[NUM_BUSINESS_TYPES];				//stores PDF for weekend errand destinations
__device__ __constant__ float infectiousness_profile[CULMINATION_PERIOD];			//stores viral shedding profiles

__device__ __constant__ kval_t kval_lookup[NUM_CONTACT_TYPES];
__device__ __constant__ int weekday_people_exclusiveScan[NUM_WEEKDAY_ERRAND_HOURS + 1];

__device__ __constant__ int business_type_max_contacts[NUM_BUSINESS_TYPES];

__device__ __constant__ int weekend_errand_contact_assignments_wholeDay[6][2];
__device__ __constant__ int weekday_errand_contact_assignments_wholeDay[3][2];

//__device__ __constant__ float infectiousness_profiles_all[6][CULMINATION_PERIOD];

#define STRAIN_COUNT 2
__device__ __constant__ float BASE_REPRODUCTION_DEVICE[STRAIN_COUNT];
float BASE_REPRODUCTION_HOST[STRAIN_COUNT];


#define BASE_R_PANDEMIC_DEVICE BASE_REPRODUCTION_DEVICE[0]
#define BASE_R_SEASONAL_DEVICE BASE_REPRODUCTION_DEVICE[1]
#define BASE_R_PANDEMIC_HOST BASE_REPRODUCTION_HOST[0]
#define BASE_R_SEASONAL_HOST BASE_REPRODUCTION_HOST[1]

#define UNSIGNED_MAX (unsigned int) -1

float workplace_type_pdf[NUM_BUSINESS_TYPES];
int h_workplace_type_offset[NUM_BUSINESS_TYPES];
int h_workplace_type_counts[NUM_BUSINESS_TYPES];
int h_workplace_max_contacts[NUM_BUSINESS_TYPES];

float h_weekday_errand_pdf[NUM_BUSINESS_TYPES];
float h_weekend_errand_pdf[NUM_BUSINESS_TYPES];
float h_infectiousness_profile[CULMINATION_PERIOD];

int h_weekday_errand_contact_assignments_wholeDay[3][2];
int h_weekend_errand_contact_assignments_wholeDay[6][2];

float h_infectiousness_profile_all[6][CULMINATION_PERIOD];
kval_t h_kval_lookup[NUM_CONTACT_TYPES];



#define CHILD_DATA_ROWS 5
float child_CDF[CHILD_DATA_ROWS];
int child_wp_types[CHILD_DATA_ROWS];

#define HH_TABLE_ROWS 9
int hh_adult_count[HH_TABLE_ROWS];
int hh_child_count[HH_TABLE_ROWS];
float hh_type_cdf[HH_TABLE_ROWS];

//the first row of the PDF with a value > 0
const int FIRST_WEEKDAY_ERRAND_ROW = 9;
const int FIRST_WEEKEND_ERRAND_ROW = 9;


PandemicSim::PandemicSim() 
{
	logging_openOutputStreams();

	if(PROFILE_SIMULATION)
		profiler.initStack();

	setup_loadParameters();
	setup_scaleSimulation();

	if(debug_log_function_calls)
		debug_print("parameters loaded");

}


PandemicSim::~PandemicSim(void)
{
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

	srand(SEED_HOST[0]);					//seed host RNG
	rand_offset = 0;				//set global rand counter to 0

	current_day = -1;

	debug_print("beginning setup");



	if(debug_log_function_calls)
		debug_print("setting up households");
	
	//setup households
	setup_generateHouseholds();	//generates according to PDFs
	setup_sizeGlobalArrays(); // only after households are generated

	if(log_people_info)
		dump_people_info();

	printf("%d people, %d households, %d workplaces\n",number_people, number_households, number_workplaces);

	setup_buildFixedLocations();	//household and workplace
	setup_initialInfected();


	//copy everything down to the GPU
	setup_pushDeviceData();

	if(log_contacts)
	{
		debug_sizeHostArrays();
		debug_copyFixedData();
	}

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
		fInfected = fopen("../debug_infected.csv", "w");
		fprintf(fInfected, "current_day, i, idx, status_p, day_p, gen_p, status_s, day_s, gen_s\n");
	}

	if(log_location_info)
	{
		fLocationInfo = fopen("../debug_location_info.csv","w");
		fprintf(fLocationInfo, "current_day, hour_index, i, offset, count, max_contacts\n");
	}

	if(log_contacts)
	{
		fContacts = fopen("../debug_contacts.csv", "w");
		fprintf(fContacts, "current_day, i, infector_idx, victim_idx, contact_type, infector_loc, victim_loc, locs_matched\n");
	}

/*	if(log_contact_kernel_setup)
	{
		fContactsKernelSetup = fopen("../debug_contacts_kernel_setup.csv", "w");
		fprintf(fContactsKernelSetup, "current_day,hour,i,infector_idx,loc,loc_offset,loc_count,contacts_desired,output_offset\n");
	}*/


	if(log_actions)
	{
		fActions = fopen("../debug_actions.csv", "w");
		fprintf(fActions, "current_day, i, type, infector, infector_status_p, infector_status_s, victim, action_gen_p, action_gen_s, y_p, thresh_p, infects_p, y_s, thresh_s, infects_s\n");
	}

	if(log_actions_filtered)
	{
		fActionsFiltered = fopen("../debug_filtered_actions.csv", "w");
		fprintf(fActionsFiltered, "current_day, i, type, victim, victim_status_p, victim_gen_p, victim_status_s, victim_gen_s\n");
	}


	fDebug = fopen("../debug.txt", "w");

	
}

void PandemicSim::setup_loadParameters()
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

	//if printing seeds is desired for debug, etc
	if(0)
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
	if(fSeed == NULL)
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
	fscanf(fConstants, "%f", &asymp_factor);
	fclose(fConstants);

	number_households = 100000;
	number_workplaces = 1300;

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
	child_CDF[0] = 0.24f;
	child_CDF[1] = 0.47f;
	child_CDF[2] = 0.72f;
	child_CDF[3] = 0.85f;
	child_CDF[4] = 1.0f;

	//what workplace type children get for this age
	child_wp_types[0] = 3;
	child_wp_types[1] = 4;
	child_wp_types[2] = 5;
	child_wp_types[3] = 6;
	child_wp_types[4] = 7;

	//workplace PDF for adults
	workplace_type_pdf[0] = 0.06586f;
	workplace_type_pdf[1] = 0.05802f;
	workplace_type_pdf[2] = 0.30227f;
	workplace_type_pdf[3] = 0.0048f;
	workplace_type_pdf[4] = 0.00997f;
	workplace_type_pdf[5] = 0.203f;
	workplace_type_pdf[6] = 0.09736f;
	workplace_type_pdf[7] = 0.10598f;
	workplace_type_pdf[8] = 0.00681f;
	workplace_type_pdf[9] = 0.02599f;
	workplace_type_pdf[10] = 0.f;
	workplace_type_pdf[11] = 0.08749f;
	workplace_type_pdf[12] = 0.03181f;
	workplace_type_pdf[13] = 0.00064f;

	//number of each type of workplace
	h_workplace_type_counts[0] = 100;
	h_workplace_type_counts[1] = 700;
	h_workplace_type_counts[2] = 240;
	h_workplace_type_counts[3] = 30;
	h_workplace_type_counts[4] = 10;
	h_workplace_type_counts[5] = 20;
	h_workplace_type_counts[6] = 10;
	h_workplace_type_counts[7] = 10;
	h_workplace_type_counts[8] = 30;
	h_workplace_type_counts[9] = 50;
	h_workplace_type_counts[10] = 0;
	h_workplace_type_counts[11] = 30;
	h_workplace_type_counts[12] = 40;
	h_workplace_type_counts[13] = 10;

	//maximum number of contacts made at each workplace type
	h_workplace_max_contacts[0] = 3;
	h_workplace_max_contacts[1] = 3;
	h_workplace_max_contacts[2] = 3;
	h_workplace_max_contacts[3] = 2;
	h_workplace_max_contacts[4] = 2;
	h_workplace_max_contacts[5] = 3;
	h_workplace_max_contacts[6] = 3;
	h_workplace_max_contacts[7] = 2;
	h_workplace_max_contacts[8] = 2;
	h_workplace_max_contacts[9] = 2;
	h_workplace_max_contacts[10] = 0;
	h_workplace_max_contacts[11] = 2;
	h_workplace_max_contacts[12] = 2;
	h_workplace_max_contacts[13] = 2;



	//pdf for weekday errand location generation
	//most entries are 0.0
	thrust::fill(h_weekday_errand_pdf, h_weekday_errand_pdf + NUM_BUSINESS_TYPES, 0.0);
	h_weekday_errand_pdf[9] = 0.61919f;
	h_weekday_errand_pdf[11] = 0.27812f;
	h_weekday_errand_pdf[12] = 0.06601f;
	h_weekday_errand_pdf[13] = 0.03668f;

	//pdf for weekend errand location generation
	//most entries are 0.0
	thrust::fill(h_weekend_errand_pdf, h_weekend_errand_pdf + NUM_BUSINESS_TYPES, 0.0f);
	h_weekend_errand_pdf[9] = 0.51493f;
	h_weekend_errand_pdf[11] = 0.25586f;
	h_weekend_errand_pdf[12] = 0.1162f;
	h_weekend_errand_pdf[13] = 0.113f;


	//how many adults in each household type
	hh_adult_count[0] = 1;
	hh_adult_count[1] = 1;
	hh_adult_count[2] = 2;
	hh_adult_count[3] = 1;
	hh_adult_count[4] = 2;
	hh_adult_count[5] = 1;
	hh_adult_count[6] = 2;
	hh_adult_count[7] = 1;
	hh_adult_count[8] = 2;

	//how many children in each household type
	hh_child_count[0] = 0;
	hh_child_count[1] = 1;
	hh_child_count[2] = 0;
	hh_child_count[3] = 2;
	hh_child_count[4] = 1;
	hh_child_count[5] = 3;
	hh_child_count[6] = 2;
	hh_child_count[7] = 4;
	hh_child_count[8] = 3;

	//the PDF of each household type
	hh_type_cdf[0] = 0.279f;
	hh_type_cdf[1] = 0.319f;
	hh_type_cdf[2] = 0.628f;
	hh_type_cdf[3] = 0.671f;
	hh_type_cdf[4] = 0.8f;
	hh_type_cdf[5] = 0.812f;
	hh_type_cdf[6] = 0.939f;
	hh_type_cdf[7] = 0.944f;
	hh_type_cdf[8] = 1.0f;

	//store all permutations of contact assignments

	//number of contacts made in each hour
	h_weekday_errand_contact_assignments_wholeDay[0][0] = 2;
	h_weekday_errand_contact_assignments_wholeDay[0][1] = 0;

	h_weekday_errand_contact_assignments_wholeDay[1][0] = 0;
	h_weekday_errand_contact_assignments_wholeDay[1][1] = 2;

	h_weekday_errand_contact_assignments_wholeDay[2][0] = 1;
	h_weekday_errand_contact_assignments_wholeDay[2][1] = 1;

	//DIFFERENT FORMAT: hours each of the 2 contacts are made in
	//2 contacts in hour 0
	h_weekend_errand_contact_assignments_wholeDay[0][0] = 0;
	h_weekend_errand_contact_assignments_wholeDay[0][1] = 0;

	//2 contacts in hour 1
	h_weekend_errand_contact_assignments_wholeDay[1][0] = 1;
	h_weekend_errand_contact_assignments_wholeDay[1][1] = 1;

	//2 contacts in hour 2
	h_weekend_errand_contact_assignments_wholeDay[2][0] = 2;
	h_weekend_errand_contact_assignments_wholeDay[2][1] = 2;

	//contact in hour 0 and hour 1
	h_weekend_errand_contact_assignments_wholeDay[3][0] = 0;
	h_weekend_errand_contact_assignments_wholeDay[3][1] = 1;

	//contact in hour 0 and hour 2
	h_weekend_errand_contact_assignments_wholeDay[4][0] = 0;
	h_weekend_errand_contact_assignments_wholeDay[4][1] = 2;

	//contact in hour 1 and 2
	h_weekend_errand_contact_assignments_wholeDay[5][0] = 1;
	h_weekend_errand_contact_assignments_wholeDay[5][1] = 2;


	//load lognorm1 as default profile - others will be used later
	h_infectiousness_profile[0] = 0.002533572f;
	h_infectiousness_profile[1] = 0.348252834f;
	h_infectiousness_profile[2] = 0.498210218f;
	h_infectiousness_profile[3] = 0.130145145f;
	h_infectiousness_profile[4] = 0.018421298f;
	h_infectiousness_profile[5] = 0.002158374f;
	h_infectiousness_profile[6] = 0.000245489f;
	h_infectiousness_profile[7] = 2.88922E-05f;
	h_infectiousness_profile[8] = 3.61113E-06f;
	h_infectiousness_profile[9] = 4.83901E-07f;


#pragma region profiles

	//gamma1
	h_infectiousness_profile_all[0][0] = 0.007339835f;
	h_infectiousness_profile_all[0][1] = 0.332600216f;
	h_infectiousness_profile_all[0][2] = 0.501192066f;
	h_infectiousness_profile_all[0][3] = 0.142183447f;
	h_infectiousness_profile_all[0][4] = 0.015675154f;
	h_infectiousness_profile_all[0][5] = 0.000967407f;
	h_infectiousness_profile_all[0][6] = 4.055E-05f;
	h_infectiousness_profile_all[0][7] = 1.29105E-06f;
	h_infectiousness_profile_all[0][8] = 3.34836E-08f;
	h_infectiousness_profile_all[0][9] = 7.41011E-10f;

	//lognorm1
	h_infectiousness_profile_all[1][0] = 0.002533572f;
	h_infectiousness_profile_all[1][1] = 0.348252834f;
	h_infectiousness_profile_all[1][2] = 0.498210218f;
	h_infectiousness_profile_all[1][3] = 0.130145145f;
	h_infectiousness_profile_all[1][4] = 0.018421298f;
	h_infectiousness_profile_all[1][5] = 0.002158374f;
	h_infectiousness_profile_all[1][6] = 0.000245489f;
	h_infectiousness_profile_all[1][7] = 2.88922E-05f;
	h_infectiousness_profile_all[1][8] = 3.61113E-06f;
	h_infectiousness_profile_all[1][9] = 4.83901E-07f;


	//weib1
	h_infectiousness_profile_all[2][0] = 0.05927385f;
	h_infectiousness_profile_all[2][1] = 0.314171688f;
	h_infectiousness_profile_all[2][2] = 0.411588802f;
	h_infectiousness_profile_all[2][3] = 0.187010054f;
	h_infectiousness_profile_all[2][4] = 0.026934715f;
	h_infectiousness_profile_all[2][5] = 0.001013098f;
	h_infectiousness_profile_all[2][6] = 7.78449E-06f;
	h_infectiousness_profile_all[2][7] = 9.29441E-09f;
	h_infectiousness_profile_all[2][8] = 1.29796E-12f;
	h_infectiousness_profile_all[2][9] = 0;

	//gamma2
	h_infectiousness_profile_all[3][0] = 0.04687299f;
	h_infectiousness_profile_all[3][1] = 0.248505983f;
	h_infectiousness_profile_all[3][2] = 0.30307952f;
	h_infectiousness_profile_all[3][3] = 0.211008627f;
	h_infectiousness_profile_all[3][4] = 0.11087006f;
	h_infectiousness_profile_all[3][5] = 0.049241932f;
	h_infectiousness_profile_all[3][6] = 0.019562658f;
	h_infectiousness_profile_all[3][7] = 0.007179076f;
	h_infectiousness_profile_all[3][8] = 0.002482875f;
	h_infectiousness_profile_all[3][9] = 0.000820094f;

	//lognorm2
	h_infectiousness_profile_all[4][0] = 0.028667712f;
	h_infectiousness_profile_all[4][1] = 0.283445338f;
	h_infectiousness_profile_all[4][2] = 0.319240133f;
	h_infectiousness_profile_all[4][3] = 0.190123057f;
	h_infectiousness_profile_all[4][4] = 0.093989959f;
	h_infectiousness_profile_all[4][5] = 0.044155659f;
	h_infectiousness_profile_all[4][6] = 0.020682822f;
	h_infectiousness_profile_all[4][7] = 0.009841839f;
	h_infectiousness_profile_all[4][8] = 0.00479234f;
	h_infectiousness_profile_all[4][9] = 0.002393665f;

	//weib2
	h_infectiousness_profile_all[5][0] = 0.087866042f;
	h_infectiousness_profile_all[5][1] = 0.223005225f;
	h_infectiousness_profile_all[5][2] = 0.258992749f;
	h_infectiousness_profile_all[5][3] = 0.208637267f;
	h_infectiousness_profile_all[5][4] = 0.127489076f;
	h_infectiousness_profile_all[5][5] = 0.061148649f;
	h_infectiousness_profile_all[5][6] = 0.023406737f;
	h_infectiousness_profile_all[5][7] = 0.007216643f;
	h_infectiousness_profile_all[5][8] = 0.001802145f;
	h_infectiousness_profile_all[5][9] = 0.00036581f;

	for(int i = 3; i < 6; i++)
		for(int j = 0; i < CULMINATION_PERIOD; i++)
			h_infectiousness_profile_all[i][j] *= asymp_factor;

#pragma endregion profiles

	//store kvals - all 1 except for no-contact
	h_kval_lookup[CONTACT_TYPE_NONE] = 0;
	for(int i = CONTACT_TYPE_NONE + 1; i < NUM_CONTACT_TYPES;i++)
		h_kval_lookup[i] = 1;
}

//push various things to device constant memory
void PandemicSim::setup_pushDeviceData()
{
	//workplace location data
	cudaMemcpyToSymbol(
		business_type_count,
		h_workplace_type_counts,
		sizeof(int) * NUM_BUSINESS_TYPES);
	cudaMemcpyToSymbol(
		business_type_count_offset,
		h_workplace_type_offset,
		sizeof(int) * NUM_BUSINESS_TYPES);
	cudaMemcpyToSymbol(
		business_type_max_contacts,
		h_workplace_max_contacts,
		sizeof(int) * NUM_BUSINESS_TYPES);

	//weekday+weekend errand PDFs
	cudaMemcpyToSymbol(
		weekday_errand_pdf,
		h_weekday_errand_pdf,
		sizeof(float) * NUM_BUSINESS_TYPES);
	cudaMemcpyToSymbol(
		weekend_errand_pdf,
		h_weekend_errand_pdf,
		sizeof(float) * NUM_BUSINESS_TYPES);

	//viral shedding profile
	cudaMemcpyToSymbol(
		infectiousness_profile,
		h_infectiousness_profile,
		sizeof(float) * CULMINATION_PERIOD);

	//reproduction numbers
	cudaMemcpyToSymbol(
		BASE_REPRODUCTION_DEVICE,
		BASE_REPRODUCTION_HOST,
		sizeof(float) * STRAIN_COUNT);

	//alternate weekend contacts_desired assignment mode
	cudaMemcpyToSymbol(
		weekend_errand_contact_assignments_wholeDay,
		h_weekend_errand_contact_assignments_wholeDay,
		sizeof(int) * 6 * 2);

	cudaMemcpyToSymbol(
		weekday_errand_contact_assignments_wholeDay,
		h_weekday_errand_contact_assignments_wholeDay,
		sizeof(int) * 3 * 2);

	//seeds
	cudaMemcpyToSymbol(
		SEED_DEVICE,
		SEED_HOST,
		sizeof(int) * SEED_LENGTH);

	//kvals
	cudaMemcpyToSymbol(
		kval_lookup,
		h_kval_lookup,
		sizeof(kval_t) * NUM_CONTACT_TYPES);

	cudaDeviceSynchronize();
}

//Sets up people's households and workplaces according to the probability functions
void PandemicSim::setup_generateHouseholds()
{
	//actual expected value: 2.5
	int expected_people = 3 * number_households;

	//stores household and workplace data for all people
	thrust::host_vector<int> h_people_hh;
	thrust::host_vector<int> h_people_wp;
	thrust::host_vector<int> h_people_age;
	h_people_hh.reserve(expected_people);
	h_people_wp.reserve(expected_people);
	h_people_age.reserve(expected_people);

	//stores the list of adults and children for weekday errands/afterschool
	thrust::host_vector<int> h_adult_indexes;
	thrust::host_vector<int> h_child_indexes;
	h_adult_indexes.reserve(expected_people);
	h_child_indexes.reserve(expected_people);

	//count number of people
	number_people = 0;

	for(int hh = 0; hh < number_households; hh++)
	{
		//fish out the type of household from CDF
		float y = (float) rand() / RAND_MAX;
		int hh_type = 0;
		while(y > hh_type_cdf[hh_type] && hh_type < HH_TABLE_ROWS - 1)
			hh_type++;

		//generate the adults for this household
		for(int i = 0; i < hh_adult_count[hh_type]; i++)
		{
			//assign adult workplace
			int wp = setup_assignWorkplace();
			h_people_wp.push_back(wp);

			//assign household
			h_people_hh.push_back(hh);

			//store as adult
			h_adult_indexes.push_back(number_people);
			h_people_age.push_back(AGE_ADULT);

			number_people++;
		}

		//generate the children for this household
		for(int i = 0; i < hh_child_count[hh_type]; i++)
		{
			//assign school
			int wp, age;
			setup_assignSchool(&wp, &age);
			h_people_wp.push_back(wp);

			//assign household
			h_people_hh.push_back(hh);

			//store as child
			h_child_indexes.push_back(number_people);
			h_people_age.push_back(age);
			

			number_people++;
		}
	}

	//trim arrays to data size, and transfer them to GPU
	h_people_wp.shrink_to_fit();
	people_workplaces = h_people_wp;

	h_people_hh.shrink_to_fit();
	people_households = h_people_hh;

	h_people_age.shrink_to_fit();
	people_ages = h_people_age;

	h_adult_indexes.shrink_to_fit();
	people_adult_indexes = h_adult_indexes;

	h_child_indexes.shrink_to_fit();
	people_child_indexes = h_child_indexes;

	number_adults = h_adult_indexes.size();
	number_children = h_child_indexes.size();

	printf("%d households, %d adults, %d children, %d total\n",
		number_households, number_adults, number_children, number_people);

	//setting up status array will be handled in setupSim()
}

int PandemicSim::setup_assignWorkplace()
{
	//fish out workplace type
	float y = (float) rand() / RAND_MAX;
	int row = 0;
	while(workplace_type_pdf[row] < y && row < NUM_BUSINESS_TYPES - 1)
	{
		y -= workplace_type_pdf[row];
		row++;
	}

	//of this workplace type, which number is this?
	float frac = y / workplace_type_pdf[row];
	int ret = frac * h_workplace_type_counts[row];  //truncate to int

	//how many other workplaces have we gone past?
	int offset = h_workplace_type_offset[row];

	//	printf("row: %d\ty: %f\tpdf[row]: %f\tfrac: %f\tret: %4d\toffset:%d\n",
	//			row, y, workplace_type_pdf[row], frac, ret, offset);


	return ret + offset;
}

void PandemicSim::setup_assignSchool(int * wp, int * age)
{
	//fish out age group and resulting school type from CDF
	int row = 0;
	float y = (float) rand() / RAND_MAX;
	while(row < CHILD_DATA_ROWS - 1 && y > child_CDF[row])
		row++;


	int wp_type = child_wp_types[row];

	//of this school type, which one will this kid be assigned to?
	float frac;
	if(row == 0)
		frac = y / (child_CDF[row]);
	else
	{
		float pdf_here = child_CDF[row] - child_CDF[row - 1];
		float y_here = y - child_CDF[row - 1];
		//	printf("y here: %f\tpdf here: %f\n", (y - child_CDF[row - 1]), pdf);
		frac =  y_here / pdf_here;
	}

	int ret = frac * h_workplace_type_counts[wp_type];

	//how many other workplaces have we gone past?
	int offset = h_workplace_type_offset[wp_type];
	(*wp) = ret + offset;
	(*age) = row;
}


//Sets up the initial infection at the beginning of the simulation
//BEWARE: you must not generate dual infections with this code, or you will end up with duplicate infected indexes
void PandemicSim::setup_initialInfected()
{
	//fill infected array with null info (not infected)
	thrust::fill(infected_days_pandemic.begin(), infected_days_pandemic.end(), DAY_NOT_INFECTED);
	thrust::fill(infected_days_seasonal.begin(), infected_days_seasonal.end(), DAY_NOT_INFECTED);
	thrust::fill(infected_generation_pandemic.begin(), infected_generation_pandemic.end(), GENERATION_NOT_INFECTED);
	thrust::fill(infected_generation_seasonal.begin(), infected_generation_seasonal.end(), GENERATION_NOT_INFECTED);

	int initial_infected = INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL;

	//get N unique indexes - they should not be sorted
	h_vec h_init_indexes(initial_infected);
	n_unique_numbers(&h_init_indexes, initial_infected, number_people);
	thrust::copy(h_init_indexes.begin(), h_init_indexes.end(), infected_indexes.begin());

	///// INFECTED PANDEMIC:
	//infect first INITIAL_INFECTED_PANDEMIC people with pandemic
	//set status to infected
	thrust::fill(
		thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes.begin()),	//begin at infected 0
		thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes.begin() + INITIAL_INFECTED_PANDEMIC),	//end at index INITIAL_INFECTED_PANDEMIC
		STATUS_INFECTED);

	//set day/generation pandemic to 0 (initial)
	thrust::fill(
		infected_days_pandemic.begin(), 		//begin
		infected_days_pandemic.begin() + INITIAL_INFECTED_PANDEMIC, //end
		INITIAL_DAY);//val
	thrust::fill(
		infected_generation_pandemic.begin(),
		infected_generation_pandemic.begin() + INITIAL_INFECTED_PANDEMIC,
		0);	//fill infected with gen 0

	///// INFECTED SEASONAL:
	//set status to infected
	thrust::fill(
		thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin()+ INITIAL_INFECTED_PANDEMIC), //begin at index INITIAL_INFECTED_PANDEMIC
		thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin() + INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL),	//end INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL
		STATUS_INFECTED);

	//set day/generation seasonal to 0
	thrust::fill(
		infected_generation_seasonal.begin() + INITIAL_INFECTED_PANDEMIC,
		infected_generation_seasonal.begin() + INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL,
		0);	//first generation
	thrust::fill(
		infected_days_seasonal.begin() + INITIAL_INFECTED_PANDEMIC,
		infected_days_seasonal.begin() + INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL,
		INITIAL_DAY);		//day: 0

	//sort array after infection complete
	thrust::sort(
		thrust::make_zip_iterator(thrust::make_tuple(			//first
		infected_indexes.begin(),
		infected_days_pandemic.begin(),infected_days_seasonal.begin(),
		infected_generation_pandemic.begin(),infected_generation_seasonal.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(			//first
		infected_indexes.begin() + initial_infected,
		infected_days_pandemic.begin() + initial_infected,infected_days_seasonal.begin() + initial_infected,
		infected_generation_pandemic.begin() + initial_infected,infected_generation_seasonal.begin() + initial_infected)),
		FiveTuple_SortByFirst_Struct());

	infected_count = initial_infected;
	reproduction_count_pandemic[0] = INITIAL_INFECTED_PANDEMIC;
	reproduction_count_seasonal[0] = INITIAL_INFECTED_SEASONAL;
}

//sets up the locations which are the same every day and do not change
//i.e. workplace and household
void PandemicSim::setup_buildFixedLocations()
{
	///////////////////////////////////////
	//home/////////////////////////////////
	household_offsets.resize(number_households + 1);
	household_people.resize(number_people);

	thrust::sequence(household_people.begin(), household_people.begin() + number_people);	//fill array with IDs to sort
	calcLocationOffsets(
		&household_people,
		people_households,
		&household_offsets,
		number_people, number_households);

	///////////////////////////////////////
	//work/////////////////////////////////
	workplace_offsets.resize(number_workplaces + 1);	//size arrays
	workplace_people.resize(number_people);

	thrust::sequence(workplace_people.begin(), workplace_people.begin() + number_people);	//fill array with IDs to sort

	calcLocationOffsets(
		&workplace_people,
		people_workplaces,
		&workplace_offsets,
		number_people, number_workplaces);

	//set up workplace max contacts
	workplace_max_contacts.resize(number_workplaces);		//size the array

	//copy the number of contacts per location type to device
	vec_t workplace_type_max_contacts(NUM_BUSINESS_TYPES);		
	thrust::copy_n(h_workplace_max_contacts, NUM_BUSINESS_TYPES, workplace_type_max_contacts.begin());

	//TODO:  make this work right with device constant memory.  For now, just make a copy in global memory
	vec_t business_type_count_vec(NUM_BUSINESS_TYPES);
	thrust::copy_n(h_workplace_type_counts,NUM_BUSINESS_TYPES,business_type_count_vec.begin());
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
}


//given an array of people's ID numbers and locations
//sort them by location, and then build the location offset/count tables
//ids_to_sort will be sorted by workplace
void PandemicSim::calcLocationOffsets(
	vec_t * ids_to_sort,
	vec_t lookup_table_copy,
	vec_t * location_offsets,
	int num_people, int num_locs)
{
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

	if(SANITY_CHECK)
	{
		debug_assert("Loc_offset not sized properly",num_locs + 1,location_offsets->size());
	}

	(*location_offsets)[num_locs] = num_people;
}

void PandemicSim::dump_people_info()
{
	h_vec h_wp = people_workplaces;
	h_vec h_hh = people_households;

	FILE * fPeopleInfo = fopen("../debug_people_info.csv.gz", "w");
	fprintf(fPeopleInfo, "i,workplace,household\n");
	for(int i = 0; i < number_people; i++)
	{
		fprintf(fPeopleInfo, "%d,%d,%d\n", i, h_wp[i], h_hh[i]);
	}
	fclose(fPeopleInfo);
}

void PandemicSim::logging_closeOutputStreams()
{
	if(log_infected_info)
	{
		fclose(fInfected);
	}

	if(log_location_info)
	{
		fclose(fLocationInfo);
	}

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

	fclose(fDebug);
	profiler.done();
} 



void PandemicSim::runToCompletion()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(-1, "runToCompletion");

	for(current_day = 0; current_day < MAX_DAYS; current_day++)
	{
		if(debug_log_function_calls)
		{
			fprintf(fDebug, "\n\n---------------------\nday %d\ninfected: %d\n---------------------\n\n", current_day, infected_count);
			fflush(fDebug);
		}

		if(CONSOLE_OUTPUT)
		{
			printf("Day %d:\tinfected: %5d\n", current_day + 1, infected_count);
		}
		daily_contacts = 0;	//start counting contacts/actions from 0 each day
		daily_actions = 0;

		/*
		//resize contacts array to fit expected number of contacts
		int contacts_expected;
		if(is_weekend())
			contacts_expected = 5; //3 home, 2 errand
		else
			contacts_expected = 8; //3 home, 2 work, 3 afterschool

		contacts_expected *= infected_count;
		daily_contact_infectors.resize(contacts_expected);
		daily_contact_victims.resize(contacts_expected);
		//daily_contact_kvals.resize(contacts_expected);*/
		
		if(debug_null_fill_daily_arrays)
			debug_nullFillDailyArrays();

		//debug: dump infected info?
		if(log_infected_info || print_infected_info)
		{
//			debug_validate_infected();
			dump_infected_info();
		}

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

		//if we're using the profiler, flush each day in case of crash
		if(PROFILE_SIMULATION)
		{
			profiler.dailyFlush();
		}
	}
	calculateFinalReproduction();

	if(PROFILE_SIMULATION)
		profiler.endFunction(-1, number_people);


	//moved to destructor for batching
	//close_output_streams();
}


//called at the end of the simulation, figures out the reproduction numbers for each generation
void PandemicSim::calculateFinalReproduction()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "calculateFinalReproduction");

	//copy to host
	thrust::host_vector<int> r_pandemic = reproduction_count_pandemic;
	thrust::host_vector<int> r_seasonal = reproduction_count_seasonal;

	FILE * out = fopen("../output_rn.csv", "w");
	fprintf(out, "gen, size_p, rn_p, size_s, rn_s\n");

	//loop and calculate reproduction
	for(int i = 0; i < MAX_DAYS - 1; i++)
	{
		int gen_size_p = r_pandemic[i];
		int gen_size_s = r_seasonal[i];

		float rn_pandemic = (float) r_pandemic[i+1] / gen_size_p;
		float rn_seasonal = (float) r_seasonal[i+1] / gen_size_s;

		fprintf(out, "%d, %d, %f, %d, %f\n",
			i, gen_size_p, rn_pandemic, gen_size_s, rn_seasonal);

		gen_size_p = r_pandemic[i];
		gen_size_s = r_seasonal[i];
	}
	fclose(out);

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, MAX_DAYS);
} 

void PandemicSim::debug_validate_infected()
{
	//ASSERT:  ALL INFECTED ARRAYS ARE THE CORRECT/SAME SIZE
	debug_assert("infected_indexes.size() != infected_count", infected_indexes.size(), infected_count);
	debug_assert("infected_days_pandemic.size() != infected_count", infected_days_pandemic.size(), infected_count);
	debug_assert("infected_days_seasonal.size() != infected_count", infected_days_seasonal.size(), infected_count);
	debug_assert("infected_generation_pandemic.size() != infected_count", infected_generation_pandemic.size(), infected_count);
	debug_assert("infected_generation_seasonal.size() != infected_count", infected_generation_seasonal.size(), infected_count);

	//ASSERT:  INFECTED INDEXES ARE SORTED
	bool sorted = thrust::is_sorted(infected_indexes.begin(), infected_indexes.begin() + infected_count);
	debug_assert(sorted, "infected indexes are not sorted");

	//ASSERT:  INFECTED INDEXES ARE UNIQUE
	d_vec unique_indexes(infected_count);
	IntIterator end = thrust::unique_copy(infected_indexes.begin(), infected_indexes.begin() + infected_count, unique_indexes.begin());
	int unique_count  = end - unique_indexes.begin();
	debug_assert("infected_indexes are not unique", infected_count, unique_count);


	//copy infected data to PC
	h_vec h_ii = infected_indexes;
	h_vec h_day_p = infected_days_pandemic;
	h_vec h_day_s = infected_days_seasonal;
	h_vec h_gen_p = infected_generation_pandemic;
	h_vec h_gen_s = infected_generation_seasonal;

	h_vec h_p_status_p = people_status_pandemic;
	h_vec h_p_status_s = people_status_seasonal;

	//begin intensive check of infected
	for(int i = 0; i < infected_count; i++)
	{
		int idx = h_ii[i];
		int day_p = h_day_p[i];
		int gen_p = h_gen_p[i];
		int day_s = h_day_s[i];
		int gen_s = h_gen_s[i];

		int status_p = h_p_status_p[idx];
		int status_s = h_p_status_s[idx];

		//ASSERT: person on infected list is infected with pandemic or seasonal
		bool has_infection = status_p || status_s;
		debug_assert(has_infection, "infected_index has no infection", idx);

		if(status_p == STATUS_INFECTED)
		{
			//check that day of pandemic infection is within bounds
			debug_assert(day_p > DAY_NOT_INFECTED, "status_p infected but day not set", idx);
			debug_assert(day_p <= current_day, "day_p is after today", idx);
			int day_of_infection = current_day - day_p;
			debug_assert(day_of_infection < CULMINATION_PERIOD, "pandemic infection should have been recovered", idx);

			//check that generation is within bounds
			debug_assert(gen_p > GENERATION_NOT_INFECTED, "status_p infected but generation not set", idx);
			debug_assert(gen_p <= current_day, "generation_p too high", idx);
		}
		else
		{
			//NOT INFECTED - these should not be set to valid data!
			debug_assert(day_p == DAY_NOT_INFECTED, "status_p not infected but day is set", idx);
			debug_assert(gen_p == GENERATION_NOT_INFECTED, "status_p not infected but gen is set", idx);
		}

		if(status_s == STATUS_INFECTED)
		{
			//check that day of seasonal infection is within bounds
			debug_assert(day_s > DAY_NOT_INFECTED, "status_s infected but day not set", idx);
			debug_assert(day_s <= current_day, "day_s is after today", idx);
			int day_of_infection = current_day - day_s;
			debug_assert(day_of_infection < CULMINATION_PERIOD, "seasonal infection should have been recovered", idx);

			//check that generation is within bounds
			debug_assert(gen_s > GENERATION_NOT_INFECTED, "status_s infected but generation not set", idx);
			debug_assert(gen_s <= current_day, "generation_s too high", idx);
		}
		else
		{
			//NOT INFECTED - these should not be set to valid data!
			debug_assert(day_s == DAY_NOT_INFECTED, "status_s not infected but day is set", idx);
			debug_assert(gen_s == GENERATION_NOT_INFECTED, "status_s not infected but gen is set", idx);
		}
	}

	fflush(fDebug);
}

//dumps all infected info to disk
//this is a big consumer of disk space, so it uses zlib to compress
//use zcat to dump the file
void PandemicSim::dump_infected_info()
{
	//copy to host PC
	h_vec h_ii(infected_count);
	thrust::copy_n(infected_indexes.begin(), infected_count, h_ii.begin());
	h_vec h_day_p(infected_count);
	thrust::copy_n(infected_days_pandemic.begin(), infected_count, h_day_p.begin());
	h_vec h_day_s(infected_count);
	thrust::copy_n(infected_days_seasonal.begin(), infected_count, h_day_s.begin());
	h_vec h_gen_p(infected_count);
	thrust::copy_n(infected_generation_pandemic.begin(), infected_count, h_gen_p.begin());
	h_vec h_gen_s(infected_count);
	thrust::copy_n(infected_generation_seasonal.begin(), infected_count, h_gen_s.begin());

	h_vec h_p_status_p = people_status_pandemic;
	h_vec h_p_status_s = people_status_seasonal;

	//iterate and dump data
	for(int i = 0; i < infected_count; i++)
	{
		int idx = h_ii[i];
		int status_p = h_p_status_p[idx];
		int status_s = h_p_status_s[idx];
		int d_p = h_day_p[i];
		int g_p = h_gen_p[i];
		int d_s = h_day_s[i];
		int g_s = h_gen_s[i];

		//day, i, idx, status_p, day_p, gen_p, status_s, day_s, gen_s
		fprintf(fInfected, "%d, %d, %d, %c, %d, %d, %c, %d, %d\n",
			current_day, i, idx, 
			status_int_to_char(status_p), d_p, g_p,
			status_int_to_char(status_s), d_s, g_s);
	}
//	gzflush(fInfected, Z_SYNC_FLUSH);
	fflush(fInfected);
	fflush(fDebug);
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
__device__ void device_assignErrandHours_weekend_wholeDay(int * hours_dest_ptr, int my_rand_offset)
{
	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} u;
	
	threefry2x64_ctr_t tf_ctr = {{((long) my_rand_offset), ((long) my_rand_offset)}};
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
	while(row < NUM_BUSINESS_TYPES - 1 && y > weekend_errand_pdf[row])
	{
		y -= weekend_errand_pdf[row];
		row++;
	}
	y = y / weekend_errand_pdf[row];
	int business_num = y * (float) business_type_count[row];
	business_num += business_type_count_offset[row];

	*output_ptr = business_num;
}

//This method consumes the accumulated contacts, and causes infections and recovery to occur
void PandemicSim::dailyUpdate()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "dailyUpdate");

	if(debug_log_function_calls)
		debug_print("beginning daily update");

	if(CONSOLE_OUTPUT)
	{
		printf("average: %.2f contacts per infected\n", (float) daily_contacts / infected_count);
	}


	//process contacts into actions
	daily_contactsToActions();

	//filter invalid actions - not susceptible, duplicate, etc
	daily_filterActions();

	if(log_actions_filtered || print_actions_filtered)
		dump_actions_filtered();

	//counts the number of each generation in today's new infections 
	daily_countReproductionNumbers(ACTION_INFECT_PANDEMIC);
	daily_countReproductionNumbers(ACTION_INFECT_SEASONAL);

	//heals infected who are reaching culmination, and adds newly infected people
	daily_rebuildInfectedArray();


	if(CONSOLE_OUTPUT)
	{
		printf("now %d infected\n",infected_count);
		printf("update complete\n");
	}

	if(debug_log_function_calls)
		debug_print("daily update complete");

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}


//once the contacts have been properly set up, this kernel determines whether each contact
//is successful, and then outputs actions
__global__ void contacts_to_actions_kernel(
		int * output_offset_arr,
		int * inf_gen_p_arr, int * inf_gen_s_arr,
		int * day_p_arr, int * day_s_arr,
		int * action_types_arr,  
		int * victim_gen_p_arr, int * victim_gen_s_arr,
		// float * rand_1, float * rand_2, float* rand_3, float * rand_4,
		int num_infected, int global_rand_offset, int current_day)
{
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;
	
	//for each infected person
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		//output_offset holds the index where this action should be stored
		int output_offset = output_offset_arr[myPos];

		//how many contacts we're trying to make
		int contacts_desired = output_offset_arr[myPos + 1] - output_offset_arr[myPos];

		//these are the day this person was infected (from 0 to MAX_DAYS) - NOT which day of infection they are on
		int day_p = day_p_arr[myPos];
		int day_s = day_s_arr[myPos];

		//if they make a successful contact, the victim will receive these generations
		int victim_gen_p = GENERATION_NOT_INFECTED;
		int victim_gen_s = GENERATION_NOT_INFECTED;
			
		//start with zero probability of infection - if the person is infected, increase it
		float inf_prob_p = 0.0;
		
		if(day_p >= 0)
		{
			inf_prob_p = (infectiousness_profile[current_day - day_p] * BASE_REPRODUCTION_DEVICE[0]) / (float) contacts_desired;
			victim_gen_p = inf_gen_p_arr[myPos] + 1;
		}
			
		//same for seasonal
		float inf_prob_s = 0.0;
	
		if(day_s >= 0)
		{
			inf_prob_s = (infectiousness_profile[current_day - day_s] * BASE_REPRODUCTION_DEVICE[1]) / (float) contacts_desired;
			victim_gen_s = inf_gen_s_arr[myPos] + 1;
		}
			
		while(contacts_desired > 0)
		{
			//we need one random number set for each contact
			int myRandOffset = output_offset + global_rand_offset;
			
			threefry2x32_ctr_t tf_ctr = {{myRandOffset,myRandOffset}};
			u.c = threefry2x32(tf_ctr, tf_k);
				
			//convert uniform to float
			float f_pandemic = (float) u.i[0] / UNSIGNED_MAX;
			float f_seasonal = (float) u.i[1] / UNSIGNED_MAX;
			
			//if the random float is less than the infection threshold, infection succeeds
			bool infects_pandemic = f_pandemic < inf_prob_p;
			bool infects_seasonal = f_seasonal < inf_prob_s;
				
			//parse bools to action type
			if(infects_pandemic && infects_seasonal)
			{
				action_types_arr[output_offset] = ACTION_INFECT_BOTH;
				victim_gen_p_arr[output_offset] = victim_gen_p;
				victim_gen_s_arr[output_offset] = victim_gen_s;
			}
			else if(infects_pandemic)
			{
				action_types_arr[output_offset] = ACTION_INFECT_PANDEMIC;
				victim_gen_p_arr[output_offset] = victim_gen_p;
			}
			else if(infects_seasonal)
			{
				action_types_arr[output_offset] = ACTION_INFECT_SEASONAL;
				victim_gen_s_arr[output_offset] = victim_gen_s;
			}
			else
			{
				action_types_arr[output_offset] = ACTION_INFECT_NONE;
			}
				
			//for debug: we can output some internal values
//			rand_1[output_offset] = inf_prob_p;
//			rand_2[output_offset] = inf_prob_s;
//			rand_3[output_offset] = f_pandemic;
//			rand_4[output_offset] = f_seasonal;
				
			output_offset++;
			contacts_desired--;
		}
	}
}

//starts the kernel to convert contacts to actions
void PandemicSim::daily_contactsToActions()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "contacts_to_actions");
	
	if(debug_log_function_calls)
		debug_print("beginning contacts-to-action setup");
	
	//size our action output arrays
	//moved to setup function, only done once to prevent fragmentation

	//get lower bound of each infector, aka index
	d_vec infector_contacts_offset (infected_count + 1);
	thrust::lower_bound(
			daily_contact_infectors.begin(),		//data.begin
			daily_contact_infectors.begin() + daily_contacts,		//data.end
			infected_indexes.begin(),		//value to search for
			infected_indexes.begin() + infected_count,			//value to search
			infector_contacts_offset.begin());
	infector_contacts_offset[infected_count] = daily_contacts;
	
	//convert thrust vectors to raw pointers
	int * inf_offset_ptr = thrust::raw_pointer_cast(infector_contacts_offset.data());
	
	int * day_p_ptr = thrust::raw_pointer_cast(infected_days_pandemic.data());
	int * day_s_ptr = thrust::raw_pointer_cast(infected_days_seasonal.data());

	int * inf_gen_p_ptr = thrust::raw_pointer_cast(infected_generation_pandemic.data());
	int * inf_gen_s_ptr = thrust::raw_pointer_cast(infected_generation_seasonal.data());

	//stores what type of action resulted
	int * actions_type_ptr = thrust::raw_pointer_cast(daily_action_type.data());

	int * victim_gen_p_ptr = thrust::raw_pointer_cast(daily_action_victim_gen_p.data());
	int * victim_gen_s_ptr = thrust::raw_pointer_cast(daily_action_victim_gen_s.data());
	
	//for debug: we can dump some internal stuff	
	float* rand1ptr = thrust::raw_pointer_cast(debug_float1.data());
	float* rand2ptr = thrust::raw_pointer_cast(debug_float2.data());
	float* rand3ptr = thrust::raw_pointer_cast(debug_float3.data());
	float* rand4ptr = thrust::raw_pointer_cast(debug_float4.data());

	if(debug_log_function_calls)
		debug_print("calling contacts-to-action kernel");

	//determine whether each infection was successful
	contacts_to_actions_kernel<<<cuda_blocks, cuda_threads>>>(
			inf_offset_ptr,
			inf_gen_p_ptr, inf_gen_s_ptr,
			day_p_ptr, day_s_ptr,
			actions_type_ptr, 
			victim_gen_p_ptr, victim_gen_s_ptr,
	//		rand1ptr, rand2ptr, rand3ptr, rand4ptr,
			infected_count, rand_offset, current_day);

	cudaDeviceSynchronize();
	//copy the IDs of the infector and victim to the action array
	thrust::copy_n(daily_contact_victims.begin(), daily_contacts, daily_action_victim_index.begin());
	
	rand_offset += daily_contacts; //increment rand counter
	daily_actions = daily_contacts; //stores # of actions to allow dumping - set in filter after filtering


	if(log_actions || print_actions)
		dump_actions(
//				rand1, rand2, rand3, rand4	//disable debug outputs from contacts kernel
				);

	/*
	if(0){
		h_vec actions = daily_action_type;
		thrust::host_vector<float> h_rand1 = rand1;
		thrust::host_vector<float> h_rand2 = rand2;
		thrust::host_vector<float> h_rand3 = rand3;
		thrust::host_vector<float> h_rand4 = rand4;
		for(int i = 0; i < daily_contacts; i++)
		{
			float t_p = h_rand1[i];
			float y_p = h_rand3[i];
			float t_s = h_rand2[i];
			float y_s = h_rand4[i];
			printf("action: %s\tthresh_p:%f \ty_p: %f\tinfects_p: %d\tthresh_s: %f\ty_s: %f\tinfects_s: %d\n", action_type_to_char(actions[i]), t_p, y_p, y_p < t_p, t_s, y_s, y_s < t_s);
		}
	} */
	

	if(debug_log_function_calls)
		debug_print("contacts_to_actions complete");
	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

//after the contacts_to_actions kernel runs, we need to filter out actions that are invalid and then remove no-ops
void PandemicSim::daily_filterActions()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "filter_actions");

	if(debug_log_function_calls)
		debug_print("filtering contacts");


	//if victim is not susceptible to the virus, turn action to none
	thrust::transform(
		daily_action_type.begin(),
		daily_action_type.begin() + daily_contacts,
		thrust::make_zip_iterator(thrust::make_tuple(
			thrust::make_permutation_iterator(people_status_pandemic.begin(), daily_contact_victims.begin()),
			thrust::make_permutation_iterator(people_status_seasonal.begin(), daily_contact_victims.begin()))),
		daily_action_type.begin(),
		filterPriorInfectedOp());
	
	//remove no infection
	ZipIntQuadIterator filter_begin = thrust::make_zip_iterator(thrust::make_tuple(
		daily_action_type.begin(), daily_action_victim_index.begin(), 
		daily_action_victim_gen_p.begin(), daily_action_victim_gen_s.begin()));

	ZipIntQuadIterator filter_end = thrust::remove_if(
		filter_begin,
		thrust::make_zip_iterator(thrust::make_tuple(
			daily_action_type.begin() + daily_contacts, daily_action_victim_index.begin() + daily_contacts,
			daily_action_victim_gen_p.begin() + daily_contacts, daily_action_victim_gen_s.begin() + daily_contacts)),
		removeNoActionOp());

	int remaining = filter_end - filter_begin;

	if(CONSOLE_OUTPUT)
	{
		printf("%d unfiltered contacts\n", remaining);
	}

	//sort by victim, then by action
	thrust::sort(
		filter_begin,
		filter_end,
		actionSortOp());
	//remove duplicate actions
	filter_end = thrust::unique(	
		filter_begin,
		filter_end,
		uniqueActionOp());

	//after actions have been filtered, count how many remain
	daily_actions = filter_end - filter_begin;

	if(CONSOLE_OUTPUT)
	{
		printf("%d final contacts\n", daily_actions);
	}

	if(debug_log_function_calls)
		debug_print("contact filtering complete");
	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, daily_contacts);
}

//given an action type, look through the filtered actions and count generations
void PandemicSim::daily_countReproductionNumbers(int action)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "daily_countReproductionNumbers");

	if(debug_log_function_calls)
		debug_print("counting reproduction");

	//tests action type
	infectionTypePred r_pred;
	r_pred.reference_val = action;


	//TODO: Fix
	//"pandemic" is a bad word for this - to do seasonal, pass it the seasonal action type
	d_vec pandemic_gens(daily_actions);
	IntIterator gens_end;
	if(action == ACTION_INFECT_PANDEMIC)
	{
		gens_end = thrust::copy_if(
			daily_action_victim_gen_p.begin(),			//copy all generations from pandemic actions
			daily_action_victim_gen_p.begin() + daily_actions,
			daily_action_type.begin(),
			pandemic_gens.begin(),
			r_pred);

	}
	else if(action == ACTION_INFECT_SEASONAL)
	{
		gens_end = thrust::copy_if(
			daily_action_victim_gen_s.begin(),		//copy all generations from seasonal actions 
			daily_action_victim_gen_s.begin() + daily_actions,
			daily_action_type.begin(),
			pandemic_gens.begin(),
			r_pred);
	}
	else
	{
		throw;
	}

	int num_matching_actions = gens_end - pandemic_gens.begin();

	//sort generations so we can count them
	thrust::sort(pandemic_gens.begin(), gens_end);

	thrust::counting_iterator<int> count_iterator(0);

	d_vec lower_bound(MAX_DAYS + 1);
	thrust::lower_bound(
		pandemic_gens.begin(),
		gens_end,
		count_iterator,
		count_iterator + MAX_DAYS,
		lower_bound.begin());
	lower_bound[MAX_DAYS] = num_matching_actions;

	//increment the generation counts in the appropriate array
	if(action == ACTION_INFECT_PANDEMIC)
	{
		thrust::transform(
			reproduction_count_pandemic.begin(),
			reproduction_count_pandemic.begin() + MAX_DAYS,
			thrust::make_transform_iterator(
				thrust::make_zip_iterator(thrust::make_tuple(lower_bound.begin(), lower_bound.begin() + 1)),
				OffsetToCountFunctor_struct()),
			reproduction_count_pandemic.begin(),
			thrust::plus<int>());
	}
	else  //seasonal
	{
		thrust::transform(
			reproduction_count_seasonal.begin(),
			reproduction_count_seasonal.begin() + MAX_DAYS,
			thrust::make_transform_iterator(
				thrust::make_zip_iterator(thrust::make_tuple(lower_bound.begin(), lower_bound.begin() + 1)),
				OffsetToCountFunctor_struct()),
			reproduction_count_seasonal.begin(),
			thrust::plus<int>());
	}

	if(debug_log_function_calls)
		debug_print("reproduction updated");
	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, daily_actions);
}

//this will dump all actions (unfiltered) to disk, which allows immediate debugging of the contacts_to_actions kernel
//If you need further detail, there are four float outputs which you can enable to debug kernels more easily
//for example, you can dump the y_vals that decide whether a contact is successful
//This uses a lot of disk space, so outputs are compressed with zlib.
//Dump the logs to disk using the 'zcat' utility
void PandemicSim::dump_actions(
		)
{
	//copy data to host - only as much data as we have contacts
	h_vec h_ac_type(daily_contacts);	//type of contact
	thrust::copy_n(daily_action_type.begin(), daily_contacts, h_ac_type.begin());

	h_vec h_ac_inf(daily_contacts);		//infector
	thrust::copy_n(daily_contact_infectors.begin(), daily_contacts, h_ac_inf.begin());

	h_vec h_ac_vic(daily_contacts);		//victim
	thrust::copy_n(daily_action_victim_index.begin(), daily_contacts, h_ac_vic.begin());

	h_vec h_vic_gen_p(daily_contacts);	//victim gen_p
	thrust::copy_n(daily_action_victim_gen_p.begin(), daily_contacts, h_vic_gen_p.begin());

	h_vec h_vic_gen_s = daily_action_victim_gen_s;	//victim gen_s
	thrust::copy_n(daily_action_victim_gen_s.begin(), daily_contacts, h_vic_gen_s.begin());
	
	//copy whole status arrays
	h_vec h_status_p = people_status_pandemic;
	h_vec h_status_s = people_status_seasonal;
	
//	thrust::host_vector<float> h_r1 = rand1;
//	thrust::host_vector<float> h_r2 = rand2;
//	thrust::host_vector<float> h_r3 = rand3;
//	thrust::host_vector<float> h_r4 = rand4;
	
	for(int i = 0; i < daily_actions; i++)
	{
		int inf = h_ac_inf[i];
		int vic = h_ac_vic[i];
		int type = h_ac_type[i];
		
		int status_p = h_status_p[inf];
		int status_s = h_status_s[inf];

		int gen_p = h_vic_gen_p[i];
		int gen_s = h_vic_gen_s[i];
		
		/*
		float thresh_p = h_r1[i];
		float y_p = h_r3[i];
		bool infects_p = y_p < thresh_p;
		
		float thresh_s = h_r2[i];
		float y_s = h_r4[i];
		bool infects_s = y_s < thresh_s;*/
		
		//current_day, i, type, infector, infector_status_p, infector_status_s, victim, y_p, thresh_p, infects_p, y_s, thresh_s, infects_s
		if(log_actions)
			fprintf(fActions, "%d, %d, %s, %d, %c, %c, %d, %d, %d\n", // %f, %f, %d, %f, %f, %d\n",
					current_day, i, action_type_to_char(type),
					inf, status_int_to_char(status_p), status_int_to_char(status_s),
					vic, gen_p, gen_s);
//					y_p, thresh_p, infects_p, y_s, thresh_s, infects_s);
		
		if(print_actions)
			printf("%2d\tinf: %6d\tstatus_p: %c\tstatus_s: %c\tvic: %6d\ttype: %s\n",
				i,  inf, status_int_to_char(status_p), status_int_to_char(status_s), vic, action_type_to_char(type));
	}
	
	fflush(fActions);
	fflush(fDebug);
}


//this will dump the actions to disk after they have been filtered for dupes, invalid infections, etc
//in combination with dump_actions(), this allows you to debug the filtering code
//This uses a lot of disk space, so outputs are compressed with zlib.
//Dump the logs to disk using the 'zcat' utility
void PandemicSim::dump_actions_filtered()
{
	//copy data to host
	h_vec h_ac_type(daily_actions);
	thrust::copy_n(daily_action_type.begin(), daily_actions, h_ac_type.begin());
	h_vec h_ac_vic(daily_actions);
	thrust::copy_n(daily_action_victim_index.begin(), daily_actions, h_ac_vic.begin());
	h_vec h_vic_gen_p(daily_actions);
	thrust::copy_n(daily_action_victim_gen_p.begin(), daily_actions, h_vic_gen_p.begin());
	h_vec h_vic_gen_s(daily_actions);
	thrust::copy_n(daily_action_victim_gen_s.begin(), daily_actions, h_vic_gen_s.begin());

	h_vec h_status_p = people_status_pandemic;
	h_vec h_status_s = people_status_seasonal;

	for(int i = 0; i < daily_actions; i++)
	{
		int vic = h_ac_vic[i];
		int type = h_ac_type[i];
		int gen_p = h_vic_gen_p[i];
		int gen_s = h_vic_gen_s[i];


		int v_status_p = h_status_p[vic];
		int v_status_s = h_status_s[vic];

		//current_day, i, type, infector, infector_status_p, infector_status_s, victim, victim_status_p, victim_status_s, gen_p, gen_s
		if(log_actions_filtered)
			fprintf(fActionsFiltered, "%d, %d, %s, %d, %c, %d, %c, %d\n",
			current_day, i, action_type_to_char(type),
			vic, 
			status_int_to_char(v_status_p), gen_p, 
			status_int_to_char(v_status_s), gen_s);

		if(print_actions_filtered)
			printf("%2d\tvic: %6d\tstatus_p: %c\tstatus_s: %c\ttype: %s\n",
			i,  vic, status_int_to_char(v_status_p), status_int_to_char(v_status_s), action_type_to_char(type));
	}

	fflush(fActionsFiltered);
	fflush(fDebug);
}


//this function controls the update process - it recovers infected people 
//and rebuilds the array for the next day
void PandemicSim::daily_rebuildInfectedArray()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "rebuild_infected_arr");

	if(debug_log_function_calls)
		debug_print("rebuilding infected array");

	//recover infected who have reached culmination
	int remaining = daily_recoverInfected();

	//get set of non-infected IDs to add to the index
	//this is to avoid creating duplicate people each with one of the infections

	//get unique action victim IDS to ensure set operation is unique
	d_vec unique_ids(daily_actions);
	IntIterator unique_ids_end = thrust::unique_copy(
		daily_action_victim_index.begin(),
		daily_action_victim_index.begin() + daily_actions,
		unique_ids.begin());

	int num_unique_ids = unique_ids_end - unique_ids.begin();
	d_vec unique_new_ids(num_unique_ids);

	//find action victims where the victim is not already in the infected array
	IntIterator unique_new_ids_end = thrust::set_difference(
		unique_ids.begin(),
		unique_ids_end,
		infected_indexes.begin(),
		infected_indexes.begin() + remaining,
		unique_new_ids.begin());

	int new_infected = unique_new_ids_end - unique_new_ids.begin();
	int new_infected_total = remaining + new_infected;

	if(CONSOLE_OUTPUT)
	{
		printf("%d still infected, %d new, %d total\n", remaining, new_infected, new_infected_total);
	}


	//copy in new indexes and set default values - will be overwritten
	thrust::copy(unique_new_ids.begin(), unique_new_ids_end, infected_indexes.begin() + remaining);
	thrust::fill(infected_days_pandemic.begin() + remaining, infected_days_pandemic.begin() + new_infected_total, DAY_NOT_INFECTED);
	thrust::fill(infected_days_seasonal.begin() + remaining, infected_days_seasonal.begin() + new_infected_total, DAY_NOT_INFECTED);
	thrust::fill(infected_generation_pandemic.begin() + remaining, infected_generation_pandemic.begin() + new_infected_total, GENERATION_NOT_INFECTED);
	thrust::fill(infected_generation_seasonal.begin() + remaining, infected_generation_seasonal.begin() + new_infected_total, GENERATION_NOT_INFECTED);

	//sort the new infected into the array
	thrust::sort(
		thrust::make_zip_iterator(thrust::make_tuple(
		infected_indexes.begin(), 
		infected_days_pandemic.begin(), infected_days_seasonal.begin(), 
		infected_generation_pandemic.begin(), infected_generation_seasonal.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(
		infected_indexes.begin() + new_infected_total,
		infected_days_pandemic.begin() + new_infected_total, infected_days_seasonal.begin() + new_infected_total,
		infected_generation_pandemic.begin() + new_infected_total, infected_generation_seasonal.begin() + new_infected_total)),
		FiveTuple_SortByFirst_Struct());

	//store new infected count
	infected_count = new_infected_total;

	//perform infection actions
	do_infection_actions(ACTION_INFECT_PANDEMIC);
	do_infection_actions(ACTION_INFECT_SEASONAL);

	if(debug_log_function_calls)
		debug_print("infected array rebuilt");

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}


//This takes the filtered actions stored in the actions array, and executes them on the infected
//It assumes that all actions are valid
//Furthermore, it assumes that new infected have been copied into the array with null data
void PandemicSim::do_infection_actions(int action)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "do_infection_actions");

	if(debug_log_function_calls)
		debug_print("processing infection actions");

	//binary search the victims we are applying the actions to
	vec_t victim_offsets(daily_actions);
	thrust::lower_bound(
		infected_indexes.begin(),
		infected_indexes.begin() + infected_count,
		daily_action_victim_index.begin(),
		daily_action_victim_index.begin() + daily_actions,
		victim_offsets.begin());

	//first, do the pandemic infections

	infectionTypePred typePred;
	typePred.reference_val = ACTION_INFECT_PANDEMIC;

	//convert victims status to infected
	thrust::replace_if(
		thrust::make_permutation_iterator(people_status_pandemic.begin(), daily_action_victim_index.begin()),
		thrust::make_permutation_iterator(people_status_pandemic.begin(), daily_action_victim_index.begin() + daily_actions),
		daily_action_type.begin(),
		typePred,
		STATUS_INFECTED);

	//copy their generation to the array
	thrust::scatter_if(
		daily_action_victim_gen_p.begin(),		//data
		daily_action_victim_gen_p.begin() + daily_actions,
		victim_offsets.begin(),			//map
		daily_action_type.begin(),		//stencil
		infected_generation_pandemic.begin(),		//output
		typePred);

	//mark tomorrow as their first day
	thrust::replace_if(
		thrust::make_permutation_iterator(infected_days_pandemic.begin(), victim_offsets.begin()),
		thrust::make_permutation_iterator(infected_days_pandemic.begin(), victim_offsets.end()),
		daily_action_type.begin(),
		typePred,
		current_day + 1);


	//do it again for seasonal
	typePred.reference_val = ACTION_INFECT_SEASONAL;
	thrust::replace_if(
		thrust::make_permutation_iterator(people_status_seasonal.begin(), daily_action_victim_index.begin()),
		thrust::make_permutation_iterator(people_status_seasonal.begin(), daily_action_victim_index.begin() + daily_actions),
		daily_action_type.begin(),
		typePred,
		STATUS_INFECTED);
	thrust::scatter_if(
		daily_action_victim_gen_s.begin(),
		daily_action_victim_gen_s.begin() + daily_actions,
		victim_offsets.begin(),
		daily_action_type.begin(),
		infected_generation_seasonal.begin(),
		typePred);
	thrust::replace_if(
		thrust::make_permutation_iterator(infected_days_seasonal.begin(), victim_offsets.begin()),
		thrust::make_permutation_iterator(infected_days_seasonal.begin(), victim_offsets.end()),
		daily_action_type.begin(),
		typePred,
		current_day + 1);

	if(debug_log_function_calls)
		debug_print("infection actions processed");

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, daily_actions);
}


//tihs function will recover infected people who have reached the culmination period
//should be called from rebuild_infected_arr()
int PandemicSim::daily_recoverInfected()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "recover_infected");

	if(debug_log_function_calls)
		debug_print("beginning recover_infected");

	recoverInfectedOp recoverOp;
	recoverOp.current_day = current_day;		//store current day in functor

	//recover pandemic strains
	thrust::transform(
		infected_days_pandemic.begin(),
		infected_days_pandemic.begin() + infected_count,
		thrust::make_zip_iterator(thrust::make_tuple(
			thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes.begin()),
			infected_days_pandemic.begin(), infected_generation_pandemic.begin())),			
		thrust::make_zip_iterator(thrust::make_tuple(
			thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes.begin()),
			infected_days_pandemic.begin(), infected_generation_pandemic.begin())),
		recoverOp);

	//recover seasonal strains
	thrust::transform(
		infected_days_seasonal.begin(),
		infected_days_seasonal.begin() + infected_count,
		thrust::make_zip_iterator(thrust::make_tuple(
			thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin()),
			infected_days_seasonal.begin(), infected_generation_seasonal.begin())),			
		thrust::make_zip_iterator(thrust::make_tuple(
			thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin()),
			infected_days_seasonal.begin(), infected_generation_seasonal.begin())),
		recoverOp);

	//remove people who are no longer infected
	d_vec infected_indexes_copy(infected_count);		//NOTE: not sure if this is necessary
	thrust::copy(infected_indexes.begin(), infected_indexes.begin() + infected_count, infected_indexes_copy.begin());	//TODO: check

	ZipIntFiveTupleIterator infected_begin = thrust::make_zip_iterator(thrust::make_tuple(
			infected_indexes.begin(), 
			infected_days_pandemic.begin(),	infected_days_seasonal.begin(), 
			infected_generation_pandemic.begin(), infected_generation_seasonal.begin()));
	ZipIntFiveTupleIterator infected_end = thrust::remove_if(
		infected_begin,
		thrust::make_zip_iterator(thrust::make_tuple(
			infected_indexes.begin() + infected_count, 
			infected_days_pandemic.begin() + infected_count, infected_days_seasonal.begin() + infected_count,
			infected_generation_pandemic.begin() + infected_count, infected_generation_seasonal.begin() + infected_count)),
		thrust::make_zip_iterator(thrust::make_tuple(
			thrust::make_permutation_iterator(people_status_pandemic.begin(), infected_indexes_copy.begin()),
			thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes_copy.begin()))),
		notInfectedPredicate());

	int infected_remaining = infected_end - infected_begin;

	if(debug_log_function_calls)
		debug_print("infected recovery complete");

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);

	return infected_remaining;
}

//tests the household/workplace locations for the first 10 infected people
void PandemicSim::test_locs()
{
	h_vec h_o = household_offsets;
	h_vec h_p = household_people;

	h_vec w_o = workplace_offsets;
	h_vec w_p = workplace_people;

	h_vec i_i = infected_indexes;
	h_vec p_hh = people_households;
	h_vec p_wp = people_workplaces; 

	for(int i = 0; i < 10; i++)
	{
		int idx = i_i[i];
		int hh = p_hh[idx];		//lookup household for 
		int wp = p_wp[idx];
		int wp_contains = 0;
		int hh_contains = 0;

/*		for(int j = 0; j < w_c[wp]; j++)
		{
			int offset = w_o[wp] + j;
			if(w_p[offset] == idx)
			{
				wp_contains = 1;
				break;
			}
		}

		for(int j = 0; j < h_c[hh]; j++)
		{
			int offset = h_o[hh] + j;
			if(h_p[offset] == idx)
			{
				hh_contains = 1;
				break;
			}
		}*/
		printf("%4d:\tID: %d\tHH: %4d\tWP: %4d\tHH_contains: %4d\tWP_contains: %4d\n",i,idx,hh,wp, hh_contains, wp_contains);
	}
}



//will resize the infected, contact, and action arrays to fit the entire population
void PandemicSim::setup_sizeGlobalArrays()
{
	//assume that at peak, about 3/4 of people will be affected
//	float peak_infection_ratio = 0.75f;
//	int expected_max_infection = (int) peak_infection_ratio * number_people;

	//setup people status:
	people_status_pandemic.resize(number_people);
	people_status_seasonal.resize(number_people);
	thrust::fill(people_status_pandemic.begin(), people_status_pandemic.end(), STATUS_SUSCEPTIBLE);
	thrust::fill(people_status_seasonal.begin(), people_status_seasonal.end(), STATUS_SUSCEPTIBLE);

	//setup output reproduction number counters
	reproduction_count_pandemic.resize(MAX_DAYS);
	reproduction_count_seasonal.resize(MAX_DAYS);
	thrust::fill(reproduction_count_pandemic.begin(), reproduction_count_pandemic.end(), 0);
	thrust::fill(reproduction_count_seasonal.begin(), reproduction_count_seasonal.end(), 0);

	//assume that worst-case everyone gets infected
	infected_indexes.resize(number_people);
	infected_days_pandemic.resize(number_people);
	infected_days_seasonal.resize(number_people);
	infected_generation_pandemic.resize(number_people);
	infected_generation_seasonal.resize(number_people);
	infected_daily_kval_sum.resize(number_people);

	int expected_max_contacts = number_people * MAX_CONTACTS_PER_DAY;

	//resize contact arrays
	daily_contact_infectors.resize(expected_max_contacts);
	daily_contact_victims.resize(expected_max_contacts);
	daily_contact_kvals.resize(expected_max_contacts);

	//resize action arrays
	daily_action_type.resize(expected_max_contacts);
	daily_action_victim_index.resize(expected_max_contacts);
	daily_action_victim_gen_p.resize(expected_max_contacts);
	daily_action_victim_gen_s.resize(expected_max_contacts);

	//weekend errands arrays tend to be very large, so pre-allocate them
	int num_weekend_errands = number_people * NUM_WEEKEND_ERRANDS;
	errand_people_table.resize(num_weekend_errands);
	errand_people_weekendHours.resize(num_weekend_errands);
	errand_people_destinations.resize(num_weekend_errands);

	errand_infected_locations.resize(num_weekend_errands);
	errand_infected_weekendHours.resize(num_weekend_errands);
	errand_infected_ContactsDesired.resize(number_people);

	errand_locationOffsets_multiHour.resize((number_workplaces * NUM_WEEKEND_ERRAND_HOURS) + 1);

	if(dump_contact_kernel_random_data)
	{
		debug_float1.resize(expected_max_contacts);
		debug_float2.resize(expected_max_contacts);
		debug_float3.resize(expected_max_contacts);
		debug_float4.resize(expected_max_contacts);
	}
}


void PandemicSim::debug_dump_array(const char * description, d_vec * gens_array, int array_count)
{
	h_vec host_array(array_count);
	thrust::copy_n(gens_array->begin(), array_count, host_array.begin());

	for(int i = 0; i < array_count; i++)
	{
		printf("%3d\t%s: %d\n",i,description, host_array[i]);
	}
}

void PandemicSim::debug_nullFillDailyArrays()
{
	thrust::fill(daily_contact_infectors.begin(), daily_contact_infectors.end(), -1);
	thrust::fill(daily_contact_victims.begin(), daily_contact_victims.end(), -1);
	thrust::fill(daily_contact_kvals.begin(), daily_contact_kvals.end(), CONTACT_TYPE_NONE);
	thrust::fill(infected_daily_kval_sum.begin(), infected_daily_kval_sum.end(), 0);

	thrust::fill(daily_action_type.begin(), daily_action_type.end(), ACTION_INFECT_NONE);
	thrust::fill(daily_action_victim_index.begin() ,daily_action_victim_index.end(), -1);
	thrust::fill(daily_action_victim_gen_p.begin(), daily_action_victim_gen_p.end(), GENERATION_NOT_INFECTED);
	thrust::fill(daily_action_victim_gen_s.begin(), daily_action_victim_gen_s.end(), GENERATION_NOT_INFECTED);

	thrust::fill(errand_infected_locations.begin(), errand_infected_locations.end(), -1);
	thrust::fill(errand_infected_weekendHours.begin(), errand_infected_weekendHours.end(), -1);
	thrust::fill(errand_infected_ContactsDesired.begin(), errand_infected_ContactsDesired.end(), -1);
}

void PandemicSim::setup_scaleSimulation()
{
	number_households = roundHalfUp_toInt(sim_scaling_factor * (double) number_households);

	int sum = 0;
	for(int business_type = 0; business_type < NUM_BUSINESS_TYPES; business_type++)
	{
		//for each type of business, scale by overall simulation scalar
		int original_type_count = roundHalfUp_toInt(h_workplace_type_counts[business_type]);
		int new_type_count = roundHalfUp_toInt(sim_scaling_factor * original_type_count);

		//if at least one business of this type existed in the original data, make sure at least one exists in the new data
		if(new_type_count == 0 && original_type_count > 0)
			new_type_count = 1;

		sum += new_type_count;
	}

	number_workplaces = sum;

	//calculate the offset of each workplace type
	thrust::exclusive_scan(
		h_workplace_type_counts,
		h_workplace_type_counts + NUM_BUSINESS_TYPES,
		h_workplace_type_offset);			
}

void PandemicSim::debug_dump_array_toTempFile(const char * filename, const char * description, d_vec * gens_array, int array_count)
{
	h_vec host_array(array_count);
	thrust::copy_n(gens_array->begin(), array_count, host_array.begin());

	FILE * fTemp = fopen(filename,"w");
	for(int i = 0; i < array_count; i++)
	{
		fprintf(fTemp,"%3d\t%s: %d\n",i,description, host_array[i]);
	}
	fclose(fTemp);
}


void PandemicSim::doWeekday_wholeDay()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "doWeekday_wholeDay");

	//generate errands and afterschool locations
	weekday_scatterAfterschoolLocations_wholeDay(&errand_people_destinations);
	weekday_scatterErrandDestinations_wholeDay(&errand_people_destinations);
	cudaDeviceSynchronize();

	debug_dump_array_toTempFile("../unsorted_dests.txt","errand dest", &errand_people_destinations, number_people * NUM_WEEKDAY_ERRAND_HOURS);

	//fish out the locations of the infected people
	weekday_doInfectedSetup_wholeDay(&errand_people_destinations, &errand_infected_locations, &errand_infected_ContactsDesired);
	if(log_contacts)
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

	debug_dump_array_toTempFile("../sorted_dests.txt", "errand_dest", &errand_people_destinations, number_people * NUM_WEEKDAY_ERRAND_HOURS);
	debug_dump_array_toTempFile("../loc_offsets.txt", "loc_offset", &errand_locationOffsets_multiHour, NUM_WEEKDAY_ERRAND_HOURS * number_workplaces);
	debug_dump_array_toTempFile("../inf_locs.txt", "loc", &errand_infected_locations, infected_count * NUM_WEEKDAY_ERRAND_HOURS);

	debug_dumpInfectedErrandLocs();

	int * infected_indexes_ptr = thrust::raw_pointer_cast(infected_indexes.data());
	kval_t * infected_daily_kval_sum_ptr = thrust::raw_pointer_cast(infected_daily_kval_sum.data());

	int * people_age_ptr = thrust::raw_pointer_cast(people_ages.data());

	int * household_lookup_ptr = thrust::raw_pointer_cast(people_households.data());
	int * household_people_ptr = thrust::raw_pointer_cast(household_people.data());
	int * household_offsets_ptr= thrust::raw_pointer_cast(household_offsets.data());

	int * workplace_lookup_ptr = thrust::raw_pointer_cast(people_workplaces.data());
	int * workplace_people_ptr = thrust::raw_pointer_cast(workplace_people.data());
	int * workplace_offsets_ptr = thrust::raw_pointer_cast(workplace_offsets.data());
	int * workplace_max_contacts_ptr = thrust::raw_pointer_cast(workplace_max_contacts.data());

	int * errand_infected_locs_ptr = thrust::raw_pointer_cast(errand_infected_locations.data());
	int * errand_loc_offsets_ptr = thrust::raw_pointer_cast(errand_locationOffsets_multiHour.data());
	int * errand_people_ptr = thrust::raw_pointer_cast(errand_people_table.data());
	int * errand_infected_contactsDesired_ptr = thrust::raw_pointer_cast(errand_infected_ContactsDesired.data());

	int * contacts_array_infector_ptr = thrust::raw_pointer_cast(daily_contact_infectors.data());
	int * contacts_array_victim_ptr = thrust::raw_pointer_cast(daily_contact_victims.data());
	int * contacts_array_kval_ptr = thrust::raw_pointer_cast(daily_contact_kvals.data());

	makeContactsKernel_weekday<<<cuda_blocks,cuda_threads>>>(
		infected_count, infected_indexes_ptr, people_age_ptr,
		household_lookup_ptr, household_offsets_ptr, household_people_ptr,
		workplace_max_contacts_ptr,workplace_lookup_ptr, 
		workplace_offsets_ptr, workplace_people_ptr,
		errand_infected_contactsDesired_ptr, errand_infected_locs_ptr,
		errand_loc_offsets_ptr,errand_people_ptr,
		number_workplaces,
		contacts_array_infector_ptr, contacts_array_victim_ptr, contacts_array_kval_ptr,
		infected_daily_kval_sum_ptr, rand_offset, number_people);

	const int rand_counts_consumed = 2;
	rand_offset += (rand_counts_consumed * infected_count);
	cudaDeviceSynchronize();

	if(log_contacts)
		validateContacts_wholeDay();

	debug_dump_array_toTempFile("../infected_kvals.txt","kval",&infected_daily_kval_sum, infected_count);

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::weekday_scatterAfterschoolLocations_wholeDay(d_vec * people_locs)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "weekday_scatterAfterschoolLocations_wholeDay");

	int * child_indexes_ptr = thrust::raw_pointer_cast(people_child_indexes.data());
	int * output_arr_ptr = thrust::raw_pointer_cast(people_locs->data());

	kernel_assignAfterschoolLocations_wholeDay<<<cuda_blocks,cuda_threads>>>(child_indexes_ptr,output_arr_ptr, number_children,number_people,rand_offset);
	rand_offset += number_children / 4;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,number_children);
}

void PandemicSim::weekday_scatterErrandDestinations_wholeDay(d_vec * people_locs)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "weekday_scatterAfterschoolLocations_wholeDay");

	int * adult_indexes_ptr = thrust::raw_pointer_cast(people_adult_indexes.data());
	int * output_arr_ptr = thrust::raw_pointer_cast(people_locs->data());

	kernel_assignErrandLocations_weekday_wholeDay<<<cuda_blocks,cuda_threads>>>(adult_indexes_ptr, number_adults, number_people ,output_arr_ptr, rand_offset);
	rand_offset += number_adults / 2;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,number_adults);
}

void PandemicSim::doWeekend_wholeDay()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "doWeekendErrands_wholeDay");

	//assign all weekend errands
	weekend_assignErrands(&errand_people_table, &errand_people_weekendHours, &errand_people_destinations);
	cudaDeviceSynchronize();

	//fish the infected errands out
	weekend_doInfectedSetup_wholeDay(&errand_people_weekendHours,&errand_people_destinations, &errand_infected_weekendHours, &errand_infected_locations, &errand_infected_ContactsDesired);
	if(log_contacts)
		debug_copyErrandLookup();
	cudaDeviceSynchronize();

	//each person gets 3 errands
	int num_weekend_errands_total = NUM_WEEKEND_ERRANDS * number_people;

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

	vec_t people_hour_offsets(NUM_WEEKEND_ERRAND_HOURS + 1);

	//find how many people are going on errands during each hour
	thrust::counting_iterator<int> count_it(0);
	thrust::lower_bound(
		errand_people_weekendHours.begin(),
		errand_people_weekendHours.begin() + num_weekend_errands_total,
		count_it,
		count_it + NUM_WEEKEND_ERRAND_HOURS,
		people_hour_offsets.begin());
	people_hour_offsets[NUM_WEEKEND_ERRAND_HOURS] = num_weekend_errands_total;

	for(int hour = 0; hour < NUM_WEEKEND_ERRAND_HOURS; hour++)
	{
		int location_offset_start = hour * number_workplaces;

		//search for the locations within this errand hour
		thrust::lower_bound(
			errand_people_destinations.begin() + people_hour_offsets[hour],
			errand_people_destinations.begin() + people_hour_offsets[hour+1],
			count_it,
			count_it + number_workplaces,
			errand_locationOffsets_multiHour.begin() + location_offset_start);
	}

	//launch kernel
	int * infected_indexes_ptr = thrust::raw_pointer_cast(infected_indexes.data());

	int * household_lookup_ptr = thrust::raw_pointer_cast(people_households.data());
	int * household_offsets_ptr = thrust::raw_pointer_cast(household_offsets.data());
	int * household_people_ptr = thrust::raw_pointer_cast(household_people.data());

	int * errand_infected_locations_ptr = thrust::raw_pointer_cast(errand_infected_locations.data());
	int * errand_infected_weekendHours_ptr = thrust::raw_pointer_cast(errand_infected_weekendHours.data());
	int * errand_infected_contactsDesired_profile_ptr = thrust::raw_pointer_cast(errand_infected_ContactsDesired.data());

	int * errand_people_ptr = thrust::raw_pointer_cast(errand_people_table.data());
	int * errand_locationOffsets_ptr = thrust::raw_pointer_cast(errand_locationOffsets_multiHour.data());
	int * errand_hour_offsets_ptr = thrust::raw_pointer_cast(people_hour_offsets.data());

	int * output_contact_infector_ptr = thrust::raw_pointer_cast(daily_contact_infectors.data());
	int * output_contact_victim_ptr = thrust::raw_pointer_cast(daily_contact_victims.data());
	int * output_contact_kval_ptr = thrust::raw_pointer_cast(daily_contact_kvals.data());
	cudaDeviceSynchronize();

	makeContactsKernel_weekend<<<cuda_blocks,cuda_threads>>>(
		infected_count, infected_indexes_ptr,
		household_lookup_ptr, household_offsets_ptr, household_people_ptr,
		errand_infected_weekendHours_ptr, errand_infected_locations_ptr, errand_infected_contactsDesired_profile_ptr,
		errand_locationOffsets_ptr,errand_people_ptr, errand_hour_offsets_ptr,
		number_workplaces,
		output_contact_infector_ptr, output_contact_victim_ptr, output_contact_kval_ptr,
		rand_offset);
	const int rand_counts_used = 2;
	rand_offset += (infected_count * rand_counts_used);
	cudaDeviceSynchronize();

	if(log_contacts)
		validateContacts_wholeDay();

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,infected_count);
}

void PandemicSim::weekday_doInfectedSetup_wholeDay(vec_t * lookup_array, vec_t * inf_locs, vec_t * inf_contacts_desired)
{
	int * inf_indexes_ptr = thrust::raw_pointer_cast(infected_indexes.data());
	int * loc_lookup_ptr = thrust::raw_pointer_cast(lookup_array->data());
	int * inf_locs_ptr = thrust::raw_pointer_cast(inf_locs->data());

	int * age_lookup_ptr = thrust::raw_pointer_cast(people_ages.data());
	int * inf_contacts_desired_ptr = thrust::raw_pointer_cast(inf_contacts_desired->data());

	kernel_doInfectedSetup_weekday_wholeDay<<<cuda_blocks, cuda_threads>>>(
		inf_indexes_ptr,infected_count,
		loc_lookup_ptr,age_lookup_ptr,number_people,
		inf_locs_ptr,inf_contacts_desired_ptr, rand_offset);

	const int rand_counts_used = infected_count / 4;
	rand_offset += rand_counts_used;
}

void PandemicSim::weekend_doInfectedSetup_wholeDay(vec_t * errand_hours, vec_t * errand_destinations, vec_t * infected_hours, vec_t * infected_destinations, vec_t * infected_contacts_desired)
{
	//first input: a list of all infected
	int * global_infected_indexes_ptr = thrust::raw_pointer_cast(infected_indexes.data());

	//second inputs: collated lookup tables for hours and destinations
	int * errand_hour_ptr = thrust::raw_pointer_cast(errand_hours->data());
	int * errand_dest_ptr = thrust::raw_pointer_cast(errand_destinations->data());

	//outputs: the hour of the errands and the destinations
	int * infected_hour_ptr = thrust::raw_pointer_cast(infected_hours->data());
	int * infected_destinations_ptr = thrust::raw_pointer_cast(infected_destinations->data());
	int * infected_contacts_desired_ptr = thrust::raw_pointer_cast(infected_contacts_desired->data());

	kernel_doInfectedSetup_weekend<<<cuda_blocks,cuda_threads>>>(
		global_infected_indexes_ptr,errand_hour_ptr,errand_dest_ptr,
		infected_hour_ptr, infected_destinations_ptr, infected_contacts_desired_ptr,
		infected_count, rand_offset);

	int rand_counts_consumed = infected_count / 4;
	rand_offset += rand_counts_consumed;
}

void PandemicSim::weekend_assignErrands(vec_t * errand_people, vec_t * errand_hours, vec_t * errand_destinations)
{
	int * errand_people_ptr = thrust::raw_pointer_cast(errand_people->data());
	int * errand_hours_ptr = thrust::raw_pointer_cast(errand_hours->data());
	int * errand_dests_ptr=  thrust::raw_pointer_cast(errand_destinations->data());

	kernel_assignErrands_weekend<<<cuda_blocks,cuda_threads>>>(errand_people_ptr,errand_hours_ptr,errand_dests_ptr, number_people,rand_offset);

	int rand_counts_consumed = 2 * number_people;
	rand_offset += rand_counts_consumed;
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

__device__ void device_assignContactsDesired_weekday_wholeDay(unsigned int rand_val, int myAge, int * output_contacts_desired)
{
	int contacts_hour[2];
	if(myAge == AGE_ADULT)
	{
		//get a profile between 0 and 2
		int contacts_profile = rand_val % 3;

		contacts_hour[0] = weekday_errand_contact_assignments_wholeDay[contacts_profile][0];
		contacts_hour[1] = weekday_errand_contact_assignments_wholeDay[contacts_profile][1];
	}
	else
	{
		contacts_hour[0] = business_type_max_contacts[BUSINESS_TYPE_AFTERSCHOOL];
		contacts_hour[1] = 0;
	}

	*(output_contacts_desired) = contacts_hour[0];
	*(output_contacts_desired) = contacts_hour[1];
}

__global__ void kernel_assignContactsDesired_weekday_wholeDay(int * infected_indexes_arr, int num_infected, int * age_lookup_arr, int * contacts_desired_arr, int rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos < num_infected / 4; myGridPos += gridDim.x * blockDim.x)
	{
		threefry2x64_ctr_t tf_ctr = {{((long) (myGridPos + rand_offset)), ((long) myGridPos + rand_offset)}};
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
		hour1 = business_type_max_contacts[BUSINESS_TYPE_AFTERSCHOOL];
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
__global__ void kernel_doInfectedSetup_weekday_wholeDay(int * infected_index_arr, int num_infected, int * loc_lookup_arr, int * ages_lookup_arr, int num_people, int * output_infected_locs, int * output_infected_contacts_desired, int rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= num_infected / 4; myGridPos += gridDim.x * blockDim.x)
	{
		//get 4 random numbers
		threefry2x64_ctr_t tf_ctr = {{((long) (myGridPos + rand_offset)), ((long) myGridPos + rand_offset)}};
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

inline char * action_type_to_char(int action)
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

struct weekend_getter : public thrust::unary_function<int, float>
{
	__device__ float operator() (const int& i)
	{
		return weekend_errand_pdf[i];
	}
};

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
										   int rand_offset)
{
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		int output_offset_base = MAX_CONTACTS_WEEKEND * myPos;

		int myIdx = infected_indexes[myPos];

		int loc_offset, loc_count;

		threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
		union{
			threefry2x64_ctr_t c;
			unsigned int i[4];
		} rand_union;
		//generate first set of random numbers

		threefry2x64_ctr_t tf_ctr_1 = {{(long) ((myPos * 2) + rand_offset), (long) ((myPos * 2) + rand_offset)}};
		rand_union.c = threefry2x64(tf_ctr_1, tf_k);

		//household: make three contacts
/*		device_lookupLocationData_singleHour(myIdx, household_lookup, household_offsets, &loc_offset, &loc_count);  //lookup location data for household
		device_selectRandomPersonFromLocation(
			myIdx, loc_offset, loc_count,rand_union.i[0], CONTACT_TYPE_HOME,
			household_people,
			output_infector_arr + output_offset_base + 0,
			output_victim_arr + output_offset_base + 0,
			output_kval_arr + output_offset_base + 0);
		device_selectRandomPersonFromLocation(
			myIdx, loc_offset, loc_count,rand_union.i[1], CONTACT_TYPE_HOME,
			household_people,
			output_infector_arr + output_offset_base + 1,
			output_victim_arr + output_offset_base + 1,
			output_kval_arr + output_offset_base + 1);
		device_selectRandomPersonFromLocation(
			myIdx, loc_offset, loc_count,rand_union.i[2], CONTACT_TYPE_HOME,
			household_people,
			output_infector_arr + output_offset_base + 2,
			output_victim_arr + output_offset_base + 2,
			output_kval_arr + output_offset_base + 2);

		//we need two more random numbers for the errands
		threefry2x32_key_t tf_k_32 = {{ SEED_DEVICE[0], SEED_DEVICE[1]}};
		threefry2x32_ctr_t tf_ctr_32 = {{((myPos * 2) + rand_offset + 1),((myPos * 2) + rand_offset + 1)}};		
		union{
			threefry2x32_ctr_t c;
			unsigned int i[2];
		} rand_union_32;
		rand_union_32.c = threefry2x32(tf_ctr_32, tf_k_32);

		int contacts_profile = infected_errand_contacts_profile[myPos];

		
		int errand_slot = weekend_errand_contact_assignments_wholeDay[contacts_profile][0]; //the errand number the contact will be made in
		device_lookupLocationData_weekendErrand(		//lookup the location data for this errand: we just need the offset and count
			myPos, errand_slot, 
			infected_errand_hours, infected_errand_destinations, 
			errand_people, number_locations, 
			errand_populationCount_exclusiveScan, 
			&loc_offset, &loc_count);
		device_selectRandomPersonFromLocation(			//select a random person at the location
			myIdx, loc_offset, loc_count, rand_union_32.i[0], CONTACT_TYPE_ERRAND,
			errand_people,
			output_infector_arr + output_offset_base + 3,
			output_victim_arr + output_offset_base + 3,
			output_kval_arr + output_offset_base + 3);

		//do it again for the second errand contact
		errand_slot = weekend_errand_contact_assignments_wholeDay[contacts_profile][1];		
		device_lookupLocationData_weekendErrand(			//lookup the location data for this errand
			myPos, errand_slot, 
			infected_errand_hours, infected_errand_destinations, 
			errand_people, number_locations, 
			errand_populationCount_exclusiveScan, 
			&loc_offset, &loc_count);
		device_selectRandomPersonFromLocation(			//select a random person at the location
			myIdx, loc_offset, loc_count, rand_union_32.i[1], CONTACT_TYPE_ERRAND,
			errand_people,
			output_infector_arr + output_offset_base + 4,
			output_victim_arr + output_offset_base + 4,
			output_kval_arr + output_offset_base + 4);*/
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

	int hour_data_position = (myPos * NUM_WEEKEND_ERRAND_HOURS) + errand_slot;

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
		*output_kval_sum += kval_lookup[contact_type];
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


__global__ void kernel_assignAfterschoolLocations_wholeDay(int * child_indexes_arr, int * output_array, int number_children, int number_people, int rand_offset)
{
	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} u;

	//get the number of afterschool locations and their offset in the business array
	int afterschool_count = business_type_count[BUSINESS_TYPE_AFTERSCHOOL];
	int afterschool_offset = business_type_count_offset[BUSINESS_TYPE_AFTERSCHOOL];

	//for each child
	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= number_children / 4; myGridPos += gridDim.x * blockDim.x)
	{
		threefry2x64_ctr_t tf_ctr = {{((long) myGridPos + rand_offset), ((long) myGridPos + rand_offset)}};
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

	int ret = frac * afterschool_count;		//find which afterschool location they're at, between 0 <= X < count
	ret = ret + afterschool_offset;		//add the offset to the first afterschool location

	*output_schedule = ret;					//store in the indicated output location
	*(output_schedule + number_people) = ret;	//children go to the same location for both hours, so put it in their second errand slot
}


__global__ void kernel_assignErrandLocations_weekday_wholeDay(int * adult_indexes_arr, int number_adults, int number_people, int * output_arr, int rand_offset)
{
	threefry2x64_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} u;

	//for each adult
	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= number_adults / 2; myGridPos += gridDim.x * blockDim.x)
	{
		threefry2x64_ctr_t tf_ctr = {{((long) myGridPos + rand_offset), ((long) myGridPos + rand_offset)}};
		u.c = threefry2x64(tf_ctr, tf_k);

		int myPos = myGridPos * 2;

		//fish out a destination
		if(myPos < number_adults)
		{
			int myAdultIdx_1 = adult_indexes_arr[myPos];
			device_fishWeekdayErrand(&u.i[0], &output_arr[myAdultIdx_1]);	//for adult index i, output the destination to arr[i]
			device_fishWeekdayErrand(&u.i[1], &output_arr[myAdultIdx_1 + number_people]);	//output a second destination to arr[i] for the second hour
		}
		//if still in bounds, assign another person
		if(myPos + 1 < number_adults)
		{
			int myAdultIdx_2 = adult_indexes_arr[myPos + 1];
			device_fishWeekdayErrand(&u.i[2], &output_arr[myAdultIdx_2]);
			device_fishWeekdayErrand(&u.i[3], &output_arr[myAdultIdx_2 + number_people]);
		}
	}
}


__device__ void device_fishWeekdayErrand(unsigned int * rand_val, int * output_destination)
{
	float yval = (float) *rand_val / UNSIGNED_MAX;

	int row = FIRST_WEEKDAY_ERRAND_ROW; //which business type

	while(yval > weekday_errand_pdf[row] && row < (NUM_BUSINESS_TYPES - 1))
	{
		yval -= weekday_errand_pdf[row];
		row++;
	}

	//figure out which business of this type we're at
	float frac = yval / weekday_errand_pdf[row];
	int business_num = frac * business_type_count[row];

	//add the offset to the first business of this type 
	int offset = business_type_count_offset[row];
	business_num += offset;

	*output_destination = business_num;
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

__global__ void kernel_assignErrands_weekend(int * people_indexes_arr, int * errand_hours_arr, int * errand_destination_arr, int num_people, int rand_offset)
{
	const int RAND_COUNTS_CONSUMED = 2;	//one for hours, one for destinations

	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_people; myPos += gridDim.x * blockDim.x)
	{
		int offset = myPos * NUM_WEEKEND_ERRANDS;
		int my_rand_offset = rand_offset + (myPos * RAND_COUNTS_CONSUMED);
		
		device_copyPeopleIndexes_weekend_wholeDay(people_indexes_arr + offset, myPos);
		device_assignErrandHours_weekend_wholeDay(errand_hours_arr + offset, my_rand_offset);
		device_assignErrandDestinations_weekend_wholeDay(errand_destination_arr + offset, my_rand_offset + 1);
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
											   int num_infected, int rand_offset)
{
	threefry2x64_key_t tf_k = {{(long) SEED_DEVICE[0], (long) SEED_DEVICE[1]}};
	union{
		threefry2x64_ctr_t c;
		unsigned int i[4];
	} rand_union;

	for(int myGridPos = blockIdx.x * blockDim.x + threadIdx.x;  myGridPos <= num_infected / 4; myGridPos += gridDim.x * blockDim.x)
	{
		int myRandOffset = rand_offset + myGridPos;
		threefry2x64_ctr_t tf_ctr = {{((long)myRandOffset), ((long)myRandOffset)}};
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
