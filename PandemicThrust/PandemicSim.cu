

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "threefry.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/pair.h>
#include <thrust/set_operations.h>

#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <time.h>


#include "simParameters.h"
#include "profiler.h"

#include "PandemicSim.h"
#include "thrust_functors.h"





#pragma region settings

//Simulation profiling master control - low performance overhead
#define PROFILE_SIMULATION 1

#define CONSOLE_OUTPUT 0

//controls master logging - everything except for profiler
#define GLOBAL_LOGGING 0
#define SANITY_CHECK 1

#define print_infected_info 0
#define log_infected_info GLOBAL_LOGGING

#define print_location_info 0
#define log_location_info GLOBAL_LOGGING

#define print_contact_kernel_setup 0
#define print_contacts 0
#define log_contacts GLOBAL_LOGGING
#define DOUBLECHECK_CONTACTS 0

#define print_actions 0
#define log_actions 0

#define print_actions_filtered 0
#define log_actions_filtered GLOBAL_LOGGING

#define log_people_info GLOBAL_LOGGING

//low overhead
#define debug_log_function_calls 1

#pragma endregion settings

int cuda_blocks = 32;
int cuda_threads = 32;



FILE * fDebug;

int SEED_HOST[SEED_LENGTH];
__device__ __constant__ int SEED_DEVICE[SEED_LENGTH];

__device__ __constant__ int business_type_count[NUM_BUSINESS_TYPES];				//stores number of each type of business
__device__ __constant__ int business_type_count_offset[NUM_BUSINESS_TYPES];			//stores location number of first business of this type
__device__ __constant__ float weekday_errand_pdf[NUM_BUSINESS_TYPES];				//stores PDF for weekday errand destinations
__device__ __constant__ float weekend_errand_pdf[NUM_BUSINESS_TYPES];				//stores PDF for weekend errand destinations
__device__ __constant__ float infectiousness_profile[CULMINATION_PERIOD];			//stores viral shedding profiles
__device__ __constant__ int weekend_errand_contact_assignments[6][3];				//stores number of 

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

float h_weekday_errand_pdf[NUM_BUSINESS_TYPES];
float h_weekend_errand_pdf[NUM_BUSINESS_TYPES];
float h_infectiousness_profile[CULMINATION_PERIOD];
int h_weekend_errand_contact_assignments[6][3];

#define CHILD_DATA_ROWS 5
float child_CDF[CHILD_DATA_ROWS];
int child_wp_types[CHILD_DATA_ROWS];

#define HH_TABLE_ROWS 9
int hh_adult_count[HH_TABLE_ROWS];
int hh_child_count[HH_TABLE_ROWS];
float hh_type_cdf[HH_TABLE_ROWS];


#define FIRST_WEEKDAY_ERRAND_ROW 9
#define FIRST_WEEKEND_ERRAND_ROW 9


PandemicSim::PandemicSim() 
{
	logging_openOutputStreams();

	if(PROFILE_SIMULATION)
		profiler.initStack();

	setup_loadParameters();

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
		profiler.beginFunction(-1,"setup_sim");
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

	if(log_people_info)
		dump_people_info();

	printf("%d people, %d households, %d workplaces\n",number_people, number_households, number_workplaces);

	
	//setup people status:
	people_status_pandemic.resize(number_people);
	people_status_seasonal.resize(number_people);
	thrust::fill(people_status_pandemic.begin(), people_status_pandemic.end(), STATUS_SUSCEPTIBLE);
	thrust::fill(people_status_seasonal.begin(), people_status_seasonal.end(), STATUS_SUSCEPTIBLE);

	
	//setup array for contacts
	daily_contact_infectors.resize(number_people * MAX_CONTACTS_PER_DAY);
	daily_contact_victims.resize(number_people * MAX_CONTACTS_PER_DAY);

	setup_initialInfected();
	setup_buildFixedLocations();	//household and workplace

	//setup output reproduction number counters
	generation_pandemic.resize(MAX_DAYS);
	generation_seasonal.resize(MAX_DAYS);
	thrust::fill(generation_pandemic.begin(), generation_pandemic.end(), 0);
	thrust::fill(generation_seasonal.begin(), generation_seasonal.end(), 0);

	//copy everything down to the GPU
	setup_pushDeviceData();

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
		fInfected = fopen("../debug_infected.csv.gz", "w");
		fprintf(fInfected, "current_day, i, idx, status_p, day_p, gen_p, status_s, day_s, gen_s\n");
	}

	if(log_location_info)
	{
		fLocationInfo = fopen("../debug_location_info.csv.gz","w");
		fprintf(fLocationInfo, "current_day, hour_index, i, offset, count, max_contacts\n");
	}

	if(log_contacts)
	{
		fContacts = fopen("../debug_contacts.csv.gz", "w");
		fprintf(fContacts, "current_day, num, contact_type, i, infector_idx, victim_idx, infector_loc, victim_loc, infector_found, victim_found\n");
	}


	if(log_actions)
	{
		fActions = fopen("../debug_actions.csv.gz", "w");
		fprintf(fActions, "current_day, i, type, infector, infector_status_p, infector_status_s, victim, y_p, thresh_p, infects_p, y_s, thresh_s, infects_s\n");
	}

	if(log_actions_filtered)
	{
		fActionsFiltered = fopen("../debug_filtered_actions.csv.gz", "w");
		fprintf(fActionsFiltered, "current_day, i, type, infector, infector_status_p, infector_status_s, victim, victim_status_p, victim_status_s\n");
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
	fscanf(fConstants, "%d%*c", &number_households);
	fscanf(fConstants, "%d", &number_workplaces);
	fclose(fConstants);


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
	child_CDF[0] = 0.24;
	child_CDF[1] = 0.47;
	child_CDF[2] = 0.72;
	child_CDF[3] = 0.85;
	child_CDF[4] = 1.0;

	//what workplace type children get for this age
	child_wp_types[0] = 3;
	child_wp_types[1] = 4;
	child_wp_types[2] = 5;
	child_wp_types[3] = 6;
	child_wp_types[4] = 7;

	//workplace PDF for adults
	workplace_type_pdf[0] = 0.06586;
	workplace_type_pdf[1] = 0.05802;
	workplace_type_pdf[2] = 0.30227;
	workplace_type_pdf[3] = 0.0048;
	workplace_type_pdf[4] = 0.00997;
	workplace_type_pdf[5] = 0.203;
	workplace_type_pdf[6] = 0.09736;
	workplace_type_pdf[7] = 0.10598;
	workplace_type_pdf[8] = 0.00681;
	workplace_type_pdf[9] = 0.02599;
	workplace_type_pdf[10] = 0;
	workplace_type_pdf[11] = 0.08749;
	workplace_type_pdf[12] = 0.03181;
	workplace_type_pdf[13] = 0.00064;

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

	//calculate the offset of each workplace type
	thrust::exclusive_scan(
		h_workplace_type_counts,
		h_workplace_type_counts + NUM_BUSINESS_TYPES,
		h_workplace_type_offset);			


	//pdf for weekday errand location generation
	//most entries are 0.0
	thrust::fill(h_weekday_errand_pdf, h_weekday_errand_pdf + NUM_BUSINESS_TYPES, 0.0);
	h_weekday_errand_pdf[9] = 0.61919;
	h_weekday_errand_pdf[11] = 0.27812;
	h_weekday_errand_pdf[12] = 0.06601;
	h_weekday_errand_pdf[13] = 0.03668;

	//pdf for weekend errand location generation
	//most entries are 0.0
	thrust::fill(h_weekend_errand_pdf, h_weekend_errand_pdf + NUM_BUSINESS_TYPES, 0.0f);
	h_weekend_errand_pdf[9] = 0.51493f;
	h_weekend_errand_pdf[11] = 0.25586f;
	h_weekend_errand_pdf[12] = 0.1162f;
	h_weekend_errand_pdf[13] = 0.113f;

	//viral shedding profile
	h_infectiousness_profile[0] = 0.002533572;
	h_infectiousness_profile[1] = 0.348252834;
	h_infectiousness_profile[2] = 0.498210218;
	h_infectiousness_profile[3] = 0.130145145;
	h_infectiousness_profile[4] = 0.018421298;
	h_infectiousness_profile[5] = 0.002158374;
	h_infectiousness_profile[6] = 0.000245489;
	h_infectiousness_profile[7] = 2.88922E-05;
	h_infectiousness_profile[8] = 3.61113E-06;
	h_infectiousness_profile[9] = 4.83901E-07;

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
	hh_type_cdf[0] = 0.279;
	hh_type_cdf[1] = 0.319;
	hh_type_cdf[2] = 0.628;
	hh_type_cdf[3] = 0.671;
	hh_type_cdf[4] = 0.8;
	hh_type_cdf[5] = 0.812;
	hh_type_cdf[6] = 0.939;
	hh_type_cdf[7] = 0.944;
	hh_type_cdf[8] = 1.0;

	//store all permutations of contact assignments
	h_weekend_errand_contact_assignments[0][0] = 2;
	h_weekend_errand_contact_assignments[0][1] = 0;
	h_weekend_errand_contact_assignments[0][2] = 0;

	h_weekend_errand_contact_assignments[1][0] = 0;
	h_weekend_errand_contact_assignments[1][1] = 2;
	h_weekend_errand_contact_assignments[1][2] = 0;

	h_weekend_errand_contact_assignments[2][0] = 0;
	h_weekend_errand_contact_assignments[2][1] = 0;
	h_weekend_errand_contact_assignments[2][2] = 2;

	h_weekend_errand_contact_assignments[3][0] = 1;
	h_weekend_errand_contact_assignments[3][1] = 1;
	h_weekend_errand_contact_assignments[3][2] = 0;

	h_weekend_errand_contact_assignments[4][0] = 1;
	h_weekend_errand_contact_assignments[4][1] = 0;
	h_weekend_errand_contact_assignments[4][2] = 1;

	h_weekend_errand_contact_assignments[5][0] = 0;
	h_weekend_errand_contact_assignments[5][1] = 1;
	h_weekend_errand_contact_assignments[5][2] = 1;
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

	//weekend errand contacts_desired assignments
	cudaMemcpyToSymbol(
		weekend_errand_contact_assignments,
		h_weekend_errand_contact_assignments,
		sizeof(int) * 6 * 3);

	//seeds
	cudaMemcpyToSymbol(
		SEED_DEVICE,
		SEED_HOST,
		sizeof(int) * SEED_LENGTH);

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
	h_people_hh.reserve(expected_people);
	h_people_wp.reserve(expected_people);

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

			number_people++;
		}

		//generate the children for this household
		for(int i = 0; i < hh_child_count[hh_type]; i++)
		{
			//assign school
			int wp = setup_assignSchool();
			h_people_wp.push_back(wp);

			//assign household
			h_people_hh.push_back(hh);

			//store as child
			h_child_indexes.push_back(number_people);

			number_people++;
		}
	}

	//trim arrays to data size, and transfer them to GPU
	h_people_wp.shrink_to_fit();
	people_workplaces = h_people_wp;

	h_people_hh.shrink_to_fit();
	people_households = h_people_hh;

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

int PandemicSim::setup_assignSchool()
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
	return ret + offset;
}


//Sets up the initial infection at the beginning of the simulation
//BEWARE: you must not generate dual infections with this code, all initial infected have one type to start
void PandemicSim::setup_initialInfected()
{
	//allocate space for initial infected
	int initial_infected = INITIAL_INFECTED_PANDEMIC + INITIAL_INFECTED_SEASONAL;
	infected_indexes.resize(initial_infected);
	infected_days_pandemic.resize(initial_infected);
	infected_days_seasonal.resize(initial_infected);
	infected_generation_pandemic.resize(initial_infected);
	infected_generation_seasonal.resize(initial_infected);

	//fill all infected with null info (not infected)
	thrust::fill(infected_days_pandemic.begin(), infected_days_pandemic.end(), DAY_NOT_INFECTED);
	thrust::fill(infected_days_seasonal.begin(), infected_days_seasonal.end(), DAY_NOT_INFECTED);
	thrust::fill(infected_generation_pandemic.begin(), infected_generation_pandemic.end(), GENERATION_NOT_INFECTED);
	thrust::fill(infected_generation_seasonal.begin(), infected_generation_seasonal.end(), GENERATION_NOT_INFECTED);

	//get N unique indexes - they should not be sorted
	h_vec h_init_indexes(initial_infected);
	n_unique_numbers(&h_init_indexes, initial_infected, number_people);
	infected_indexes = h_init_indexes;

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
		thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.end()),	//end at real end
		STATUS_INFECTED);

	//set day/generation seasonal to 0
	thrust::fill(
		infected_generation_seasonal.begin() + INITIAL_INFECTED_PANDEMIC,
		infected_generation_seasonal.end(),
		0);	//first generation
	thrust::fill(
		infected_days_seasonal.begin() + INITIAL_INFECTED_PANDEMIC,
		infected_days_seasonal.end(),
		INITIAL_DAY);		//day: 0

	//sort array after infection complete
	thrust::sort(
		thrust::make_zip_iterator(thrust::make_tuple(			//first
		infected_indexes.begin(),
		infected_days_pandemic.begin(),infected_days_seasonal.begin(),
		infected_generation_pandemic.begin(),infected_generation_seasonal.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(			//first
		infected_indexes.end(),
		infected_days_pandemic.end(),infected_days_seasonal.end(),
		infected_generation_pandemic.end(),infected_generation_seasonal.end())),
		FiveTuple_SortByFirst_Struct());

	infected_count = initial_infected;
}

//sets up the locations which are the same every day and do not change
//i.e. workplace and household
void PandemicSim::setup_buildFixedLocations()
{
	///////////////////////////////////////
	//work/////////////////////////////////
	workplace_offsets.resize(number_workplaces + 1);	//size arrays
	workplace_people.resize(number_people);

	thrust::sequence(workplace_people.begin(), workplace_people.end());	//fill array with IDs to sort

	calcLocationOffsets(
		&workplace_people,
		people_workplaces,
		&workplace_offsets,
		number_people, number_workplaces);

	//TODO:  max contacts are currently 3 for all workplaces
	workplace_max_contacts.resize(number_workplaces);
	thrust::fill(workplace_max_contacts.begin(), workplace_max_contacts.end(), 3);	//fill max contacts


	///////////////////////////////////////
	//home/////////////////////////////////
	household_offsets.resize(number_households + 1);
	household_people.resize(number_people);

	thrust::sequence(household_people.begin(), household_people.end());	//fill array with IDs to sort
	calcLocationOffsets(
		&household_people,
		people_households,
		&household_offsets,
		number_people, number_households);

	household_max_contacts.resize(number_households);
	thrust::fill(household_max_contacts.begin(), household_max_contacts.end(), 2);

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
		lookup_table_copy.end(),
		(*ids_to_sort).begin());

	//build count/offset table
	thrust::counting_iterator<int> count_iterator(0);
	thrust::lower_bound(		//find lower bound of each location
		lookup_table_copy.begin(),
		lookup_table_copy.end(),
		count_iterator,
		count_iterator + num_locs,
		(*location_offsets).begin());

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
	fflush(fPeopleInfo);
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

		//debug: dump infected info?
		if(log_infected_info || print_infected_info)
		{
			debug_validate_infected();
			dump_infected_info();
		}

		//MAKE CONTACTS DEPENDING ON TYPE OF DAY
		if(is_weekend())
		{
			doWeekend();
		}
		else
		{
			doWeekday();
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
		profiler.beginFunction(current_day, "calculate_final_reproduction");

	//copy to host
	thrust::host_vector<int> r_pandemic = generation_pandemic;
	thrust::host_vector<int> r_seasonal = generation_seasonal;

	FILE * out = fopen("../output_rn.csv", "w");
	fprintf(out, "gen, size_p, rn_p, size_s, rn_s\n");

	//loop and calculate reproduction
	int gen_size_p = INITIAL_INFECTED_PANDEMIC;
	int gen_size_s = INITIAL_INFECTED_SEASONAL;
	for(int i = 0; i < MAX_DAYS; i++)
	{
		float rn_pandemic = (float) r_pandemic[i] / gen_size_p;
		float rn_seasonal = (float) r_seasonal[i] / gen_size_s;

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
	bool sorted = thrust::is_sorted(infected_indexes.begin(), infected_indexes.end());
	debug_assert(sorted, "infected indexes are not sorted");

	//ASSERT:  INFECTED INDEXES ARE UNIQUE
	d_vec unique_indexes(infected_count);
	IntIterator end = thrust::unique_copy(infected_indexes.begin(), infected_indexes.end(), unique_indexes.begin());
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
	h_vec h_ii = infected_indexes;
	h_vec h_day_p = infected_days_pandemic;
	h_vec h_day_s = infected_days_seasonal;
	h_vec h_gen_p = infected_generation_pandemic;
	h_vec h_gen_s = infected_generation_seasonal;

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

//generate one weekday's worth of contacts
void PandemicSim::doWeekday()
{
	//tests household and workplaces for each infected person
	if(0)
		test_locs();

	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "doWeekday");

	//make workplace contacts
	makeContacts_byLocationMax(
		"workplace",
		&infected_indexes, infected_count,
		&workplace_people, &workplace_max_contacts,
		&workplace_offsets, number_workplaces, &people_workplaces);

	if(debug_log_function_calls)
		debug_print("workplace contacts complete"); 

	//do afterschool for children, and errands for adults
	doWeekdayErrands();

	if(debug_log_function_calls)
		debug_print("errand contacts complete");

	//make household contacts
	makeContacts_byLocationMax(
		"household",
		&infected_indexes, infected_count,
		&household_people, &household_max_contacts,
		&household_offsets, number_households,
		&people_households);

	if(debug_log_function_calls)
		debug_print("household contacts complete");

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

//Makes one day of contacts according to weekend schedule
void PandemicSim::doWeekend()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "doWeekend");

	//everyone makes household contacts
	makeContacts_byLocationMax(
		"household",
		&infected_indexes, infected_count,
		&household_people, &household_max_contacts,
		&household_offsets, number_households,
		&people_households); //hh

	//each person will make errand contacts on 3 of 6 possible errand hours
	doWeekendErrands();

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

//generates contacts for the 6 errand hours on a weekend
void PandemicSim::doWeekendErrands()
{
	
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "doWeekendErrands");

	//each person gets 3 errands
	int num_weekend_errands_total = NUM_WEEKEND_ERRANDS * number_people;

	//allocate arrays to store the errand locations
	vec_t errand_people(num_weekend_errands_total);
	vec_t errand_hours(num_weekend_errands_total);
	vec_t errand_locations(num_weekend_errands_total);

	//copy people's IDs and their 3 unique hours
	weekend_copyPeopleIndexes(&errand_people);
	weekend_generateThreeUniqueHours(&errand_hours);
	weekend_generateErrandDestinations(&errand_locations);
	cudaDeviceSynchronize();

	//extract a list of infected hours and destinations
	int infected_errand_count = infected_count * 3;
	vec_t infected_present_indexes(infected_errand_count);
	vec_t infected_destinations(infected_errand_count);
	vec_t infected_contacts_desired(infected_errand_count);
	vec_t infected_hour_offsets(NUM_WEEKEND_ERRAND_HOURS + 1);

	//set up the infected for the errands
	//copies the errand hours and destinations for infected into a separate array,
	//finds the number of infected making an errand each hour, and assigns the number of contacts desired
	weekendErrand_doInfectedSetup(
		&errand_hours, &errand_locations,
		&infected_present_indexes, &infected_destinations, &infected_contacts_desired,
		&infected_hour_offsets);

	//now sort the errand_people array into a large multi-hour location table
	thrust::sort_by_key(
		thrust::make_zip_iterator(thrust::make_tuple(errand_hours.begin(), errand_locations.begin())),		//key.begin
		thrust::make_zip_iterator(thrust::make_tuple(errand_hours.end(), errand_locations.end())),		//key.end
		errand_people.begin(),
		Pair_SortByFirstThenSecond_struct());									//data

	//count the number of people running errands on each hour
	//compute as count and offset for each hour
	vec_t errand_people_hour_offsets(NUM_WEEKEND_ERRAND_HOURS + 1);
	thrust::lower_bound(
		errand_hours.begin(),		//data.first
		errand_hours.end(),			//data.last
		thrust::counting_iterator<int>(0),		//search_val.first
		thrust::counting_iterator<int>(NUM_WEEKEND_ERRAND_HOURS), //search_val.last
		errand_people_hour_offsets.begin());
	errand_people_hour_offsets[NUM_WEEKEND_ERRAND_HOURS] = num_weekend_errands_total;



	if(print_location_info || log_location_info){
		printf("dumping weekend errand setup...\n");
		cudaDeviceSynchronize();
		dump_weekend_errands(errand_people, errand_hours, errand_locations, 5, number_people);
	}

	//for each hour, set up location arrays and make contacts
	for(int hour = 0; hour < NUM_WEEKEND_ERRAND_HOURS; hour++)
	{
		if(debug_log_function_calls)
		{
			fprintf(fDebug, "---------------------\nbeginning errand hour %d\n---------------------\n", hour);
			fflush(fDebug);
		}

		//get fancy string for debug output
		std::ostringstream s;
		s << "weekend_errand_";
		s << hour;
		std::string str = s.str();

		int people_offset = errand_people_hour_offsets[hour];		//index of first person for this hour
		int people_count = errand_people_hour_offsets[hour+1] - errand_people_hour_offsets[hour];		//number of people out on an errand this hour

		int infected_offset = infected_hour_offsets[hour];		//offset into infected_present of first infected person making contacts this hour
		int infected_present_count = infected_hour_offsets[hour+1] - infected_offset;	//number of infected making contacts this hour

		//build location offset table
		vec_t location_offsets(number_workplaces + 1);
		thrust::counting_iterator<int> count_iterator(0);
		thrust::lower_bound(
			errand_locations.begin() + people_offset,
			errand_locations.begin() + people_offset + people_count,
			count_iterator,
			count_iterator + number_workplaces,
			location_offsets.begin());
		location_offsets[number_workplaces] = people_count;

		clipContactsDesired_byLocationCount(
			infected_destinations.begin() + infected_offset,		//infected locations for this hour
			infected_count, &location_offsets,	//number of infected and the location offset table
			infected_contacts_desired.begin() + infected_offset);


		launchContactsKernel(
			str.c_str(),
			infected_present_indexes.begin() + infected_offset,		//iterator to first infected index
			infected_destinations.begin() + infected_offset,		//iterator to first infected destination
			infected_contacts_desired.begin() + infected_offset,	//iterator to first infected contacts_desired
			infected_present_count,									//number of infected present
			thrust::raw_pointer_cast(errand_people.begin() + people_offset),	//pointer to first person in the location_people table
			&location_offsets,		//pointer to location offset table
			number_workplaces);		//number_locations

//		TODO: validate contacts

		if(debug_log_function_calls)
			debug_print("errand hour complete");
	}

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}


//copies ID numbers into array for weekend errands
//we want three copies of each name, spaced out in collation style
__global__ void copy_weekend_errand_indexes_kernel(int * id_array, int N)
{	

	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{
		id_array[myPos] = myPos;
		id_array[myPos + N] = myPos;
		id_array[myPos + N + N] = myPos;
	}
}

//copies indexes 3 times into array, i.e. for IDS 1-3 produces array:
// 1 2 3 1 2 3 1 2 3
void PandemicSim::weekend_copyPeopleIndexes(vec_t * index_arr)
{
	int * index_arr_ptr = thrust::raw_pointer_cast(index_arr->data());
	copy_weekend_errand_indexes_kernel<<<cuda_blocks, cuda_threads>>>(index_arr_ptr, number_people);
}

//gets three UNIQUE errand hours 
__global__ void weekend_errand_hours_kernel(int * hours_array, int N, int rand_offset)
{
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;
	const int RNG_COUNTS_CONSUMED = 2;

	//for each person in simulation
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{
		threefry2x32_ctr_t tf_ctr = {{(myPos * RNG_COUNTS_CONSUMED) + rand_offset, (myPos * RNG_COUNTS_CONSUMED) + rand_offset }};
		u.c = threefry2x32(tf_ctr, tf_k);

		int first, second, third;

		//get first hour
		first = u.i[0] % NUM_WEEKEND_ERRAND_HOURS;

		//get second hour, if it matches then increment
		second = u.i[1] % NUM_WEEKEND_ERRAND_HOURS;
		if(second == first)
			second = (second + 1) % NUM_WEEKEND_ERRAND_HOURS;

		threefry2x32_ctr_t tf_ctr_2 = {{(myPos * RNG_COUNTS_CONSUMED) + rand_offset + 1, (myPos * RNG_COUNTS_CONSUMED) + rand_offset + 1}};
		u.c = threefry2x32(tf_ctr_2, tf_k);

		//get third hour, increment until it no longer matches
		third = u.i[0] % NUM_WEEKEND_ERRAND_HOURS;
		while(third == first || third == second)
		{
			third = (third + 1 ) % NUM_WEEKEND_ERRAND_HOURS;
		}

		//store in output array
		hours_array[myPos] = first;
		hours_array[myPos + N] = second;
		hours_array[myPos + N + N] = third;
	}
}

//gets 3 DIFFERENT errand hours for each person, collated order
//i.e. 1 2 3 1 2 3 1 2 3
void PandemicSim::weekend_generateThreeUniqueHours(vec_t * hours_array)
{
	int * loc_arr_ptr = thrust::raw_pointer_cast(hours_array->data());
	weekend_errand_hours_kernel<<<cuda_blocks, cuda_threads>>>(loc_arr_ptr, number_people, rand_offset);
	rand_offset += number_people * 2;
}

__device__ int device_fishWeekendErrandDestination(float y)
{
	int row = FIRST_WEEKEND_ERRAND_ROW;
	while(row < NUM_BUSINESS_TYPES - 1 && y > weekend_errand_pdf[row])
	{
		y -= weekend_errand_pdf[row];
		row++;
	}
	y = y / weekend_errand_pdf[row];
	int business_num = y * (float) business_type_count[row];
	business_num += business_type_count_offset[row];

	return business_num;
}

//gets three errand locations for each person in collation style
__global__ void weekend_errand_locations_kernel(int * location_array, int N, int rand_offset)
{
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;

	//the number of times we will call the RNG
	const int RNG_COUNTS_CONSUMED = 2;

	//for each person in simulation
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{
		//set up random number generator
		threefry2x32_ctr_t tf_ctr = {{(myPos * RNG_COUNTS_CONSUMED) + rand_offset, (myPos * RNG_COUNTS_CONSUMED) + rand_offset}};
		u.c = threefry2x32(tf_ctr, tf_k);

		//get first location - fish the type from the PDF
		float y_a =  (float) u.i[0] / UNSIGNED_MAX;
		int first = device_fishWeekendErrandDestination(y_a);

		//second 
		float y_b = (float) u.i[1] / UNSIGNED_MAX;
		int second = device_fishWeekendErrandDestination(y_b);

		//set up a second RNG run - use one number higher than the last RNG run
		threefry2x32_ctr_t tf_ctr_2 = {{(myPos * RNG_COUNTS_CONSUMED) + (rand_offset + 1), (myPos * RNG_COUNTS_CONSUMED) + (rand_offset + 1)}};
		u.c = threefry2x32(tf_ctr_2, tf_k);

		//third
		float y_c = (float) u.i[0] / UNSIGNED_MAX;
		int third = device_fishWeekendErrandDestination(y_c);

		location_array[myPos] = first;
		location_array[myPos + N] = second;
		location_array[myPos + N + N] = third;
	}
}

//gets 3 errand locations for each person according to PDF with collated order
//i.e. 1 2 3 1 2 3 1 2 3
void PandemicSim::weekend_generateErrandDestinations(vec_t * location_array)
{
	const int RNG_COUNTS_CONSUMED = 2;
	int * loc_arr_ptr = thrust::raw_pointer_cast(location_array->data());
	weekend_errand_locations_kernel<<<cuda_blocks, cuda_threads>>>(loc_arr_ptr, number_people, rand_offset);
	rand_offset += number_people * RNG_COUNTS_CONSUMED;
}

//prints some of the weekend errands to console
void PandemicSim::dump_weekend_errands(d_vec people, d_vec hours, d_vec locations, int num_to_print, int N)
{
	h_vec h_people = people;
	h_vec h_hours = hours;
	h_vec h_locs = locations;

	for(int i = 0; i < num_to_print; i++)
	{
		printf("i: %d\tidx: %6d\thour: %d\tloc: %3d\n",
			i, h_people[i], h_hours[i], h_locs[i]);
		printf("i: %d\tidx: %6d\thour: %d\tloc: %3d\n",
			i, h_people[i + N], h_hours[i + N], h_locs[i + N]);
		printf("i: %d\tidx: %6d\thour: %d\tloc: %3d\n",
			i, h_people[i + (N + N)], h_hours[i + (N + N)], h_locs[i + (N + N)]);
	}
}

//helper function that will automatically generate a list of contacts_desired
//uses the max_contacts number for each location
void PandemicSim::makeContacts_byLocationMax(const char * hour_string,
										vec_t *infected_list, int infected_list_count,
										vec_t *loc_people, vec_t *loc_max_contacts,
										vec_t *loc_offsets, int num_locs,
										vec_t *people_lookup)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "makeContacts_byLocationMax");

	//get the locations of infected people
	//ASSUMES: people_locations[i] contains the location of person index i (all people are present)
	vec_t infected_locations(infected_list_count);
	thrust::gather(
		(*infected_list).begin(),	//map.begin
		(*infected_list).begin() + infected_list_count,		//map.end
		(*people_lookup).begin(),
		infected_locations.begin());

	//get contacts desired for each infected person
	//return max_contacts, or count if count < max_contacts, or 0 if count == 1
	vec_t contacts_desired(infected_list_count);
	buildContactsDesired_byLocationMax(
		&infected_locations, infected_list_count,
		loc_offsets, loc_max_contacts,
		&contacts_desired);

	//get total number of contacts this hour 
	int num_new_contacts = thrust::reduce(contacts_desired.begin(), contacts_desired.end());

	//get raw pointer into the location table
	int * location_people_ptr = thrust::raw_pointer_cast((*loc_people).data());

	//make contacts
	launchContactsKernel(
		hour_string,
		infected_list, &infected_locations, 
		&contacts_desired, infected_list_count,
		location_people_ptr, loc_offsets, num_locs);

	//validate the contacts
	if(log_contacts || print_contacts)
	{
		validate_contacts(hour_string, loc_people, people_lookup, loc_offsets, num_new_contacts);
	}

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}
//method to set up afterschool activities and errands and make contacts
//children go to one randomly selected afterschool activity for 2 hours
//adults will go to two randomly selected errands, and make 2 contacts split between them
void PandemicSim::doWeekdayErrands()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "doWeekdayErrands");

	//errand arrays, will hold adults and children together
	vec_t errand_people_lookup(number_people);

	//generate child afterschool activities and first set of adult errands
	weekday_scatterAfterschoolLocations(&errand_people_lookup);
	weekday_scatterErrandLocations(&errand_people_lookup); 
	cudaDeviceSynchronize();

	//generate list of IDs to sort by location
	vec_t errand_location_people(number_people);
	thrust::sequence(errand_location_people.begin(), errand_location_people.end());

	//allocate and build the location offset array
	vec_t errand_location_offsets(number_workplaces + 1);
	calcLocationOffsets(&errand_location_people, errand_people_lookup, &errand_location_offsets, number_people, number_workplaces);

	if(debug_log_function_calls)
		debug_print("first errand locations built");

	//make children errands first
	vec_t infected_children(infected_count);
	filterInfectedByPopulationGroup("afterschool", &people_child_indexes, &infected_children);
	makeContacts_byLocationMax(
		"afterschool",
		&infected_children, infected_children.size(),
		&errand_location_people, &workplace_max_contacts,
		&errand_location_offsets, number_workplaces,
		&errand_people_lookup);


	/////////////now do adult errands, first hour

	//get a list of infected adults
	vec_t infected_adults(infected_count);
	filterInfectedByPopulationGroup("errand1", &people_adult_indexes, &infected_adults);
	int infected_adults_count = infected_adults.size();

	//assign 2 contacts randomly between the 2 errands
	vec_t errand_contacts_desired(infected_adults_count);
	assign_weekday_errand_contacts(&errand_contacts_desired, infected_adults_count);

	makeContacts_byContactsDesiredArray(
		"errand1",
		&infected_adults, infected_adults_count,
		&errand_location_people, &errand_contacts_desired,
		&errand_location_offsets, number_workplaces,
		&errand_people_lookup);

	if(debug_log_function_calls)
		debug_print("first errand complete and validated");


	//////////////generate second errand hour and make contacts

	//get new locations for adults
	weekday_scatterErrandLocations(&errand_people_lookup);
	cudaDeviceSynchronize();
	
	//rebuild the location offset array
	thrust::sequence(errand_location_people.begin(), errand_location_people.end());
	calcLocationOffsets(&errand_location_people, errand_people_lookup, &errand_location_offsets,number_people, number_workplaces);

	if(debug_log_function_calls)
		debug_print("second errand location array built");

	//reassign contacts: 2 - [contacts on first errand]
	thrust::transform(
		thrust::constant_iterator<int>(2),			//first.begin
		thrust::constant_iterator<int>(2) + infected_adults_count,	//first.end
		errand_contacts_desired.begin(),		//second.begin
		errand_contacts_desired.begin(),		//output - in place
		thrust::minus<int>());		

	makeContacts_byContactsDesiredArray(
		"errand2",		
		&infected_adults, infected_adults_count,
		&errand_location_people, &errand_contacts_desired,
		&errand_location_offsets, number_workplaces,
		&errand_people_lookup);

	if(debug_log_function_calls)
		debug_print("second errand complete and validated");

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

//kernel gets a random afterschool location for each child
__global__
	void get_afterschool_locations_kernel(int * child_indexes_arr, int * output_array, int number_children, int rand_offset)
{
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;

	//get the number of afterschool locations and their offset in the business array
	int afterschool_count = business_type_count[BUSINESS_TYPE_AFTERSCHOOL];
	int afterschool_offset = business_type_count_offset[BUSINESS_TYPE_AFTERSCHOOL];

	//for each child
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < number_children; myPos += gridDim.x * blockDim.x)
	{
		threefry2x32_ctr_t tf_ctr = {{myPos + rand_offset, myPos + rand_offset }};
		u.c = threefry2x32(tf_ctr, tf_k);

		//get a random float
		float frac = (float) u.i[0] / UNSIGNED_MAX;
		int ret = frac * afterschool_count;		//find which afterschool location they're at, between 0 <= X < count
		ret = ret + afterschool_offset;		//add the offset to the first afterschool location

		//scatter the afterschool locations into the lookup table according to the children's indexes
		int output_offset = child_indexes_arr[myPos];
		output_array[output_offset] = ret;
	}
}

//starts the kernel that gets a random afterschool location for each child.
void PandemicSim::weekday_scatterAfterschoolLocations(vec_t * people_locs)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "get_afterschool_locations");

	int * children_idxes_ptr = thrust::raw_pointer_cast(people_child_indexes.data());
	int * output_arr_ptr = thrust::raw_pointer_cast(people_locs->data());

	get_afterschool_locations_kernel<<<cuda_blocks, cuda_threads>>>(children_idxes_ptr, output_arr_ptr, number_children, rand_offset);

	rand_offset += number_children;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, number_children);
}


__device__ int device_fishWeekdayErrand(float yval)
{
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

	return business_num;
}

//gets a random weekday errand location for each adult, where N = number_adults
__global__ void get_weekday_errand_locations_kernel(int * adult_indexes_arr, int * output_arr, int number_adults, int rand_offset)
{

	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;

	//for each adult
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < number_adults; myPos += gridDim.x * blockDim.x)
	{
		int myRandOffset = myPos + rand_offset;

		threefry2x32_ctr_t tf_ctr = {{myRandOffset,myRandOffset}};
		u.c = threefry2x32(tf_ctr, tf_k);

		//fish out a business type
		float yval = (float) u.i[0] / UNSIGNED_MAX;
		int errand_destination = device_fishWeekdayErrand(yval);

		int output_offset = adult_indexes_arr[myPos];
		output_arr[output_offset] = errand_destination;

		//TODO:  use other rand number
	}
}

//gets a random weekday errand location for each adult
void PandemicSim::weekday_scatterErrandLocations(d_vec * people_lookup)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "get_weekday_errand_locations");

	int * adult_indexes_ptr = thrust::raw_pointer_cast((*people_adult_indexes.data()));
	int * output_arr_ptr = thrust::raw_pointer_cast((*people_lookup).data());

	//start kernel
	get_weekday_errand_locations_kernel<<<cuda_blocks, cuda_threads>>>(adult_indexes_ptr, output_arr_ptr, number_adults, rand_offset);
	rand_offset += number_adults;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, number_adults);
}


//for each infected adult, get the number of contacts they will make at their first errand
//valid outputs are {0,1,2}
__global__ void errand_contacts_kernel(int * array, int N, int global_rand_offset)
{		
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;

	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{

		int myRandOffset = myPos + global_rand_offset;

		threefry2x32_ctr_t tf_ctr = {{myRandOffset,0xfacecafe}};
		u.c = threefry2x32(tf_ctr, tf_k);

		array[myPos] = u.i[0] % 3;


		//TODO:  use other rand number
	}
}

//for each infected adult, get the number of contacts they will make at their first errand
//valid outputs are {0,1,2}
void PandemicSim::assign_weekday_errand_contacts(d_vec * contacts_desired, int num_infected_adults)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "assign_weekday_errand_contacts");

	int * arr_ptr = thrust::raw_pointer_cast(contacts_desired->data());

	//start kernel
	errand_contacts_kernel<<<cuda_blocks, cuda_threads>>>(arr_ptr, num_infected_adults, rand_offset);
	cudaDeviceSynchronize();

	rand_offset += num_infected_adults;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, num_infected_adults);
}

__global__ void build_contacts_desired_kernel(
			int *infected_location_arr, 
			int *loc_offset_arr, int *loc_max_contacts_arr, 
			int *contacts_desired_arr,
			int num_infected)
{
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		int myLoc = infected_location_arr[myPos];

		int desired = loc_max_contacts_arr[myPos];
		int loc_count = loc_offset_arr[myLoc + 1] - loc_offset_arr[myLoc];

		if(loc_count == 1)
			contacts_desired_arr[myPos] = 0;
		else
			contacts_desired_arr[myPos] = __min(desired, loc_count);
	}
}

void PandemicSim::buildContactsDesired_byLocationMax(
	vec_t *infected_locations, int num_infected,
	vec_t *loc_offsets,
	vec_t *loc_max_contacts,
	vec_t *contacts_desired)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "build_contacts_desired");

	int * infected_loc_ptr = thrust::raw_pointer_cast((*infected_locations).data());
	int * loc_offsets_ptr = thrust::raw_pointer_cast((*loc_offsets).data());
	int * max_contacts_ptr = thrust::raw_pointer_cast((*loc_max_contacts).data());
	int * contacts_desired_ptr = thrust::raw_pointer_cast((*contacts_desired).data());

	build_contacts_desired_kernel<<<cuda_blocks,cuda_threads>>>(infected_loc_ptr,loc_offsets_ptr,max_contacts_ptr,contacts_desired_ptr,num_infected);
	cudaDeviceSynchronize();

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, num_infected);
}

//prints some of the weekend errands to console
void dump_weekend_errands(d_vec people, d_vec hours, d_vec locations, int num_to_print, int N)
{
	h_vec h_people = people;
	h_vec h_hours = hours;
	h_vec h_locs = locations;

	for(int i = 0; i < num_to_print; i++)
	{
		printf("i: %d\tidx: %6d\thour: %d\tloc: %3d\n",
			i, h_people[i], h_hours[i], h_locs[i]);
		printf("i: %d\tidx: %6d\thour: %d\tloc: %3d\n",
			i, h_people[i + N], h_hours[i + N], h_locs[i + N]);
		printf("i: %d\tidx: %6d\thour: %d\tloc: %3d\n",
			i, h_people[i + (N + N)], h_hours[i + (N + N)], h_locs[i + (N + N)]);
	}
}

//Randomly select people at the same location as the infector as contacts
//NOTE: assumes that a location count of 1 means that contacts_desired = 0 
//		for anyone at that location otherwise, someone could select themselves
__global__ void victim_index_kernel(
	int * infector_indexes_arr, int * contacts_desired_arr, int * output_offset_arr, int * infector_loc_arr,
	int * location_offsets_arr, int * location_people_arr,
	int * contact_infector_arr, int * contact_idx_arr,
	int N, int rand_offset)
{
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;

	//for each infector
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{
		//info about the infector 
		int infector_idx = infector_indexes_arr[myPos];
		int output_offset = output_offset_arr[myPos];
		int contacts_desired = contacts_desired_arr[myPos];	//stores number of contacts we still want

		//info about the location
		int myLoc = infector_loc_arr[myPos];
		int loc_offset = location_offsets_arr[myLoc];
		int loc_count = location_offsets_arr[myLoc+1] - loc_offset;	//

		//get N contacts
		while(contacts_desired > 0)
		{
			threefry2x32_ctr_t tf_ctr = {{output_offset + rand_offset, output_offset + rand_offset + 1}};
			u.c = threefry2x32(tf_ctr, tf_k);

			//randomly select person at location
			int victim_offset = u.i[0] % loc_count;
			int victim_idx = location_people_arr[loc_offset + victim_offset];

			//if we have selected the infector, we need to get a different person
			if(victim_idx == infector_idx)
			{
				//get the next person
				victim_offset = (victim_offset + 1);
				if(victim_offset == loc_count)		//wrap around if needed
					victim_offset = 0;
				victim_idx = location_people_arr[loc_offset + victim_offset];
			}

			//save contact into output array
			contact_infector_arr[output_offset] = infector_idx;
			contact_idx_arr[output_offset] = victim_idx;
			output_offset++;	//move to next output slot
			contacts_desired--;	//decrement contacts remaining

			//R123 returns 2x32 bit numbers, so if we need another one we can use that
			if(contacts_desired > 0)
			{
				//randomly select person at that location
				victim_offset = u.i[1] % loc_count;
				int victim_idx = location_people_arr[loc_offset + victim_offset];

				//if we have selected the infector, get the next person
				if(victim_idx == infector_idx)
				{
					victim_offset = (victim_offset + 1);
					if(victim_offset == loc_count)		//wrap around
						victim_offset = 0;
					victim_idx = location_people_arr[loc_offset + victim_offset];
				}

				//save contact into output array
				contact_infector_arr[output_offset] = infector_idx;
				contact_idx_arr[output_offset] = victim_idx;
				output_offset++;
				contacts_desired--;
			}

		}  //end while(contacts_desired)
	}	//end for each infector
}



//this function does final setup and then calls the kernel to make contacts
//contacts can be validated afterwards if desired
void PandemicSim::launchContactsKernel(
	const char * hour_string,
	IntIterator infected_indexes_present_begin, IntIterator infected_locations_begin, 
	IntIterator infected_contacts_desired_begin, int infected_present_count,
	int * loc_people_ptr, vec_t *location_offsets, int num_locs)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "make_contacts");

	if(debug_log_function_calls)
		debug_print("inside make_contacts");

	//each infected needs to know where to put their contacts in the contacts array
	vec_t output_offsets(infected_present_count);
	thrust::exclusive_scan(
		infected_contacts_desired_begin,
		infected_contacts_desired_begin + infected_present_count,
		output_offsets.begin());

	//how many contacts we are about to make in total
	int new_contacts = thrust::reduce(infected_contacts_desired_begin, infected_contacts_desired_begin + infected_present_count);

	//if the contacts array is too small, resize it to fit
	if(daily_contacts + new_contacts > daily_contact_infectors.size())
	{
		printf("warning:  contacts too small, old size: %d new size: %d\n", daily_contact_infectors.size(), daily_contacts + new_contacts);
		daily_contact_infectors.resize(daily_contacts + new_contacts);
		daily_contact_victims.resize(daily_contacts + new_contacts);
		//daily_contact_kvals.resize(daily_contacts + new_contacts);
	}

	//get raw pointers to infector data
	int * infected_idx_ptr = thrust::raw_pointer_cast(infected_indexes_present_begin);
	int * infected_loc_ptr = thrust::raw_pointer_cast(infected_locations_begin);
	int * infected_contacts_ptr = thrust::raw_pointer_cast(infected_contacts_desired_begin);
	int * output_offsets_ptr = thrust::raw_pointer_cast(output_offsets.data());

	//get raw pointers to location data
	int * loc_offsets_ptr = thrust::raw_pointer_cast((*location_offsets).data());

	//get raw pointers into output array, and advance past spots already filled
	int * contact_infector_ptr = thrust::raw_pointer_cast(daily_contact_infectors.data());
	contact_infector_ptr += daily_contacts;
	int * contact_victim_ptr = thrust::raw_pointer_cast(daily_contact_victims.data());
	contact_victim_ptr += daily_contacts;

	if(print_contact_kernel_setup)
	{
		dump_contact_kernel_setup(
			hour_string,
			infected_indexes_present_begin, infected_locations_begin, 
			infected_contacts_desired_begin, &output_offsets,
			loc_people_ptr, location_offsets, 
			infected_present_count);
	}

	if(CONSOLE_OUTPUT)
	{
		printf("daily contacts: %d\t new_contacts: %d\t rand_offset: %d\n", daily_contacts, new_contacts, rand_offset);

		//	printf("infected_present size: %d\ninfected present: %d\ninfected_locations size: %d\ninfected_contacts_desired size: %d\nloc_offsets size: %d\nloc_counts size: %d\nnum_locs: %d\n",
		//			infected_indexes_present.size(), infected_present, infected_locations.size(), infected_contacts_desired.size(), location_offsets.size(), location_counts.size(), num_locs);

	}


	//call the kernel
	if(debug_log_function_calls)
		debug_print("calling contacts kernel");
	victim_index_kernel<<<cuda_blocks, cuda_threads>>>(
		infected_idx_ptr, infected_contacts_ptr, output_offsets_ptr, infected_loc_ptr,
		loc_offsets_ptr, loc_people_ptr, 
		contact_infector_ptr, contact_victim_ptr, 
		infected_present_count, rand_offset);
	cudaDeviceSynchronize();

	if(debug_log_function_calls)
		debug_print("contacts kernel sync'd");

	//increment the contacts counter and the RNG counter
	rand_offset += new_contacts;
	daily_contacts += new_contacts;

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_present_count); 
}

//dumps the setup that is passed to the make_contacts kernel
void PandemicSim::dump_contact_kernel_setup(
	const char * hour_string,
	IntIterator infected_indexes_present_begin, IntIterator infected_locations_begin,
	IntIterator infected_contacts_desired_begin, d_vec * output_offsets,
	int * location_people_ptr, d_vec *location_offsets,
	int num_infected)
{
	h_vec i_idx(num_infected);
	thrust::copy(infected_indexes_present_begin, infected_indexes_present_begin + num_infected, i_idx.begin());

	h_vec i_loc(num_infected);
	thrust::copy(infected_locations_begin, infected_locations_begin + num_infected, i_loc.begin());

	h_vec i_contacts_desired(num_infected);
	thrust::copy(infected_contacts_desired_begin, infected_contacts_desired_begin + num_infected, i_contacts_desired.begin());

	h_vec o_o = (*output_offsets);

	h_vec loc_o = (*location_offsets);

	for(int i = 0; i < num_infected; i++)
	{
		int idx = i_idx[i];
		int loc = i_loc[i];
		int c_d = i_contacts_desired[i];
		int offset = o_o[i];

		int loc_offset = loc_o[loc];
		int loc_count = loc_o[loc+1] - loc_offset;

		//		int contains_person = 0;
		//		for(int j = loc_offset; j < loc_offset + loc_count; j++)
		//		{
		//			if(loc_p[j] == idx)
		//			{
		//				contains_person = 1;
		//				break;
		//			}
		//		}

		printf("inf %3d\tidx: %4d\tloc: %3d\tloc_offset: %5d\tloc_count: %5d\tdesired: %d\tout_offset: %d\n",
			i, idx, loc, loc_offset, loc_count, c_d, offset);
	}

}



//This method checks that the N contacts most recently generated are valid
//It should be called immediately after make_contacts
//Specifically, it checks that the infector and victim have the same location
//if the DOUBLECHECK_CONTACTS define is set to true, it will actually look in the location table to be sure (very slow)
void PandemicSim::validate_contacts(const char * hour_string, d_vec *d_people, d_vec *d_lookup, d_vec *d_offsets, int N)
{	
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "validate_contacts");
	if(debug_log_function_calls)
		debug_print("validating contacts");

	//copy data to host
	h_vec h_people = (*d_people);
	h_vec h_lookup = (*d_lookup);
	h_vec h_offsets = (*d_offsets);

	h_vec contact_infectors = daily_contact_infectors;
	h_vec contact_victims = daily_contact_victims;

	if(print_contacts)
		printf("validating %d contacts...\n", N);



	if(0)
	{
		printf("h_people size: %d\n", h_people.size());
		printf("h_lookup size: %d\n", h_lookup.size());
		printf("h_offsets size: %d\n", h_offsets.size());
		printf("c_i size: %d\n", contact_infectors.size());
		printf("c_v size: %d\n", contact_victims.size());
		printf("daily contacts: %d\n", daily_contacts);
	}


	for(int i = daily_contacts - N; i < daily_contacts; i++)
	{
		int infector_idx = contact_infectors[i];
		int i_loc = h_lookup[infector_idx];

		int victim_idx = contact_victims[i];
		int v_loc = h_lookup[victim_idx];

		int i_found = 0;
		int v_found = 0;

		if(DOUBLECHECK_CONTACTS)
		{
			if(i_loc == v_loc)
			{
				int loc_o = h_offsets[i_loc];
				int loc_c = h_offsets[i_loc + 1] - loc_o;
				for(int k = loc_o; k < loc_o + loc_c; k++)
				{
					if(h_people[k] == infector_idx)
						i_found = 1;
					if(h_people[k] == victim_idx)
						v_found = 1;

					if(i_found && v_found)
						break;
				}
			}
		}
		else
		{
			i_found = i_loc == v_loc;
			v_found = i_loc == v_loc;
		}



		//current_day, i, infector_idx, victim_idx, infector_loc, victim_loc, infector_found, victim_found
		if(log_contacts)
			fprintf(fContacts, "%d, %d, %s, %d, %d, %d, %d, %d, %d\n",
			current_day, i, hour_string, infector_idx, victim_idx, i_loc, v_loc, i_found, v_found);
		if(print_contacts)
			printf("%d\tinf_idx: %5d\tvic_idx: %5d\ti_loc: %5d\tv_loc: %5d\ti_found: %d v_found: %d\n",
			i, infector_idx, victim_idx, i_loc, v_loc, i_found, v_found);
	}

	if(debug_log_function_calls)
		debug_print("contact validation complete");

	fflush(fContacts);
	fflush(fDebug);

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, N);
}


//This method consumes the accumulated contacts, and causes infections and recovery to occur
void PandemicSim::dailyUpdate()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "dailyUpdate");

	if(debug_log_function_calls)
		debug_print("beginning daily update");

	//TODO:  ??? not sure what I was going for here
	//sorts the k-vals in parallel once they are coded
	//	vec_t infector_copy(daily_contacts);
	//	thrust::copy(daily_infector_indexes.begin(), daily_infector_indexes.end(), infector_copy.begin());
	//	thrust::sort_by_key(
	//			infector_copy.begin(),
	//			infector_copy.end(),
	//			daily_contact_kvals.begin());

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

	//copies the generation from the infector to the new victim
	daily_assignVictimGenerations();

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

//after the contacts have been converted to actions, build an array of generations for the victims
void PandemicSim::daily_assignVictimGenerations()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "build_action_generations");

	if(debug_log_function_calls)
		debug_print("building generations for actions");

	//size the action generation arrays
	daily_victim_gen_p.resize(daily_actions);
	daily_victim_gen_s.resize(daily_actions);

	//	thrust::fill(daily_victim_gen_p.begin(), daily_victim_gen_p.end(), -2);
	//	thrust::fill(daily_victim_gen_s.begin(), daily_victim_gen_s.end(), -2);


	//find the ID of the infector
	d_vec inf_offset(daily_actions);
	thrust::lower_bound(
		infected_indexes.begin(),
		infected_indexes.end(),
		daily_action_infectors.begin(),
		daily_action_infectors.begin() + daily_actions,
		inf_offset.begin());

	//get the generation of the infector, add one, store in array
	thrust::transform(
		daily_action_type.begin(),
		daily_action_type.begin() + daily_actions,
		thrust::make_zip_iterator(thrust::make_tuple(
		thrust::make_permutation_iterator(infected_generation_pandemic.begin(), inf_offset.begin()),
		thrust::make_permutation_iterator(infected_generation_seasonal.begin(), inf_offset.begin()))),
		thrust::make_zip_iterator(thrust::make_tuple(
		daily_victim_gen_p.begin(), daily_victim_gen_s.begin())),
		generationOp());


	if(debug_log_function_calls)
		debug_print("generation actions built");

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, daily_actions);

}

//once the contacts have been properly set up, this kernel determines whether each contact
//is successful, and then outputs actions
__global__ void contacts_to_actions_kernel(
		int * contacts_desired_arr, int * output_offset_arr,
		int * day_p_arr, int * day_s_arr,
		int * action_types_arr,  // float * rand_1, float * rand_2, float* rand_3, float * rand_4,
		int N, int global_rand_offset, int current_day)
{
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;
	
	//for each infected person
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{
		//how many contacts we're tyring to make
		int contacts_desired = contacts_desired_arr[myPos];
		if(contacts_desired > 0)
		{
			//output_offset holds the index where this action should be stored
			int output_offset = output_offset_arr[myPos];
			
			//these are the day this person was infected (from 0 to MAX_DAYS) - NOT which day of infection they are on
			int day_p = day_p_arr[myPos];
			int day_s = day_s_arr[myPos];
			
			//start with zero probability of infection - if the person is infected, increase it
			float inf_prob_p = 0.0;
		
			if(day_p >= 0)
			{
				inf_prob_p = (infectiousness_profile[current_day - day_p] * BASE_REPRODUCTION_DEVICE[0]) / (float) contacts_desired;
//				inf_prob_p = day_p;
			}
			
			//same for seasonal
			float inf_prob_s = 0.0;
	
			if(day_s >= 0)
			{
				inf_prob_s = (infectiousness_profile[current_day - day_s] * BASE_REPRODUCTION_DEVICE[1]) / (float) contacts_desired;
//				inf_prob_s = day_s;
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
				}
				else if(infects_pandemic)
				{
					action_types_arr[output_offset] = ACTION_INFECT_PANDEMIC;
				}
				else if(infects_seasonal)
				{
					action_types_arr[output_offset] = ACTION_INFECT_SEASONAL;
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
}

//starts the kernel to convert contacts to actions
void PandemicSim::daily_contactsToActions()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "contacts_to_actions");
	
	if(debug_log_function_calls)
		debug_print("beginning contacts-to-action setup");
	
	//sort contacts by infector
	thrust::sort_by_key(
			daily_contact_infectors.begin(),
			daily_contact_infectors.begin() + daily_contacts,
			daily_contact_victims.begin());
	
	//size our actino output arrays
	daily_action_type.resize(daily_contacts);
	daily_action_infectors.resize(daily_contacts);
	daily_action_victims.resize(daily_contacts);

	//get lower bound of each infector, aka index
	d_vec infector_contacts_offset (infected_count);
	thrust::lower_bound(
			daily_contact_infectors.begin(),		//data.begin
			daily_contact_infectors.begin() + daily_contacts,		//data.end
			infected_indexes.begin(),		//value to search for
			infected_indexes.end(),			//value to search
			infector_contacts_offset.begin());

	//get upper bound of each infector
	d_vec infector_contacts_count(infected_count);
	thrust::upper_bound(
			daily_contact_infectors.begin(),		//data.begin
			daily_contact_infectors.begin() + daily_contacts,		//data.end
			infected_indexes.begin(),
			infected_indexes.end(),
			infector_contacts_count.begin());

	//difference is the count
//	vec_t infector_contacts_today(infected_count);
	thrust::transform(
			infector_contacts_count.begin(),
			infector_contacts_count.end(),
			infector_contacts_offset.begin(),
			infector_contacts_count.begin(),
			thrust::minus<int>());
	
	
	//convert thrust vectors to raw pointers
	
	//get victim 
//	int * contact_victims_ptr = thrust::raw_pointer_cast(daily_contact_victims.data());
	
	int * inf_offset_ptr = thrust::raw_pointer_cast(infector_contacts_offset.data());
	int * inf_count_ptr = thrust::raw_pointer_cast(infector_contacts_count.data());
	
	//get 
	int * day_p_ptr = thrust::raw_pointer_cast(infected_days_pandemic.data());
	int * day_s_ptr = thrust::raw_pointer_cast(infected_days_seasonal.data());
	
	//stores what type of action resulted
	int * actions_type_ptr = thrust::raw_pointer_cast(daily_action_type.data());
	
	//for debug: we can dump some internal stuff
//	thrust::device_vector<float> rand1(daily_contacts);
//	thrust::device_vector<float> rand2(daily_contacts);
//	thrust::device_vector<float> rand3(daily_contacts);
//	thrust::device_vector<float> rand4(daily_contacts);
	
//	float* rand1ptr = thrust::raw_pointer_cast(rand1.data());
//	float* rand2ptr = thrust::raw_pointer_cast(rand2.data());
//	float* rand3ptr = thrust::raw_pointer_cast(rand3.data());
//	float* rand4ptr = thrust::raw_pointer_cast(rand4.data());

	if(debug_log_function_calls)
		debug_print("calling contacts-to-action kernel");
	
	//determine whether each infection was successful
	contacts_to_actions_kernel<<<cuda_blocks, cuda_threads>>>(
			inf_count_ptr, inf_offset_ptr,
			day_p_ptr, day_s_ptr,
			actions_type_ptr, // rand1ptr, rand2ptr, rand3ptr, rand4ptr,
			infected_count, rand_offset, current_day);

	//copy the IDs of the infector and victim to the action array
	thrust::copy(daily_contact_infectors.begin(), daily_contact_infectors.begin() + daily_contacts, daily_action_infectors.begin());
	thrust::copy(daily_contact_victims.begin(), daily_contact_victims.begin() + daily_contacts, daily_action_victims.begin());
	cudaDeviceSynchronize();
	
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
		daily_action_type.end(),
		thrust::make_zip_iterator(thrust::make_tuple(
			thrust::make_permutation_iterator(people_status_pandemic.begin(), daily_contact_victims.begin()),
			thrust::make_permutation_iterator(people_status_seasonal.begin(), daily_contact_victims.begin()))),
		daily_action_type.begin(),
		filterPriorInfectedOp());

	//remove no infection
	ZipIntTripleIterator filter_begin = thrust::make_zip_iterator(thrust::make_tuple(
		daily_action_type.begin(), daily_action_infectors.begin(), daily_action_victims.begin()));
	ZipIntTripleIterator filter_end = thrust::remove_if(
		filter_begin,
		thrust::make_zip_iterator(thrust::make_tuple(daily_action_type.end(), daily_action_infectors.end(), daily_action_victims.end())),
		removeNoActionOp());

	int remaining = filter_end - filter_begin;

	if(CONSOLE_OUTPUT)
	{
		printf("%d unfiltered contacts\n", remaining);
	}

	//remove duplicate actions
	thrust::sort(
		filter_begin,		//sort by victim, then by action
		filter_end,
		actionSortOp());
	filter_end = thrust::unique(
		filter_begin,
		filter_end,
		uniqueActionOp());

	daily_actions = filter_end - filter_begin;

	if(CONSOLE_OUTPUT)
	{
		printf("%d final contacts\n", daily_actions);
	}

	if(debug_log_function_calls)
		debug_print("contact filtering complete");
	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, daily_action_type.end() - daily_action_type.begin());
}


//given an action type, look through the filtered actions and count generations
void PandemicSim::daily_countReproductionNumbers(int action)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "count_reproduction");

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
			daily_victim_gen_p.begin(),			//copy all generations from pandemic actions
			daily_victim_gen_p.begin() + daily_actions,
			daily_action_type.begin(),
			pandemic_gens.begin(),
			r_pred);

	}
	else if(action == ACTION_INFECT_SEASONAL)
	{
		gens_end = thrust::copy_if(
			daily_victim_gen_s.begin(),		//copy all generations from seasonal actions 
			daily_victim_gen_s.begin() + daily_actions,
			daily_action_type.begin(),
			pandemic_gens.begin(),
			r_pred);
	}
	else
	{
		throw;
	}

	int num_matching_actions = gens_end - pandemic_gens.begin();

	//unique requires sort, sort generations
	thrust::sort(pandemic_gens.begin(), gens_end);

	//copy all unique generations
	d_vec unique_gens(num_matching_actions);
	IntIterator unique_gens_end = thrust::unique_copy(
		pandemic_gens.begin(), gens_end, unique_gens.begin());

	int num_unique_gens = unique_gens_end - unique_gens.begin();

	d_vec upper_bound(num_unique_gens);
	//get count:  UPPER - LOWER BOUND FOR EACH UNIQUE GENERATION IN INFECTION LIST
	thrust::upper_bound(
		pandemic_gens.begin(),
		gens_end,
		unique_gens.begin(),
		unique_gens_end,
		upper_bound.begin());

	d_vec lower_bound(num_unique_gens);
	thrust::lower_bound(
		pandemic_gens.begin(),
		gens_end,
		unique_gens.begin(),
		unique_gens_end,
		lower_bound.begin());

	//in place transformation into lower bound
	thrust::transform(
		upper_bound.begin(),
		upper_bound.end(),
		lower_bound.begin(),
		lower_bound.begin(),
		thrust::minus<int>());
	//upper - lower equals count

	//increment the generation counts in the appropriate array
	if(action == ACTION_INFECT_PANDEMIC)
	{
		thrust::transform(
			thrust::make_permutation_iterator(generation_pandemic.begin() - 1, unique_gens.begin()),		//first.begin
			thrust::make_permutation_iterator(generation_pandemic.begin() - 1, unique_gens_end),			//first.end
			lower_bound.begin(),	//count																//second.begin: count
			thrust::make_permutation_iterator(generation_pandemic.begin() - 1, unique_gens.begin()),		//output back to g_p
			thrust::plus<int>());																	//op: add
	}
	else if(action == ACTION_INFECT_SEASONAL)
	{

		thrust::transform(
			thrust::make_permutation_iterator(generation_seasonal.begin() - 1, unique_gens.begin()),		//first.begin
			thrust::make_permutation_iterator(generation_seasonal.begin() - 1, unique_gens_end),			//first.end
			lower_bound.begin(),	//count																//second.begin: count
			thrust::make_permutation_iterator(generation_seasonal.begin() - 1, unique_gens.begin()),	//output back to g_s
			thrust::plus<int>());																	//op: add
	}
	else{
		throw; //action should be either seasonal or pandemic
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
//		thrust::device_vector<float> rand1, thrust::device_vector<float> rand2, thrust::device_vector<float> rand3, thrust::device_vector<float> rand4
		)
{
	//copy data to host
	h_vec h_ac_type = daily_action_type;
	h_vec h_ac_inf = daily_action_infectors;
	h_vec h_ac_vic = daily_action_victims;
	h_vec h_vic_gen_p = daily_victim_gen_p;
	h_vec h_vic_gen_s = daily_victim_gen_s;
	
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
		
		/*
		float thresh_p = h_r1[i];
		float y_p = h_r3[i];
		bool infects_p = y_p < thresh_p;
		
		float thresh_s = h_r2[i];
		float y_s = h_r4[i];
		bool infects_s = y_s < thresh_s;*/
		
		//current_day, i, type, infector, infector_status_p, infector_status_s, victim, y_p, thresh_p, infects_p, y_s, thresh_s, infects_s
		if(log_actions)
			fprintf(fActions, "%d, %d, %s, %d, %c, %c, %d\n", // %f, %f, %d, %f, %f, %d\n",
					current_day, i, action_type_to_char(type),
					inf, status_int_to_char(status_p), status_int_to_char(status_s),
					vic);
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
	h_vec h_ac_type = daily_action_type;
	h_vec h_ac_inf = daily_action_infectors;
	h_vec h_ac_vic = daily_action_victims;
	h_vec h_vic_gen_p = daily_victim_gen_p;
	h_vec h_vic_gen_s = daily_victim_gen_s;

	h_vec h_status_p = people_status_pandemic;
	h_vec h_status_s = people_status_seasonal;

	for(int i = 0; i < daily_actions; i++)
	{
		int inf = h_ac_inf[i];
		int vic = h_ac_vic[i];
		int type = h_ac_type[i];
		//	int gen_p = h_vic_gen_p[i];
		//	int gen_s = h_vic_gen_s[i];

		int status_p = h_status_p[inf];
		int status_s = h_status_s[inf];

		int v_status_p = h_status_p[vic];
		int v_status_s = h_status_s[vic];

		//current_day, i, type, infector, infector_status_p, infector_status_s, victim, victim_status_p, victim_status_s, gen_p, gen_s
		if(log_actions_filtered)
			fprintf(fActionsFiltered, "%d, %d, %s, %d, %c, %c, %d, %c, %c\n",
			current_day, i, action_type_to_char(type),
			inf, status_int_to_char(status_p), status_int_to_char(status_s), 
			vic, status_int_to_char(v_status_p), status_int_to_char(v_status_s));

		if(print_actions_filtered)
			printf("%2d\tinf: %6d\tstatus_p: %c\tstatus_s: %c\tvic: %6d\ttype: %s\n",
			i,  inf, status_int_to_char(status_p), status_int_to_char(status_s), vic, action_type_to_char(type));
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
		daily_action_victims.begin(),
		daily_action_victims.begin() + daily_actions,
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


	//resize arrays 
	infected_indexes.resize(new_infected_total);
	infected_days_pandemic.resize(new_infected_total);
	infected_days_seasonal.resize(new_infected_total);
	infected_generation_pandemic.resize(new_infected_total);
	infected_generation_seasonal.resize(new_infected_total);

	//copy in new indexes and set default values - will be overwritten
	thrust::copy(unique_new_ids.begin(), unique_new_ids_end, infected_indexes.begin() + remaining);
	thrust::fill(infected_days_pandemic.begin() + remaining, infected_days_pandemic.end(), DAY_NOT_INFECTED);
	thrust::fill(infected_days_seasonal.begin() + remaining, infected_days_seasonal.end(), DAY_NOT_INFECTED);
	thrust::fill(infected_generation_pandemic.begin() + remaining, infected_generation_pandemic.end(), GENERATION_NOT_INFECTED);
	thrust::fill(infected_generation_seasonal.begin() + remaining, infected_generation_seasonal.end(), GENERATION_NOT_INFECTED);

	//sort the new infected into the array
	thrust::sort(
		thrust::make_zip_iterator(thrust::make_tuple(
		infected_indexes.begin(), 
		infected_days_pandemic.begin(), infected_days_seasonal.begin(), 
		infected_generation_pandemic.begin(), infected_generation_seasonal.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(
		infected_indexes.end(),
		infected_days_pandemic.end(), infected_days_seasonal.end(),
		infected_generation_pandemic.end(), infected_generation_seasonal.end())),
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
		infected_indexes.end(),
		daily_action_victims.begin(),
		daily_action_victims.begin() + daily_actions,
		victim_offsets.begin());

	//first, do the pandemic infections

	infectionTypePred typePred;
	typePred.reference_val = ACTION_INFECT_PANDEMIC;

	//convert victims status to infected
	thrust::replace_if(
		thrust::make_permutation_iterator(people_status_pandemic.begin(), daily_action_victims.begin()),
		thrust::make_permutation_iterator(people_status_pandemic.begin(), daily_action_victims.begin() + daily_actions),
		daily_action_type.begin(),
		typePred,
		STATUS_INFECTED);

	//copy their generation to the array
	thrust::scatter_if(
		daily_victim_gen_p.begin(),
		daily_victim_gen_p.end(),
		victim_offsets.begin(),
		daily_action_type.begin(),
		infected_generation_pandemic.begin(),
		typePred);

	//mark today as their first day
	thrust::replace_if(
		thrust::make_permutation_iterator(infected_days_pandemic.begin(), victim_offsets.begin()),
		thrust::make_permutation_iterator(infected_days_pandemic.begin(), victim_offsets.end()),
		daily_action_type.begin(),
		typePred,
		current_day + 1);


	//do it again for seasonal
	typePred.reference_val = ACTION_INFECT_SEASONAL;
	thrust::replace_if(
		thrust::make_permutation_iterator(people_status_seasonal.begin(), daily_action_victims.begin()),
		thrust::make_permutation_iterator(people_status_seasonal.begin(), daily_action_victims.begin() + daily_actions),
		daily_action_type.begin(),
		typePred,
		STATUS_INFECTED);
	thrust::scatter_if(
		daily_victim_gen_s.begin(),
		daily_victim_gen_s.end(),
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
		infected_days_pandemic.end(),
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
		infected_days_seasonal.end(),
		thrust::make_zip_iterator(thrust::make_tuple(
		thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin()),
		infected_days_seasonal.begin(), infected_generation_seasonal.begin())),			
		thrust::make_zip_iterator(thrust::make_tuple(
		thrust::make_permutation_iterator(people_status_seasonal.begin(), infected_indexes.begin()),
		infected_days_seasonal.begin(), infected_generation_seasonal.begin())),
		recoverOp);

	//remove people who are no longer infected
	d_vec infected_indexes_copy(infected_count);		//NOTE: not sure if this is necessary
	thrust::copy(infected_indexes.begin(), infected_indexes.end(), infected_indexes_copy.begin());	//TODO: check

	ZipIntFiveTupleIterator infected_begin = thrust::make_zip_iterator(thrust::make_tuple(
			infected_indexes.begin(), 
			infected_days_pandemic.begin(),	infected_days_seasonal.begin(), 
			infected_generation_pandemic.begin(), infected_generation_seasonal.begin()));
	ZipIntFiveTupleIterator infected_end = thrust::remove_if(
		infected_begin,
		thrust::make_zip_iterator(thrust::make_tuple(
			infected_indexes.end(), 
			infected_days_pandemic.end(), infected_days_seasonal.end(),
			infected_generation_pandemic.end(), infected_generation_seasonal.end())),
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
	h_vec h_c = household_counts;
	h_vec h_p = household_people;

	h_vec w_o = workplace_offsets;
	h_vec w_c = workplace_counts;
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

		for(int j = 0; j < w_c[wp]; j++)
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
		}
		printf("%4d:\tID: %d\tHH: %4d\tWP: %4d\tHH_contains: %4d\tWP_contains: %4d\n",i,idx,hh,wp, hh_contains, wp_contains);
	}
}

void PandemicSim::setup_configureLogging()
{

}

void PandemicSim::filterInfectedByPopulationGroup(const char * hour_string, vec_t * population_group, vec_t * infected_present)
{
	if(PROFILE_SIMULATION)
	{
		profiler.beginFunction(current_day,"filterInfectedByPopulationGroup");
	}

	IntIterator infected_present_end = thrust::set_intersection(
		(*population_group).begin(),
		(*population_group).end(),
		infected_indexes.begin(),
		infected_indexes.end(),
		(*infected_present).begin()
		);

	int infected_present_count  = infected_present_end - (*infected_present).begin();

	if(SANITY_CHECK)
	{
		debug_assert(infected_present_count >= 0,"negative infected present count");
		debug_assert(infected_present_count < infected_count, "too many infected present");
	}

	if(CONSOLE_OUTPUT)
	{
		printf("%c\tinfected present: %d\n", hour_string, infected_present_count);
	}

	if(PROFILE_SIMULATION)
	{
		profiler.endFunction(current_day, infected_count);
	}
}

__global__ void clipContactsDesired_byLocationCount_kernel(int * infected_loc_arr, int * loc_offset_arr, int * contacts_desired_arr, int num_infected)
{
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		int myLoc = infected_loc_arr[myPos];		//location of infected

		int loc_count = loc_offset_arr[myLoc + 1] - loc_offset_arr[myLoc];

		if(loc_count == 1)
			contacts_desired_arr[myPos] = 0;
	}

}

void PandemicSim::clipContactsDesired_byLocationCount(IntIterator infected_locations_begin, int num_infected, vec_t *loc_offsets, IntIterator contacts_desired_begin)
{
	if(PROFILE_SIMULATION)
	{
		profiler.beginFunction(current_day,"buildContactsDesired_fromArray");
	}

	int * infected_loc_arr_ptr = thrust::raw_pointer_cast(infected_locations_begin);
	int * loc_offset_arr_ptr = thrust::raw_pointer_cast((*loc_offsets).data());
	int * contacts_desired_arr_ptr = thrust::raw_pointer_cast(contacts_desired_begin);

	clipContactsDesired_byLocationCount_kernel<<<cuda_blocks,cuda_threads>>>(infected_loc_arr_ptr, loc_offset_arr_ptr, contacts_desired_arr_ptr,num_infected);
	cudaDeviceSynchronize();


	if(PROFILE_SIMULATION)
	{
		profiler.endFunction(current_day, num_infected);
	}
}

void PandemicSim::makeContacts_byContactsDesiredArray(
	const char * hour_string, 
	vec_t *infected_list, int infected_list_count, 
	vec_t *loc_people, vec_t *contacts_desired,
	vec_t *loc_offsets, int num_locs, 
	vec_t *people_lookup)
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "makeContacts_byContactsDesiredArray");

	//get the locations of infected people
	//ASSUMES: people_locations[i] contains the location of person index i (all people are present)
	vec_t infected_locations(infected_list_count);
	thrust::gather(
		(*infected_list).begin(),	//map.begin
		(*infected_list).begin() + infected_list_count,		//map.end
		(*people_lookup).begin(),
		infected_locations.begin());

	//get contacts desired for each infected person
	//this will simply clip contacts_desired to zero if they are alone at a location
	clipContactsDesired_byLocationCount(
		infected_locations.begin(), infected_list_count,
		loc_offsets, contacts_desired->begin());

	//get total number of contacts this hour 
	int num_new_contacts = thrust::reduce((*contacts_desired).begin(), (*contacts_desired).end());

	//get raw pointer into the location table
	int * location_people_ptr = thrust::raw_pointer_cast((*loc_people).data());

	//make contacts
	launchContactsKernel(
		hour_string,
		infected_list, &infected_locations, 
		contacts_desired, infected_list_count,
		location_people_ptr, loc_offsets, num_locs);

	//validate the contacts
	if(log_contacts || print_contacts)
	{
		validate_contacts(hour_string, loc_people, people_lookup, loc_offsets, num_new_contacts);
	}

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}



__global__ void weekendErrand_getInfectedHoursAndDestinations_kernel(int * input_infected_indexes_ptr, int * input_errand_hours_ptr, int * input_errand_destinations_ptr,
																	 int * output_infected_present_ptr, int * output_infected_hour_ptr, 
																	 int * output_infected_dest_ptr, int * output_contacts_desired_ptr,
																	 int num_infected, int num_people, int rand_offset)
{
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;

	//for each infected index i, their first errand destination and hour will be at i, their second at i+N, the third at i+N+N
	//where N = num_people
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < num_infected; myPos += gridDim.x * blockDim.x)
	{
		//assign 2 contacts between 3 hours randomly
		threefry2x32_ctr_t tf_ctr = {{myPos + rand_offset, myPos + rand_offset }};
		u.c = threefry2x32(tf_ctr, tf_k);
		int contacts_assignment = u.i[0] % 6;		//parse uint to a number between 0 <= X < 6

		int myIdx = input_infected_indexes_ptr[myPos];

		int myInputOffset = myIdx;
		int output_offset = myPos * NUM_WEEKEND_ERRANDS;

		output_infected_present_ptr[output_offset] = myIdx;									//copy infected index
		output_infected_hour_ptr[output_offset] = input_errand_hours_ptr[myInputOffset];	//copy the hour of the errand
		output_infected_dest_ptr[output_offset] = input_errand_destinations_ptr[myInputOffset];			//copy the errand destination
		output_contacts_desired_ptr[output_offset] = weekend_errand_contact_assignments[contacts_assignment][0];	//assign contacts to this errand

		myInputOffset += num_people;	//move input slot by num_people
		output_offset++;				//move output slot over by one
		output_infected_present_ptr[output_offset] = myIdx;
		output_infected_hour_ptr[output_offset] = input_errand_hours_ptr[myInputOffset];
		output_infected_dest_ptr[output_offset] = input_errand_destinations_ptr[myInputOffset];
		output_contacts_desired_ptr[output_offset] = weekend_errand_contact_assignments[contacts_assignment][1];

		myInputOffset += num_people;
		output_offset++;
		output_infected_present_ptr[output_offset] = myIdx;
		output_infected_hour_ptr[output_offset] = input_errand_hours_ptr[myInputOffset];
		output_infected_dest_ptr[output_offset] = input_errand_destinations_ptr[myInputOffset];
		output_contacts_desired_ptr[output_offset] = weekend_errand_contact_assignments[contacts_assignment][2];
	}
}

void PandemicSim::weekendErrand_doInfectedSetup(vec_t * errand_hours, vec_t * errand_destinations, vec_t * infected_present, vec_t * infected_locations, vec_t * infected_contacts_desired, vec_t * infected_hour_offsets)
{
	if(SANITY_CHECK)
	{
		int all_errands_count = number_people * NUM_WEEKEND_ERRANDS;
		debug_assert("input errand_hours.size is wrong", all_errands_count, errand_hours->size());
		debug_assert("input errand_destinations.size is wrong", all_errands_count, errand_destinations->size());

		int infected_errands_count = infected_count * NUM_WEEKEND_ERRANDS;
		debug_assert("output infected_present.size is wrong", infected_errands_count,infected_present->size());
		debug_assert("output infected_locations.size is wrong", infected_errands_count,infected_locations->size());
		debug_assert("output contacts_desired.size is wrong", infected_errands_count, infected_contacts_desired->size());

		debug_assert("output infected_hour_offsets.size is wrong", NUM_WEEKEND_ERRAND_HOURS + 1, infected_hour_offsets->size());
	}

	//first input: a list of all infected
	int * global_infected_indexes_ptr = thrust::raw_pointer_cast(infected_indexes.data());

	//second inputs: collated lookup tables for hours and destinations
	int * errand_hour_ptr = thrust::raw_pointer_cast((*errand_hours).data());
	int * errand_dest_ptr = thrust::raw_pointer_cast((*errand_destinations).data());

	//outputs: a list of infected indexes, the hour of the errands, the destinations
	int * infected_present_ptr = thrust::raw_pointer_cast(infected_present->data());
	int * infected_destinations_ptr = thrust::raw_pointer_cast(infected_locations->data());
	int * infected_contacts_desired_ptr = thrust::raw_pointer_cast(infected_contacts_desired->data());

	//use temporary array to store the infected hours - we don't need to pass this back, as the data is conveyed in the offset array
	vec_t infected_hour(infected_count * NUM_WEEKEND_ERRANDS);
	int * infected_hour_ptr = thrust::raw_pointer_cast(infected_hour.data());

	weekendErrand_getInfectedHoursAndDestinations_kernel<<<cuda_blocks,cuda_threads>>>(
		global_infected_indexes_ptr,errand_hour_ptr,errand_dest_ptr,
		infected_present_ptr, infected_hour_ptr, 
		infected_destinations_ptr, infected_contacts_desired_ptr,
		infected_count, number_people, rand_offset);
	cudaDeviceSynchronize();
	rand_offset += infected_count;

	//sort the list of infected indexes present and the destinations by hour
	//since the array is built pre-sorted by infected index, this will result in
	//an array sorted by hour, with each hour additionally sorted by infected index
	thrust::stable_sort_by_key(
		infected_hour.begin(),
		infected_hour.end(),
		thrust::make_zip_iterator(thrust::make_tuple(
			infected_present->begin(), infected_locations->begin(), infected_contacts_desired->begin())));

	//count the number of infected in each hour	
	thrust::lower_bound(
		infected_hour.begin(),		//data.first
		infected_hour.end(),			//data.last
		thrust::counting_iterator<int>(0),		//search val first
		thrust::counting_iterator<int>(NUM_WEEKEND_ERRAND_HOURS), //search val last
		infected_hour_offsets->begin());
	(*infected_hour_offsets)[NUM_WEEKEND_ERRAND_HOURS] = infected_count * NUM_WEEKEND_ERRANDS;

	if(SANITY_CHECK)
	{
		bool hours_sorted = thrust::is_sorted(infected_hour.begin(), infected_hour.end());
		debug_assert(hours_sorted, "infected hours did not sort properly!");

		for(int hour = 0; hour < NUM_WEEKEND_ERRAND_HOURS; hour++)
		{
			int offset = (*infected_hour_offsets)[hour];
			int count = (*infected_hour_offsets)[hour+1] - offset;

			debug_assert(count > 0, "warning: zero infected found in weekend errand hour ", hour);
			debug_assert(count < infected_count, "more than infected_count infected found in hour ", hour);

			//check that the infected indexes are still sorted for each hour
			bool infected_are_sorted =thrust::is_sorted(infected_present->begin() + offset, infected_present->begin() + offset + count);
			debug_assert(infected_are_sorted, "infected not sorted for weekend errand hour ", hour);
		}
	}
}

//helper method that will adapt older code to the IntIterator pointers
void PandemicSim::launchContactsKernel(const char * hour_string, vec_t *infected_indexes_present, vec_t *infected_locations, vec_t *infected_contacts_desired, int infected_present, int * loc_people_ptr, vec_t *location_offsets, int num_locs)
{
	if(SANITY_CHECK)
	{
		if((*location_offsets).size() != num_locs + 1)
		{
			printf("WARNING: Need to convert old-style loc_offset in function %c\n",profiler.getCurrentFuncName());
			exit(1);
		}

		//theoretically, the size of the list of infected people present should equal the parameter
		//If infected_preent < size(), this will be OK - it will just select the first N people
		//however, it really implies that something else went bad along the way
		debug_assert("infected present size mismatch", infected_indexes_present->size(), infected_present);
	}

	launchContactsKernel(
		hour_string,
		infected_indexes_present->begin(), infected_locations->begin(), 
		infected_contacts_desired->begin(), infected_present, 
		loc_people_ptr, location_offsets, num_locs);
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

