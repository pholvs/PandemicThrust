#include "PandemicSim.h"

#include "simParameters.h"
#include "profiler.h"
#include "thrust_functors.h"

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


#pragma region settings

//Simulation profiling master control - low performance overhead
#define PROFILE_SIMULATION 1

//controls master logging - everything except for profiler
#define GLOBAL_LOGGING 0
#define SANITY_CHECK 1

#define print_infected_info 0
#define log_infected_info GLOBAL_LOGGING

#define print_location_info 0
#define log_location_info 0

#define print_contacts 0
#define log_contacts GLOBAL_LOGGING
#define DOUBLECHECK_CONTACTS 0

#define print_actions 0
#define log_actions GLOBAL_LOGGING

#define print_actions_filtered 0
#define log_actions_filtered GLOBAL_LOGGING

#define log_people_info GLOBAL_LOGGING

//low overhead
#define debug_log_function_calls 1

#pragma endregion settings


FILE * fDebug;
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

int SEED_HOST[SEED_LENGTH];
__device__ __constant__ int SEED_DEVICE[SEED_LENGTH];

__device__ __constant__ int business_type_count[NUM_BUSINESS_TYPES];
__device__ __constant__ int business_type_count_offset[NUM_BUSINESS_TYPES];
__device__ __constant__ float weekday_errand_pdf[NUM_BUSINESS_TYPES];
__device__ __constant__ float weekend_errand_pdf[NUM_BUSINESS_TYPES];
__device__ __constant__ float infectiousness_profile[CULMINATION_PERIOD];

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

#define CHILD_DATA_ROWS 5
float child_CDF[CHILD_DATA_ROWS];
int child_wp_types[CHILD_DATA_ROWS];

#define HH_TABLE_ROWS 9
int hh_adult_count[HH_TABLE_ROWS];
int hh_child_count[HH_TABLE_ROWS];
float hh_type_cdf[HH_TABLE_ROWS];




//generates N unique numbers between 0 and max, exclusive
//assumes array is big enough
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

struct weekend_getter : public thrust::unary_function<int, float>
{
	__device__ float operator() (const int& i)
	{
		return weekend_errand_pdf[i];
	}
};


PandemicSim::PandemicSim() 
{
	open_debug_streams();

	if(PROFILE_SIMULATION)
		profiler.initStack();

	setupLoadParameters();

	if(debug_log_function_calls)
		debug_print("parameters loaded");

}


PandemicSim::~PandemicSim(void)
{
	close_output_streams();
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


	printf("%d people, %d households, %d workplaces\n",number_people, number_households, number_workplaces);


	if(debug_log_function_calls)
		debug_print("setting up households");
	
	//setup households
	setupHouseholdsNew();	//generates according to PDFs

	if(log_people_info)
		dump_people_info();
	
	
	//setup people status:
	people_status_pandemic.resize(number_people);
	people_status_seasonal.resize(number_people);
	thrust::fill(people_status_pandemic.begin(), people_status_pandemic.end(), STATUS_SUSCEPTIBLE);
	thrust::fill(people_status_seasonal.begin(), people_status_seasonal.end(), STATUS_SUSCEPTIBLE);

	
	//setup array for contacts
	daily_contact_infectors.resize(number_people * MAX_CONTACTS_PER_DAY);
	daily_contact_victims.resize(number_people * MAX_CONTACTS_PER_DAY);

	setupInitialInfected();
	setupFixedLocations();	//household and workplace

	//setup output reproduction number counters
	generation_pandemic.resize(MAX_DAYS);
	generation_seasonal.resize(MAX_DAYS);
	thrust::fill(generation_pandemic.begin(), generation_pandemic.end(), 0);
	thrust::fill(generation_seasonal.begin(), generation_seasonal.end(), 0);

	//copy everything down to the GPU
	setupDeviceData();

	if(PROFILE_SIMULATION)
	{
		profiler.endFunction(-1, number_people);
	}

	if(debug_log_function_calls)
		debug_print("simulation setup complete");

}

void PandemicSim::open_debug_streams()
{
	
	fInfected = fopen("../debug_infected.csv.gz", "w");
	fprintf(fInfected, "current_day, i, idx, status_p, day_p, gen_p, status_s, day_s, gen_s\n");

	fLocationInfo = fopen("../debug_location_info.csv.gz","w");
	fprintf(fLocationInfo, "current_day, hour_index, i, offset, count, max_contacts\n");

	fContacts = fopen("../debug_contacts.csv.gz", "w");
	fprintf(fContacts, "current_day, num, contact_type, i, infector_idx, victim_idx, infector_loc, victim_loc, infector_found, victim_found\n");

	fActions = fopen("../debug_actions.csv.gz", "w");
	fprintf(fActions, "current_day, i, type, infector, infector_status_p, infector_status_s, victim, y_p, thresh_p, infects_p, y_s, thresh_s, infects_s\n");

	fActionsFiltered = fopen("../debug_filtered_actions.csv.gz", "w");
	fprintf(fActionsFiltered, "current_day, i, type, infector, infector_status_p, infector_status_s, victim, victim_status_p, victim_status_s\n");

	fDebug = fopen("../debug.txt", "w");

	fWeekendLocations = fopen("../debug_weekend_locations.csv.gz", "w");
	fprintf(fWeekendLocations, "day,i,loc\n");
	
}


void PandemicSim::setupLoadParameters()
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
}

//push various things to device constant memory
void PandemicSim::setupDeviceData()
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

	//seeds
	cudaMemcpyToSymbol(
		SEED_DEVICE,
		SEED_HOST,
		sizeof(int) * SEED_LENGTH);

	cudaDeviceSynchronize();

}

//Sets up people's households and workplaces according to the probability functions
void PandemicSim::setupHouseholdsNew()
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
			int wp = setupAssignWorkplace();
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
			int wp = setupAssignSchool();
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

int PandemicSim::setupAssignWorkplace()
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

int PandemicSim::setupAssignSchool()
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
void PandemicSim::setupInitialInfected()
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
void PandemicSim::setupFixedLocations()
{
	///////////////////////////////////////
	//work/////////////////////////////////
	workplace_counts.resize(number_workplaces);
	workplace_offsets.resize(number_workplaces);	//size arrays
	workplace_people.resize(number_people);

	thrust::sequence(workplace_people.begin(), workplace_people.end());	//fill array with IDs to sort

	vec_t stencil(number_people);		//get a copy of people's workplaces to sort by
	thrust::copy(people_workplaces.begin(), people_workplaces.end(), stencil.begin());

	calcLocationOffsets(
		&workplace_people,
		&stencil,
		&workplace_offsets,
		&workplace_counts,
		number_people, number_workplaces);

	//TODO:  max contacts are currently 3 for all workplaces
	workplace_max_contacts.resize(number_workplaces);
	thrust::fill(workplace_max_contacts.begin(), workplace_max_contacts.end(), 3);	//fill max contacts


	///////////////////////////////////////
	//home/////////////////////////////////
	household_counts.resize(number_households);
	household_offsets.resize(number_households);
	household_people.resize(number_people);

	thrust::sequence(household_people.begin(), household_people.end());	//fill array with IDs to sort
	thrust::copy(people_households.begin(), people_households.end(), stencil.begin());

	calcLocationOffsets(
		&household_people,
		&stencil,
		&household_offsets,
		&household_counts,
		number_people, number_households);

	household_max_contacts.resize(number_households);
	thrust::fill(household_max_contacts.begin(), household_max_contacts.end(), 2);

}


//given an array of people's ID numbers and location numbers
//sort them by location, and then build the location offset/count tables
//location_people will be sorted by workplace
void PandemicSim::calcLocationOffsets(
	vec_t * location_people,
	vec_t * stencil,
	vec_t * location_offsets,
	vec_t * location_counts,
	int num_people, int num_locs)
{

	//sort people by workplace
	thrust::sort_by_key(
		(*stencil).begin(),
		(*stencil).end(),
		(*location_people).begin());

	//build count/offset table
	thrust::counting_iterator<int> count_iterator(0);
	thrust::lower_bound(		//find lower bound of each location
		(*stencil).begin(),
		(*stencil).end(),
		count_iterator,
		count_iterator + num_locs,
		(*location_offsets).begin());
	thrust::upper_bound(		//find upper bound of each location
		(*stencil).begin(),
		(*stencil).end(),
		count_iterator,
		count_iterator + num_locs,
		(*location_counts).begin());

	//upper - lower = count
	thrust::transform(
		(*location_counts).begin(),
		(*location_counts).end(),
		(*location_offsets).begin(),
		(*location_counts).begin(),
		thrust::minus<int>());
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

void PandemicSim::close_output_streams()
{

	fclose(fDebug);
	fclose(fInfected);
	fclose(fLocationInfo);
	fclose(fContacts);
	fclose(fActions);
	fclose(fActionsFiltered);
	fclose(fWeekendLocations);

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

		printf("Day %d:\tinfected: %5d\n", current_day + 1, infected_count);
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
	calculate_final_reproduction();

	if(PROFILE_SIMULATION)
		profiler.endFunction(-1, number_people);


	//moved to destructor for batching
	//close_output_streams();
}


//called at the end of the simulation, figures out the reproduction numbers for each generation
void PandemicSim::calculate_final_reproduction()
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
	make_weekday_contacts(
		"workplace",
		workplace_offsets, workplace_counts,
		workplace_people, workplace_max_contacts,
		people_workplaces, number_workplaces); //wp

	if(debug_log_function_calls)
		debug_print("workplace contacts complete"); 

	//do afterschool for children, and errands for adults
	doWeekdayErrands();

	if(debug_log_function_calls)
		debug_print("errand contacts complete");

	//make household contacts
	make_weekday_contacts(
		"household",
		household_offsets, household_counts,
		household_people, household_max_contacts,
		people_households, number_households);	//hh

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
	make_weekday_contacts(
		"household",
		household_offsets, household_counts,
		household_people, household_max_contacts,
		people_households, number_households); //hh

	//each person will make errand contacts on 3 of 6 possible errand hours
	doWeekendErrands();

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

void PandemicSim::doWeekendErrands()
{

}

void PandemicSim::make_weekday_contacts(const char *, vec_t, vec_t, vec_t, vec_t, vec_t, int)
{

}

void PandemicSim::doWeekdayErrands()
{

}




void PandemicSim::dailyUpdate()
{

}

void PandemicSim::test_locs()
{

}

