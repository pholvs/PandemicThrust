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

int cuda_blocks;
int cuda_threads;



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

void profile_begin_function(int current_day, char * msg)
{
	throw;


}
void profile_begin_function(int current_day, const char * msg)
{
	throw;


}

void profile_end_function(int current_day, int size)
{

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

//generates contacts for the 6 errand hours on a weekend
void PandemicSim::doWeekendErrands()
{
	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "doWeekendErrands");

	//each person gets 3 errands
	int num_weekend_errands_total = NUM_WEEKEND_ERRANDS * number_people;

	//allocate arrays to store the errand locations
	vec_t errand_hours(num_weekend_errands_total);
	vec_t errand_locations(num_weekend_errands_total);
	vec_t errand_people(num_weekend_errands_total);

	//copy people's IDs and their 3 unique hours
	copy_weekend_errand_indexes(&errand_people);
	get_weekend_errand_hours(&errand_hours);
	cudaDeviceSynchronize();

	//begin computing their errand locations
	get_weekend_errand_locs(&errand_locations);

	if(print_location_info || log_location_info){
		printf("dumping weekend errand setup...\n");
		cudaDeviceSynchronize();
		dump_weekend_errands(errand_people, errand_hours, errand_locations, 5, number_people);
	}

	//sort the people IDs by the hours of their errands
	thrust::sort_by_key(
		errand_hours.begin(),
		errand_hours.end(),
		errand_people.begin());

	//count the number of people running errands on each hour
	//compute as count and offset for each hour
	vec_t errand_hour_offsets(NUM_WEEKEND_ERRAND_HOURS);
	vec_t errand_hour_counts(NUM_WEEKEND_ERRAND_HOURS);

	thrust::lower_bound(
		errand_hours.begin(),		//data.first
		errand_hours.end(),			//data.last
		thrust::counting_iterator<int>(0),		//search val first
		thrust::counting_iterator<int>(NUM_WEEKEND_ERRAND_HOURS), //search val last
		errand_hour_offsets.begin());
	thrust::upper_bound(
		errand_hours.begin(),
		errand_hours.end(),
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(NUM_WEEKEND_ERRAND_HOURS),
		errand_hour_counts.begin());
	thrust::transform(
		errand_hour_counts.begin(),
		errand_hour_counts.end(),
		errand_hour_offsets.begin(),
		errand_hour_counts.begin(),
		thrust::minus<int>());

	//finish computing the locations
	cudaDeviceSynchronize();

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

		int offset = errand_hour_offsets[hour];		//index of first person for this hour
		int count = errand_hour_counts[hour];		//number of people out on an errand this hour

		//helper function to handle each hour
		make_contacts_WeekendErrand(str.c_str(), &errand_people, &errand_locations, offset, count);
		if(debug_log_function_calls)
			debug_print("errand hour complete");
	}

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, infected_count);
}

/*
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
}*/

//copies indexes 3 times into array, i.e. for IDS 1-3 produces array:
// 1 2 3 1 2 3 1 2 3
void PandemicSim::copy_weekend_errand_indexes(vec_t * index_arr)
{
	throw;
	int * index_arr_ptr = thrust::raw_pointer_cast(index_arr->data());
	//copy_weekend_errand_indexes_kernel<<<cuda_blocks, cuda_threads>>>(index_arr_ptr, number_people);
}

//gets three UNIQUE errand hours 
/*
__global__ void weekend_errand_hours_kernel(int * hours_array, int N, int rand_offset)
{
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;

	//for each person in simulation
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{
		threefry2x32_ctr_t tf_ctr = {{(myPos * 2) + rand_offset, (myPos * 2) + rand_offset }};
		u.c = threefry2x32(tf_ctr, tf_k);

		int first, second, third;

		//get first hour
		first = u.i[0] % NUM_WEEKEND_ERRAND_HOURS;

		//get second hour, if it matches then increment
		second = u.i[1] % NUM_WEEKEND_ERRAND_HOURS;
		if(second == first)
			second = (second + 1) % NUM_WEEKEND_ERRAND_HOURS;

		threefry2x32_ctr_t tf_ctr_2 = {{(myPos * 2) + rand_offset, (myPos * 2) + rand_offset}};
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
}*/

//gets 3 DIFFERENT errand hours for each person, collated order
//i.e. 1 2 3 1 2 3 1 2 3
void PandemicSim::get_weekend_errand_hours(vec_t * hours_array)
{
	throw;
	int * loc_arr_ptr = thrust::raw_pointer_cast(hours_array->data());
//	weekend_errand_hours_kernel<<<cuda_blocks, cuda_threads>>>(loc_arr_ptr, number_people, rand_offset);
	rand_offset += number_people * 2;
}

//gets three errand locations for each person in collation style
/*__global__ void weekend_errand_locations_kernel(int * location_array, int N, int rand_offset)
{
	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;

	//for each person in simulation
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{
		threefry2x32_ctr_t tf_ctr = {{(myPos * 2) + rand_offset, (myPos * 2) + rand_offset}};
		u.c = threefry2x32(tf_ctr, tf_k);

		int first, second, third;

		//get first location - fish the type from the PDF
		float y =  (float) u.i[0] / UNSIGNED_MAX;
		int row_a = 9;
		while(row_a < NUM_BUSINESS_TYPES - 1 && y > weekend_errand_pdf[row_a])
		{
			y -= weekend_errand_pdf[row_a];
			row_a++;
		}
		y = y / weekend_errand_pdf[row_a];
		int business_num = y * (float) business_type_count[row_a];
		first = business_num + business_type_count_offset[row_a];

		//second 
		y = (float) u.i[1] / UNSIGNED_MAX;
		int row_b = 9;
		while(row_b < NUM_BUSINESS_TYPES - 1 && y > weekend_errand_pdf[row_b])
		{
			y -= weekend_errand_pdf[row_b];
			row_b++;
		}
		y = y / weekend_errand_pdf[row_b];
		business_num = y * (float) business_type_count[row_b];
		second = business_num + business_type_count_offset[row_b];

		threefry2x32_ctr_t tf_ctr_2 = {{(myPos * 2) + (rand_offset + 1), (myPos * 2) + (rand_offset + 1)}};
		u.c = threefry2x32(tf_ctr_2, tf_k);

		//third
		y = (float) u.i[0] / UNSIGNED_MAX;
		int row_c = 9;
		while(row_c < NUM_BUSINESS_TYPES - 1 && y > weekend_errand_pdf[row_c])
		{
			y -= weekend_errand_pdf[row_c];
			row_c++;
		}
		y = y / weekend_errand_pdf[row_c];
		business_num = y * (float) business_type_count[row_c];
		third = business_num + business_type_count_offset[row_c];

		location_array[myPos] = first;
		location_array[myPos + N] = second;
		location_array[myPos + N + N] = third;

	}
}*/

//gets 3 errand locations for each person according to PDF with collated order
//i.e. 1 2 3 1 2 3 1 2 3
void PandemicSim::get_weekend_errand_locs(vec_t * location_array)
{
	throw;
	int * loc_arr_ptr = thrust::raw_pointer_cast(location_array->data());
//	weekend_errand_locations_kernel<<<cuda_blocks, cuda_threads>>>(loc_arr_ptr, number_people, rand_offset);
	rand_offset += number_people * 2;
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

//For making weekend contacts
//We are given a large array containing all errands for the day
//We want to make contacts for all infected people between indexes [offset] and [offset+count]
//The array has not been preprocessed at all except for sorting by the hour of the errand
void PandemicSim::make_contacts_WeekendErrand(const char * hour_string, vec_t * people, vec_t * locations, int offset, int count)
{
	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "make_contacts_WeekendErrand");

	//make_contacts needs both the raw table of people (sorted by location) and a list of infected people present

	//start by building and sorting a lookup table
	d_vec people_lookup_locations(count);
	d_vec people_lookup_indexes(count);

	thrust::copy(				//copy the locations into the lookup table
		locations->begin() + offset, 
		locations->begin() + (offset + count), 
		people_lookup_locations.begin());
	thrust::copy(				//copy people IDs into the lookup table
		people->begin() + offset,
		people->begin() + (offset + count),
		people_lookup_indexes.begin());

	thrust::sort_by_key(	//sort the lookup table by people ID
		people_lookup_indexes.begin(),
		people_lookup_indexes.end(),
		people_lookup_locations.begin());

	if(SANITY_CHECK)
	{
		bool sort = thrust::is_sorted(people_lookup_indexes.begin(), people_lookup_indexes.end());
		debug_assert(sort, "weekend errand: people lookup indexes are not sorted!");
	}

	//sort the location array in place by location
	thrust::sort_by_key(
		locations->begin() + offset,
		locations->begin() + (offset + count),
		people->begin() + offset);

	if(SANITY_CHECK)
	{
		bool sort = thrust::is_sorted(locations->begin() + offset, locations->begin() + (offset + count));
		debug_assert(sort, "weekend errand: location table is not sorted by location!");
	}

	//build the location offset and count tables
	d_vec location_offsets(number_workplaces);
	d_vec location_counts(number_workplaces);
	thrust::counting_iterator<int> counting_iterator(0);
	thrust::lower_bound(
		locations->begin() + offset,
		locations->begin() + (offset + count),
		counting_iterator,
		counting_iterator + number_workplaces,
		location_offsets.begin());
	thrust::upper_bound(
		locations->begin() + offset,
		locations->begin() + (offset + count),
		counting_iterator,
		counting_iterator + number_workplaces,
		location_counts.begin());
	thrust::transform(
		location_counts.begin(),
		location_counts.end(),
		location_offsets.begin(),
		location_counts.begin(),
		thrust::minus<int>()); 

	//get set of infected present
	d_vec infected_present(infected_count);
	IntIterator infected_present_end = thrust::set_intersection(
		infected_indexes.begin(),
		infected_indexes.begin() + infected_count,
		people_lookup_indexes.begin(),
		people_lookup_indexes.end(),
		infected_present.begin());
	int infected_present_count = infected_present_end - infected_present.begin();
	printf("weekend errand: %d infected present\n", infected_present_count);
	if(1)
	{
		fprintf(fDebug, "weekend errand: %d infected present\n", infected_present_count);
		fflush(fDebug);
	}

	//technically does nothing, but provides a sanity check for make_contacts
	infected_present.resize(infected_present_count);

	//search the offsets of the infected present
	vec_t search_offsets(infected_present_count);
	thrust::lower_bound(
		people_lookup_indexes.begin(),
		people_lookup_indexes.end(),
		infected_present.begin(),
		infected_present.begin() + infected_present_count,
		search_offsets.begin());

	//copy their location for this hour
	vec_t infected_locations(infected_present_count);
	thrust::gather(
		search_offsets.begin(),
		search_offsets.end(),
		people_lookup_locations.begin(),
		infected_locations.begin());

	if(debug_log_function_calls)
		debug_print("calculating contacts desired");

	vec_t contacts_desired(infected_present_count);

	//build contacts_desired array
	build_contacts_desired(
		infected_locations,
		location_counts.begin(),
		workplace_max_contacts.begin(),
		&contacts_desired);

	//how many contacts we make in total this hour
	int new_contacts_count = thrust::reduce(contacts_desired.begin(), contacts_desired.end());

	if(debug_log_function_calls)
		debug_print("test-4");

	//get a pointer to the people at the first location
	int * loc_people_ptr = thrust::raw_pointer_cast(people->data());	//ptr to first element
	loc_people_ptr += offset;						//offset the pointer into the array

	//make contacts
	make_contacts(
		infected_present, infected_present_count,
		infected_locations, contacts_desired,
		loc_people_ptr, location_offsets, location_counts, number_workplaces);


	if(log_contacts || print_contacts)
	{
		d_vec val_loc_people(count);
		thrust::copy(
			people->begin() + offset,
			people->begin() + (offset + count),
			val_loc_people.begin());

		d_vec big_lookup(number_people);
		thrust::fill(big_lookup.begin(), big_lookup.end(), -1);
		thrust::scatter(
			people_lookup_locations.begin(),
			people_lookup_locations.end(),
			people_lookup_indexes.begin(),
			big_lookup.begin());

		h_vec h_lookup = big_lookup;
		h_vec h_inf = infected_present;
		h_vec h_inf_loc = infected_locations;
		h_vec h_c_i = daily_contact_infectors;
		h_vec h_c_v = daily_contact_victims;

		for(int i = 0; i < 5; i++)
		{
			printf("%d:\t T6d\tvic: %6d\n", i, h_c_i[i], h_c_v[i]);
		}
		if(print_contacts || log_contacts)
			validate_contacts(
			hour_string,
			val_loc_people, big_lookup,
			location_offsets, location_counts,
			new_contacts_count);


	}

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, infected_present_count);
}

//actually works for any array that has been pre-computed AND where all population is present (i.e. not errands)
void PandemicSim::make_weekday_contacts(const char * contact_type,
										vec_t loc_offsets, vec_t loc_counts,
										vec_t loc_people, vec_t loc_max_contacts,
										vec_t people_locations, int num_locs
										)
{

	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "make_weekday_contacts");


	//get the locations of infected people
	//ASSUMES: people_locations[i] contains the location of person index i (all people are present)
	vec_t infected_locations(infected_count);
	thrust::gather(
		infected_indexes.begin(),	//map.begin
		infected_indexes.end(),		//map.end
		people_locations.begin(),
		infected_locations.begin());

	//get contacts desired for each person
	//return max_contacts, or count if count < max_contacts, or 0 if count == 1
	vec_t contacts_desired(infected_count);
	build_contacts_desired(
		infected_locations,
		loc_counts.begin(),
		loc_max_contacts.begin(),
		&contacts_desired);

	//get total number of contacts this hour 
	int num_new_contacts = thrust::reduce(contacts_desired.begin(), contacts_desired.end());

	//get raw pointer into the location table
	int * location_people_ptr = thrust::raw_pointer_cast(loc_people.data());

	//make contacts
	make_contacts(
		infected_indexes, infected_count, infected_locations, contacts_desired,
		location_people_ptr, loc_offsets, loc_counts, num_locs);

	//validate the contacts
	if(log_contacts || print_contacts)
	{
		validate_contacts(contact_type, loc_people, people_locations, loc_offsets, loc_counts, num_new_contacts);
	}

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, infected_count);
}
//method to set up afterschool activities and errands and make contacts
//children go to one randomly selected afterschool activity for 2 hours
//adults will go to two randomly selected errands, and make 2 contacts split between them
void PandemicSim::doWeekdayErrands()
{
	//GENERAL APPROACH:  since children go to the same afterschool for both hours,
	//		pregenerate their locations and just copy them in for the second hour


	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "doWeekdayErrands");

	//generate child afterschool activities
	vec_t child_locs(number_children);
	get_afterschool_locations(&child_locs);

	//errand arrays, will hold adults and children together
	vec_t errand_people_lookup(number_people);
	vec_t errand_location_people(number_people);
	vec_t errand_location_offsets(number_workplaces);
	vec_t errand_location_counts(number_workplaces);

	//copy the children into the array, generate adult locations,
	//sort and build the location arrays
	build_weekday_errand_locations(child_locs,
		&errand_people_lookup,
		&errand_location_people,
		&errand_location_offsets, &errand_location_counts);

	if(debug_log_function_calls)
		debug_print("first errand locations built");

	//make children contacts
	make_contacts_where_present(
		"afterschool",
		people_child_indexes,
		errand_people_lookup, errand_location_people,
		errand_location_offsets, errand_location_counts);

	//get set of infected_adults
	vec_t infected_adults(infected_count);
	IntIterator infected_adults_end = thrust::set_intersection(
		people_adult_indexes.begin(),
		people_adult_indexes.end(),
		infected_indexes.begin(),
		infected_indexes.end(),
		infected_adults.begin());

	int infected_adults_count = infected_adults_end - infected_adults.begin();
	infected_adults.resize(infected_adults_count);		//trim arrays

	//look up their locations
	vec_t infected_adult_locations(infected_adults_count);
	thrust::gather(
		infected_adults.begin(),
		infected_adults_end,
		errand_people_lookup.begin(),
		infected_adult_locations.begin());


	//assign 2 contacts randomly between the 2 errands
	vec_t errand_contacts_desired(infected_adults_count);
	assign_weekday_errand_contacts(&errand_contacts_desired, infected_adults_count);

	if(debug_log_function_calls)
		debug_print("errand contacts assigned");


	//TODO: FIX

	//if there is only one person present at that location, replace contacts_desired with 0
	vec_t inf_location_count(infected_adults_count);
	thrust::gather(
		infected_adult_locations.begin(),
		infected_adult_locations.end(),
		errand_location_counts.begin(),
		inf_location_count.begin());
	thrust::replace(
		inf_location_count.begin(),
		inf_location_count.end(),
		1,
		0);
	thrust::transform(
		errand_contacts_desired.begin(),
		errand_contacts_desired.end(),
		inf_location_count.begin(),
		errand_contacts_desired.begin(),
		thrust::minimum<int>());

	//find how many contacts we will make the first hour
	int first_errand_contacts = thrust::reduce(errand_contacts_desired.begin(), errand_contacts_desired.end());

	printf("rand_offset: %d\n", rand_offset);

	//make contacts for first errand hour - adults
	int * errand_location_people_ptr = thrust::raw_pointer_cast(errand_location_people.data());
	make_contacts(
		infected_adults, infected_adults_count,
		infected_adult_locations, errand_contacts_desired,
		errand_location_people_ptr, errand_location_offsets, errand_location_counts, number_workplaces);

	if(log_contacts || print_contacts)
	{
		printf("validating first errand contacts\n");
		validate_contacts("weekday_errand_1", errand_location_people, errand_people_lookup, errand_location_offsets, errand_location_counts, first_errand_contacts);
	}
	if(debug_log_function_calls)
		debug_print("first errand validated");

	//shouldn't be necessary at this point
	if(0){
		errand_people_lookup.clear();
		errand_people_lookup.resize(number_people);
		thrust::fill(errand_people_lookup.begin(), errand_people_lookup.end(), 0);
		errand_location_people.clear();
		errand_location_people.resize(number_people);
		errand_location_offsets.clear();
		errand_location_offsets.resize(number_workplaces);
		thrust::fill(errand_location_offsets.begin(), errand_location_offsets.end(), 0);
		errand_location_counts.clear();
		errand_location_counts.resize(number_workplaces);
		thrust::fill(errand_location_counts.begin(), errand_location_counts.end(), 0);
	}


	////rebuild array for second hour

	//generate new ID/location array
	build_weekday_errand_locations(child_locs,
		&errand_people_lookup,
		&errand_location_people,
		&errand_location_offsets, &errand_location_counts);

	if(debug_log_function_calls)
		debug_print("second errand locations built");


	//get the locations of infected adults
	thrust::gather(
		infected_adults.begin(),
		infected_adults_end,
		errand_people_lookup.begin(),
		infected_adult_locations.begin());

	//operation: 2 - (# contacts)
	thrust::transform(
		thrust::constant_iterator<int>(2),			//first.begin
		thrust::constant_iterator<int>(2) + infected_adults_count,	//first.end
		errand_contacts_desired.begin(),		//second.begin
		errand_contacts_desired.begin(),		//output - in place
		thrust::minus<int>());		


	//TODO: fix
	//again, we need to check there's more than one person there
	thrust::gather(
		infected_adult_locations.begin(),
		infected_adult_locations.end(),
		errand_location_counts.begin(),
		inf_location_count.begin());
	thrust::replace(
		inf_location_count.begin(),
		inf_location_count.end(),
		1,
		0);
	thrust::transform(
		errand_contacts_desired.begin(),
		errand_contacts_desired.end(),
		inf_location_count.begin(),
		errand_contacts_desired.begin(),
		thrust::minimum<int>());

	int second_errand_contacts_count = thrust::reduce(errand_contacts_desired.begin(), errand_contacts_desired.end());

	//make remaining adult contacts
	errand_location_people_ptr = thrust::raw_pointer_cast(errand_location_people.data());
	make_contacts(
		infected_adults, infected_adults_count,
		infected_adult_locations, errand_contacts_desired,
		errand_location_people_ptr, errand_location_offsets, errand_location_counts, number_workplaces);

	if(log_contacts || print_contacts)
	{
		validate_contacts("weekday_errand_2", errand_location_people, errand_people_lookup, errand_location_offsets, errand_location_counts, second_errand_contacts_count);
	}
	if(debug_log_function_calls)
		debug_print("second errand validated");

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, infected_count);
}

//kernel gets a random afterschool location for each child
/*__global__
	void get_afterschool_locations_kernel(int * location_array, int N, int rand_offset)
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
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{

		threefry2x32_ctr_t tf_ctr = {{myPos + rand_offset, myPos + rand_offset }};
		u.c = threefry2x32(tf_ctr, tf_k);

		//get a random float
		float frac = (float) u.i[0] / UNSIGNED_MAX;
		int ret = frac * afterschool_count;		//find which afterschool location they're at, between 0 and count

		location_array[myPos] = ret + afterschool_offset; //add the offset into the location array
	}
}*/

//starts the kernel that gets a random afterschool location for each child.
void PandemicSim::get_afterschool_locations(vec_t * child_locs)
{
	throw;
	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "get_afterschool_locations");

	int * arr_ptr = thrust::raw_pointer_cast(child_locs->data());
	//get_afterschool_locations_kernel<<<cuda_blocks, cuda_threads>>>(arr_ptr, number_children, rand_offset);
	cudaDeviceSynchronize();
	rand_offset += number_children;

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, number_children);
}

//Given a vector of child locations, generate adult locations
// 		and set up the count/offset arrays
void PandemicSim::build_weekday_errand_locations(
	vec_t child_locs,
	vec_t * errand_people_lookup,
	vec_t * errand_location_people,
	vec_t * errand_location_offsets, vec_t * errand_location_counts)
{
	//	NOTE:  array will initially be set up with adults in the front, and children in the back

	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "build_weekday_errand_locations");

	//get locations for adults and copy in their indexes
	get_weekday_errand_locations(errand_people_lookup);		//copy to FRONT
	thrust::copy(people_adult_indexes.begin(), people_adult_indexes.end(), errand_location_people->begin());

	//copy in afterschool locations and indexes for children
	//note that these don't change between errand hours so they are generated outside the program
	thrust::copy(child_locs.begin(), child_locs.end(), errand_people_lookup->begin() + number_adults);	//copy into END
	thrust::copy(people_child_indexes.begin(), people_child_indexes.end(), errand_location_people->begin() + number_adults);

	//copy the arrays to produce the location arrays
	vec_t sorting_stencil(number_people);
	thrust::copy(errand_people_lookup->begin(), errand_people_lookup->end(), sorting_stencil.begin());

	vec_t seq(number_people);
	thrust::copy(errand_location_people->begin(), errand_location_people->end(), seq.begin());

	//sort the sorting stencil and location table, and use it to get offset and counts
	calcLocationOffsets(
		errand_location_people,
		&sorting_stencil,
		errand_location_offsets,
		errand_location_counts,
		number_people, number_workplaces);

	//sort by person ID to produce the lookup table
	thrust::sort_by_key(
		seq.begin(),
		seq.end(),
		errand_people_lookup->begin());

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, number_people);
}


//gets a random weekday errand location for each adult, where N = number_adults
/*__global__ void get_weekday_errand_locations_kernel(int * locations_arr, int N, int global_rand_offset)
{

	threefry2x32_key_t tf_k = {{SEED_DEVICE[0], SEED_DEVICE[1]}};
	union{
		threefry2x32_ctr_t c;
		unsigned int i[2];
	} u;

	//for each adult
	for(int myPos = blockIdx.x * blockDim.x + threadIdx.x;  myPos < N; myPos += gridDim.x * blockDim.x)
	{

		int myRandOffset = myPos + global_rand_offset;

		threefry2x32_ctr_t tf_ctr = {{myRandOffset,myRandOffset}};
		u.c = threefry2x32(tf_ctr, tf_k);

		//fish out a business type
		float yval = (float) u.i[0] / UNSIGNED_MAX;
		int row = 0; //which business type

		while(yval > weekday_errand_pdf[row] && row < (NUM_BUSINESS_TYPES - 1))
		{
			yval -= weekday_errand_pdf[row];
			row++;
		}

		//figure out which business of this type we're at
		float frac = yval / weekday_errand_pdf[row];
		int offset = business_type_count_offset[row];
		int business_num = frac * business_type_count[row];

		//store in output array
		locations_arr[myPos] = business_num + offset;

		//TODO:  use other rand number
	}
}*/

//gets a random weekday errand location for each adult
void PandemicSim::get_weekday_errand_locations(d_vec * locations_array)
{
	throw;
	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "get_weekday_errand_locations");

	int * locations_array_ptr = thrust::raw_pointer_cast((*locations_array).data());

	//start kernel
//	get_weekday_errand_locations_kernel<<<cuda_blocks, cuda_threads>>>(locations_array_ptr, number_adults, rand_offset);
	cudaDeviceSynchronize(); //need to sync to be sure child locations are finished

	rand_offset += number_adults;

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, number_adults);
}


//for each infected adult, get the number of contacts they will make at their first errand
//valid outputs are {0,1,2}
/*__global__ void errand_contacts_kernel(int * array, int N, int global_rand_offset)
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
}*/

//for each infected adult, get the number of contacts they will make at their first errand
//valid outputs are {0,1,2}
void PandemicSim::assign_weekday_errand_contacts(d_vec * contacts_desired, int num_infected_adults)
{
	throw;
	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "assign_weekday_errand_contacts");

	int * arr_ptr = thrust::raw_pointer_cast(contacts_desired->data());

	//start kernel
//	errand_contacts_kernel<<<cuda_blocks, cuda_threads>>>(arr_ptr, num_infected_adults, rand_offset);
	cudaDeviceSynchronize();

	rand_offset += num_infected_adults;

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, num_infected_adults);
}


void PandemicSim::build_contacts_desired(
	vec_t infected_locations, 
	IntIterator loc_counts_begin, 
	IntIterator loc_max_contacts_begin, 
	vec_t *contacts_desired)
{
	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "build_contacts_desired");

	thrust::gather(
		infected_locations.begin(),		//gather location count to desired
		infected_locations.end(),
		loc_counts_begin,
		contacts_desired->begin());

	thrust::replace(					//if location count is 1, replace with 0
		contacts_desired->begin(),
		contacts_desired->end(),
		1,
		0);
	thrust::transform(				//contacts_desired = min(count or 0, max_contacts)
		contacts_desired->begin(),
		contacts_desired->end(),
		thrust::make_permutation_iterator(loc_max_contacts_begin, infected_locations.begin()),
		contacts_desired->begin(),
		thrust::minimum<int>());


	if(PROFILE_SIMULATION)
		profile_end_function(current_day, infected_locations.end() - infected_locations.begin());
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

//Randomly select people at the same location as the infector as contacts
//NOTE: assumes that a location count of 1 means that contacts_desired = 0 
//		for anyone at that location otherwise, someone could select themselves
/*__global__ void victim_index_kernel(
	int * infector_indexes_arr, int * contacts_desired_arr, int * output_offset_arr, int * infector_loc_arr,
	int * location_offsets_arr, int * location_counts_arr, int * location_people_arr,
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
		int loc_count = location_counts_arr[myLoc];
		int loc_offset = location_offsets_arr[myLoc];

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
}*/



//this function does final setup and then calls the kernel to make contacts
//contacts can be validated afterwards if desired
void PandemicSim::make_contacts(
	vec_t infected_indexes_present, int infected_present,
	vec_t infected_locations, vec_t infected_contacts_desired,
	int * loc_people_ptr, vec_t location_offsets, vec_t location_counts, int num_locs)
{
	throw;
	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "make_contacts");

	if(debug_log_function_calls)
		debug_print("inside make_contacts");

	//get raw pointers to infected data
	int * infected_idx_ptr = thrust::raw_pointer_cast(infected_indexes_present.data());
	int * infected_loc_ptr = thrust::raw_pointer_cast(infected_locations.data());
	int * infected_contacts_ptr = thrust::raw_pointer_cast(infected_contacts_desired.data());

	//get raw pointers to location data
	int * loc_offsets_ptr = thrust::raw_pointer_cast(location_offsets.data());
	int * loc_counts_ptr = thrust::raw_pointer_cast(location_counts.data());

	//build output offset array:  exclusive scan
	vec_t output_offsets(infected_indexes_present.size());
	thrust::exclusive_scan(
		infected_contacts_desired.begin(),
		infected_contacts_desired.end(),
		output_offsets.begin());
	int * output_offsets_ptr = thrust::raw_pointer_cast(output_offsets.data());

	//how many contacts we are about to make in total
	int new_contacts = thrust::reduce(infected_contacts_desired.begin(), infected_contacts_desired.end());

	//if the array is too small, this will go poorly
	if(daily_contacts + new_contacts > daily_contact_infectors.size())
	{
		printf("warning:  contacts too small, old size: %d new size: %d\n", daily_contact_infectors.size(), daily_contacts + new_contacts);
		daily_contact_infectors.resize(daily_contacts + new_contacts);
		daily_contact_victims.resize(daily_contacts + new_contacts);
		//daily_contact_kvals.resize(daily_contacts + new_contacts);
	}

	//get raw pointers into output array, and advance past spots already filled
	int * contact_infector_ptr = thrust::raw_pointer_cast(daily_contact_infectors.data());
	contact_infector_ptr += daily_contacts;
	int * contact_victim_ptr = thrust::raw_pointer_cast(daily_contact_victims.data());
	contact_victim_ptr += daily_contacts;

//	if(0)
//		dump_contact_kernel_setup(
//		infected_indexes_present, infected_locations, 
//		infected_contacts_desired, output_offsets,
//		loc_people_ptr, location_offsets, location_counts,
//		infected_present);


	printf("daily contacts: %d\t new_contacts: %d\t rand_offset: %d\n", daily_contacts, new_contacts, rand_offset);
	//	printf("infected_present size: %d\ninfected present: %d\ninfected_locations size: %d\ninfected_contacts_desired size: %d\nloc_offsets size: %d\nloc_counts size: %d\nnum_locs: %d\n",
	//			infected_indexes_present.size(), infected_present, infected_locations.size(), infected_contacts_desired.size(), location_offsets.size(), location_counts.size(), num_locs);

	//theoretically, the size of the list of infected people present should equal the parameter
	//If infected_preent < size(), this will be OK - it will just select the first N people
	//however, it really implies that something else went bad along the way
	if(SANITY_CHECK)
	{
		debug_assert("infected present size mismatch", infected_indexes_present.size(), infected_present);
	}


	//call the kernel
	if(debug_log_function_calls)
		debug_print("calling contacts kernel");
//	victim_index_kernel<<<cuda_blocks, cuda_threads>>>(
//		infected_idx_ptr, infected_contacts_ptr, output_offsets_ptr, infected_loc_ptr,
//		loc_offsets_ptr, loc_counts_ptr, loc_people_ptr,
//		contact_infector_ptr, contact_victim_ptr,
//		infected_present, rand_offset);
	cudaDeviceSynchronize();

	if(debug_log_function_calls)
		debug_print("contacts kernel sync'd");

	//adjust global parameters - increment random offset and the # of contacts made today
	rand_offset += new_contacts;
	daily_contacts += new_contacts;

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, infected_present); 
}

void PandemicSim::make_contacts_where_present(const char * hour_string, vec_t population_group, vec_t errand_people_lookup, vec_t errand_location_people, vec_t errand_location_offsets, vec_t errand_location_counts)
{
	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, hour_string);

	//get set of children present
	vec_t infected_present(infected_count);
	IntIterator infected_present_end = thrust::set_intersection(
		population_group.begin(),
		population_group.end(),
		infected_indexes.begin(),
		infected_indexes.end(),
		infected_present.begin()
		);
	int infected_present_count  = infected_present_end - infected_present.begin();
	printf("infected present: %d\n", infected_present_count);

	//get the locations of infected children
	vec_t infected_locations(infected_present_count);
	thrust::gather(
		infected_present.begin(),
		infected_present_end,
		errand_people_lookup.begin(),
		infected_locations.begin());

	//assign their desired contacts
	vec_t contacts_desired(infected_present_count);
	build_contacts_desired(
		infected_locations,
		errand_location_counts.begin(),
		workplace_max_contacts.begin(),
		&contacts_desired);
	int num_new_contacts = thrust::reduce(contacts_desired.begin(), contacts_desired.end());

	infected_present.resize(infected_present_count);


	int * location_people_ptr = thrust::raw_pointer_cast(errand_location_people.data());
	make_contacts(
		infected_present, infected_present_count,
		infected_locations, contacts_desired,
		location_people_ptr,
		errand_location_offsets, errand_location_counts, number_workplaces);


	if(log_contacts || print_contacts)
	{
		printf("validating children afterschool\n");
		validate_contacts(hour_string, errand_location_people, errand_people_lookup, errand_location_offsets, errand_location_counts, num_new_contacts);
	}

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, infected_present_count);
}

//This method checks that the N contacts most recently generated are valid
//It should be called immediately after make_contacts
//Specifically, it checks that the infector and victim have the same location
//if the DOUBLECHECK_CONTACTS define is set to true, it will actually look in the location table to be sure (very slow)
void PandemicSim::validate_contacts(const char * contact_type, d_vec d_people, d_vec d_lookup, d_vec d_offsets, d_vec d_counts, int N)
{	
	if(PROFILE_SIMULATION)
		profile_begin_function(current_day, "validate_contacts");
	if(debug_log_function_calls)
		debug_print("validating contacts");

	//copy data to host
	h_vec h_people = d_people;
	h_vec h_lookup = d_lookup;
	h_vec h_offsets = d_offsets;
	h_vec h_counts = d_counts;

	h_vec contact_infectors = daily_contact_infectors;
	h_vec contact_victims = daily_contact_victims;

	if(print_contacts)
		printf("validating %d contacts...\n", N);



	if(0)
	{
		printf("h_people size: %d\n", h_people.size());
		printf("h_lookup size: %d\n", h_lookup.size());
		printf("h_offsets size: %d\n", h_offsets.size());
		printf("h_counts size: %d\n", h_counts.size());
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
				int loc_c = h_counts[i_loc];
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
			current_day, i, contact_type, infector_idx, victim_idx, i_loc, v_loc, i_found, v_found);
		if(print_contacts)
			printf("%d\tinf_idx: %5d\tvic_idx: %5d\ti_loc: %5d\tv_loc: %5d\ti_found: %d v_found: %d\n",
			i, infector_idx, victim_idx, i_loc, v_loc, i_found, v_found);
	}

	if(debug_log_function_calls)
		debug_print("contact validation complete");

	fflush(fContacts);
	fflush(fDebug);

	if(PROFILE_SIMULATION)
		profile_end_function(current_day, N);
}


void PandemicSim::dailyUpdate()
{

}

void PandemicSim::test_locs()
{

}

