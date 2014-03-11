#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "profiler.h"
#include "indirect.h"

#define CULMINATION_PERIOD 10
#define NUM_BUSINESS_TYPES 14

#define MAX_CONTACTS_PER_DAY 10

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
	void open_debug_streams();
	void setupLoadParameters();
	void setupHouseholdsNew();
	int setupAssignWorkplace();
	int setupAssignSchool();
	void setupDeviceData();
	void setupInitialInfected();
	void setupFixedLocations();

	void calcLocationOffsets(vec_t * location_people, vec_t * stencil,	vec_t * location_offsets,	vec_t * location_counts, int num_people, int num_locs);

	CudaProfiler profiler;


	void runToCompletion();
	void dump_people_info();
	void debug_validate_infected();
	void dump_infected_info();
	void test_locs();

	void close_output_streams();

	void doWeekday();
	void make_weekday_contacts(const char *, vec_t, vec_t, vec_t, vec_t, vec_t, int);
	void get_afterschool_locations(vec_t * child_locs);
	void build_weekday_errand_locations(
		vec_t child_locs,
		vec_t * errand_people_lookup,
		vec_t * errand_location_people,
		vec_t * errand_location_offsets, vec_t * errand_location_counts);
	void make_contacts_where_present(const char * hour_string, vec_t population_group, vec_t errand_people_lookup, vec_t errand_location_people, vec_t errand_location_offsets, vec_t errand_location_counts);
	void assign_weekday_errand_contacts(d_vec * contacts_desired, int num_infected_adults);
	void doWeekdayErrands();
	void get_weekday_errand_locations(d_vec * locations_array);

	void doWeekend();
	void doWeekendErrands();
	void copy_weekend_errand_indexes(vec_t * errand_people);
	void get_weekend_errand_hours(vec_t * errand_hours);
	void get_weekend_errand_locs(vec_t * errand_locations);
	void dump_weekend_errands(vec_t errand_people, vec_t errand_hours, vec_t errand_locations, int num_to_print, int num_people);
	void make_contacts_WeekendErrand(const char * hour_string, vec_t * errand_people, vec_t *errand_locations, int offset, int count);

	void build_contacts_desired(vec_t infected_locations, IntIterator loc_counts_begin, IntIterator loc_max_contacts_begin, vec_t *contacts_desired);
	void make_contacts(
		vec_t infected_indexes_present, int infected_present,
		vec_t infected_locations, vec_t infected_contacts_desired,
		int * loc_people_ptr, vec_t location_offsets, vec_t location_counts, int num_locs);
	void validate_contacts(const char * contact_type, d_vec d_people, d_vec d_lookup, d_vec d_offsets, d_vec d_counts, int N);

	void dailyUpdate();
	void build_action_generations();
	void contacts_to_action();
	void dump_actions();
	void filter_actions();
	void do_infection_actions(int action);
	int recover_infected();
	void countReproduction(int action_type);
	void dump_actions_filtered();
	void rebuild_infected_arr();

	void calculate_final_reproduction();

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
	vec_t daily_action_infectors;
	vec_t daily_action_victims;
	vec_t daily_victim_gen_p;
	vec_t daily_victim_gen_s;

//	ZipIntQuadIterator daily_actions_begin;
//	ZipIntQuadIterator daily_actions_end;

	//stores number of infections by generation
	vec_t generation_pandemic;
	vec_t generation_seasonal;

	FILE *fInfected, *fLocationInfo, *fContacts, *fActions, *fActionsFiltered, *fWeekendLocations;




	vec_t workplace_counts;
	vec_t workplace_offsets;
	vec_t workplace_people;
	vec_t workplace_max_contacts;

	vec_t household_counts;
	vec_t household_offsets;
	vec_t household_people;
	vec_t household_max_contacts;
};

#define is_weekend() (current_day % 7 < 5 ? 0 : 1)

