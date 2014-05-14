#include "stdafx.h"

#include "simParameters.h"
#include "profiler.h"

#include "PandemicSim.h"
#include "thrust_functors.h"
#include <algorithm>



h_vec h_infected_indexes;
int h_infected_indexes_freshness = -2;

thrust::host_vector<status_t> h_people_status_pandemic;
thrust::host_vector<status_t> h_people_status_seasonal;
thrust::host_vector<day_t> h_people_days_pandemic;
thrust::host_vector<day_t> h_people_days_seasonal;
thrust::host_vector<gen_t> h_people_gens_pandemic;
thrust::host_vector<gen_t> h_people_gens_seasonal;
int h_people_status_data_freshness = -2;


thrust::host_vector<personId_t> h_contacts_infector;
thrust::host_vector<personId_t> h_contacts_victim;
thrust::host_vector<personId_t> h_contacts_kval;
thrust::host_vector<errandSchedule_t> h_contacts_location;
int h_contacts_freshness = -2;

thrust::host_vector<action_t> h_action_types;
thrust::host_vector<float> h_actions_rand1, h_actions_rand2, h_actions_rand3, h_actions_rand4;
int h_actions_freshness = -2;

thrust::host_vector<errandSchedule_t> h_people_errands;
int errand_lookup_freshness = -2;


thrust::host_vector<personId_t> h_errand_people_table;
thrust::host_vector<errandSchedule_t> h_infected_errands_array;
thrust::host_vector<errandContactsProfile_t> h_errand_infected_contactsDesired;
h_vec h_errand_locationOffsets_multiHour;
int h_errand_data_freshness = -2;


//////////////////////////////////////////
//fixed data
h_vec h_people_households;
h_vec h_people_workplaces;
thrust::host_vector<age_t> h_people_age;

h_vec h_workplace_offsets;
thrust::host_vector<personId_t> h_workplace_people;
h_vec h_workplace_max_contacts;

h_vec h_household_offsets;
thrust::host_vector<personId_t> h_household_people;

#define DOUBLECHECK_CONTACTS 0

void PandemicSim::debug_copyFixedData()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "debug_copyFixedData");

	thrust::copy_n(people_households.begin(), number_people, h_people_households.begin());
	thrust::copy_n(people_workplaces.begin(), number_people, h_people_workplaces.begin());
	thrust::copy_n(people_ages.begin(), number_people, h_people_age.begin());

	thrust::copy_n(workplace_offsets.begin(), number_workplaces + 1, h_workplace_offsets.begin());
	thrust::copy_n(workplace_people.begin(), number_people, h_workplace_people.begin());
	thrust::copy_n(workplace_max_contacts.begin(), number_workplaces, h_workplace_max_contacts.begin());

	thrust::copy_n(household_offsets.begin(), number_households + 1, h_household_offsets.begin());

	if(SIM_PROFILING)
		profiler.endFunction(current_day, number_people);
}

void PandemicSim::debug_sizeHostArrays()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "debug_sizeHostArrays");

	h_people_households.resize(people_households.size());
	h_people_workplaces.resize(people_workplaces.size());
	h_people_age.resize(people_ages.size());

	h_infected_indexes.resize(infected_indexes.size());

	h_people_status_pandemic.resize(people_status_pandemic.size());
	h_people_status_seasonal.resize(people_status_seasonal.size());
	h_people_days_pandemic.resize(people_days_pandemic.size());
	h_people_days_seasonal.resize(people_days_seasonal.size());
	h_people_gens_pandemic.resize(people_gens_pandemic.size());
	h_people_gens_seasonal.resize(people_gens_seasonal.size());

	h_workplace_offsets.resize(workplace_offsets.size());
	h_workplace_people.resize(workplace_people.size());
	h_workplace_max_contacts.resize(workplace_max_contacts.size());

	h_household_offsets.resize(household_offsets.size());

	h_contacts_infector.resize(daily_contact_infectors.size());
	h_contacts_victim.resize(daily_contact_victims.size());
	h_contacts_kval.resize(daily_contact_kval_types.size());
	h_contacts_location.resize(daily_contact_locations.size());

	h_action_types.resize(daily_action_type.size());

	h_actions_rand1.resize(debug_contactsToActions_float1.size());
	h_actions_rand2.resize(debug_contactsToActions_float2.size());
	h_actions_rand3.resize(debug_contactsToActions_float3.size());
	h_actions_rand4.resize(debug_contactsToActions_float4.size());

	h_errand_people_table.resize(errand_people_table.size());
	h_people_errands.resize(people_errands.size());

	h_infected_errands_array.resize(infected_errands.size());
	h_errand_infected_contactsDesired.resize(errand_infected_ContactsDesired.size());

	h_errand_locationOffsets_multiHour.resize(errand_locationOffsets.size());

	if(SIM_PROFILING)
		profiler.endFunction(current_day,number_people);
}

void PandemicSim::validateContacts_wholeDay()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "validateContacts_wholeDay");

	debug_freshenContacts();
	debug_freshenInfected();
	debug_freshenErrands();

	/*if(DOUBLECHECK_CONTACTS)
	{
		//copy errand data
		int num_errands_to_copy = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS * number_people : NUM_WEEKDAY_ERRAND_HOURS * number_people;
		thrust::copy_n(errand_people_table.begin(), num_errands_to_copy, h_errand_people_table.begin());

		int num_errand_location_offsets = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS * number_workplaces : NUM_WEEKDAY_ERRAND_HOURS * number_workplaces;
		thrust::copy_n(errand_locationOffsets_multiHour.begin(), num_errand_location_offsets, h_errand_locationOffsets_multiHour.begin());
	}*/


	debug_validateErrandSchedule();
	debug_validateInfectedLocArrays();

	int contacts_per_person = is_weekend() ?  MAX_CONTACTS_WEEKEND : MAX_CONTACTS_WEEKDAY;
	int weekend = is_weekend();

	//do verification
	for(int i = 0; i < infected_count; i++)
	{
		int infected_index = h_infected_indexes[i];

		int infector_age = h_people_age[infected_index];
		int infected_hh = h_people_households[infected_index];
		int infected_wp = h_people_workplaces[infected_index];


		int contact_offset = i * contacts_per_person;

		for(int contact = 0; contact < contacts_per_person; contact++)
		{
			int contact_type = h_contacts_kval[contact_offset + contact];
			int contact_infector = h_contacts_infector[contact_offset + contact];
			int contact_victim = h_contacts_victim[contact_offset + contact];

			errandSchedule_t contact_location = h_contacts_location[contact_offset + contact];

			int locs_matched = 0;

			//the following represent major errors and the resulting contact cannot be checked
			bool abort_check = false;
			if(contact_infector != infected_index)
			{
				debug_print("error: contact_infector improperly set");
				abort_check = true;
			}
			if(contact_infector < 0 || contact_infector >= number_people)
			{
//				debug_print("contact_infector is out of range");
				abort_check = true;
			}
			if(contact_victim < 0 || contact_victim >= number_people)
			{
//				debug_print("contact_victim is out of range");
				abort_check = true;
			}

			if(contact_type == CONTACT_TYPE_NONE)
			{
				if(contact_location == NULL_ERRAND)
				{
					debug_assert("nulled contact, but victim is not null", NULL_PERSON_INDEX, contact_victim);
				}
				else
				{
					int loc_count = contact_victim;
					debug_assert("infector wasn't alone at location, but failed to make contacts, loc_count",1,loc_count);
					
					contact_victim = -1;
				}

				locs_matched = 1;
				abort_check = true;
			}


			int infector_loc = -1;
			int victim_loc = -1;

			//only perform these checks if we did not reach a failure condition before
			if(!abort_check)
			{
				int victim_age = h_people_age[contact_victim];

				//begin type-specific checks
				if (contact_type == CONTACT_TYPE_WORKPLACE)
				{
					infector_loc = h_people_workplaces[contact_infector];
					victim_loc = h_people_workplaces[contact_victim];

					locs_matched = (infector_loc == victim_loc);

					//infector must be adults
					debug_assert(infector_age == AGE_ADULT, "work-type contact but infector is a child, infector",infected_index);

					//contact_loc and infector_loc must match
					debug_assert(contact_location == infector_loc, "workplace contact: contact_location does not match infector location, person",infected_index);

					//infector location must be valid
					debug_assert(infector_loc >= 0 && infector_loc < number_workplaces, "infector location out of bounds in workplace contact, infector",infected_index);

					//victim location must be valid
					debug_assert(victim_loc >= 0 && victim_loc < number_workplaces, "victim location out of bounds in workplace contact, infector",infected_index);

					//the location must be the infector's workplace
					debug_assert(infected_wp == infector_loc, "workplace contact is not made at infector's workplace, infector",infected_index);

					//the infector and the victim must have the same location
					debug_assert(infector_loc == victim_loc, "workplace contact between people with different workplaces, infector",infected_index);

					//workplace contacts cannot happen on weekends
					debug_assert(!weekend, "work contact during weekend, infector",infected_index);
				}
				else if(contact_type == CONTACT_TYPE_SCHOOL)
				{
					infector_loc = h_people_workplaces[contact_infector];
					victim_loc = h_people_workplaces[contact_victim];
					locs_matched = (infector_loc == victim_loc);

					//infector must be a child
					debug_assert(infector_age != AGE_ADULT, "school-type contact but infector is an adult, infector",infected_index);

					//the contact must take place at the infector_loc
					debug_assert(contact_location == infector_loc, "school contact: contact does not take place at infector's school, person",infected_index);

					//the location must be a valid location that is a school
					debug_assert(infector_loc >= WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_PRESCHOOL] && infector_loc < WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_UNIVERSITY+1], "infector loc for school contact is not school locationtype, infector",infected_index);
					debug_assert(victim_loc >= WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_PRESCHOOL] && victim_loc < WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_UNIVERSITY+1], "victim loc for school contact is not school locationtype, infector",infected_index);

					//the school must be the infector's assigned WP
					debug_assert(infected_wp == infector_loc, "school contact is not made at infector's school, infector",infected_index);

					//the infector and the victim must have the same location
					debug_assert(infector_loc == victim_loc, "school contact between people with different schools, infector",infected_index);

					//school contacts cannot happen on weekends
					debug_assert(!weekend, "school contact during weekend, infector",infected_index);
				}
				else if(contact_type == CONTACT_TYPE_ERRAND)
				{
					//weekend contact:  see if the infector and the victim go on an errand to the same location during the same hour
					if(weekend)
					{
						int inf_hour_arr[NUM_WEEKEND_ERRANDS];
						int inf_loc_arr[NUM_WEEKEND_ERRANDS];

						int vic_hour_arr[NUM_WEEKEND_ERRANDS];
						int vic_loc_arr[NUM_WEEKEND_ERRANDS];

						//iterate errands
						for(int infector_errand = 0; infector_errand < NUM_WEEKEND_ERRANDS && locs_matched == 0; infector_errand++)
						{
							//fish out the infector's errand
							int inf_errand_lookup_offset = (infected_index * NUM_WEEKEND_ERRANDS) + infector_errand;
							errandSchedule_t errand = h_people_errands[inf_errand_lookup_offset];

							inf_hour_arr[infector_errand] = get_hour_from_errandSchedule_t(errand);
							inf_loc_arr[infector_errand] = get_location_from_errandSchedule_t(errand);

							if(errand == contact_location)
								infector_loc = get_location_from_errandSchedule_t(errand);
						}

						//iterate victim errands
						for(int victim_errand = 0; victim_errand < NUM_WEEKEND_ERRANDS && locs_matched == 0; victim_errand++)
						{
							//fish out victim's errand
							int victim_errand_lookup_offset = (contact_victim * NUM_WEEKEND_ERRANDS) + victim_errand;
							errandSchedule_t errand = h_people_errands[victim_errand_lookup_offset];

							vic_hour_arr[victim_errand] = get_hour_from_errandSchedule_t(errand);
							vic_loc_arr[victim_errand] = get_location_from_errandSchedule_t(errand);

							if(errand == contact_location)
								victim_loc = get_location_from_errandSchedule_t(errand);
						}

						if(infector_loc >= 0 && victim_loc >= 0 && infector_loc == victim_loc)
							locs_matched = 1;


						if(!locs_matched)
							printf("");

						//the location must be a valid location # of type that has errands
						if(infector_loc != -1)
							debug_assert(infector_loc >= WORKPLACE_TYPE_OFFSET_HOST[FIRST_WEEKEND_ERRAND_ROW] && infector_loc < number_workplaces, "infector location is not valid for weekend errand PDF, infector",infected_index);
						if(victim_loc != -1)
							debug_assert(victim_loc >= WORKPLACE_TYPE_OFFSET_HOST[FIRST_WEEKEND_ERRAND_ROW] && victim_loc < number_workplaces, "victim location is not valid for weekend errand PDF, infector",infected_index);
					
						//the locations must be the same
						debug_assert(locs_matched, "infector and victim errand destinations do not match, infector",infected_index);
					}
				
					//weekday errand: see if infector and victim go on an errand to the same location during the same hour
					else
					{
						//individuals must be adults
						debug_assert(infector_age == AGE_ADULT, "weekday errand contact type but infector is a child, infector",infected_index);
						debug_assert(victim_age == AGE_ADULT, "weekday errand contact type but victim is a child, victim", contact_victim);

						for(int hour = 0; hour < NUM_WEEKDAY_ERRANDS && locs_matched == 0; hour++)
						{
							//fish out the locations for this hour
							int infector_errand_loc = h_people_errands[(hour * number_people) + infected_index] % number_workplaces;
							int victim_errand_loc = h_people_errands[(hour * number_people) + contact_victim] % number_workplaces;
						
							//if the locations (and implicitly hours) match, save and break
							if(infector_errand_loc == victim_errand_loc)
							{
								locs_matched = 1;
								infector_loc = infector_errand_loc;
								victim_loc = victim_errand_loc;
								break;
							}
						}

						//the location must be valid and be a destination for errands
						if(infector_loc != -1)
							debug_assert(infector_loc >= WORKPLACE_TYPE_OFFSET_HOST[FIRST_WEEKDAY_ERRAND_ROW] && infector_loc < number_workplaces, "infector location is not valid for weekday errand PDF, infector", infected_index);
						if(victim_loc != -1)
							debug_assert(victim_loc >= WORKPLACE_TYPE_OFFSET_HOST[FIRST_WEEKDAY_ERRAND_ROW] && victim_loc < number_workplaces, "victim location is not valid for weekday errand PDF, infector", infected_index);

						if(DOUBLECHECK_CONTACTS && !locs_matched)
						{
//							debug_doublecheckContact_usingPeopleTable(i,NUM_WEEKDAY_ERRANDS,contact_infector, contact_victim);
						}


						//the infector and victim must have gone on an errand to the same place
						debug_assert(locs_matched, "infector and victim errand destinations do not match, infector",infected_index);
					}
				}
				else if(contact_type == CONTACT_TYPE_AFTERSCHOOL)
				{
					//fish out afterschool location from the first hour slot of "errands"
					infector_loc = h_people_errands[infected_index] % number_workplaces;
					victim_loc = h_people_errands[contact_victim] % number_workplaces;
					locs_matched = (infector_loc == victim_loc);

					//individuals must not be adults
					debug_assert(infector_age != AGE_ADULT, "afterschool contact but infector is an adult, infector",infected_index);
					debug_assert(victim_age != AGE_ADULT, "afterschool contact but victim is an adult, victim", contact_victim);

					//contact must be at infector's afterschool loc
					debug_assert(contact_location == infector_loc, "contact is not at infector's scheduled afterschool, infector",infected_index);

					//location must be valid and location_type == AFTERSCHOOL
					debug_assert(infector_loc >= WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_AFTERSCHOOL] && infector_loc < WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_AFTERSCHOOL + 1], "infector afterschool destination is not afterschool location type, infector", infected_index);
					debug_assert(victim_loc >= WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_AFTERSCHOOL] && victim_loc < WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_AFTERSCHOOL + 1], "victim afterschool destination is not afterschool location type, infector", infected_index);

					//the infector and victim must have the same location
					debug_assert(locs_matched, "infector and victim afterschool destinations do not match, infector",infected_index);

					//afterschool contacts cannot occur on a weekend
					debug_assert(!weekend, "afterschool contact on weekend, infector",infected_index);
				}
				else if(contact_type == CONTACT_TYPE_HOME)
				{
					//lookup victim location
					infector_loc = h_people_households[infected_index];
					victim_loc = h_people_households[contact_victim];
					locs_matched = (infector_loc == victim_loc);

					//contact must be at infector's household
					debug_assert(contact_location == infector_loc, "household contact not at infector's household, infector",infected_index);

					//locations must be valid household indexes
					debug_assert(infector_loc >= 0 && infector_loc < number_households, "infector household index out of range, infector",infected_index);
					debug_assert(victim_loc >= 0 && victim_loc < number_households, "victim household index out of range, infector",infected_index);

					//infector and victim must have the same location
					debug_assert(locs_matched, "household contact between people with different households, infector",infected_index);
				}
				else
					debug_print("error: invalid contact type");
			}
			if(log_contacts)
				fprintf(fContacts,"%d,%d,%d,%d,%s,%d,%d,%d,%d\n",
					current_day, contact_offset + contact, 
					contact_infector, contact_victim, lookup_contact_type(contact_type),
					contact_location, infector_loc, victim_loc, locs_matched);
		}
	}
	if(log_contacts)
		fflush(fContacts);
	
	if(SIM_PROFILING)
		profiler.endFunction(current_day, infected_count);
}

void PandemicSim::debug_copyErrandLookup()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"debug_copyErrandLookup");

	int num_errands = number_people * errands_per_person_today();
	thrust::copy_n(people_errands.begin(), num_errands,h_people_errands.begin());

	if(SIM_PROFILING)
		profiler.endFunction(current_day,num_errands);
}


void PandemicSim::debug_dumpInfectedErrandLocs()
{

	int errand_dests_to_copy;
	int loc_offsets_to_copy;
	if(is_weekend())
	{
		errand_dests_to_copy = infected_count * NUM_WEEKDAY_ERRAND_HOURS;
		loc_offsets_to_copy = number_workplaces * NUM_WEEKDAY_ERRAND_HOURS;
	}
	else
	{
		errand_dests_to_copy = infected_count * NUM_WEEKEND_ERRAND_HOURS;
		loc_offsets_to_copy = number_workplaces * NUM_WEEKEND_ERRAND_HOURS;
	}

	thrust::copy_n(infected_indexes.begin(), infected_count, h_infected_indexes.begin());
	thrust::copy_n(infected_errands.begin(), errand_dests_to_copy, h_infected_errands_array.begin());
	thrust::copy_n(errand_locationOffsets.begin(), loc_offsets_to_copy, h_errand_locationOffsets_multiHour.begin());

	int errands_per_person = is_weekend() ? NUM_WEEKEND_ERRANDS : NUM_WEEKDAY_ERRANDS;

	FILE * f_locs = fopen("../location_data.csv","w");
	fprintf(f_locs,"index,loc,offset,count\n");
	for(int i = 0; i < infected_count; i++)
	{
		int i_i = h_infected_indexes[i];

		for(int hour = 0; hour < errands_per_person; hour++)
		{
			int i_loc = h_infected_errands_array[(i * NUM_WEEKDAY_ERRAND_HOURS) + hour] % number_workplaces;

			int loc_offset = h_errand_locationOffsets_multiHour[(hour * number_workplaces) + i_loc];
			int loc_count;
			if(i_loc == number_workplaces - 1)
				loc_count = number_people - loc_offset;
			else
				loc_count = h_errand_locationOffsets_multiHour[(hour * number_workplaces) + i_loc + 1] - loc_offset;
			fprintf(f_locs, "%d, %d, %d, %d\n",
				i_i, i_loc, loc_offset, loc_count);
		}
		
	}

	fclose(f_locs);
}


void PandemicSim::debug_validateInfectedLocArrays()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "debug_validateInfectedLocArrays");

	int weekend = is_weekend();
	int errands_per_day = is_weekend() ? NUM_WEEKEND_ERRANDS : NUM_WEEKDAY_ERRANDS;

	for(int pos = 0; pos < infected_count; pos++)
	{
		int inf_idx = h_infected_indexes[pos];

		for(int errand = 0; errand < errands_per_day; errand++)
		{
			int errand_loc_array_val;
			if(weekend)
			{
				errand_loc_array_val = h_people_errands[(inf_idx * errands_per_day) + errand];
			}
			else
			{
				errand_loc_array_val = h_people_errands[(errand * number_people) + inf_idx];
			}
			int inf_loc_array_val  = h_infected_errands_array[(pos * errands_per_day) + errand];

			debug_assert("infected_loc value does not match errand_dest val", errand_loc_array_val,inf_loc_array_val);
		}
	}

	if(SIM_PROFILING)
		profiler.endFunction(current_day, infected_count);
}

void PandemicSim::debug_doublecheckContact_usingPeopleTable(int pos, int number_hours, int infector, int victim)
{
	int valid_contact = 0;

	throw;
	/*
	for(int hour = 0; hour < number_hours; hour++)
	{
		int loc = h_in[(number_hours * pos) + hour];


		int loc_offset_pos = (number_workplaces * hour) + loc;
		int loc_offset = h_errand_locationOffsets_multiHour[loc_offset_pos];
		int loc_count = 
			loc == number_workplaces - 1 ? 
			h_errand_locationOffsets_multiHour[loc_offset_pos+1] - loc_offset :
		number_people - loc_offset;

		//offset by number of hours
		loc_offset += (number_people * hour);

		int found_infector = 0;
		int found_victim = 0;

		for(int pos =  loc_offset; pos < loc_offset + loc_count && !(found_infector && found_victim); pos++)
		{
			int person_idx = h_errand_people_table[pos];
			if(person_idx == infector)
				found_infector = 1;
			else if (person_idx == victim)
				found_victim = 1;
		}

		debug_assert(found_infector,"doublecheck: could not find infector for given location in errand lookup table, index",infector);

		if(found_infector && found_victim)
			valid_contact = 1;
	}

	debug_assert(valid_contact, "doublecheck: could not find matching errand location between infector and victim, infector index", infector);*/
}

void PandemicSim::debug_validateErrandSchedule()
{
	int weekend = is_weekend();
	int errands_per_person = errands_per_person_today();
	int first_errand_row = is_weekend() ? FIRST_WEEKEND_ERRAND_ROW : FIRST_WEEKDAY_ERRAND_ROW;

	for(int myIdx = 0; myIdx < number_people; myIdx++)
	{
		int myAge = h_people_age[myIdx]; 

		for(int errand = 0; errand < errands_per_person; errand++)
		{
			errandSchedule_t scheduled_errand;

			if(weekend)
			{
				scheduled_errand = h_people_errands[(myIdx * NUM_WEEKEND_ERRANDS) + errand];
			}
			else
			{
				scheduled_errand = h_people_errands[(errand * number_people) + myIdx];
			}

			int errand_loc = get_location_from_errandSchedule_t(scheduled_errand);
			int errand_hour = get_hour_from_errandSchedule_t(scheduled_errand);

			//validate hour
			if(weekend)
			{
				debug_assert(errand_hour >= 0 && errand_hour < NUM_WEEKEND_ERRAND_HOURS, "scheduled weekend errand has invalid hour, person",myIdx);
			}
			else
			{
				debug_assert("scheduled weekday errand has unexpected hour",errand,errand_hour);
			}

			//validate location
			if(myAge == AGE_ADULT || weekend)
			{
				debug_assert(errand_loc >= WORKPLACE_TYPE_OFFSET_HOST[first_errand_row] && errand_loc < number_workplaces, "person location is not valid for errand PDF, person", myIdx);
			}
			else
			{
				debug_assert(errand_loc >= WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_AFTERSCHOOL] && errand_loc < WORKPLACE_TYPE_OFFSET_HOST[BUSINESS_TYPE_AFTERSCHOOL + 1], "person afterschool destination is not afterschool location type, person", myIdx);
			}
		}
	}
}



void PandemicSim::debug_dumpWeekendErrandTables(h_vec * h_sorted_people, h_vec * h_sorted_hours, h_vec * h_sorted_dests)
{
	int num_errands_total = NUM_WEEKEND_ERRANDS * number_people;

	FILE * fErrands = fopen("../weekend_errands.csv","w");
	fprintf(fErrands, "hour,dest,person\n");
	for(int i = 0; i < num_errands_total; i++)
	{
		int hour = (*h_sorted_hours)[i];
		int dest = (*h_sorted_dests)[i];
		int person = (*h_sorted_people)[i];
		fprintf(fErrands,"%d,%d,%d\n",
			hour,dest,person);
	}
	fclose(fErrands);
}




void PandemicSim::debug_validateLocationArrays()
{
	int num_errand_hours = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS : NUM_WEEKDAY_ERRAND_HOURS;
	int num_errands_total = num_all_errands_today();

	thrust::host_vector<errandSchedule_t> h_sorted_errands(num_errands_total);
	thrust::copy_n(people_errands.begin(), num_errands_total, h_sorted_errands.begin());

	thrust::copy_n(errand_locationOffsets.begin(), number_workplaces * errand_hours_today(), h_errand_locationOffsets_multiHour.begin());

	//test that the hours transition properly
	for(int hour = 0; hour < num_errand_hours; hour++)
	{
		int first_loc_offset_index = hour * number_workplaces;
		if(hour > 0)
		{
			int offset_to_last_loc_of_previous_hour = h_errand_locationOffsets_multiHour[first_loc_offset_index - 1];
			errandSchedule_t errand_at_offset = h_sorted_errands[offset_to_last_loc_of_previous_hour];

			int hour_of_errand = get_hour_from_errandSchedule_t(errand_at_offset);
			int locId_of_errand = get_location_from_errandSchedule_t(errand_at_offset);

			debug_assert("location_offset_array: last hour of previous errand does not have expected value",hour - 1,hour_of_errand);
		}

		//check that first location has the proper value
		{
			int offset_to_first_loc = h_errand_locationOffsets_multiHour[first_loc_offset_index];
			errandSchedule_t errand_at_offset = h_sorted_errands[offset_to_first_loc];

			int hour_of_errand = get_hour_from_errandSchedule_t(errand_at_offset);
			int locId_of_errand = get_location_from_errandSchedule_t(errand_at_offset);

			debug_assert("location_offset_array: first errand of hour does not have expected value",hour,hour_of_errand);
		}
	}
	
	for(int hour = 0; hour < num_errand_hours; hour++)
	{
		for(int loc = 0; loc < number_workplaces; loc++)
		{
			//index to where this location offset is stored
			int loc_offset_index = (hour * number_workplaces) + loc;

			//offset of the first person/errand at this location
			int loc_offset = h_errand_locationOffsets_multiHour[loc_offset_index];

			int loc_count = h_errand_locationOffsets_multiHour[loc_offset_index + 1] - loc_offset;

			if(loc_count > 0)
			{
				errandSchedule_t first_errand_at_offset = h_sorted_errands[loc_offset];
				int hour_of_first_errand = get_hour_from_errandSchedule_t(first_errand_at_offset);
				int locId_of_first_errand = get_location_from_errandSchedule_t(first_errand_at_offset);

				debug_assert("location_offset_array: hour at offset[0] does not match",hour,hour_of_first_errand);
				debug_assert("location_offset_array: locId at offset[0] does not match",loc,locId_of_first_errand);

				errandSchedule_t last_errand_at_offset = h_sorted_errands[loc_offset + loc_count - 1];
				int hour_of_last_errand = get_hour_from_errandSchedule_t(last_errand_at_offset);
				int locId_of_last_errand = get_location_from_errandSchedule_t(last_errand_at_offset);

				debug_assert("location_offset_array: hour at offset[last] does not match",hour,hour_of_last_errand);
				debug_assert("location_offset_array: locId at offset[last] does not match",loc,locId_of_last_errand);
			}
		}
	}

}


void PandemicSim::debug_validatePeopleSetup()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"debug_validatePeopleSetup");

	bool households_are_sorted = thrust::is_sorted(people_households.begin(), people_households.begin() + number_people);
	debug_assert(households_are_sorted, "household indexes are not monotonically increasing");

	int children_count = 0;
	int adult_count = 0;

	for(int myIdx = 0; myIdx < number_people; myIdx++)
	{
		int myAge = h_people_age[myIdx];
		int hh = h_people_households[myIdx];
		int wp = h_people_workplaces[myIdx];

		debug_assert(hh >= 0 && hh < number_households, "invalid household ID assigned to person", myIdx);
		debug_assert(wp >= 0 && wp < number_workplaces, "invalid workplace ID assigned to person", myIdx);

		if(myAge == AGE_ADULT)
		{
			adult_count++;
		}
		else if(myAge >= 0 && myAge < AGE_ADULT)
		{

			//workplace is assigned to valid school for age type
			int school_loc_type = CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[myAge];
			int school_offset = WORKPLACE_TYPE_OFFSET_HOST[school_loc_type];
			int school_count = WORKPLACE_TYPE_COUNT_HOST[school_loc_type];
			debug_assert(wp >= school_offset && wp < school_offset + school_count, "school is not valid for age, person",myIdx);

			children_count++;
		}
		else
			debug_assert(false,"invalid age code for person",myIdx);
	}

	debug_assert(number_adults == adult_count, "number_adults/arraysize does not match num of adults actually counted");
	debug_assert(number_children == children_count, "number_children/arraysize does not match num of adults actually counted");

	if(SIM_PROFILING)
		profiler.endFunction(current_day,number_people);
}

void PandemicSim::debug_validateInfectionStatus()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day,"debug_validateInfectionStatus");

	/*thrust::pair<int, int> extrema = thrust::minmax_element(infected_indexes.begin(), infected_indexes.begin() + infected_count);
	int minIdx = extrema.first;
	int maxIdx = extrema.second;
	debug_assert(minIdx >= 0, "negative ID in infected list, value",minIdx);
	debug_assert(maxIdx < number_people, "idx in infected list exceeds number_people, value", maxIdx);*/

	debug_freshenInfected();
	debug_freshenPeopleStatus();

	int h_infected_count = 0;

	for(int myIdx = 0; myIdx < number_people; myIdx++)
	{
		int status_p = h_people_status_pandemic[myIdx];
		int status_s = h_people_status_seasonal[myIdx];
		int day_p = h_people_days_pandemic[myIdx];
		int day_s = h_people_days_seasonal[myIdx];
		int gen_p = h_people_gens_pandemic[myIdx];
		int gen_s = h_people_gens_seasonal[myIdx];

		bool active_p = status_is_infected(status_p) && get_profile_from_status(status_p) < NUM_SHEDDING_PROFILES;
		bool active_s = status_is_infected(status_s) && get_profile_from_status(status_s) < NUM_SHEDDING_PROFILES;
		bool found_in_infected = std::binary_search(h_infected_indexes.begin(), h_infected_indexes.begin() + infected_count, myIdx);

		if(active_p || active_s)
		{
			h_infected_count++;
			debug_assert(found_in_infected, "active infection, but index not in infected_indexes, person", myIdx);
		}
		else
			debug_assert(!found_in_infected, "no active infection, but index in infected_indexes, person", myIdx);

		if(status_p == STATUS_SUSCEPTIBLE)
		{
			debug_assert(day_p == DAY_NOT_INFECTED, "susceptible to pandemic but day is set, person", myIdx);
			debug_assert(gen_p == GENERATION_NOT_INFECTED, "susceptible to pandemic but generation is set, person", myIdx);
		}
		else if(status_p == STATUS_RECOVERED)
		{
			debug_assert(day_p >= 0 && day_p < current_day, "recovered from pandemic but day is invalid");
			debug_assert(gen_p >= 0 && gen_p < current_day, "recovered from pandemic but gen is invalid");
		}
		else
		{
			debug_assert(status_is_infected(status_p) && get_profile_from_status(status_p) < NUM_SHEDDING_PROFILES, "invalid pandemic status/profile code, person index", myIdx);

			int profile_day = current_day - day_p;
			debug_assert(profile_day >= 0 && profile_day < CULMINATION_PERIOD, "pandemic day is outside of valid range");
		}

		if(status_s == STATUS_SUSCEPTIBLE)
		{
			debug_assert(day_s == DAY_NOT_INFECTED, "susceptible to seasonal but day is set, person", myIdx);
			debug_assert(gen_s == GENERATION_NOT_INFECTED, "susceptible to seasonal but generation is set, person", myIdx);
		}
		else if(status_s == STATUS_RECOVERED)
		{
			debug_assert(day_s >= 0 && day_s < current_day, "recovered from seasonal but day is invalid");
			debug_assert(gen_s >= 0 && gen_s < current_day, "recovered from seasonal but gen is invalid");
		}
		else
		{
			debug_assert(status_is_infected(status_s) && get_profile_from_status(status_s) < NUM_SHEDDING_PROFILES, "invalid seasonal status/profile code, person");

			int profile_day = current_day - day_s;
			debug_assert(profile_day >= 0 && profile_day < CULMINATION_PERIOD, "seasonal day is outside of valid range");
		}
	}

	debug_assert(h_infected_count == infected_count, "validation count of infected disagrees with sim");

	if(SIM_PROFILING)
		profiler.endFunction(current_day,number_people);
}


void PandemicSim::debug_freshenPeopleStatus()
{
	if(h_people_status_data_freshness < current_day)
	{
		thrust::copy_n(people_status_pandemic.begin(), number_people, h_people_status_pandemic.begin());
		thrust::copy_n(people_days_pandemic.begin(), number_people, h_people_days_pandemic.begin());
		thrust::copy_n(people_gens_pandemic.begin(), number_people, h_people_gens_pandemic.begin());

		thrust::copy_n(people_status_seasonal.begin(), number_people, h_people_status_seasonal.begin());
		thrust::copy_n(people_days_seasonal.begin(), number_people, h_people_days_seasonal.begin());
		thrust::copy_n(people_gens_seasonal.begin(), number_people, h_people_gens_seasonal.begin());

		h_people_status_data_freshness = current_day;
	}
}

void PandemicSim::debug_freshenInfected()
{
	//copy infected information
	if(h_infected_indexes_freshness < current_day)
	{
		thrust::copy_n(infected_indexes.begin(), infected_count, h_infected_indexes.begin());
		h_infected_indexes_freshness = current_day;
	}
}

void PandemicSim::debug_freshenContacts()
{
	//copy contact arrays
	if(h_contacts_freshness < current_day)
	{
		int num_contacts = num_infected_contacts_today();
		thrust::copy_n(daily_contact_infectors.begin(), num_contacts, h_contacts_infector.begin());
		thrust::copy_n(daily_contact_victims.begin(), num_contacts, h_contacts_victim.begin());
		thrust::copy_n(daily_contact_kval_types.begin(), num_contacts, h_contacts_kval.begin());
		thrust::copy_n(daily_contact_locations.begin(), num_contacts, h_contacts_location.begin());

		h_contacts_freshness = current_day;
	}
}

void PandemicSim::debug_freshenActions()
{
	if(h_actions_freshness < current_day)
	{
		int num_contacts = num_infected_contacts_today();
		thrust::copy_n(daily_action_type.begin(), num_contacts, h_action_types.begin());
		thrust::copy_n(debug_contactsToActions_float1.begin(), num_contacts, h_actions_rand1.begin());
		thrust::copy_n(debug_contactsToActions_float2.begin(), num_contacts, h_actions_rand2.begin());
		thrust::copy_n(debug_contactsToActions_float3.begin(), num_contacts, h_actions_rand3.begin());
		thrust::copy_n(debug_contactsToActions_float4.begin(), num_contacts, h_actions_rand4.begin());

		h_actions_freshness = current_day;
	}
}

void PandemicSim::debug_freshenErrands()
{
	if(h_errand_data_freshness < current_day)
	{
		int infected_errands_to_copy = num_infected_errands_today();
		thrust::copy_n(infected_errands.begin(),infected_errands_to_copy,h_infected_errands_array.begin());
		thrust::copy_n(errand_infected_ContactsDesired.begin(), infected_count, h_errand_infected_contactsDesired.begin());

		h_errand_data_freshness = current_day;
	}
}

void PandemicSim::debug_validateActions()
{
	if(SIM_PROFILING)
		profiler.beginFunction(current_day, "debug_validateActions");

	int contacts_per_person = contacts_per_person();
	int total_contacts = contacts_per_person * infected_count;

	debug_freshenPeopleStatus();
	debug_freshenInfected();
	debug_freshenContacts();
	debug_freshenActions();

	for(int inf_pos = 0; inf_pos < infected_count; inf_pos++)
	{
		int offset_base = inf_pos * contacts_per_person;

		for(int contact = 0; contact < contacts_per_person; contact++)
		{
			int i = offset_base + contact;

			personId_t infector = h_contacts_infector[i];
			personId_t victim = h_contacts_victim[i];

			float y_p = h_actions_rand1[i];
			float inf_prob_p = h_actions_rand2[i];
			float y_s = h_actions_rand3[i];
			float inf_prob_s = h_actions_rand4[i];

			int action_type = h_action_types[i];

			int infector_status_p = h_people_status_pandemic[infector];
			int infector_day_p = h_people_days_pandemic[infector];
			int infector_status_s = h_people_status_seasonal[infector];
			int infector_day_s = h_people_days_seasonal[infector];

			//test that all floats are 0 <= x <= 1
			debug_assert(y_p >= 0.f && y_p <= 1.0f, "y_p out of valid range, person", infector);
			debug_assert(y_s >= 0.f && y_s <= 1.0f, "y_s out of valid range, person", infector);

			if(status_is_infected(infector_status_p) && infector_day_p <= current_day)
			{
				debug_assert(inf_prob_p >= 0.f, "infector has pandemic but no chance of infection, infector ", infector);
			}
			else
			{
				//note: <=, or else you will get warnings when non-infected people have contact_type_none
				debug_assert(inf_prob_p <= 0.f, "infector does not have pandemic, but chance is > 0, infector ", infector);
			}

			if(status_is_infected(infector_status_s) && infector_day_s <= current_day)
			{
				debug_assert(inf_prob_s >= 0.f, "infector has seasonal but no chance of infection, infector ", infector);
			}
			else
			{
				//note: <=, or else you will get warnings when non-infected people have contact_type_none
				debug_assert(inf_prob_s <= 0.f, "infector does not have seasonal, but chance is > 0, infector", infector);
			}

			//this is fine, just clearing a warning
			/*bool infects_p = y_p < inf_prob_p;
			bool infects_s = y_s < inf_prob_s;*/

			/*bool victim_susceptible_pandemic = h_people_status_pandemic[victim] == STATUS_SUSCEPTIBLE;
			bool victim_susceptible_seasonal = h_people_status_seasonal[victim] == STATUS_SUSCEPTIBLE;

			if(action_type == ACTION_INFECT_NONE)
			{
				//either the threshold was not met or the contact was filtered (victim status is not susceptible)
				debug_assert(!infects_p || !victim_susceptible_pandemic, "ACTION_TYPE_NONE, but successful attempt and susceptible victim for pandemic");
				debug_assert(!infects_s || !victim_susceptible_seasonal, "ACTION_TYPE_NONE, but successful attempt and susceptible victim for seasonal");
			}
			else if(action_type == ACTION_INFECT_PANDEMIC)
			{
				debug_assert(infects_p && victim_susceptible_pandemic, "ACTION_TYPE_PANDEMIC, but unsuccessful attempt or non-susceptible victim for pandemic");
				debug_assert(!infects_s || !victim_susceptible_seasonal, "ACTION_TYPE_PANDEMIC, but successful attempt and susceptible victim for seasonal");
			}
			else if(action_type == ACTION_INFECT_SEASONAL)
			{
				debug_assert(!infects_p || !victim_susceptible_pandemic, "ACTION_TYPE_SEASONAL, but successful attempt and susceptible victim for pandemic");
				debug_assert(infects_s && victim_susceptible_seasonal, "ACTION_TYPE_SEASONAL, but unsuccessful attempt or non-susceptible victim for seasonal");
			}
			else if(action_type == ACTION_INFECT_BOTH)
			{
				debug_assert(infects_p && victim_susceptible_pandemic, "ACTION_TYPE_BOTH, but unsuccessful attempt or non-susceptible victim for pandemic");
				debug_assert(infects_s && victim_susceptible_seasonal, "ACTION_TYPE_BOTH, but unsuccessful attempt or non-susceptible victim for seasonal");

			}
			else
				debug_print("invalid action code");*/

			if(log_actions)
				fprintf(fActions, "%d,%d,%d,%d,%d,%s\n",
				current_day,i,infector,victim,action_type,action_type_to_string(action_type));
		}
	}

	if(log_actions)
		fflush(fActions);

	if(SIM_PROFILING)
		profiler.endFunction(current_day, total_contacts);
}
//fprintf(fActions,"%d,%d,%f,%f,%f,%f\n",
//	i, i2, rand1[i], rand2[i], rand3[i], rand4[i]);