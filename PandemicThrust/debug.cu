#include "stdafx.h"

#include "simParameters.h"
#include "profiler.h"

#include "PandemicSim.h"
#include "thrust_functors.h"
#include <algorithm>



h_vec h_infected_indexes;
int h_infected_indexes_freshness = -2;

h_vec h_people_status_pandemic;
h_vec h_people_status_seasonal;
h_vec h_people_days_pandemic;
h_vec h_people_days_seasonal;
h_vec h_people_gens_pandemic;
h_vec h_people_gens_seasonal;
int h_people_status_data_freshness = -2;


h_vec h_contacts_infector;
h_vec h_contacts_victim;
h_vec h_contacts_kval;
h_vec h_infected_kval_sums;
int h_contacts_freshness = -2;

h_vec h_action_types;
thrust::host_vector<float> h_actions_rand1, h_actions_rand2, h_actions_rand3, h_actions_rand4;
int h_actions_freshness = -2;

h_vec h_errand_people_weekendHours;
h_vec h_errand_people_destinations;
int errand_lookup_freshness = -2;


h_vec h_errand_people_table;
h_vec h_errand_infected_locations;
h_vec h_errand_infected_weekendHours;
h_vec h_errand_infected_contactsDesired;
h_vec h_errand_locationOffsets_multiHour;
h_vec h_errand_hourOffsets_weekend;
int h_errand_data_freshness = -2;


//////////////////////////////////////////
//fixed data
h_vec h_people_households;
h_vec h_people_workplaces;
h_vec h_people_age;

h_vec h_workplace_offsets;
h_vec h_workplace_people;
h_vec h_workplace_max_contacts;

h_vec h_household_offsets;
h_vec h_household_people;

#define DOUBLECHECK_CONTACTS 0

void PandemicSim::debug_copyFixedData()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "debug_copyFixedData");

	thrust::copy_n(people_households.begin(), number_people, h_people_households.begin());
	thrust::copy_n(people_workplaces.begin(), number_people, h_people_workplaces.begin());
	thrust::copy_n(people_ages.begin(), number_people, h_people_age.begin());

	thrust::copy_n(workplace_offsets.begin(), number_workplaces + 1, h_workplace_offsets.begin());
	thrust::copy_n(workplace_people.begin(), number_people, h_workplace_people.begin());
	thrust::copy_n(workplace_max_contacts.begin(), number_workplaces, h_workplace_max_contacts.begin());

	thrust::copy_n(household_offsets.begin(), number_households + 1, h_household_offsets.begin());
	thrust::copy_n(household_people.begin(), number_people, h_household_people.begin());

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, number_people);
}

void PandemicSim::debug_sizeHostArrays()
{
	if(PROFILE_SIMULATION)
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
	h_household_people.resize(household_people.size());

	h_contacts_infector.resize(daily_contact_infectors.size());
	h_contacts_victim.resize(daily_contact_victims.size());
	h_contacts_kval.resize(daily_contact_kval_types.size());

	h_action_types.resize(daily_action_type.size());
	h_infected_kval_sums.resize(infected_daily_kval_sum.size());
	h_actions_rand1.resize(debug_contactsToActions_float1.size());
	h_actions_rand2.resize(debug_contactsToActions_float2.size());
	h_actions_rand3.resize(debug_contactsToActions_float3.size());
	h_actions_rand4.resize(debug_contactsToActions_float4.size());

	h_errand_people_table.resize(errand_people_table.size());
	h_errand_people_weekendHours.resize(errand_people_weekendHours.size());
	h_errand_people_destinations.resize(errand_people_destinations.size());

	h_errand_infected_locations.resize(errand_infected_locations.size());
	h_errand_infected_weekendHours.resize(errand_infected_weekendHours.size());
	h_errand_infected_contactsDesired.resize(errand_infected_ContactsDesired.size());

	h_errand_locationOffsets_multiHour.resize(errand_locationOffsets_multiHour.size());
	h_errand_hourOffsets_weekend.reserve(errand_hourOffsets_weekend.size());

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,number_people);
}

void PandemicSim::validateContacts_wholeDay()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "validateContacts_wholeDay");

	int num_contacts = is_weekend() ? infected_count * MAX_CONTACTS_WEEKEND : infected_count * MAX_CONTACTS_WEEKDAY;

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
	int errands_per_person = is_weekend() ? NUM_WEEKEND_ERRANDS : NUM_WEEKDAY_ERRANDS;
	int errand_hours = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS : NUM_WEEKDAY_ERRAND_HOURS;
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

			int locs_matched = 0;

			//the following represent major errors and the resulting contact cannot be checked
			bool abort_check = false;
			if(contact_infector != infected_index)
			{
//				debug_print("error: contact_infector improperly set");
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
//				debug_assert(contact_victim == NULL_PERSON_INDEX, "null contact but victim index is set");
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

					//individuals must be adults
					debug_assert(infector_age == AGE_ADULT, "work-type contact but infector is a child, infector",infected_index);

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
							int infector_errand_loc = h_errand_people_destinations[inf_errand_lookup_offset];
							int infector_errand_hour = h_errand_people_weekendHours[inf_errand_lookup_offset];

							inf_hour_arr[infector_errand]=infector_errand_hour;
							inf_loc_arr[infector_errand] = infector_errand_loc;

							//iterate victim errands
							for(int victim_errand = 0; victim_errand < NUM_WEEKEND_ERRANDS && locs_matched == 0; victim_errand++)
							{
								//fish out victim's errand
								int victim_errand_lookup_offset = (contact_victim * NUM_WEEKEND_ERRANDS) + victim_errand;
								int victim_errand_loc = h_errand_people_destinations[victim_errand_lookup_offset];
								int victim_errand_hour = h_errand_people_weekendHours[victim_errand_lookup_offset];


								vic_hour_arr[victim_errand] = victim_errand_hour;
								vic_loc_arr[victim_errand] = victim_errand_loc;

								//if the infector and victim locations and hours match, save and break
								if(infector_errand_loc == victim_errand_loc && infector_errand_hour == victim_errand_hour)
								{
									locs_matched = 1;
									infector_loc = infector_errand_loc;
									victim_loc = victim_errand_loc;
								}
							}
						}

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
							int infector_errand_loc = h_errand_people_destinations[(hour * number_people) + infected_index];
							int victim_errand_loc = h_errand_people_destinations[(hour * number_people) + contact_victim];
						
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
					infector_loc = h_errand_people_destinations[infected_index];
					victim_loc = h_errand_people_destinations[contact_victim];
					locs_matched = (infector_loc == victim_loc);

					//individuals must not be adults
					debug_assert(infector_age != AGE_ADULT, "afterschool contact but infector is an adult, infector",infected_index);
					debug_assert(victim_age != AGE_ADULT, "afterschool contact but victim is an adult, victim", contact_victim);

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
				fprintf(fContacts,"%d,%d,%d,%d,%s,%d,%d,%d\n",
					current_day, contact_offset + contact, 
					contact_infector, contact_victim, lookup_contact_type(contact_type),
					infector_loc, victim_loc, locs_matched);
		}
	}
	if(log_contacts)
		fflush(fContacts);
	
	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

void PandemicSim::debug_copyErrandLookup()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day,"debug_copyErrandLookup");

	int num_errands = number_people * errands_per_person();
	if(is_weekend())
	{
		thrust::copy_n(errand_people_destinations.begin(), num_errands, h_errand_people_destinations.begin());
		thrust::copy_n(errand_people_weekendHours.begin(), num_errands, h_errand_people_weekendHours.begin());
	}
	else
	{
		thrust::copy_n(errand_people_destinations.begin(), num_errands, h_errand_people_destinations.begin());
	}

	if(PROFILE_SIMULATION)
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
	thrust::copy_n(errand_infected_locations.begin(), errand_dests_to_copy, h_errand_infected_locations.begin());
	thrust::copy_n(errand_locationOffsets_multiHour.begin(), loc_offsets_to_copy, h_errand_locationOffsets_multiHour.begin());

	int errands_per_person = is_weekend() ? NUM_WEEKEND_ERRANDS : NUM_WEEKDAY_ERRANDS;

	FILE * f_locs = fopen("../location_data.csv","w");
	fprintf(f_locs,"index,loc,offset,count\n");
	for(int i = 0; i < infected_count; i++)
	{
		int i_i = h_infected_indexes[i];

		for(int hour = 0; hour < errands_per_person; hour++)
		{
			int i_loc = h_errand_infected_locations[(i * NUM_WEEKDAY_ERRAND_HOURS) + hour];

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
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day, "debug_validateInfectedLocArrays");

	int weekend = is_weekend();
	int errands_per_day = is_weekend() ? NUM_WEEKEND_ERRANDS : NUM_WEEKDAY_ERRANDS;

	for(int pos = 0; pos < infected_count; pos++)
	{
		int inf_idx = h_infected_indexes[pos];

		for(int errand = 0; errand < errands_per_day; errand++)
		{
			int errand_loc_array_val;
			int inf_loc_array_val = h_errand_infected_locations[(pos * errands_per_day) + errand];

			if(weekend)
			{
				errand_loc_array_val = h_errand_people_destinations[(inf_idx * errands_per_day) + errand];
			}
			else
			{
				errand_loc_array_val = h_errand_people_destinations[(errand * number_people) + inf_idx];
			}

			debug_assert("infected_loc value does not match errand_dest val", errand_loc_array_val,inf_loc_array_val);
		}
	}

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

void PandemicSim::debug_doublecheckContact_usingPeopleTable(int pos, int number_hours, int infector, int victim)
{
	int valid_contact = 0;

	for(int hour = 0; hour < number_hours; hour++)
	{
		int loc = h_errand_infected_locations[(number_hours * pos) + hour];


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

	debug_assert(valid_contact, "doublecheck: could not find matching errand location between infector and victim, infector index", infector);
}

void PandemicSim::debug_validateErrandSchedule()
{
	int weekend = is_weekend();
	int errands_per_person = errands_per_person();
	int first_errand_row = is_weekend() ? FIRST_WEEKEND_ERRAND_ROW : FIRST_WEEKDAY_ERRAND_ROW;

	for(int myIdx = 0; myIdx < number_people; myIdx++)
	{
		int myAge = h_people_age[myIdx]; 

		for(int errand = 0; errand < errands_per_person; errand++)
		{
			int errand_loc;

			if(weekend)
			{
				errand_loc = h_errand_people_destinations[(myIdx * NUM_WEEKEND_ERRANDS) + errand];
			}
			else
			{
				errand_loc = h_errand_people_destinations[(errand * number_people) + myIdx];
			}
			
			if(myAge == AGE_ADULT || weekend)
			{
				bool errand_loc_test = errand_loc >= WORKPLACE_TYPE_OFFSET_HOST[first_errand_row];
				bool errand_loc_test_2 = errand_loc < number_workplaces;
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
	int weekend = is_weekend();
	int num_errand_hours = weekend ? NUM_WEEKEND_ERRAND_HOURS : NUM_WEEKDAY_ERRAND_HOURS;

	h_vec h_sorted_hours;
	int num_errands_total;

	if(weekend)
	{
		num_errands_total = NUM_WEEKEND_ERRANDS * number_people;

		h_sorted_hours.resize(num_errands_total);
		thrust::copy_n(errand_people_weekendHours.begin(), num_errands_total, h_sorted_hours.begin());

		thrust::copy_n(errand_hourOffsets_weekend.begin(), NUM_WEEKEND_ERRAND_HOURS + 1, h_errand_hourOffsets_weekend.begin());
	}
	else
	{
		num_errands_total = NUM_WEEKDAY_ERRANDS * number_people;
	}

	h_vec h_sorted_people(num_errands_total);
	thrust::copy_n(errand_people_table.begin(), num_errands_total, h_sorted_people.begin());
	h_vec h_sorted_dests(num_errands_total);
	thrust::copy_n(errand_people_destinations.begin(), num_errands_total, h_sorted_dests.begin());

//	debug_dumpWeekendErrandTables(&h_sorted_people, &h_sorted_hours, &h_sorted_dests);
	thrust::copy_n(errand_locationOffsets_multiHour.begin(),number_workplaces * num_errand_hours, h_errand_locationOffsets_multiHour.begin());



	//validate hour offsets
	if(weekend)
		for(int hour = 0; hour < num_errand_hours + 1; hour++)
		{
			int hour_offset = h_errand_hourOffsets_weekend[hour];
			if(hour > 0)
			{
				int hour_before_offset = h_sorted_hours[hour_offset - 1];
				debug_assert("hour before offset is wrong", hour - 1, hour_before_offset);
			}

			if(hour != num_errand_hours)
			{
				int hour_at_offset = h_sorted_hours[hour_offset];
				debug_assert("hour at offset is wrong", hour, hour_at_offset);
			}
		}
	

	for(int hour = 0; hour < num_errand_hours; hour++)
	{
		int test_last_loc_offset = h_errand_locationOffsets_multiHour[(hour * number_workplaces) + (number_workplaces - 1)];

		int hour_offset = weekend ? h_errand_hourOffsets_weekend[hour] : hour * number_people;

		for(int loc = 0; loc < number_workplaces; loc++)
		{
			int loc_offset_pos = (hour * number_workplaces) + loc;

			int loc_offset = h_errand_locationOffsets_multiHour[loc_offset_pos];
			int next_loc_offset;

			if(loc == number_workplaces - 1)
			{
				if(weekend)
				{
					int people_present_this_hour = h_errand_hourOffsets_weekend[hour+1] - h_errand_hourOffsets_weekend[hour];
					next_loc_offset = people_present_this_hour;
				}
				else
				{
					next_loc_offset = number_people;
				}
			}
			else
				next_loc_offset = h_errand_locationOffsets_multiHour[loc_offset_pos+1];

			int loc_count = next_loc_offset - loc_offset;

			if(weekend)
				loc_offset += h_errand_hourOffsets_weekend[hour];
			else
				loc_offset += (hour * number_people);

			if(loc_count > 0)
			{
				int location_at_offset = h_sorted_dests[loc_offset];

				if(loc != location_at_offset)
					printf("");

				debug_assert("location at offset is wrong", loc, location_at_offset);

				int last_location_here = h_sorted_dests[loc_offset + loc_count - 1];
				debug_assert("last location in table is wrong", loc, last_location_here);
			}
		}
	}

}


void PandemicSim::debug_validatePeopleSetup()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day,"debug_validatePeopleSetup");



	bool households_are_sorted = thrust::is_sorted(people_households.begin(), people_households.begin() + number_people);
	debug_assert(households_are_sorted, "household indexes are not monotonically increasing");

	h_vec h_child_indexes = people_child_indexes;
	h_vec h_adult_indexes = people_adult_indexes;

	debug_assert(people_adult_indexes.size() == number_adults, "people_adult_indexes array missized");
	debug_assert(people_child_indexes.size() == number_children, "people_child_indexes array missized");

	int children_count = 0;
	int adult_count = 0;

	for(int myIdx = 0; myIdx < number_people; myIdx++)
	{
		int myAge = h_people_age[myIdx];
		int hh = h_people_households[myIdx];
		int wp = h_people_workplaces[myIdx];

		debug_assert(hh >= 0 && hh < number_households, "invalid household ID assigned to person", myIdx);
		debug_assert(wp >= 0 && wp < number_workplaces, "invalid workplace ID assigned to person", myIdx);

		if(myAge < 0)
		{
			debug_print("invalid age assignment");
		} 
		else if(myAge == AGE_ADULT)
		{
			//index is in adult array
			bool found_adult_index = std::binary_search(h_adult_indexes.begin(), h_adult_indexes.end(), myIdx);
			debug_assert(found_adult_index, "could not find adult index in people_adult_index array, person", myIdx);

			adult_count++;
		}
		else
		{
			//index is in child array
			bool found_child_index = std::binary_search(h_child_indexes.begin(), h_child_indexes.end(), myIdx);
			debug_assert(found_child_index, "could not find child index in people_child_index array, person", myIdx);

			//workplace is assigned to valid school for age type
			int school_loc_type = CHILD_AGE_SCHOOLTYPE_LOOKUP_HOST[myAge];
			int school_offset = WORKPLACE_TYPE_OFFSET_HOST[school_loc_type];
			int school_count = WORKPLACE_TYPE_COUNT_HOST[school_loc_type];
			debug_assert(wp >= school_offset && wp < school_offset + school_count, "school is not valid for age, person",myIdx);

			children_count++;
		}
	}

	debug_assert(number_adults == adult_count, "number_adults/arraysize does not match num of adults actually counted");
	debug_assert(number_children == children_count, "number_children/arraysize does not match num of adults actually counted");

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day,number_people);
}

void PandemicSim::debug_validateInfectionStatus()
{
	if(PROFILE_SIMULATION)
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

		bool active_p = status_p != STATUS_SUSCEPTIBLE && status_p != STATUS_RECOVERED;
		bool active_s = status_s != STATUS_SUSCEPTIBLE && status_s != STATUS_RECOVERED;
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
			debug_assert(status_p >= 0 && status_p < NUM_PROFILES, "invalid status/profile code");

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
			debug_assert(status_s >= 0 && status_s < NUM_PROFILES, "invalid status/profile code");

			int profile_day = current_day - day_s;
			debug_assert(profile_day >= 0 && profile_day < CULMINATION_PERIOD, "seasonal day is outside of valid range");
		}
	}

	debug_assert(h_infected_count == infected_count, "validation count of infected disagrees with sim");

	if(PROFILE_SIMULATION)
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
		thrust::copy_n(infected_daily_kval_sum.begin(), infected_count, h_infected_kval_sums.begin());

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
		thrust::copy_n(errand_infected_locations.begin(), infected_errands_to_copy, h_errand_infected_locations.begin());
		thrust::copy_n(errand_infected_weekendHours.begin(), infected_errands_to_copy, h_errand_infected_weekendHours.begin());
		thrust::copy_n(errand_infected_ContactsDesired.begin(), infected_count, h_errand_infected_contactsDesired.begin());

		h_errand_data_freshness = current_day;
	}
}

void PandemicSim::debug_validateActions()
{
	if(PROFILE_SIMULATION)
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
			int infector_status_s = h_people_status_seasonal[infector];

			//test that all floats are 0 <= x <= 1
			debug_assert(y_p >= 0.f && y_p <= 1.0f, "y_p out of valid range, person", infector);
			debug_assert(y_s >= 0.f && y_s <= 1.0f, "y_s out of valid range, person", infector);
			
			if(infector == 10643)
				printf("");

			if(infector_status_p >= 0)
			{
				debug_assert(inf_prob_p >= 0.f, "infector has pandemic but no chance of infection, infector ", infector);
			}
			else
			{
				//note: <=, or else you will get warnings when non-infected people have contact_type_none
				debug_assert(inf_prob_p <= 0.f, "infector does not have pandemic, but chance is > 0, infector ", infector);
			}

			if(infector_status_s >= 0)
			{
				debug_assert(inf_prob_s >= 0.f, "infector has seasonal but no chance of infection, infector ", infector);
			}
			else
			{
				//note: <=, or else you will get warnings when non-infected people have contact_type_none
				debug_assert(inf_prob_s <= 0.f, "infector does not have seasonal, but chance is > 0, infector", infector);
			}

			bool infects_p = y_p < inf_prob_p;
			bool infects_s = y_s < inf_prob_s;

			bool victim_susceptible_pandemic = h_people_status_pandemic[victim] == STATUS_SUSCEPTIBLE;
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
				debug_print("invalid action code");

			if(log_actions)
				fprintf(fActions, "%d,%d,%d,%d,%d,%s\n",
				current_day,i,infector,victim,action_type,action_type_to_string(action_type));
		}
	}

	if(log_actions)
		fflush(fActions);

	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, total_contacts);
}
//fprintf(fActions,"%d,%d,%f,%f,%f,%f\n",
//	i, i2, rand1[i], rand2[i], rand3[i], rand4[i]);