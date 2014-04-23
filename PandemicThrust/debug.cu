#include "stdafx.h"

#include "simParameters.h"
#include "profiler.h"

#include "PandemicSim.h"
#include "thrust_functors.h"
#include <algorithm>


h_vec h_people_households;
h_vec h_people_workplaces;
h_vec h_people_age;

h_vec h_infected_indexes;

h_vec h_workplace_offsets;
h_vec h_workplace_people;
h_vec h_workplace_max_contacts;

h_vec h_household_offsets;
h_vec h_household_people;

h_vec h_contacts_infector;
h_vec h_contacts_victim;
h_vec h_contacts_kval;

h_vec h_errand_people_table;
h_vec h_errand_people_weekendHours;
h_vec h_errand_people_destinations;

h_vec h_errand_infected_locations;
h_vec h_errand_infected_weekendHours;
h_vec h_errand_infected_contactsDesired;

h_vec h_errand_locationOffsets_multiHour;

#define DOUBLECHECK_CONTACTS 0

void PandemicSim::debug_copyFixedData()
{
	thrust::copy_n(people_households.begin(), number_people, h_people_households.begin());
	thrust::copy_n(people_workplaces.begin(), number_people, h_people_workplaces.begin());
	thrust::copy_n(people_ages.begin(), number_people, h_people_age.begin());

	thrust::copy_n(workplace_offsets.begin(), number_workplaces + 1, h_workplace_offsets.begin());
	thrust::copy_n(workplace_people.begin(), number_people, h_workplace_people.begin());
	thrust::copy_n(workplace_max_contacts.begin(), number_workplaces, h_workplace_max_contacts.begin());

	thrust::copy_n(household_offsets.begin(), number_households + 1, h_household_offsets.begin());
	thrust::copy_n(household_people.begin(), number_people, h_household_people.begin());
}

void PandemicSim::debug_sizeHostArrays()
{
	h_people_households.resize(people_households.size());
	h_people_workplaces.resize(people_workplaces.size());
	h_people_age.resize(people_ages.size());

	h_infected_indexes.resize(infected_indexes.size());

	h_workplace_offsets.resize(workplace_offsets.size());
	h_workplace_people.resize(workplace_people.size());
	h_workplace_max_contacts.resize(workplace_max_contacts.size());

	h_household_offsets.resize(household_offsets.size());
	h_household_people.resize(household_people.size());

	h_contacts_infector.resize(daily_contact_infectors.size());
	h_contacts_victim.resize(daily_contact_victims.size());
	h_contacts_kval.resize(daily_contact_kvals.size());

	h_errand_people_table.resize(errand_people_table.size());
	h_errand_people_weekendHours.resize(errand_people_weekendHours.size());
	h_errand_people_destinations.resize(errand_people_destinations.size());

	h_errand_infected_locations.resize(errand_infected_locations.size());
	h_errand_infected_weekendHours.resize(errand_infected_weekendHours.size());
	h_errand_infected_contactsDesired.resize(errand_infected_ContactsDesired.size());

	h_errand_locationOffsets_multiHour.resize(errand_locationOffsets_multiHour.size());
}

void PandemicSim::validateContacts_wholeDay()
{
	if(PROFILE_SIMULATION)
	{
		if(is_weekend())
			profiler.beginFunction(current_day, "validateContacts_wholeDay_weekend");
		else
			profiler.beginFunction(current_day, "validateContacts_wholeDay_weekday");
	}

	int num_contacts;
	if(is_weekend())
		num_contacts = infected_count * MAX_CONTACTS_WEEKEND;
	else
		num_contacts = infected_count * MAX_CONTACTS_WEEKDAY;

	//copy contact arrays
	thrust::copy_n(daily_contact_infectors.begin(), num_contacts, h_contacts_infector.begin());
	thrust::copy_n(daily_contact_victims.begin(), num_contacts, h_contacts_victim.begin());
	thrust::copy_n(daily_contact_kvals.begin(), num_contacts, h_contacts_kval.begin());

	if(DOUBLECHECK_CONTACTS)
	{
		//copy errand data
		int num_errands_to_copy = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS * number_people : NUM_WEEKDAY_ERRAND_HOURS * number_people;
		thrust::copy_n(errand_people_table.begin(), num_errands_to_copy, h_errand_people_table.begin());

		int num_errand_location_offsets = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS * number_workplaces : NUM_WEEKDAY_ERRAND_HOURS * number_workplaces;
		thrust::copy_n(errand_locationOffsets_multiHour.begin(), num_errand_location_offsets, h_errand_locationOffsets_multiHour.begin());
	}

	//copy infected information
	thrust::copy_n(infected_indexes.begin(), infected_count, h_infected_indexes.begin());

	int infected_errands_to_copy = is_weekend() ? NUM_WEEKEND_ERRANDS * infected_count : NUM_WEEKDAY_ERRAND_HOURS * infected_count;
	thrust::copy_n(errand_infected_locations.begin(), infected_errands_to_copy, h_errand_infected_locations.begin());
	thrust::copy_n(errand_infected_weekendHours.begin(), infected_errands_to_copy, h_errand_infected_weekendHours.begin());
	thrust::copy_n(errand_infected_ContactsDesired.begin(), infected_count, h_errand_infected_contactsDesired.begin());


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

		int locs_matched = 0;

		int contact_offset = i * contacts_per_person;

		for(int contact = 0; contact < contacts_per_person; contact++)
		{
			int contact_type = h_contacts_kval[contact_offset + contact];
			int contact_infector = h_contacts_infector[contact_offset + contact];
			int contact_victim = h_contacts_victim[contact_offset + contact];

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

			int victim_age = h_people_age[contact_victim];

			int infector_loc = -1;
			int victim_loc = -1;

			//only perform these checks if we did not reach a failure condition before
			if(!abort_check)
			{
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
					debug_assert(infector_loc >= h_workplace_type_offset[BUSINESS_TYPE_PRESCHOOL] && infector_loc < h_workplace_type_offset[BUSINESS_TYPE_UNIVERSITY+1], "infector loc for school contact is not school locationtype, infector",infected_index);
					debug_assert(victim_loc >= h_workplace_type_offset[BUSINESS_TYPE_PRESCHOOL] && victim_loc < h_workplace_type_offset[BUSINESS_TYPE_UNIVERSITY+1], "victim loc for school contact is not school locationtype, infector",infected_index);

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
						//iterate errands
						for(int infector_errand = 0; infector_errand < NUM_WEEKEND_ERRANDS && locs_matched == 0; infector_errand++)
						{
							//fish out the infector's errand
							int errand_lookup_offset = (infector_errand * number_people) + infected_index;
							int infector_errand_loc = h_errand_people_destinations[errand_lookup_offset];
							int infector_errand_hour = h_errand_people_weekendHours[errand_lookup_offset];

							//iterate victim errands
							for(int victim_errand = 0; victim_errand < NUM_WEEKEND_ERRANDS && locs_matched == 0; victim_errand++)
							{
								//fish out victim's errand
								errand_lookup_offset = (victim_errand * number_people) + contact_victim;
								int victim_errand_loc = h_errand_people_destinations[errand_lookup_offset];
								int victim_errand_hour = h_errand_people_weekendHours[errand_lookup_offset];

								//if the infector and victim locations and hours match, save and break
								if(infector_errand_loc == victim_errand_loc && infector_errand_hour == victim_errand_hour)
								{
									locs_matched = 1;
									infector_loc = infector_errand_loc;
									victim_loc = victim_errand_loc;
								}
							}
						}

						//the location must be a valid location # of type that has errands
						if(infector_loc != -1)
							debug_assert(infector_loc >= h_workplace_type_offset[FIRST_WEEKEND_ERRAND_ROW] && infector_loc < number_workplaces, "infector location is not valid for weekend errand PDF, infector",infected_index);
						if(victim_loc != -1)
							debug_assert(victim_loc >= h_workplace_type_offset[FIRST_WEEKEND_ERRAND_ROW] && victim_loc < number_workplaces, "victim location is not valid for weekend errand PDF, infector",infected_index);
					
						//the locations must be the same
						debug_assert(locs_matched, "infector and victim errand destinations do not match, infector",infected_index);
					}
				
					//weekday errand: see if infector and victim go on an errand to the same location during the same hour
					else
					{
						//individuals must be adults
						debug_assert(infector_age == AGE_ADULT, "weekday errand contact type but infector is a child, infector",infected_index);
						debug_assert(victim_age == AGE_ADULT, "weekday errand contact type but victim is a child, victim", contact_victim);

						for(int hour = 0; hour < NUM_WEEKDAY_ERRAND_HOURS && locs_matched == 0; hour++)
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
							debug_assert(infector_loc >= h_workplace_type_offset[FIRST_WEEKDAY_ERRAND_ROW] && infector_loc < number_workplaces, "infector location is not valid for weekday errand PDF, infector", infected_index);
						if(victim_loc != -1)
							debug_assert(victim_loc >= h_workplace_type_offset[FIRST_WEEKDAY_ERRAND_ROW] && victim_loc < number_workplaces, "victim location is not valid for weekday errand PDF, infector", infected_index);

						if(DOUBLECHECK_CONTACTS && !locs_matched)
						{
							debug_doublecheckContact_usingPeopleTable(i,NUM_WEEKDAY_ERRANDS,contact_infector, contact_victim);
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
					debug_assert(infector_loc >= h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL] && infector_loc < h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL + 1], "infector afterschool destination is not afterschool location type, infector", infected_index);
					debug_assert(victim_loc >= h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL] && victim_loc < h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL + 1], "victim afterschool destination is not afterschool location type, infector", infected_index);

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
			fprintf(fContacts,"%d,%d,%d,%d,%s,%d,%d,%d\n",
				current_day, contact_offset + contact, 
				contact_infector, contact_victim, lookup_contact_type(contact_type),
				infector_loc, victim_loc, locs_matched);
		}
	}
	fflush(fContacts);
	
	if(PROFILE_SIMULATION)
		profiler.endFunction(current_day, infected_count);
}

void PandemicSim::debug_copyErrandLookup()
{
	if(PROFILE_SIMULATION)
		profiler.beginFunction(current_day,"debug_copyErrandLookup");

	int num_errands;
	if(is_weekend())
	{
		num_errands = NUM_WEEKEND_ERRANDS * number_people;
		thrust::copy_n(errand_people_destinations.begin(), num_errands, h_errand_people_destinations.begin());
		thrust::copy_n(errand_people_weekendHours.begin(), num_errands, h_errand_people_weekendHours.begin());
	}
	else
	{
		num_errands = NUM_WEEKDAY_ERRAND_HOURS * number_people;
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

	FILE * f_locs = fopen("../location_data.csv","w");
	fprintf(f_locs,"index,loc,offset,count\n");
	for(int i = 0; i < infected_count; i++)
	{
		int i_i = h_infected_indexes[i];

		for(int hour = 0; hour < NUM_WEEKDAY_ERRAND_HOURS; hour++)
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
	int errands_per_day = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS : NUM_WEEKDAY_ERRAND_HOURS;
	for(int pos = 0; pos < infected_count; pos++)
	{
		int inf_idx = h_infected_indexes[pos];

		for(int errand = 0; errand < errands_per_day; errand++)
		{
			int errand_loc_array_val = h_errand_people_destinations[(errand * number_people) + inf_idx];
			int inf_loc_array_val = h_errand_infected_locations[(pos * errands_per_day) + errand];

			debug_assert("infected_loc value does not match errand_dest val", errand_loc_array_val,inf_loc_array_val);
		}
	}
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
	int errands_per_person = is_weekend()? NUM_WEEKEND_ERRANDS : NUM_WEEKDAY_ERRANDS;
	int first_errand_row = is_weekend() ? FIRST_WEEKEND_ERRAND_ROW : FIRST_WEEKDAY_ERRAND_ROW;

	for(int myIdx = 0; myIdx < number_people; myIdx++)
	{
		int myAge = h_people_age[myIdx]; 

		for(int errand = 0; errand < errands_per_person; errand++)
		{
			int errand_loc = h_errand_people_destinations[(errand * number_people) + myIdx];

			if(myAge == AGE_ADULT || weekend)
			{
				debug_assert(errand_loc >= h_workplace_type_offset[first_errand_row] && errand_loc < number_workplaces, "infector location is not valid for errand PDF, infector", myIdx);
			}
			else
			{
				debug_assert(errand_loc >= h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL] && errand_loc < h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL + 1], "infector afterschool destination is not afterschool location type, infector", myIdx);
			}
		}
	}
}