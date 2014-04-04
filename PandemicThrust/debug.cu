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

	//copy errand data
	int num_errands_to_copy = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS * number_people : NUM_WEEKDAY_ERRAND_HOURS * number_people;
	thrust::copy_n(errand_people_table.begin(), num_errands_to_copy, h_errand_people_table.begin());

	int num_errand_location_offsets = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS * number_workplaces : NUM_WEEKDAY_ERRAND_HOURS * number_workplaces;
	thrust::copy_n(errand_locationOffsets_multiHour.begin(), num_errand_location_offsets, h_errand_locationOffsets_multiHour.begin());

	//copy infected information
	thrust::copy_n(infected_indexes.begin(), infected_count, h_infected_indexes.begin());

	int infected_errands_to_copy = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS * infected_count : NUM_WEEKDAY_ERRAND_HOURS * infected_count;
	thrust::copy_n(errand_infected_locations.begin(), infected_errands_to_copy, h_errand_infected_locations.begin());
	thrust::copy_n(errand_infected_weekendHours.begin(), infected_errands_to_copy, h_errand_infected_weekendHours.begin());
	thrust::copy_n(errand_infected_ContactsDesired.begin(), infected_count, h_errand_infected_contactsDesired.begin());

	int contacts_per_person = is_weekend() ?  MAX_CONTACTS_WEEKEND : MAX_CONTACTS_WEEKDAY;
	int errands_per_person = is_weekend() ? NUM_WEEKEND_ERRANDS : NUM_WEEKDAY_ERRANDS;
	int errand_hours = is_weekend() ? NUM_WEEKEND_ERRAND_HOURS : NUM_WEEKDAY_ERRAND_HOURS;
	int weekend = is_weekend();

	//do verification
	for(int i = 0; i < infected_count; i++)
	{
		int infected_index = h_infected_indexes[i];

		int infected_age = h_people_age[infected_index];
		int infected_hh = h_people_households[infected_index];
		int infected_wp = h_people_workplaces[infected_index];

		int contact_offset = i * contacts_per_person;

		for(int contact = 0; contact < contacts_per_person; contact++)
		{
			int contact_type = h_contacts_kval[contact_offset + contact];
			int contact_infector = h_contacts_infector[contact_offset + contact];
			int contact_victim = h_contacts_victim[contact_offset + contact];

			int infector_loc = -1;
			int victim_loc = -1;
			int locs_matched = 0;


			if(contact_infector != infected_index)
			{
				debug_print("error: contact_infector improperly set");
			}
			if(contact_infector < 0 || contact_infector >= number_people)
			{
				debug_print("contact_infector is out of range");
			}
			if(contact_victim < 0 || contact_victim >= number_people)
			{
				debug_print("contact_victim is out of range");
			}

			if(contact_type == CONTACT_TYPE_NONE)
			{
				debug_assert(contact_victim == NULL_PERSON_INDEX, "null contact but victim index is set");
			}
			else if (contact_type == CONTACT_TYPE_WORKPLACE)
			{
				infector_loc = h_people_workplaces[contact_infector];
				victim_loc = h_people_workplaces[contact_victim];

				debug_assert(infector_loc >= 0 && infector_loc < number_workplaces, "infector location out of bounds in workplace contact");
				debug_assert(victim_loc >= 0 && victim_loc < number_workplaces, "victim location out of bounds in workplace contact");
				debug_assert(infected_wp == infector_loc, "workplace contact is not made at infector's workplace");
				debug_assert(infector_loc == victim_loc, "workplace contact between people with different workplaces");
				debug_assert(!weekend, "work contact during weekend");
			}
			else if(contact_type == CONTACT_TYPE_SCHOOL)
			{
				infector_loc = h_people_workplaces[contact_infector];
				victim_loc = h_people_workplaces[contact_victim];

				debug_assert(infector_loc >= h_workplace_type_offset[BUSINESS_TYPE_PRESCHOOL] && infector_loc < h_workplace_type_offset[BUSINESS_TYPE_UNIVERSITY+1], "infector loc for school contact is not school locationtype");
				debug_assert(victim_loc >= h_workplace_type_offset[BUSINESS_TYPE_PRESCHOOL] && victim_loc < h_workplace_type_offset[BUSINESS_TYPE_UNIVERSITY+1], "victim loc for school contact is not school locationtype");
				debug_assert(infected_wp == infector_loc, "school contact is not made at infector's school");
				debug_assert(infected_wp == victim_loc, "school contact between people with different schools");
				debug_assert(!weekend, "school contact during weekend");
			}
			else if(contact_type == CONTACT_TYPE_ERRAND)
			{
				if(weekend)
				{
					for(int errand = 0; errand < NUM_WEEKEND_ERRAND_HOURS && locs_matched == 0; errand++)
					{
						int infector_errand_loc = h_errand_people_destinations[(errand * number_people) + infected_index];
						int infector_errand_hour = h_errand_people_weekendHours[(errand * number_people) + infected_index];

						for(int j = 0; j < NUM_WEEKEND_ERRAND_HOURS && locs_matched == 0; j++)
						{
							int victim_errand_loc = h_errand_people_destinations[(j * number_people) + contact_victim];
							int victim_errand_hour = h_errand_people_weekendHours[(j * number_people) + contact_victim];

							if(infector_errand_loc == victim_errand_loc && infector_errand_hour == victim_errand_hour)
							{
								locs_matched = 1;
								infector_loc = infector_errand_loc;
								victim_loc = victim_errand_loc;
							}
						}
					}

					debug_assert(infector_loc >= h_workplace_type_offset[FIRST_WEEKEND_ERRAND_ROW] && infector_loc < number_workplaces, "infector location is not valid for weekend errand PDF");
					debug_assert(victim_loc >= h_workplace_type_offset[FIRST_WEEKEND_ERRAND_ROW] && victim_loc < number_workplaces, "victim location is not valid for weekend errand PDF");
					debug_assert(locs_matched, "infector and victim errand destinations do not match");
				}
				else
				{
					for(int hour = 0; hour < NUM_WEEKDAY_ERRAND_HOURS; hour++)
					{
						int infector_errand_loc = h_errand_people_destinations[(hour * number_workplaces) + infected_index];
						int victim_errand_loc = h_errand_people_destinations[(hour * number_workplaces) + contact_victim];
						if(infector_errand_loc == victim_errand_loc)
						{
							locs_matched = 1;
							infector_loc = infector_errand_loc;
							victim_loc = victim_errand_loc;
							break;
						}
					}

					debug_assert(infector_loc >= h_workplace_type_offset[FIRST_WEEKDAY_ERRAND_ROW] && infector_loc < number_workplaces, "infector location is not valid for weekday errand PDF");
					debug_assert(victim_loc >= h_workplace_type_offset[FIRST_WEEKDAY_ERRAND_ROW] && victim_loc < number_workplaces, "victim location is not valid for weekday errand PDF");
					debug_assert(locs_matched, "infector and victim errand destinations do not match");
				}
			}
			else if(contact_type == CONTACT_TYPE_AFTERSCHOOL)
			{
				int infector_loc = h_errand_people_destinations[infected_index];
				int victim_loc = h_errand_people_destinations[contact_victim];

				debug_assert(infector_loc >= h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL] && infector_loc < h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL + 1], "infector afterschool destination is not afterschool location type");
				debug_assert(victim_loc >= h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL] && victim_loc < h_workplace_type_offset[BUSINESS_TYPE_AFTERSCHOOL + 1], "victim afterschool destination is not afterschool location type");

				debug_assert(infector_loc == victim_loc, "infector and victim afterschool destinations do not match");
				debug_assert(!weekend, "afterschool contact on weekend");
			}
			else if(contact_type == CONTACT_TYPE_HOME)
			{
				int victim_hh = h_people_households[contact_victim];
				debug_assert(infected_hh == victim_hh, "household contact between people with different households");
			}
			else
				debug_print("error: invalid contact type");

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
	if(is_weekend())
	{
		int num_errands = NUM_WEEKEND_ERRANDS * number_people;
		thrust::copy_n(errand_people_destinations.begin(), num_errands, h_errand_people_destinations.begin());
		thrust::copy_n(errand_people_weekendHours.begin(), num_errands, h_errand_people_weekendHours.begin());
	}
	else
	{
		int num_errands = NUM_WEEKDAY_ERRAND_HOURS * number_people;
		thrust::copy_n(errand_people_destinations.begin(), num_errands, h_errand_people_destinations.begin());
	}
}
