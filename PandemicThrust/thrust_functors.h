#pragma once

#include "stdafx.h"

//given a ThreeTuple of integers, sort by the first one
struct Triplet_SortByFirst_Struct
{
	__host__ __device__
		bool operator() (thrust::tuple<int,int,int,int,int> a, thrust::tuple<int,int,int,int,int> b)
	{
		int a_idx = thrust::get<0>(a);
		int b_idx = thrust::get<0>(b);
		return a_idx < b_idx;
	}
};


//given a FiveTuple of integers, sort by the first one
struct FiveTuple_SortByFirst_Struct
{
	__host__ __device__
		bool operator() (thrust::tuple<int,int,int,int,int> a, thrust::tuple<int,int,int,int,int> b)
	{
		int a_idx = thrust::get<0>(a);
		int b_idx = thrust::get<0>(b);
		return a_idx < b_idx;
	}
};

//transform functor - given an action (infect both/pandemic/seasonal/none), and the victim's status,
//	return an appropriate action type.  This filters out victims who are not susceptible
struct filterPriorInfectedOp
{
	__device__
		int operator() (int action, thrust::tuple<int,int> status)
	{
		if(action == ACTION_INFECT_BOTH)
		{
			int status_p = thrust::get<0>(status);
			int status_s = thrust::get<1>(status);

			if(status_p == STATUS_SUSCEPTIBLE && status_s == STATUS_SUSCEPTIBLE)
				return action;		//everything is OK
			else if(status_p == STATUS_SUSCEPTIBLE)
				return ACTION_INFECT_PANDEMIC;		//not susceptible seasonal, return pandemic
			else if(status_s == STATUS_SUSCEPTIBLE)
				return ACTION_INFECT_SEASONAL;	//not susceptible pandemic, return seasonal
			else
				return ACTION_INFECT_NONE;		//susceptible to neither
		}
		else if(action == ACTION_INFECT_PANDEMIC)
		{
			int status_p = thrust::get<0>(status);
			if(status_p == STATUS_SUSCEPTIBLE)
				return action;					//everything is OK
			else 
				return ACTION_INFECT_NONE;		//victim not susceptible
		}
		else if(action == ACTION_INFECT_SEASONAL)
		{
			int status_s = thrust::get<1>(status);
			if(status_s == STATUS_SUSCEPTIBLE)
				return action;			//everything is ok
			else
				return ACTION_INFECT_NONE;	//victim not susceptible
		}
		else			//no op
			return ACTION_INFECT_NONE;
	}
};



//predicate functor - return true if action == NONE
struct removeNoActionOp
{
	__device__
		bool operator () (thrust::tuple<int,int,int> action_tuple)
	{
		int action = thrust::get<0>(action_tuple);
		return action == ACTION_INFECT_NONE;
	}
};

//we will use the unique command to remove duplicate infections, but this requires sorting
//given a three-tuple of (action, infector, victim) sort by victim (ascending), and then sort by action (descending)
struct actionSortOp
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

//predicate - filters out duplicate actions.  A person may have one pandemic and one seasonal infection action
//  or one "both" infection action, but may not receive the same strain twice
struct uniqueActionOp
{
	__device__
		bool operator () (thrust::tuple<int,int,int> a, thrust::tuple<int,int,int> b)
	{
		//if victim idxs are different, the actions are not duplicate
		int victim_a = thrust::get<2>(a);
		int victim_b = thrust::get<2>(b);

		if(victim_a != victim_b)
			return false;

		//indexes are identical - are the actions different?
		int op_a = thrust::get<0>(a);
		if(op_a == ACTION_INFECT_BOTH)		//any type of infection is duplicate with a "both" action
			return true;

		int op_b = thrust::get<0>(b);
		if(op_a == op_b)
			return true;
		else
			return false;

	}

};



//transformation functor - when people reach the culmination period, set their status to not infected
struct recoverInfectedOp
{
	int current_day;
	__device__
		thrust::tuple<int,int,int> operator () (int disease_infection_day, thrust::tuple<int,int,int> infection_triple)
	{
		if(disease_infection_day == DAY_NOT_INFECTED)
			return infection_triple;

		if(current_day + 1 - disease_infection_day == CULMINATION_PERIOD)
		{
			thrust::get<0>(infection_triple) = STATUS_RECOVERED;
			thrust::get<1>(infection_triple) = DAY_NOT_INFECTED;
			thrust::get<2>(infection_triple) = GENERATION_NOT_INFECTED;
		}
		return infection_triple;
	}
};

//predicate functor - determines whether this person has at least one active infection or not
struct notInfectedPredicate{
	__device__
		bool operator() (thrust::tuple<int, int> status)
	{
		int status_p = thrust::get<0>(status);
		int status_s = thrust::get<1>(status);
		if(status_p == STATUS_INFECTED || status_s == STATUS_INFECTED)
			return false;
		else
			return true;
	}
};



//returns true if the action type matches both or the virus strain we're looking at
struct infectionTypePred
{
	int reference_val;
	__device__ bool operator() (int action_type)
	{
		if(action_type == ACTION_INFECT_BOTH || action_type == reference_val)
			return true;
		else return false;
	}
};


//sets the generation to (parent + 1) or to null generation for each action
struct generationOp
{
	__device__
		thrust::tuple<int,int> operator () (int action, thrust::tuple<int, int> parent_gen)
	{
		thrust::tuple<int,int> ret;
		if(action == ACTION_INFECT_BOTH || action == ACTION_INFECT_PANDEMIC)
		{
			int gen_p_parent = thrust::get<0>(parent_gen);
			thrust::get<0>(ret) = gen_p_parent + 1;
		}
		else
			thrust::get<0>(ret) = GENERATION_NOT_INFECTED;
		if(action == ACTION_INFECT_BOTH || action == ACTION_INFECT_SEASONAL)
		{
			int gen_s_parent = thrust::get<1>(parent_gen);
			thrust::get<1>(ret) = gen_s_parent + 1;
		}
		else
			thrust::get<1>(ret) = GENERATION_NOT_INFECTED;

		return ret;
	}
};

struct Pair_SortByFirstThenSecond_struct
{
	__host__ __device__
		bool operator() (thrust::tuple<int,int> a, thrust::tuple<int,int> b)
	{
		int a_first= thrust::get<0>(a);
		int b_first = thrust::get<0>(b);

		int a_second = thrust::get<1>(a);
		int b_second = thrust::get<1>(b);

		if(a_first != b_first)
			return a_first < b_first;
		else
			return a_second < b_second;
	}
};