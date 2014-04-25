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


const int ACTION_TUPLE_VICTIM_IDX = 1;
const int ACTION_TUPLE_ACTION_IDX = 0;



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
		if(op_a == ACTION_INFECT_BOTH)        //any type of infection is duplicate with a "both" action
			return true;

		int op_b = thrust::get<0>(b);
		if(op_a == op_b)
			return true;
		else
			return false;
	}

};
