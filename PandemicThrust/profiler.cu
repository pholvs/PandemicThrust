

#include "profiler.h"

#include <stdio.h>
#include <string>

//call once to initialize the
void CudaProfiler::initStack()
{
	fProfileLog = fopen("../profile_log.csv", "w");
	fprintf(fProfileLog, "day,stack_depth,function_name,problem_size,inclusive_time_milliseconds,exclusive_time_milliseconds\n");
	fFunctionCalls = fopen("../profile_function_log.csv", "w");
	
	stack_depth = -1;
	
	for(int i = 0; i < MAX_STACK_DEPTH; i++)
	{
		cudaEventCreate(&profile_eventStack_start[i]);
		cudaEventCreate(&profile_eventStack_end[i]);
		profile_timeInChildren[i] = 0.0;
	}
}

//call when you begin a function - pushes a new timer event onto stack
void CudaProfiler::beginFunction(int current_day, const char * function_name)
{
	//check whether the stack is full
	if(stack_depth == MAX_STACK_DEPTH - 1)
	{
		fprintf(fProfileLog,"Error: profiler stack overflowed!\n");
		fflush(fProfileLog);
		exit(1);
	}
	
	stack_depth++;
	profile_timeInChildren[stack_depth] = 0.0;
	profile_functionName[stack_depth] = function_name;

	fprintf(fFunctionCalls, "%d,%d,beginning %s\n",current_day, stack_depth, function_name);
	fflush(fFunctionCalls);
	cudaEventRecord(profile_eventStack_start[stack_depth]);
}

//call when you end a function - pops a timer event and pushes the inclusive time upwards
void CudaProfiler::endFunction(int current_day, int problem_size)
{
	//record the event and synchronize
	cudaEventRecord(profile_eventStack_end[stack_depth]);
	cudaEventSynchronize(profile_eventStack_end[stack_depth]);
	
	//get the elapsed time between the events
	float inclusive_milliseconds;
	cudaEventElapsedTime(
			&inclusive_milliseconds, 
			profile_eventStack_start[stack_depth], 
			profile_eventStack_end[stack_depth]);
	
	//if this is the child of another function, add this time to the parent's child counter
	if(stack_depth > 0)
		profile_timeInChildren[stack_depth - 1] += inclusive_milliseconds;
	
	//if this function called any children, their time will be subtracted from this function's total time
	double exclusive_milliseconds = inclusive_milliseconds - profile_timeInChildren[stack_depth];
	
	//write log
	fprintf(fProfileLog, "%d, %d, %s, %d, %f, %lf\n",
			current_day, 
			stack_depth, profile_functionName[stack_depth],
			problem_size,
			inclusive_milliseconds, exclusive_milliseconds);
	fflush(fProfileLog);

	fprintf(fFunctionCalls, "%d,%d,ending %s\n", current_day, stack_depth, profile_functionName[stack_depth]);
	fflush(fFunctionCalls);
	//rotate to parent
	stack_depth--;
}

void CudaProfiler::done()
{
	//close streams
	fclose(fFunctionCalls);
	fclose(fProfileLog);

	//destroy event objects
	for(int i = 0; i < MAX_STACK_DEPTH; i++)
	{
		cudaEventDestroy(profile_eventStack_start[i]);
		cudaEventDestroy(profile_eventStack_end[i]);
	}
}

void CudaProfiler::dailyFlush()
{
	fflush(fProfileLog);
	if(stack_depth > 0)
	{
		printf("ERROR:  missed profile_end_function call somewhere\n");
		exit(1);
	}
	if(stack_depth < 0)
	{
		printf("ERROR: too many profile_end_function calls somewhere\n");
		exit(1);
	}
}

const char * CudaProfiler::getCurrentFuncName()
{
	if(stack_depth >= 0 && stack_depth < MAX_STACK_DEPTH)
		return profile_functionName[stack_depth];
	else 
		throw std::runtime_error(std::string("No function on stack"));
}
