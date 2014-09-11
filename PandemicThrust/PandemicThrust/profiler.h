#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <stdlib.h>

#define MAX_STACK_DEPTH 128

class CudaProfiler
{
public:
	void initStack(const char * profile_filename, const char * log_filename);
	void beginFunction(int current_day, const char * function_name);
	void endFunction(int current_day, int problem_size);
	void dailyFlush();
	void done();
	const char * getCurrentFuncName();

private:

	FILE  *fProfileLog, *fFunctionCalls;

	cudaEvent_t profile_eventStack_start[MAX_STACK_DEPTH];
	cudaEvent_t profile_eventStack_end[MAX_STACK_DEPTH];
	double profile_timeInChildren[MAX_STACK_DEPTH];
	const char * profile_functionName[MAX_STACK_DEPTH];

	int stack_depth;
};
