#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>

void logging_pollMemUsage_doSetup(bool pollMemory, bool outputFilesInParentDir);
void logging_setSimData(float people_scale, float loc_scale, const char * sim_type, const char * device, int core_seed);
void logging_pollMemoryUsage_takeSample(int day);
void logging_pollMemoryUsage_done();

