#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void logging_pollMemUsage_doSetup(bool pollMemory, bool outputFilesInParentDir);
void logging_setSimScale(float people_scale, float loc_scale);
void logging_pollMemoryUsage_takeSample(int day);
void logging_pollMemoryUsage_done();

