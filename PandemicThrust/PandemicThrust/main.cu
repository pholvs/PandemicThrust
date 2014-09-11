#include "stdafx.h"

#include "PandemicSim.h"

//#include "indirect.h"
#include "resource_logging.h"

#if CUDA_PROFILER_ENABLE == 1
#include "cuda_profiler_api.h"
#endif

#ifdef _MSC_VER
#include <Windows.h>

void delay_start()
{
	int milliseconds = 1000 * MAIN_DELAY_SECONDS;
	Sleep(milliseconds);
}
#else
#include <unistd.h>

void delay_start()
{
	sleep(MAIN_DELAY_SECONDS);
}
#endif


int main()
{
	if(MAIN_DELAY_SECONDS > 0)
		delay_start();

	if(CUDA_PROFILER_ENABLE)
		cudaProfilerStart();

	logging_pollMemUsage_doSetup(POLL_MEMORY_USAGE, OUTPUT_FILES_IN_PARENTDIR);

	PandemicSim sim;
	sim.setupSim();
	sim.runToCompletion();

	logging_pollMemoryUsage_done();

	if(CUDA_PROFILER_ENABLE)
		cudaProfilerStop();

//	cudaDeviceReset();

	return 0;
}
