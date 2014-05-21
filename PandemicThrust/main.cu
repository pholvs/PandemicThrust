#include "stdafx.h"

#include "PandemicSim.h"

//#include "indirect.h"
#include "resource_logging.h"

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

	logging_pollMemUsage_doSetup(POLL_MEMORY_USAGE, OUTPUT_FILES_IN_PARENTDIR);

	PandemicSim sim;
	sim.setupSim();
	sim.runToCompletion();

	logging_pollMemoryUsage_done();

	return 0;
}
