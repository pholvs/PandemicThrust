#include "stdafx.h"

#include "PandemicSim.h"

//#include "indirect.h"
#include "resource_logging.h"



int main()
{
	logging_pollMemUsage_doSetup(POLL_MEMORY_USAGE, OUTPUT_FILES_IN_PARENTDIR);

	PandemicSim sim;
	sim.setupSim();
	sim.runToCompletion();

	logging_pollMemoryUsage_done();

	return 0;
}
