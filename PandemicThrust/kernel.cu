
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "PandemicSim.h"

int main()
{
	PandemicSim sim;

	sim.setupSim();
	sim.runToCompletion();

	return 0;
}
