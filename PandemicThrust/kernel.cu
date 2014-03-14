#include "stdafx.h"

#include "PandemicSim.h"

#include "indirect.h"

int main()
{
	PandemicSim sim;

	sim.setupSim();
	sim.runToCompletion();

	return 0;
}
