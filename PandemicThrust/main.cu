#include "stdafx.h"

#include "PandemicSim.h"

#include "indirect.h"

int main()
{
	PandemicSim sim;

	sim.setupSim();
	sim.runToCompletion();
	/*
	try{
	}
	catch (std::runtime_error &e)
	{
		std::cerr << "Program crashed: " << e.what() << std::endl;

	}*/


	return 0;
}
