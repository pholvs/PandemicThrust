### What is this repository for?

This is the repository for PandemicThrust, an agent-based influenza simulation which runs on GPUs.

Since this is a stochastic simulation, batch control is provided by the PythonBatcher program.  Sample scripts to write the configuration files for enqueueing are provided in a separate directory.

### How do I get set up?

This repo should contain all code libraries needed to get started.  These libraries may be outdated, but the application works as presented.

However, you will need the following:
* Visual Studio 2012 or g++
* NVIDIA CUDA Toolkit v5.5

To build the project using g++:
* Copy constants.csv and seed.txt from the subdirectory into the root of the project
* Edit the makefile, setting the variable "arch=sm_35" to your proper compute capability.  The application will fail to run if your device does not support the given compute capability.
* Type make, and run the application

To build the application using Visual Studio:
* Edit the project settings.  Set the target CUDA architecture to your proper compute capability settings.

### Project Description

#### Main.cu

Contains only the main method needed to launch the simulation.

#### SimParameters.h

Contains important defines, typedefs, and infrequently-changed configuration options that are used across multiple files.

#### PandemicSim.h

Prototypes for the main simulation code, as well as frequently-changed configuration options.  Profiles for different devices (to suit available shared memory per processor, etc) can be selected using the commented lines at the top.

#### PandemicSim.cu

Contains main simulation code.  For a more detailed description of program flow, pseudocode, etc, consult thesis document.

The justification for not breaking this out further into separate files is compatibility.  CUDA Version 1.0 and 1.1 are not capable of linking between multiple source files, which is referred to as "relocatable device code". All kernels and device functions/variables must be in one source file.

To try and make that a little more maintainable, we use the following naming convention.  Use an IDE that gives you a list of methods to navigate the file.

* PandemicSim namespace - contains high-level functions that control the overall flow of the simulation.  Anything in this namespace is executed on the CPU.  However, function calls within these methods may invoke GPU computation.

* No namespace - Contains mostly GPU code and Thrust Functors (for GPU) as well as a few small "helper" routines.

* GPU code has method names prefixed with the type of function:  methods beginning with kernel_ are high-level GPU programs that handle iteration, etc, while methods that begin with device_ are lower-level that are called by kernels.

* Function names also usually incorporate what part of the program calls them - for example, "setup", "daily", "weekday", "weekend", "final", "debug" or "logging"

#### debug.cu/.h and host_functions.cpp

Contains unit tests to ensure that GPU output is valid, as well as some equivalent Random123 code for the CPU.

Because Visual Studio treats the CUDA framework differently, I couldn't get the standard MS unit test suite to work with it.  Instead, the program strategically copies some internal GPU variables back to CPU memory at key points during the simulation, and validates these instead.  This isn't totally ideal, but it was simple and cross-platform.

#### resource_logging.cu

Contains methods to monitor runtime and memory usage of simulations.

#### Profiler.cu

This program was written during the switchover from CUDA 5.5 to 6.0.  As a result, I was using the beta releases of the NSIGHT tool suite. I had problems getting this beta version of the visual profiler to connect to CUDA 5.5 simulations properly.  Also, the CUDA profiler logs are quite large and would be difficult to justify saving during routine runs.

I wrote a simple recursive-stack profiler to give myself an idea where the program runtime was concentrated.  Call the beginFunction method when you want to start timing a region, and the endFunction method when you want to stop timing a region, and it will output inclusive and exclusive time.  These files are quite small and the overhead is low, so they can be routinely used during computation for live performance measurements.

#### ThrustFunctors.h

Deprecated functors from older implementations.