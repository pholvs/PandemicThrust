### What is this repository for? ###

This is the repository for PandemicThrust, an agent-based influenza simulation which runs on GPUs.

### How do I get set up? ###

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