all: pandemicthrust pandemicthrust_o3 pandemicthrust_validation

pandemicthrust: .FORCE
	/opt/nvidia/cuda-5.5/bin/nvcc PandemicThrust/main.cu PandemicThrust/PandemicSim.cu PandemicThrust/profiler.cu PandemicThrust/debug.cu PandemicThrust/resource_logging.cu PandemicThrust/host_functions.cpp -arch=sm_35 -o pandemicthrust -I Random123 -DSIM_VALIDATION=0

pandemicthrust_o3: .FORCE
	/opt/nvidia/cuda-5.5/bin/nvcc PandemicThrust/main.cu PandemicThrust/PandemicSim.cu PandemicThrust/profiler.cu PandemicThrust/debug.cu PandemicThrust/resource_logging.cu PandemicThrust/host_functions.cpp -arch=sm_35 -o pandemicthrust_o3 -I Random123 -DSIM_VALIDATION=0 --compiler-options "-O3" -use_fast_math


pandemicthrust_validation: .FORCE
	/opt/nvidia/cuda-5.5/bin/nvcc PandemicThrust/main.cu PandemicThrust/PandemicSim.cu PandemicThrust/profiler.cu PandemicThrust/debug.cu PandemicThrust/resource_logging.cu PandemicThrust/host_functions.cpp -arch=sm_35 -o pandemicthrust_validation -I Random123 -DSIM_VALIDATION=1


.FORCE:
	
