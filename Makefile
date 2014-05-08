all: pandemicthrust pandemicthrust_validation

pandemicthrust: .FORCE
	/opt/nvidia/cuda-5.5/bin/nvcc PandemicThrust/main.cu PandemicThrust/PandemicSim.cu PandemicThrust/profiler.cu PandemicThrust/debug.cu PandemicThrust/resource_logging.cu -arch=sm_35 -o pandemicthrust -I Random123 -DSIM_VALIDATION=0

pandemicthrust_validation: .FORCE
	/opt/nvidia/cuda-5.5/bin/nvcc PandemicThrust/main.cu PandemicThrust/PandemicSim.cu PandemicThrust/profiler.cu PandemicThrust/debug.cu PandemicThrust/resource_logging.cu -arch=sm_35 -o pandemicthrust_validation -I Random123 -DSIM_VALIDATION=1


.FORCE:
	
