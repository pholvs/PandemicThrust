pandemicthrust: .FORCE
	/opt/nvidia/cuda-5.5/bin/nvcc PandemicThrust/main.cu PandemicThrust/PandemicSim.cu PandemicThrust/profiler.cu PandemicThrust/debug.cu PandemicThrust/resource_logging.cu -arch=sm_35 -o pandemicthrust -I Random123

.FORCE:
	
