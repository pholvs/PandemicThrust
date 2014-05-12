#include "resource_logging.h"

#ifdef _MSC_VER
#define VISUAL_STUDIO 1
#else
#define VISUAL_STUDIO 0
#endif

FILE * fMemory = NULL;
size_t initial_free_bytes;
size_t initial_total_bytes;

size_t max_memory_used = 0;

cudaEvent_t event_start, event_stop;

float p_scale=0.f, l_scale=0.f;

void logging_pollMemUsage_doSetup(bool log_memory_usage, bool outputFilesInParentDir)
{
	//create events
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);

	cudaMemGetInfo(&initial_free_bytes, &initial_total_bytes);

	if(log_memory_usage){
		//do initial setup for memory log
		if(outputFilesInParentDir)
			fMemory = fopen("../output_mem.csv","w");
		else
			fMemory = fopen("output_mem.csv","w");
		fprintf(fMemory, "day,freeBytes,bytesUsed,totalBytes,megabytesUsed\n");

		if(VISUAL_STUDIO)
			fprintf(fMemory,"INITIAL,%Iu,0,%Iu,0\n");
		else
			fprintf(fMemory, "INITIAL,%zu,0,%zu,0\n", initial_free_bytes, initial_total_bytes);
	}

	//record the start event
	cudaEventRecord(event_start);
}

void logging_pollMemoryUsage_takeSample(int day)
{
	size_t current_free_bytes, current_total_bytes;

	cudaMemGetInfo(&current_free_bytes, &current_total_bytes);

	size_t bytes_used = initial_free_bytes - current_free_bytes;
	size_t megabytes_used = bytes_used >> 20;

	if(bytes_used > max_memory_used)
		max_memory_used = bytes_used;
	
	if(VISUAL_STUDIO)
		fprintf(fMemory,"%d,%Iu,%Iu,%Iu,%Iu\n",
			day, current_free_bytes, bytes_used, current_total_bytes, megabytes_used);
	else
		fprintf(fMemory, "%d,%zu,%zu,%zu,%zu\n",
			day, current_free_bytes, bytes_used, current_total_bytes, megabytes_used);
}

void logging_pollMemoryUsage_done()
{
	cudaEventRecord(event_stop);
	cudaEventSynchronize(event_stop);

	//calculate elapsed time
	float elapsed_milliseconds;
	cudaEventElapsedTime(&elapsed_milliseconds, event_start, event_stop);
	float elapsed_seconds = (float) elapsed_milliseconds / 1000;

	size_t current_free_bytes, current_total_bytes;
	cudaMemGetInfo(&current_free_bytes, &current_total_bytes);

	size_t bytes_used = initial_free_bytes - current_free_bytes;
	if(bytes_used > max_memory_used)
		max_memory_used = bytes_used;

	size_t max_megabytes_used = max_memory_used >> 20;


	FILE * fResourceLog = fopen("output_resource_log.csv", "w");
	fprintf(fResourceLog, "people_sim_scale,location_sim_scale,runtime_milliseconds,runtime_seconds,bytes_used,megabytes_used\n");

	if(VISUAL_STUDIO)
		fprintf(fResourceLog,"%f,%f,%f,%f,%Iu,%Iu\n",
			p_scale,l_scale,elapsed_milliseconds,elapsed_seconds,max_memory_used,max_megabytes_used);
	else
		fprintf(fResourceLog, "%f,%f,%f,%f,%zu,%zu\n",
			p_scale,l_scale,elapsed_milliseconds,elapsed_seconds,max_memory_used,max_megabytes_used);
	fclose(fResourceLog);


	if(fMemory != NULL)
		fclose(fMemory);

	cudaEventDestroy(event_start);
	cudaEventDestroy(event_stop);
}

void logging_setSimScale(float people_scale, float loc_scale)
{
	p_scale = people_scale;
	l_scale = loc_scale;

}
