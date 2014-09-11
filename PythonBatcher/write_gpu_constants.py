import os
import os.path as path

output_path = "./gpu_experiment_files"

def write_constants_file(experiment_name, max_days=100, rn_pandemic=1.8,rn_seasonal=1.3,initial_pandemic=30,initial_seasonal=30,people_scale=1,location_scale=1,percent_symptomatic=1,asymp_factor=0):
  experiment_dir = path.join(output_path,experiment_name)
  if path.exists(experiment_dir) == False:
    os.makedirs(experiment_dir)
  
  constants_path = path.join(experiment_dir, "constants.csv")
  f = open(constants_path, "w")
  f.write("constants,max_days,reproduction_number_pandemic,reproduction_number_seasonal,initial_infected_pandemic,initial_infected_seasonal,people_scaling_factor,location_scaling_factor,percent_symptomatic,asymp_factor\n")
  f.write("constants," + str(max_days) + "," + str(rn_pandemic) + "," + str(rn_seasonal) + "," + str(initial_pandemic) + "," + str(initial_seasonal) + "," + str(people_scale) + "," + str(location_scale) + "," + str(percent_symptomatic) + "," + str(asymp_factor) + "\n")
  f.close()
  
  return


  
if __name__ == "__main__":
  #make the directory
  if path.exists(output_path) == False:
    os.makedirs(output_path)
  
  #open the experiment name and config files
  f_experiment_files = open(path.join(output_path,"experiment_files.txt"), "w")
  f_experiment_names = open(path.join(output_path,"experiment_names.txt"), "w")
  
  scale_arr = [[0.1,0.1],[0.1,1],[1,1],[1,10],[10,10],[10,100],[20,20],[20,200]]
  init_p_arr = [30, 60]
  init_s_arr = [30, 60]
  
  
  for p_scale,l_scale in scale_arr:
    for init_p in init_p_arr:
      for init_s in init_s_arr:
        #come up with a descriptive name for this experiment configuration
        experiment_name = "p"+str(p_scale)+"_l" + str(l_scale) + "_" + str(init_p) + "_" + str(init_s)
        print experiment_name
        
        #write the constants file for this experiment configuration
        write_constants_file(experiment_name, initial_pandemic=init_p, initial_seasonal=init_s, people_scale=p_scale,location_scale=l_scale)
        
        #add this experiment to the global lists
        f_experiment_files.write(path.join(output_path,experiment_name,"constants.csv") + "\n")
        f_experiment_names.write(experiment_name + "\n")
        
  f_experiment_files.close()
  f_experiment_names.close()