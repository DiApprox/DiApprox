from query_processing import __regular_query_execution__,__sampling_query_execution_laplace__,__get_clusters__,__one_table_query__

from adult_preprocessing import __main_adult_preprocessing__,__main_adult_preprocessing_synth__
from bitcoin_preprocessing import __main_bitcoin_preprocessing__
from amazon_preprocessing import __main_amazon_preprocessing__

from workload.domain import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import pickle5 as pickle


def __prepare_the_data__():

    __main_adult_preprocessing__(.01)
    __main_adult_preprocessing_synth__(.001)

    __main_bitcoin_preprocessing__(.01)
    __main_amazon_preprocessing__(.001)
def __run_experiments__(runs_per_query=10):
    
    __prepare_the_data__()
    
    epsilons = [1]

    for e in tqdm(np.arange(len(epsilons)), " epsilon loop : ",leave=True):
        epsilon  = epsilons[e]
        e_N_Q = 0.1
        epsilon = epsilon -e_N_Q
        datasets=["adult","bitcoin","adult_synth","amazon"]
        operators = ["sum","count"]
        workloads_path = os.getcwd()+"/src/Workloads"


        for d in  tqdm(np.arange(len(datasets)), " Dataset loop : ",leave=True) :
                

            dataset = datasets[d]
                

            for o in tqdm(np.arange(len(operators)), " Operator loop : ",leave=True):
                operator = operators[o]

                
                S,N,D =[0,0,0]
                ##################################
                ## Read dataset meta-metadata from file
                ##################################
                file_to_open= dataset+"_meta.metadata"
                file_path = os.path.join(workloads_path, file_to_open)
                
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    S = data[0]
                    N = data[1]
                    D = data[2]
                
                N_min = N*.3 # 30 % of the clusters

                ##################################

                upper_bound =1
                if operator == "avg":
                    upper_bound = D
                elif operator== "sum":
                    upper_bound =D
                


                workload = []

                ##################################
                ## Read the workload from file
                ##################################
                dataset_workload = dataset
                # adult_synth uses the same worload as adult
                if "adult" in dataset_workload:
                    dataset_workload = "adult"
                
                file_to_open= dataset_workload+".wrkld"
                file_path = os.path.join(workloads_path, file_to_open)
                
                with open(file_path, 'rb') as f:
                        workload = pickle.load(f)
                ##################################
                
                results_normal=[]
                
                results_stratified=[]
                
                results_approx=[]
                
                time_ap_inclu=[]
                
                results_dp =[]
                
                time_normal=[]
                time_stratified=[]

                for i in tqdm(np.arange(len(workload)), " Query loop : ",leave=True):
                    w_query=workload[i]
                    query,query_index = w_query.__query_dicts__()
                    query_stratas_nums,_ = __get_clusters__(dataset,query,query_index)
                    N_ = len(query_stratas_nums)



                    res,time_e = __one_table_query__(dataset,operator,query_index)
                    results_normal.append(res)
                    time_normal.append(time_e)
                        

                    res,time_e = __regular_query_execution__(dataset,operator,query,query_index)
                    results_stratified.append(res)
                    time_stratified.append(time_e)
                        
                    res_approx=0
                    res_dp =0
                    time_e=0
                    for _ in range(runs_per_query):
                        r_a,r_dp,t =  __sampling_query_execution_laplace__(dataset,operator,query,query_index,N_min,epsilon,e_N_Q,upper_bound)
                        if r_dp == r_dp:
                            res_dp+=r_dp/runs_per_query
                            res_approx+= int(r_a/runs_per_query)
                        else:
                            res+= 0
                            res_dp+=0
                        time_e += t / runs_per_query
                        
                    results_approx.append(res_approx)
                    results_dp.append(res_dp)
                    time_ap_inclu.append(time_e)
                
                
                relative_error =[]
                relative_error_lap=[]
                se=[]
                se_lap=[]
                speed_up =[]
                for i in range(len(results_normal)):
                        if results_approx[i] > 0:
                            re = np.nan_to_num(np.abs(np.array(results_dp[i])-np.array(results_stratified[i]))/np.array(results_stratified[i]),1)
                            relative_error_lap.append(re)
                            relative_error.append(np.nan_to_num(np.abs(np.array(results_approx[i])-np.array(results_stratified[i]))/np.array(results_stratified[i]),1))
                            
                            se.append((results_approx[i] - results_stratified[i])**2 )
                            se_lap.append(((results_dp[i]) - results_stratified[i])**2 )
                            speed_up.append(np.array(time_normal[i])/np.array(time_ap_inclu[i]))
                data= {
                     "re" : relative_error,
                     "re_lap" : relative_error_lap,
                     "se" : se,
                     "se_lap" :se_lap,
                     "speed" : speed_up

                }
                df = pd.DataFrame(data)

                if dataset == "adult_synth":
                    df.to_csv(os.getcwd()+"/src/Comp_Res/Scalability/"+dataset+"_operator_"+operator+".csv")
                else:
                    df.to_csv(os.getcwd()+"/src/Comp_Res/Reel_Dataset/"+dataset+"_operator_"+operator+".csv")

__run_experiments__()