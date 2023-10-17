
from data_preprocessing import __aggregate_table_to_temp__,__clean_db__,__insert_synth_data__,__dims_declaration_with_types__,__dims_aggregation__,__dims_declaration__,__dims_cast_name__,__create_index__,__create_clusters__,__load_data_file__,__add_dbms_indexes_to_tables__, __dims_concat_pre_suf_fix__
import numpy as np
import math
import pickle5 as pickle
import os


# how to get for each number between 0 and 25 it's corresponding letter ? 
# Those are dataset dependant inputs : (name:str,dims:[int],add_dims[int], strata_size:int,path:str)


def __dump_meta_metadata_to_file__(dataset_name,S,N,D):
    directory_path = os.getcwd()+"/src/Workloads"
    with open(directory_path+"/"+dataset_name+"_meta.metadata", 'wb') as outp:
        pickle.dump([S,N,D], outp, pickle.HIGHEST_PROTOCOL)




def __main_adult_preprocessing__(strata_portion):
    
    name ="adult"
    adult_data_size= int(48839 + np.random.laplace(0,1))
    strata_size = int(strata_portion*adult_data_size)
    
    path = os.getcwd()+"/src/Data/Adult/data_0.csv"
    
    dims = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    full_domain = [74, 9, 100, 16, 16, 7, 15, 6, 5,2, 100,100,99,42, 2]
    
    reduced_domain = [2]
    
    aggregation_dims = [0,1,2,3,4,5,6,7,8,10,11,12,13,14]
    domain_agg =[74, 9, 100, 16, 16, 7, 15, 6, 5,100,100,99,42,2]

    # converting dims to strings for sql queries
    dims_declaration  = __dims_declaration__(dims,"int","","_")
    agg_dims_declaration = __dims_declaration__(aggregation_dims,"int","","_")
    ### For the index 
    agg_dims_declaration_min = __dims_declaration__(aggregation_dims,"int","","_min")
    agg_dims_declaration_max = __dims_declaration__(aggregation_dims,"int","","_max")
    # for rdbms index
    agg_dims_min = " , ".join(__dims_concat_pre_suf_fix__(aggregation_dims,"int","","_min"))
    agg_dims_max = " , ".join(__dims_concat_pre_suf_fix__(aggregation_dims,"int","","_max"))
    ### End for index
    agg_dims = __dims_aggregation__(aggregation_dims,"int","","_")
    # For stratas
    agg_dims_declaration_s = __dims_declaration__(aggregation_dims,"int","_","_")
    agg_dims_s = __dims_aggregation__(aggregation_dims,"int","_","_")
    dims_passage_temp_to_strata = __dims_cast_name__([aggregation_dims,aggregation_dims],["int","int"],["","_"],["_","_"])


    ### Main Data preprocessing
    # Clean First
    __clean_db__(name)
    __load_data_file__(name,dims_declaration,path,",","HEADER")
    
    __aggregate_table_to_temp__(name,agg_dims_declaration,agg_dims)

    __create_index__(name,agg_dims_declaration_min,agg_dims_declaration_max)

    number_cluster = __create_clusters__(name,strata_size,agg_dims_declaration_s,dims_passage_temp_to_strata,agg_dims_s,domain_agg,True,True)

   
    __dump_meta_metadata_to_file__(name,strata_size,number_cluster,math.prod(reduced_domain))

    return strata_size,number_cluster



def __main_adult_preprocessing_synth__(strata_portion):
    
    name ="adult_synth"
    number_random=10**9
    adult_data_size= 10**9
    strata_size = int(strata_portion*adult_data_size)
    
    
    
    dims = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    full_domain = [74, 9, 100, 16, 16, 7, 15, 6, 5,2, 100,100,99,42, 2]
    
    reduced_domain = [9,7,6,5,2,2]
    
    aggregation_dims = [0,2,3,4,6,10,11,12,13]
    domain_agg =[73,99,15,15,14,99,99,98,41]

    # converting dims to strings for sql queries
    dim_dec_ss = __dims_aggregation__(dims,"int","","_")
    dims_declaration  = __dims_declaration__(dims,"int","","_")
    agg_dims_declaration = __dims_declaration__(aggregation_dims,"int","","_")
    ### For the index 
    agg_dims_declaration_min = __dims_declaration__(aggregation_dims,"int","","_min")
    agg_dims_declaration_max = __dims_declaration__(aggregation_dims,"int","","_max")
    # for rdbms index
    agg_dims_min = " , ".join(__dims_concat_pre_suf_fix__(aggregation_dims,"int","","_min"))
    agg_dims_max = " , ".join(__dims_concat_pre_suf_fix__(aggregation_dims,"int","","_max"))
    ### End for index
    agg_dims = __dims_aggregation__(aggregation_dims,"int","","_")
    # For stratas
    agg_dims_declaration_s = __dims_declaration__(aggregation_dims,"int","_","_")
    agg_dims_s = __dims_aggregation__(aggregation_dims,"int","_","_")
    dims_passage_temp_to_strata = __dims_cast_name__([aggregation_dims,aggregation_dims],["int","int"],["","_"],["_","_"])


    ### Main Data preprocessing
    # Clean First
    __clean_db__(name)
    __insert_synth_data__(name,dims_declaration,dim_dec_ss,full_domain,number_random)
    
    __aggregate_table_to_temp__(name,agg_dims_declaration,agg_dims)

    __create_index__(name,agg_dims_declaration_min,agg_dims_declaration_max)

    number_stratas = __create_clusters__(name,strata_size,agg_dims_declaration_s,dims_passage_temp_to_strata,agg_dims_s,domain_agg,True,True)

   
    __dump_meta_metadata_to_file__(name,strata_size,number_stratas,math.prod(reduced_domain))

    return strata_size,number_stratas
