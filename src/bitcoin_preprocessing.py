
from data_preprocessing import __aggregate_table_to_temp__,__clean_db__,__dims_declaration_with_types__,__dims_aggregation__,__dims_declaration__,__dims_cast_name__,__create_index__,__create_clusters__,__load_data_file__,__add_dbms_indexes_to_tables__, __dims_concat_pre_suf_fix__
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


def __main_bitcoin_preprocessing__(strata_portion):
    name ="bitcoin"
    adult_data_size= int(17965+ np.random.laplace(0,1))
    strata_size = int(strata_portion*adult_data_size)
    
    path = "/Data/Bitcoin/bitcoin.csv"
    
    dims = ["year","day","length","weight","count","looped","neighbors","income","label"]
    aggregation_dims = ["year","day","length","weight","count","looped","income"]
    domain =[7,29,29,13,29,29,7,29,1]
    reduced_domain = [7,1]
    domain_agg = [7,29,29,13,29,29,29]

    # converting dims to strings for sql queries
    dims_declaration  = __dims_declaration__(dims,"str","","_")
    agg_dims_declaration = __dims_declaration__(aggregation_dims,"str","","_")
    ### For the index 
    agg_dims_declaration_min = __dims_declaration__(aggregation_dims,"str","","_min")
    agg_dims_declaration_max = __dims_declaration__(aggregation_dims,"str","","_max")
    # for rdbms index
    agg_dims_min = " , ".join(__dims_concat_pre_suf_fix__(aggregation_dims,"str","","_min"))
    agg_dims_max = " , ".join(__dims_concat_pre_suf_fix__(aggregation_dims,"str","","_max"))
    ### End for index
    agg_dims = __dims_aggregation__(aggregation_dims,"str","","_")
    # For stratas
    agg_dims_declaration_s = __dims_declaration__(aggregation_dims,"str","_","_")
    agg_dims_s = __dims_aggregation__(aggregation_dims,"str","_","_")
    dims_passage_temp_to_strata = __dims_cast_name__([aggregation_dims,aggregation_dims],["str","str"],["","_"],["_","_"])


    ### Main Data preprocessing
    # Clean First
    __clean_db__(name)
    __load_data_file__(name,dims_declaration,path,",","HEADER")
    
    __aggregate_table_to_temp__(name,agg_dims_declaration,agg_dims)

    __create_index__(name,agg_dims_declaration_min,agg_dims_declaration_max)

    number_stratas = __create_clusters__(name,strata_size,agg_dims_declaration_s,dims_passage_temp_to_strata,agg_dims_s,domain_agg,True,True)

    
    __dump_meta_metadata_to_file__(name,strata_size,number_stratas,math.prod(reduced_domain))
    return strata_size,number_stratas
