
from Connection import Connection 

import sql_data
from Index import sql_index

import numpy as np
import scipy.stats as stats
from typing import Dict
from time import time
import random

# Optimization packages

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.factory import get_termination
from pymoo.optimize import minimize



def __range_to_sql__(query:Dict[str,np.array])->str:
    """Convert a dictionary of ranges to SQL format"""
    sql_range_array =[str(key)+" between "+str(val[0])+ " and " +str(val[1]) for key,val in query.items()]
    
    return " and ".join(sql_range_array)
def __range_to_sql_index__(query:Dict[str,np.array],operation:str)->str:
    operator = ""
    index =0
    if operation == "min":
        operator = " >= "
        index =1
    else:
        operator = " <= "
    sql_range_array =[str(key)+operation+operator+str(val[index]) for key,val in query.items()]
    return " and ".join(sql_range_array)
    
    
def __range_condition_on_index__(index_query:Dict[str,np.array],operation:str) ->str:
    index =0
    if operation == "min": ## to compute the End
        operator = " < "
        index =1
    else:                  ## to compute the Begin
        operator = " > "
    condition = ""

    size = len(index_query.keys())
    items = list(index_query.items())
    for i in range(size):
        key,val = items[i]
        if i + 1 == size:
            condition = condition + "(("+str(key)+operation+operator+str(val[index])+" and "+str(key)+operation+" != "+str(val[index])+" ) or ("+str(key)+operation+" = "+str(val[index])
        else:
            condition = condition + "(("+str(key)+operation+operator+str(val[index])+" and "+str(key)+operation+" != "+str(val[index])+" ) or ("+str(key)+operation+" = "+str(val[index])+" and "

    condition = condition + ")"*size*2
    return condition
def __range_on_index__(index_query:Dict[str,np.array]) ->str:
    conditions=[]
    for key,val in index_query.items():
        condition = "("+str(key)+"max >= "+str(val[0])+" and "+str(key)+"min <= "+str(val[1])+")"
        conditions.append(condition)
    return " and ".join(conditions)


def __one_table_query__(dataset:str,operator:str,query:Dict[str,np.array]):
    time_s = time()
    sql_range = __range_to_sql__(query)
    
    cnx = Connection()
    cur = cnx.conn.cursor()
    
    query_strata = sql_data.__execute_query_one_table__(dataset,operator,sql_range)
    cur.execute(query_strata)
    x = cur.fetchone()[0]
    

    return x ,time() - time_s

def __regular_query_execution__(dataset:str,operator:str,query:Dict[str,np.array],index_query:Dict[str,np.array]):
    s_time = time()
    sql_range = __range_to_sql__(query)
    
    cnx = Connection()
    cur = cnx.conn.cursor()
    
    query_stratas_nums,_ = __get_clusters__(dataset,query,index_query)
    res = 0
    count =0
    for i in query_stratas_nums:
        query_strata = sql_data.__execute_query_cluster__(dataset,operator,i,sql_range)
        cur.execute(query_strata)
        x = cur.fetchone()[0]
        if x:
            count+=1
            res+= int(x)

    return res,time() - s_time



def __sampling_query_execution_laplace__(dataset:str,operator:str,query:Dict[str,np.array],index_query:Dict[str,np.array],N_min,epsilon,e_N_Q,upper_bound_answer):
    s_time = time()
    num_,weight_ = __get_clusters__(dataset,query,index_query)

    N_Q = len(num_)
    
    # No condition because all queries trigger approx

    N_Q = np.max([N_Q + np.random.laplace(loc=0,scale=1/e_N_Q),N_min])
    sens_q = upper_bound_answer
    
    # Solve problem to split epsilon and find the best Sampling Rate
    problem = Epsilon_SamplingRate_problem(N_min,N_Q,epsilon,sens_q) 

    algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True)
    termination = get_termination("n_gen", 80)
    
    res = minimize(problem,
            algorithm,
            termination,
            seed=1,
            save_history=False,
            verbose=False)
    X = res.X
    F = res.F
    

    
    exps_dis = [exp[0]+exp[1] for exp in F]
    index = np.argmin(exps_dis)
    solution = X[index]
    
    gamma = solution[0]
    sample_size= int(np.max([int(solution[1]),1]))

    p_epsilon = epsilon*gamma
    q_epsilon = epsilon-p_epsilon

    
    res_clusters,p_s_dp,p_s = __sampling_laplace__(dataset,operator,N_min,N_Q,num_,weight_,query,p_epsilon,int(sample_size))
     
    # Estimator
    
    scale = (upper_bound_answer*sample_size)/q_epsilon
    
    res_approx = []
    res_dp =[]
    for i in range(len(res_clusters)):
        res_approx.append(float(res_clusters[i])/p_s[i])
        noise = np.random.laplace(loc=0,scale=scale)
        perturbed_res = float(res_clusters[i]) + noise
        perturbed_res = perturbed_res/p_s_dp[i]
        res_dp.append(perturbed_res)

    
    return np.mean(res_approx),np.mean(res_dp), time() - s_time


def __sampling_laplace__(dataset:str,operator:str,N_min,N_Q,num_,weigth_,query:Dict[str,np.array],epsilon:float,b:int):
    p_sensitivity = 0
    if len(num_) == 0:
        return [0],[p_sensitivity],[p_sensitivity]
    elif len(num_) == 1:
        sql_range = __range_to_sql__(query)
        v =__result_query_cluster__(dataset,operator,num_[0],sql_range)
        if v:
            return [int(float(v))],[p_sensitivity],[p_sensitivity]
        return [0],[p_sensitivity],[p_sensitivity]
    else:
        p_sensitivity = 1/(N_min*(N_min+1))
        s_R = np.sum(weigth_)
        p_s =[w/s_R for w in weigth_]
        
        #Adding noise
        epsilon = epsilon/np.max([len(num_),N_Q])
        p_dp = [np.max([p+ np.random.laplace(p_sensitivity/epsilon),0]) for p in p_s]
            ## normalize this probs (Post_proceess of DP)
        p_dp=[p/np.sum(p_dp) for p in p_dp]
        
        # Creating cumulative ranges requires Sorting
        sorted_num =[]
        sorted_p_s=[]
        sorted_p_dp = []
        for  i in sorted(enumerate(p_dp),key=lambda x:x[1]):
            sorted_num.append(num_[i[0]])
            sorted_p_s.append(p_s[i[0]])
            sorted_p_dp.append(i[1])   
        # create cumulative ranges
        ranges=[]
        for i in range(len(sorted_p_dp)):
            w = sorted_p_dp[i]
            r = np.sum(sorted_p_dp[0:i]) + w
            if r > w:
                ranges.append(r)
            else:
                ranges.append(w)
        
        p_dp_clusters=[]
        p_s_clusters= []
        results_each_cluster =[]
        for _ in range(b):
            x = random.random()
            
            cluster_index = [num for num,range in enumerate(ranges) if range > x ][0]
            
            # Save probability sampled cluster
            p_s_clusters.append(sorted_p_s[cluster_index])
            p_dp_clusters.append(sorted_p_dp[cluster_index])
            
            sql_range = __range_to_sql__(query)
            v =__result_query_cluster__(dataset,operator,sorted_num[cluster_index],sql_range)
            if v != v or v is None:
                v= 0
            # Save result sampled cluster
            results_each_cluster.append(v)

        return results_each_cluster,p_dp_clusters,p_s_clusters



def __get_clusters__(dataset:str,query:Dict[str,np.array],index_query:Dict[str,np.array]):
    
    sql_index_range_min = __range_condition_on_index__(index_query,"min")
    sql_index_range_max = __range_condition_on_index__(index_query,"max")
    
    cnx = Connection()
    cur = cnx.conn.cursor()
    # execute the index to get all stratified samples that satisfy the given conditions
    
    query_nums_stratas = sql_index.__clusters_in_range__(dataset,sql_index_range_min,sql_index_range_max)
    cur.execute(query_nums_stratas)
    rows = cur.fetchall()
    
    num_,weight_ = __cluster_weight_with_inclusion_probability(dataset,rows,query)
    return num_,weight_
    

class Epsilon_SamplingRate_problem(ElementwiseProblem):

    def __init__(self,N_min,N_Q,epsilon,sens_q):
        self.N_min = N_min
        self.N_Q = N_Q
        self.epsilon = epsilon
        self.sens_q =sens_q
        self.min_value = 1/(10**5) # This can be changed, it was a choice to define smallest number > 0
        self.max_value = 1 - self.min_value
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=[self.min_value,1],
                         xu=[self.max_value,self.N_Q - 1 ])

    def _evaluate(self, x, out, *args, **kwargs):
        delta_p = (1/(self.N_min*(self.N_min + 1))) * (self.N_Q/(x[0]*self.epsilon))
        delta_q = (self.sens_q * x[1])/ (1 -x[0]*self.epsilon)

        out["F"] = [delta_p, delta_q]



def __result_query_cluster__(name:str,operator:str,strata:int,sql_range:str):
    cnx = Connection()
    cur = cnx.conn.cursor()
    query_strata = sql_data.__execute_query_cluster__(name,operator,strata,sql_range)
    cur.execute(query_strata)
    return cur.fetchone()[0]



def __cluster_weight_with_inclusion_probability(table_name:str,rows,query):
    cnx = Connection()
    cursor = cnx.conn.cursor()
    num_=[]
    weight_ =[]
    for row in rows:
        num_s = row[0]
        p = 1.0
        for dim,range_ in query.items():
            q = sql_index.__get_probability_dim_value_greater__(table_name,num_s,dim,str(range_[0]))
            cursor.execute(q)
            p_g = float(cursor.fetchone()[0])
            q = sql_index.__get_probability_dim_value_greater__(table_name,num_s,dim,str(range_[1]))
            cursor.execute(q)
            p_g_2 = float(cursor.fetchone()[0])
            p= p * (p_g- p_g_2)
        if p>0:
            num_.append(num_s)
            weight_.append(p)
    return num_,weight_

