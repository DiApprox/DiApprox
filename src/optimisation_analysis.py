
import numpy as np
import pandas as pd
import os
# Optimization packages

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.factory import get_termination
from pymoo.optimize import minimize

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


def __solve_problem__(N_min,N_Q,epsilon,sens_q) ->(float,float):        
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

        scale_q = (sens_q*sample_size)/q_epsilon
        scale_p = (1/(N_Q+1))* (1/p_epsilon)

        return scale_q,scale_p


def __problem_analysis_N_Q__(sens_q = 1,N_min=.1):
        N_Q = 100
        N_min = int(N_min * N_Q)
        while N_Q < 10001:
                
                scales_q =[]
                scales_p =[]
                epsilons = []
                for epsilon in range(100,3100,100):
                        epsilon = epsilon/1000
                        scale_q,scale_p =__solve_problem__(N_min,N_Q,epsilon,sens_q)
               
                        scales_q.append(scale_q)
                        scales_p.append(scale_p)
                        epsilons.append(epsilon)
                data={"epsilons":epsilons,"scales_q":scales_q,"scales_p":scales_p}
                df = pd.DataFrame(data)
                df.to_csv(os.getcwd()+"/src/Comp_Res/OP_Analysis/op_analysis_N_Q_"+str(N_Q)+"_.csv")
                if N_Q < 1000:
                      N_Q+=100
                else:
                      N_Q+=1000
        
def __problem_analysis_N_min__(N_Q = 100,sens_q = 1):
        N_min = .1
        while N_min < .6:
                
                scales_q =[]
                scales_p =[]
                epsilons = []
                for epsilon in range(100,3100,100):
                        epsilon = epsilon/1000
                        scale_q,scale_p =__solve_problem__(N_min,N_Q,epsilon,sens_q)
               
                        scales_q.append(scale_q)
                        scales_p.append(scale_p)
                        epsilons.append(epsilon)
                data={"epsilons":epsilons,"scales_q":scales_q,"scales_p":scales_p}
                df = pd.DataFrame(data)
                df.to_csv(os.getcwd()+"/src/Comp_Res/OP_Analysis/op_analysis_N_min_"+str(N_min)+"_.csv")
                
                N_min+=.1
               
          
def __problem_analysis_sens_q__(N_Q = 100,N_min=.2):
        N_min = int(N_min * N_Q)
        sens_q = 1
        while sens_q < 101:
                
                scales_q =[]
                scales_p =[]
                epsilons = []
                for epsilon in range(100,3100,100):
                        epsilon = epsilon/1000
                        scale_q,scale_p =__solve_problem__(N_min,N_Q,epsilon,sens_q)
               
                        scales_q.append(scale_q)
                        scales_p.append(scale_p)
                        epsilons.append(epsilon)
                data={"epsilons":epsilons,"scales_q":scales_q,"scales_p":scales_p}
                df = pd.DataFrame(data)
                df.to_csv(os.getcwd()+"/src/Comp_Res/OP_Analysis/op_analysis_sens_q_"+str(sens_q)+"_.csv")
                if sens_q < 10:
                      sens_q+=1
                else:
                      sens_q+=10
                #sens_q+=.1

def __problem_analysis__():
    
        __problem_analysis_N_Q__()
        __problem_analysis_sens_q__()
        __problem_analysis_N_min__()

__problem_analysis__()
    
