# -*- coding: utf-8 -*-
"""
    this module is written by users
    this module defines Design Varas,circuit,and all settings...
"""
import re

# Project
prj_name = "ThreeStageOpamp"  
## prj_name = Project name

# Design Variables
DX = [('W1', 0.22e-6, 2.4e-6, 1e-7, 1e-6,  'NO'),
      ('L1', 0.18e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),
      ('W3', 0.22e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),
      ('L3', 0.18e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),
      ('W5', 0.4e-6, 4.8e-6, 1e-7, 2e-6, 'NO'),
      ('L5', 0.18e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),

      ('W6', 1.2e-6, 14.4e-6, 1e-7, 6e-6, 'NO'),
      ('L6', 0.18e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),
      ('W8', 0.22e-6, 1.92e-6, 1e-7, 0.8e-6, 'NO'),
      ('L8', 0.18e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),
      ('W10', 2.4e-6,28.8e-6, 1e-7, 12e-6, 'NO'),
      ('L10', 0.18e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),

      ('W11', 12e-6, 144e-6, 1e-6, 60e-6, 'NO'),
      ('L11', 0.18e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),
      ('W12', 60e-6, 720e-6, 1e-6, 300e-6, 'NO'),
      ('L12', 0.18e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),

      ('W13', 0.22e-6, 2.4e-6, 1e-7, 1e-6, 'NO'),
      ('L13', 0.18e-6, 1.2e-6, 1e-7, 0.5e-6, 'NO'),
      #('m',1,5,1,2,'NO'),
      #('m2',1,5,1,2,'NO')
     ]
## DX = [('name',L,U,step,init,[discrete list]),....] if there is no discrete, do not write

# Setting
setting_1 = ['sim.sh', 'result.lis', 
[
("gain", None, None, ">", "80", "extract_perf(file,'gain')", "10"),
("pm", None, None, ">", "50", "extract_perf(file,'pm')", "10"),
("gbw", None, None, ">", "6e6", "extract_perf(file,'gbw')", "10"),
("power", None, None, None, None, "extract_perf(file,'power')", "10"),
]
]
##setting_1 = ['test1.sp', 'test1.lis', [("pm",None, None, ">","90","extract_pm(file)", "5"),]]
##setting_x = ['test name', 'result file', [(per1, </empty/<=,num1,>/empty/>=,num2, "extract function", weight),...]]
setting = [setting_1]
##setting = [setting_1, setting_2, ...]
FOM = ['power', "5e-3", None]
##FOM = ['-pm', "-90", 10]
##FOM = ['-gain', "-95", None]
##FOM = ['fuc', num1/None, weight/None] :minimize FOM, if FOM has constraint, the form must be "FOM < num"

# extract performance from result file
def extract_perf(file, perf):
    pattern_str = perf+'=\s*([\d.eE+\-]+)'
    pattern = re.compile(pattern_str)
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = pattern.search(line)
            if result:
                return result.group(1)
        return "0"

# Control prrameter
DEL_OUT_FOLDER = True
Mode = "Spice"
CPU_CORES = 2
##for DE, GA, PSO, SA, turbo, this parameter can be more according to the Init_num

# set the optimization algorithm
'''
# if using src/scripts/bak
def get_totNum(Init_num, Max_eval, Algorithm):
    if Algorithm in ("Bayes", "BOc", "weibo_py", "bobyqa_py"):
        totNum = Init_num + Max_eval
    elif Algorithm in ("PSO", "GA"):
        totNum = Init_num * (Max_eval+2)
    elif Algorithm == "SA":
        totNum = Init_num * Max_eval+1
    elif Algorithm == "pycma":
        totNum = Init_num * Max_eval+2
    elif Algorithm == "DE":
        totNum = Init_num * Max_eval * 2
    elif Algorithm in ("SQP", "bobyqa"):
        totNum = 600
    elif Algorithm in ("tssbo"):
        totNum = Init_num * Max_eval + 1
    elif Algorithm in ("random"):
        totNum = Init_num * Max_eval
    elif Algorithm in ("turbo"):
        totNum = Init_num*Tr_num + Max_eval
    elif Algorithm in ("pyVTS"):
        totNum = Init_num + Max_eval
    elif Algorithm in ("bbgp"):
        totNum = 4*Init_num + 10*Max_eval
    return totNum
'''

# Algorithm
Algorithm = "tssbo" 
##"Bayes", "BOc", "weibo_py", "DE", "GA", "PSO", "SQP", "bobyqa", "SA", "random", "turbo", "pycma", "bobyqa_py", "pyVTS"
Init_num = 20
Max_eval = 10
Tr_num = 5 # for trust region related algorithms
##"Bayes": (20, 80),
##"BOc": (20, 80),
##"weibo_py": (20, 80),
##"DE": (20, 20),
##"GA": (20, 20),
##"PSO": (20, 20),
##"SQP": (200, 400),
##"bobyqa": (200, 400),
##"SA": (20, 20),
##"random": (100, 100),
##"turbo": (2*dim, 100)
##"pycma": (4+3*np.log(dim), 100+150*(dim+3)**2//init_num**0.5)
##"bobyqa_py": (2*dim+1, 100)
##"pyVTS": (2*dim, 100)
'''
# if using src/scripts/bak
TOT_NUM = get_totNum(Init_num, Max_eval, Algorithm)
'''
WITH_CONS = False
##for python algos this parameter can be True 
##for cxx algos, this parameter is useless
##but note that BOc always carries out constrained optimization
##and note that turbo can only carry out cost optimization
WITH_INIT = False
##for python algos this parameter can be True
##for cxx algos, this parameter is useless (always False)
