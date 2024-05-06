from src import *
from tests import *
import problexity as px
import numpy as np

######## conduct performance test vs problexity library and plot graphs ##############

#conduct analysis and plot results for F-based metrics

#set experiment variables
# functions_list  = [compute_F2_imbalanced_gpu,compute_F3_imbalanced_gpu,compute_F4_imbalanced_gpu]
# functions_prob  = [px.f2,px.f3,px.f4]
# values_samples  = [10**2, 10**3, 10**4, 10**5,10**6,10**7]
# values_features = np.arange(5)
# metrics_labels   = ['F2','F3','F4']

# results_samples, results_features = performance_analysis(functions_list,functions_prob,values_samples,values_features,10**5)
# plotting_results(results_samples,values_samples,results_features,2**values_features,metrics_labels)

#conduct analysis and plot results for L-based metrics

#set experiment variables
functions_list  = [compute_L1_imbalanced_gpu,compute_L2_imbalanced_gpu,compute_L3_imbalanced_gpu]
functions_prob  = [px.l1,px.l2,px.l3]
values_samples  = [10**2, 10**3, 10**4, 10**5]
values_features = np.arange(5)
metrics_labels   = ['L1','L2','L3']

results_samples, results_features = performance_analysis(functions_list,functions_prob,values_samples,values_features,10**4)
plotting_results(results_samples,values_samples,results_features,2**values_features,metrics_labels)



