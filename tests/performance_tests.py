
## iterate over datasets and run GPU-based functions and CPU (problexity) based functions

import time
import numpy as np
import matplotlib.pyplot as plt

##### testing performance functions #####

def create_gaussian_dataset(n_samples=1000, imbalance_ratio=0.5, overlap_degree=0.5, random_state=None):
    """
    Create a 2D dataset with a predefined degree of class imbalance and class overlap using Gaussian distribution.
    
    Parameters:
    - n_samples: Total number of samples.
    - imbalance_ratio: Ratio of samples in class 1. (0 <= imbalance_ratio <= 1)
    - overlap_degree: Degree of overlap between classes. Lower values mean more overlap.
    - random_state: Seed for reproducibility.
    
    Returns:
    - X: Data matrix.
    - y: Class labels.
    """
    
    #np.random.seed(random_state)
    
    # Calculate number of samples for each class based on imbalance_ratio
    n_samples_class_0 = int(n_samples * (1 - imbalance_ratio))
    n_samples_class_1 = n_samples - n_samples_class_0
    
    # Generate samples for class 0
    mean_class_0 = [0, 0]
    cov_class_0 = [[1, overlap_degree], [overlap_degree, 1]]
    X_class_0 = np.random.multivariate_normal(mean_class_0, cov_class_0, n_samples_class_0)
    
    # Generate samples for class 1
    mean_class_1 = [2, 2]  # Shifted mean to create separation between classes
    cov_class_1 = [[1, overlap_degree], [overlap_degree, 1]]
    X_class_1 = np.random.multivariate_normal(mean_class_1, cov_class_1, n_samples_class_1)
    
    # Combine the samples and labels
    X = np.vstack((X_class_0, X_class_1))
    y = np.hstack((np.zeros(n_samples_class_0), np.ones(n_samples_class_1)))
    
    return X, y

def performance_analysis(functions_list,functions_prob,samples_values,feature_values,feature_nsamples):

    results_samples = np.zeros((len(functions_list),2,len(samples_values))) 

    for t,function in enumerate(functions_list):

        for i,value in enumerate(samples_values):

            X, y = create_gaussian_dataset(n_samples=value,imbalance_ratio=0.5)

            results_temp_pro, results_temp_gpu = [], []

            for _ in range(10):

                #### run GPU code

                start_time = time.time()

                F2_gpu = functions_list[t](X,y)

                end_time = time.time()

                results_temp_gpu.append(end_time-start_time)

                #### run problexity code

                start_time = time.time()

                F2_gpu = functions_prob[t](X,y)

                end_time = time.time()

                results_temp_pro.append(end_time-start_time)

            results_samples[t,0,i] = np.mean(np.array(results_temp_gpu))
            results_samples[t,1,i] = np.mean(np.array(results_temp_pro))


            print(f'Conducted Analysis on number of samples {value} and function: {function}')

    results_features = np.zeros((len(functions_list),2,len(feature_values))) 

    for t, function in enumerate(functions_list):

        for i, value in enumerate(feature_values):

            X, y = create_gaussian_dataset(n_samples=feature_nsamples,imbalance_ratio=0.5)

            results_temp_pro, results_temp_gpu = [], []
            
            #adding more features
            for _ in range(value):

                X = np.hstack([X,X])

            for _ in range(10):

                #### run GPU code

                start_time = time.time()

                F2_gpu = functions_list[t](X,y)

                end_time = time.time()

                results_temp_gpu.append(end_time-start_time)

                #### run problexity code

                start_time = time.time()

                F2_gpu = functions_prob[t](X,y)

                end_time = time.time()

                results_temp_pro.append(end_time-start_time)

            results_features[t,0,i] = np.mean(np.array(results_temp_gpu))
            results_features[t,1,i] = np.mean(np.array(results_temp_pro))


            print(f'Conducted Analysis with 2 to the power of {value} features and function: {function}')

    return results_samples, results_features

##### plotting results ################

def plotting_results(results_samples,values_samples,results_features,values_features,metrics_labels):
            
    #### n_samples plot
    
    fig,axes = plt.subplots(figsize=(16,10),ncols=2,sharey=True)

    for i in range(results_samples.shape[0]):

        axes[0].plot(values_samples,results_samples[i,0,:],'--', label = metrics_labels[i] + "(GPU implementation)")

    for i in range(results_samples.shape[0]):

        axes[0].plot(values_samples,results_samples[i,1,:],'-',label= metrics_labels[i] + "(Problexity implementation)")

    ## graph params

    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(axis='both',which='both',linestyle='--',alpha=0.5)
    axes[0].set_title('Function Execution Time vs. Number of Samples',fontsize=14)
    axes[0].set_ylabel('Runtime (s)',fontsize=12)
    axes[0].set_xlabel('# of samples',fontsize=12)
    axes[0].set_xlim([values_samples[0],values_samples[-1]])

    #### n_features plot ####

    for i in range(results_features.shape[0]):

        axes[1].plot(values_features,results_features[i,0,:],'--', label = metrics_labels[i] + "(GPU implementation)")

    for i in range(results_features.shape[0]):

        axes[1].plot(values_features,results_features[i,1,:],'-',label=metrics_labels[i] + "(Problexity implementation)")

    a = axes[1].legend(loc="upper left", bbox_to_anchor=(1.05, 0.5),edgecolor='k') 

    ## graph params

    axes[1].set_yscale('log')
    axes[1].grid(axis='both',which='both',linestyle='--',alpha=0.5)
    axes[1].set_title('Function Execution Time vs. Number of Features',fontsize=14)
    axes[1].set_xlabel('# of features',fontsize=12)
    axes[1].set_xlim([1,values_features[-1]])

    fig.suptitle('Runtime analysis between GPU and CPU based metrics implementation\n',fontsize=16,fontweight='bold')
    plt.tight_layout()
    plt.show()