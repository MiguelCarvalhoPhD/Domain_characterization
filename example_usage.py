import src
import tests
import numpy as np

#list all available functions
all_attributes = dir(src)
functions = [attr for attr in all_attributes if attr.startswith('compute')]

#create a balanced dataset and an imbalanced dataset
X_imbalanced, y_imbalanced = tests.create_gaussian_dataset(n_samples=10**4,imbalance_ratio=0.01,overlap_degree=0.5,random_state=1)
X_normal    , y_normal     = tests.create_gaussian_dataset(n_samples=10**4,imbalance_ratio=0.5,overlap_degree=0.5,random_state=1)


#iterate over functions and 
for func_name in functions:

    func = getattr(src, func_name)

    print(func)

    metric_value_imbalanced = func(X_imbalanced,y_imbalanced)
    metric_value_balanced   = func(X_normal,y_normal)

    if not isinstance(metric_value_imbalanced,float):

        metric_value_imbalanced = metric_value_imbalanced[-1]
        metric_value_balanced   = metric_value_balanced[-1]

    print(f'{func}: Imbalanced dataset: {np.round(metric_value_imbalanced,3)}, Balanced dataset: {np.round(metric_value_balanced,3)}')