import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
from sklearn.preprocessing import LabelEncoder 
import joblib 
import random 
import argparse 
import os 
import pyswarms as ps
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.preprocessing import StandardScaler
random.seed(42) 
np.random.seed(42) 

 
CRITERION = ["gini", "entropy"] 
SPLITTER = ["best", "random"] 
MAX_DEPTH = [None] + list(range(3, 51, 3)) 
MIN_SAMPLES_SPLIT = list(range(2, 21)) 
MIN_SAMPLES_LEAF = list(range(1, 21)) 
MIN_WEIGHT_FRACTION_LEAF = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] 
MAX_FEATURES = ["auto", "sqrt", "log2", None] 
MAX_LEAF_NODES = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
MIN_IMPURITY_DECREASE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] 
CCP_ALPHA = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05] 


class ParticleSwarmOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def f(self, x):
#         print(x)
        j = evalModel(x, self.X_train, self.y_train, self.X_test, self.y_test)
#         print("yououo")
#         print(j)
#         print("jfaksjd")
        return -np.array(j)

    def optimize(self):
        options = {'c1': 1.5, 'c2': 2, 'w': 0.9}
        lower_bounds = [0 for _ in [CRITERION, SPLITTER, MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF, MIN_WEIGHT_FRACTION_LEAF, MAX_FEATURES, MAX_LEAF_NODES, MIN_IMPURITY_DECREASE, CCP_ALPHA]]
        upper_bounds = [len(option)-1 for option in [CRITERION, SPLITTER, MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF, MIN_WEIGHT_FRACTION_LEAF, MAX_FEATURES, MAX_LEAF_NODES, MIN_IMPURITY_DECREASE, CCP_ALPHA]]
        bounds = (lower_bounds, upper_bounds)
        optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=10, options=options, bounds=bounds)
        cost, pos = optimizer.optimize(self.f, iters=50, n_processes=2) # Set n_processes to the number of cores you wish to use
#         print(cost,pos)
        return cost, pos

def load_and_preprocess(filepath): 
    df = pd.read_csv(filepath, index_col=[0]) 
    df=df[[]'SrcWin', 'sHops', 'sTtl', 'dTtl', 'SrcBytes', 'DstBytes', 'Dur', 'TotBytes', 'Rate','Label']]
    print("loading data") 
    X = df.iloc[:,:-1] 
    y = df.iloc[:,-1] 
    return X, y 

def individual_to_params(individual): 
    


    # Convert individual to params 
    criterion = CRITERION[int(np.round(individual[0]))] 
    splitter = SPLITTER[int(np.round(individual[1]))] 
    max_depth = MAX_DEPTH[int(np.round(individual[2]))] 
    min_samples_split = MIN_SAMPLES_SPLIT[int(np.round(individual[3]))] 
    min_samples_leaf = MIN_SAMPLES_LEAF[int(np.round(individual[4]))] 
    min_weight_fraction_leaf = MIN_WEIGHT_FRACTION_LEAF[int(np.round(individual[5]))] 
    max_features = MAX_FEATURES[int(np.round(individual[6]))] 
    max_leaf_nodes = MAX_LEAF_NODES[int(np.round(individual[7]))] 
    min_impurity_decrease = MIN_IMPURITY_DECREASE[int(np.round(individual[8]))] 
    ccp_alpha = CCP_ALPHA[int(np.round(individual[9]))] 
    params = {"criterion": criterion, "splitter": splitter, "max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf, "min_weight_fraction_leaf": min_weight_fraction_leaf, "max_features": max_features, "max_leaf_nodes": max_leaf_nodes, "min_impurity_decrease": min_impurity_decrease, "ccp_alpha": ccp_alpha} 

    return params 

def single_model_creation(params, X_train, y_train):   
    clf = DecisionTreeClassifier(random_state=42, **params) 
    clf.fit(X_train, y_train) 
    return clf
    
    

def createModel(individual, X_train, y_train): 
    my_clf_list=[]
    for i in individual:
        params = individual_to_params(i) 
        clf = DecisionTreeClassifier(random_state=42, **params) 
        clf.fit(X_train, y_train) 
        my_clf_list.append(clf)
    return my_clf_list

def evalModel(individual, X_train, y_train, X_test, y_test): 
    result_list=[]
    clf_list = createModel(individual, X_train, y_train) 
    for clf in clf_list:
        predictions = clf.predict(X_test) 
        f1 = f1_score(y_test, predictions) 
        accuracy = accuracy_score(y_test, predictions) 
        result_list.append(0.5 * f1 + 0.5 * accuracy)
    print(result_list)
    return result_list

  
  
def main(): 

    parser = argparse.ArgumentParser(description='Optimize Decision Tree hyperparameters using Particle Swarm Optimization.') 
    parser.add_argument('--dataset', type=str, help='Dataset name: "iscx", "isot", or "ctu"') 
    parser.add_argument('--data_path', type=str, help='Path to the dataset directory') 
    args = parser.parse_args()
    dataset_name = args.dataset.lower()
    data_path = args.data_path
    scaler = StandardScaler()
#    dataset_name = 'iscx' 
#    data_path = './data/' 
    if dataset_name == 'iscx': 
        print("entered in iscx") 
        train_file = os.path.join(data_path, 'ISCX_training.csv') 
        test_file = os.path.join(data_path, 'ISCX_Testing.csv') 
        X_train, y_train = load_and_preprocess(train_file) 
        X_test, y_test = load_and_preprocess(test_file) 
        
        # Scale the features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        
    elif dataset_name in ['isot_botnet', 'ctu_final']:  
        single_file = os.path.join(data_path, f'{dataset_name}.csv') 
        X, y = load_and_preprocess(single_file) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
        
        if dataset_name == 'ctu_final':
            majority_indices = np.where(y_train == 0)[0]
            # Find the indices of the minority class
            minority_indices = np.where(y_train == 1)[0]

            # Downsample the majority class indices
            downsampled_majority_indices = np.random.choice(majority_indices, size=len(minority_indices), replace=False)

            # Combine the downsampled majority indices with the original minority indices
            resampled_indices = np.concatenate([downsampled_majority_indices, minority_indices])
            print(resampled_indices)

            # Get the corresponding rows from X_train and y_train
            X_train = X_train.values[resampled_indices]
            y_train = y_train.values[resampled_indices]
        
        # Scale the features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    else: 
        raise ValueError('Unsupported dataset name. Please specify "iscx", "isot", or "ctu".') 
    print("dataset loading finished") 
    
    

    pso = ParticleSwarmOptimizer(X_train, y_train, X_test, y_test)
    cost, pos = pso.optimize()
    params = individual_to_params(pos) 
    print(params) 
    clf = single_model_creation(params, X_train, y_train) 
    predictions = clf.predict(X_test) 
    accuracy = accuracy_score(y_test, predictions) 
    joblib.dump(clf, 'path'+dataset_name+'.pkl') 

 
    print("Best parameters: ", params) 
    print("Accuracy: ", accuracy_score(y_test, predictions)) 
    print("Precision: ", precision_score(y_test, predictions)) 
    print("Recall: ", recall_score(y_test, predictions)) 
    print("F1 score: ", f1_score(y_test, predictions)) 
    print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))
    
    print("final optimize",0.5 * f1_score(y_test, predictions) + 0.5 * accuracy_score(y_test, predictions))

if __name__ == "__main__": 

    main() 