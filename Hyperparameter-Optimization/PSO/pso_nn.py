import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
from sklearn.preprocessing import LabelEncoder 
import joblib 
import random 
import argparse 
import os 
import pyswarms as ps
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, Hinge


random.seed(42) 
np.random.seed(42) 

# The optimizers dict mapping
OPTIMIZERS_DICT = {
    0: Adam(),
    1: SGD(),
    2: RMSprop()
}

# The losses dict mapping
LOSSES_DICT = {
    0: BinaryCrossentropy(),
    1: Hinge()
}

# Defining constants for the hyperparameters
ACTIVATIONS = ['relu', 'sigmoid', 'tanh']
OPTIMIZERS = ['adam', 'sgd', 'rmsprop']
LOSSES = ['binary_crossentropy', 'hinge']
MAX_HIDDEN_LAYERS = [1, 2, 3, 4, 5]
MAX_LAYER_NODES = [10, 20, 30, 40, 50]


def load_and_preprocess(filepath): 
    df = pd.read_csv(filepath, index_col=[0]) 
#     df = df[['SrcWin', 'sHops', 'sTtl', 'dTtl', 'SrcBytes', 'DstBytes', 'Dur', 'TotBytes', 'Rate', 'Label']] 
    df=df[['SrcWin','sHops','dHops','sTtl','dTtl','SynAck','SrcBytes','DstBytes','SAppBytes',\
                       'Dur','TotPkts','TotBytes','TotAppByte','Rate','SrcRate','Label']]
    print("loading data") 
    X = df.iloc[:,:-1] 
    y = df.iloc[:,-1] 
    return X, y 

def individual_to_params(individual):
    activation = ACTIVATIONS[int(np.round(individual[0]))]
    optimizer = OPTIMIZERS[int(np.round(individual[1]))]
    loss = LOSSES[int(np.round(individual[2]))]
    hidden_layers = MAX_HIDDEN_LAYERS[int(np.round(individual[3]))]
    layer_nodes = MAX_LAYER_NODES[int(np.round(individual[4]))]
    params = {
        'activation': activation, 
        'optimizer': optimizer, 
        'loss': loss, 
        'hidden_layers': hidden_layers, 
        'layer_nodes': layer_nodes,
        'input_nodes': 15
    }
    return params

def single_model_creation(params, X_train, y_train):
    model = Sequential()
    model.add(Dense(params['input_nodes'], input_dim=X_train.shape[1], activation=params['activation']))
    for _ in range(params['hidden_layers']-1):
        model.add(Dense(params['layer_nodes'], activation=params['activation']))
    model.add(Dense(1, activation='sigmoid')) # For binary classification
    
    model.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, verbose=0)
    return model
    
    
def createModel(individual, X_train, y_train): 
    my_model_list = []
    for i in individual:
        params = individual_to_params(i) 
        model = single_model_creation(params, X_train, y_train) 
        my_model_list.append(model)
    return my_model_list


def evalModel(individual, X_train, y_train, X_test, y_test): 
    result_list = []
    model_list = createModel(individual, X_train, y_train) 
    for model in model_list:
        predictions = (model.predict(X_test) > 0.5).astype("int32") 
        f1 = f1_score(y_test, predictions) 
        accuracy = accuracy_score(y_test, predictions) 
        result_list.append(0.5 * f1 + 0.5 * accuracy)
    print(result_list)
    return result_list


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
        lower_bounds = [0 for _ in [ACTIVATIONS, OPTIMIZERS, LOSSES, MAX_HIDDEN_LAYERS, MAX_LAYER_NODES]]
        upper_bounds = [len(option)-1 for option in [ACTIVATIONS, OPTIMIZERS, LOSSES, MAX_HIDDEN_LAYERS, MAX_LAYER_NODES]]
        bounds = (lower_bounds, upper_bounds)
        optimizer = ps.single.GlobalBestPSO(n_particles=40, dimensions=5, options=options, bounds=bounds)
        cost, pos = optimizer.optimize(self.f, iters=50, n_processes=5) # Set n_processes to the number of cores you wish to use
        return cost, pos


  
  
def main(): 

    parser = argparse.ArgumentParser(description='Optimize Decision Tree hyperparameters using Particle Swarm Optimization.') 
    parser.add_argument('--dataset', type=str, help='Dataset name: "iscx", "isot", or "ctu"') 
    parser.add_argument('--data_path', type=str, help='Path to the dataset directory') 
    args = parser.parse_args()
    dataset_name = args.dataset.lower()
    data_path = args.data_path
#     dataset_name = 'iscx' 
#     data_path = './data/' 
    scaler = StandardScaler()
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
        
        # Scale the features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
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
            X_train = X_train[resampled_indices]
            y_train = y_train.values[resampled_indices]
        
    else: 
        raise ValueError('Unsupported dataset name. Please specify "iscx", "isot", or "ctu".') 
    print("dataset loading finished") 
    
    

    pso = ParticleSwarmOptimizer(X_train, y_train, X_test, y_test)
    cost, pos = pso.optimize()
    params = individual_to_params(pos) 
    print(params) 
    clf = single_model_creation(params, X_train, y_train) 
    predictions = clf.predict(X_test)
    predictions = [round(x[0]) for x in predictions]  # rounding off the prediction values as they might be probabilities

    accuracy = accuracy_score(y_test, predictions)
    clf.save('path'+dataset_name+'.h5')  # for keras models, .h5 format is used

 
    print("Best parameters: ", params) 
    print("Accuracy: ", accuracy_score(y_test, predictions)) 
    print("Precision: ", precision_score(y_test, predictions)) 
    print("Recall: ", recall_score(y_test, predictions)) 
    print("F1 score: ", f1_score(y_test, predictions)) 
    print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))
    
    print("final optimize",0.5 * f1_score(y_test, predictions) + 0.5 * accuracy_score(y_test, predictions))

if __name__ == "__main__": 

    main() 