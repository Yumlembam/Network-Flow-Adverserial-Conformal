import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from deap import creator, base, tools, algorithms
import random
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

random.seed(42)
np.random.seed(42)

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, index_col=[0])
#     df = df[['SrcWin', 'sHops', 'sTtl', 'dTtl', 'SrcBytes', 'DstBytes', 'Dur', 'TotBytes', 'Rate', 'Label']]
    df=df[['SrcWin', 'sHops', 'sTtl', 'dTtl', 'SrcBytes', 'DstBytes', 'Dur', 'TotBytes', 'Rate','Label']]
    
    #Le = LabelEncoder()
    #df['Label'] = le.fit_transform(df['Label'])
    print("loading data")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    return X, y

def individual_to_params(individual):
    n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, bootstrap,ccp_alpha= individual
    params = {"n_estimators": n_estimators, 
              "criterion": criterion, 
              "max_depth": max_depth, 
              "min_samples_split": min_samples_split, 
              "min_samples_leaf": min_samples_leaf, 
              "min_weight_fraction_leaf": min_weight_fraction_leaf, 
              "max_features": max_features, 
              "max_leaf_nodes": max_leaf_nodes, 
              "min_impurity_decrease": min_impurity_decrease, 
              "bootstrap": bootstrap, 
              "ccp_alpha": ccp_alpha,
              "n_jobs": -1
              }
    
    return params


def createModel(individual, X_train, y_train):
    params = individual_to_params(individual)
    clf = RandomForestClassifier(random_state=42,**params)
    clf.fit(X_train, y_train)
    return clf

def evalModel(individual, X_train, y_train, X_test, y_test):
    clf = createModel(individual, X_train, y_train)
    predictions = clf.predict(X_test)
    f1 = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    return (f1,accuracy,)

def makeEvalModel(X_train, y_train, X_test, y_test):
    def evalModelWrapper(individual):
        return evalModel(individual, X_train, y_train, X_test, y_test)
    return evalModelWrapper

# Define constants:
POPULATION_SIZE = 100
P_CROSSOVER = 0.7
P_MUTATION = 0.3
NUM_GENERATIONS = 100
HALL_OF_FAME_SIZE = 10

# # Genetic Algorithm constants:
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)rm 
creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

N_ESTIMATORS = [1,2,3,4,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191]
CRITERION = ["gini", "entropy"]
MAX_DEPTH = [None] + list(range(3, 51, 3))
MIN_SAMPLES_SPLIT = range(2, 21)
MIN_SAMPLES_LEAF = range(1, 21)
MIN_WEIGHT_FRACTION_LEAF = [0] + [i/20.0 for i in range(1, 11)]
MAX_FEATURES = [None,"sqrt", "log2"] + list(range(2, 10))
MAX_LEAF_NODES = [None] + list(range(10, 101, 10))
MIN_IMPURITY_DECREASE = [i/10.0 for i in range(0, 11)]
BOOTSTRAP = [True, False]
CCP_ALPHA = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
# OOB_SCORE = [True, False]

# Attribute generator 
# toolbox.register("attr_n_estimators", random.randint, 0, len(N_ESTIMATORS)-1)
# toolbox.register("attr_criterion", random.randint, 0, len(CRITERION)-1)
# toolbox.register("attr_max_depth", random.randint, 0, len(MAX_DEPTH)-1)
# toolbox.register("attr_min_samples_split", random.randint, 0, len(MIN_SAMPLES_SPLIT)-1)
# toolbox.register("attr_min_samples_leaf", random.randint, 0, len(MIN_SAMPLES_LEAF)-1)
# toolbox.register("attr_min_weight_fraction_leaf", random.randint, 0, len(MIN_WEIGHT_FRACTION_LEAF)-1)
# toolbox.register("attr_max_features", random.randint, 0, len(MAX_FEATURES)-1)
# toolbox.register("attr_max_leaf_nodes", random.randint, 0, len(MAX_LEAF_NODES)-1)
# toolbox.register("attr_min_impurity_decrease", random.randint, 0, len(MIN_IMPURITY_DECREASE)-1)
# toolbox.register("attr_bootstrap", random.randint, 0, len(BOOTSTRAP)-1)
# toolbox.register("attr_oob_score", random.randint, 0, len(OOB_SCORE)-1)

# Attribute generator 
toolbox.register("attr_n_estimators", random.choice, N_ESTIMATORS)
toolbox.register("attr_criterion", random.choice, CRITERION)
toolbox.register("attr_max_depth", random.choice, MAX_DEPTH)
toolbox.register("attr_min_samples_split", random.choice, MIN_SAMPLES_SPLIT)
toolbox.register("attr_min_samples_leaf", random.choice, MIN_SAMPLES_LEAF)
toolbox.register("attr_min_weight_fraction_leaf", random.choice, MIN_WEIGHT_FRACTION_LEAF)
toolbox.register("attr_max_features", random.choice, MAX_FEATURES)
toolbox.register("attr_max_leaf_nodes", random.choice, MAX_LEAF_NODES)
toolbox.register("attr_min_impurity_decrease", random.choice, MIN_IMPURITY_DECREASE)
toolbox.register("attr_bootstrap", random.choice, BOOTSTRAP)
toolbox.register("attr_ccp_alpha", random.choice, CCP_ALPHA)
# toolbox.register("attr_oob_score", random.choice, OOB_SCORE)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_n_estimators, toolbox.attr_criterion, toolbox.attr_max_depth, 
                  toolbox.attr_min_samples_split, toolbox.attr_min_samples_leaf, 
                  toolbox.attr_min_weight_fraction_leaf, toolbox.attr_max_features, 
                  toolbox.attr_max_leaf_nodes, toolbox.attr_min_impurity_decrease, 
                  toolbox.attr_bootstrap,toolbox.attr_ccp_alpha), n=1)
                  
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("select", tools.selTournament, tournsize=5)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", tools.cxUniform, indpb=0.5)

def custom_mutate(individual):
    gene = random.randint(0,10) # Select which parameter to mutate
    if gene == 0:
        individual[0] = toolbox.attr_n_estimators()
    elif gene == 1:
        individual[1] = toolbox.attr_criterion()
    elif gene == 2:
        individual[2] = toolbox.attr_max_depth()
    elif gene == 3:
        individual[3] = toolbox.attr_min_samples_split()
    elif gene == 4:
        individual[4] = toolbox.attr_min_samples_leaf()
    elif gene == 5:
        individual[5] = toolbox.attr_min_weight_fraction_leaf()
    elif gene == 6:
        individual[6] = toolbox.attr_max_features()
    elif gene == 7:
        individual[7] = toolbox.attr_max_leaf_nodes()
    elif gene == 8:
        individual[8] = toolbox.attr_min_impurity_decrease()
    elif gene == 9:
        individual[9] = toolbox.attr_bootstrap()
    elif gene == 10:
        individual[10] = toolbox.attr_ccp_alpha()
    return individual,

toolbox.register("mutate", custom_mutate)


def main():
    parser = argparse.ArgumentParser(description='Optimize Decision Tree hyperparameters using Genetic Algorithm.')
    parser.add_argument('--dataset', type=str, help='Dataset name: "iscx", "isot", or "ctu"')
    parser.add_argument('--data_path', type=str, help='Path to the dataset directory')
    args = parser.parse_args()
    # Load your training and test data
    dataset_name = args.dataset.lower()
    data_path = args.data_path
    scaler = StandardScaler()
    global X_train, y_train, X_test, y_test
    # If the dataset is ISCX, we expect separate training and testing files
    if dataset_name == 'iscx':
        train_file = os.path.join(data_path, 'ISCX_training.csv')
        test_file = os.path.join(data_path, 'ISCX_Testing.csv')
        X_train, y_train = load_and_preprocess(train_file)
        X_test, y_test = load_and_preprocess(test_file)
        
        # Scale the features
#        X_train = scaler.fit_transform(X_train)
#        X_test = scaler.transform(X_test)
        
    elif dataset_name in ['isot_botnet', 'ctu_final']:  # For ISOT and CTU, we split a single dataset into train and test
        single_file = os.path.join(data_path, f'{dataset_name}.csv')
        X, y = load_and_preprocess(single_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale the features
 #       X_train = scaler.fit_transform(X_train)
 #       X_test = scaler.transform(X_test)
        
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
    # Initialize a population and evolve it
    toolbox.register("evaluate", makeEvalModel(X_train, y_train, X_test, y_test))
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0]) # for the first objective
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    stats2 = tools.Statistics(lambda ind: ind.fitness.values[1]) # for the second objective
    stats2.register("avg", np.mean)
    stats2.register("std", np.std)
    stats2.register("min", np.min)
    stats2.register("max", np.max)
    
    mstats = tools.MultiStatistics(fitness=stats, fitness2=stats2)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, 
                                        ngen=NUM_GENERATIONS, stats=mstats, halloffame=hof, verbose=True)

    # Get the best individual from the Hall of Fame
    best_ind = hof[0]
    print("Best individual: %s\nwith fitness: %s" % (best_ind, best_ind.fitness))

    # Train and test the best individual on the full data
    clf = createModel(best_ind, X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save the best model
    joblib.dump(clf,'path_to_file'+'.pkl')

    # Print out its metrics
    predictions = clf.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, predictions))
    print("Precision: ", precision_score(y_test, predictions))
    print("Recall: ", recall_score(y_test, predictions))
    print("F1 score: ", f1_score(y_test, predictions))
    print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()

