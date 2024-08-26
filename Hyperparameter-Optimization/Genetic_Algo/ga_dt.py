import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from deap import creator, base, tools, algorithms
from sklearn.preprocessing import StandardScaler
import random
import argparse
import os

random.seed(42)
np.random.seed(42)

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, index_col=[0])
    # df=df[['SrcWin','sHops','dHops','sTtl','dTtl','SynAck','SrcBytes','DstBytes','SAppBytes',\
    #                    'Dur','TotPkts','TotBytes','TotAppByte','Rate','SrcRate','Label']]
    df=df[['SrcWin', 'sHops', 'sTtl', 'dTtl', 'SrcBytes', 'DstBytes', 'Dur', 'TotBytes', 'Rate','Label']]
    
    #Le = LabelEncoder()
    #df['Label'] = le.fit_transform(df['Label'])
    print("loading data")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    return X, y

def individual_to_params(individual):
    criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, ccp_alpha = individual
    
    params = {"criterion": criterion, "splitter": splitter, "max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf, "min_weight_fraction_leaf": min_weight_fraction_leaf, "max_features": max_features, "max_leaf_nodes": max_leaf_nodes, "min_impurity_decrease": min_impurity_decrease, "ccp_alpha": ccp_alpha}
    
    return params

def createModel(individual, X_train, y_train):
    params = individual_to_params(individual)
    clf = DecisionTreeClassifier(random_state=42,**params)
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
P_CROSSOVER = 0.5
P_MUTATION = 0.5
NUM_GENERATIONS = 100
HALL_OF_FAME_SIZE = 10
print("c")

# Genetic Algorithm constants:
creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# ['entropy', 'random', None, 11, 2, 0, None, 90, 0.0, 0.0]
print("yo")
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

# Attribute generator 
toolbox.register("attr_criterion", random.choice, CRITERION)
toolbox.register("attr_splitter", random.choice, SPLITTER)
toolbox.register("attr_max_depth", random.choice, MAX_DEPTH)
toolbox.register("attr_min_samples_split", random.choice, MIN_SAMPLES_SPLIT)
toolbox.register("attr_min_samples_leaf", random.choice, MIN_SAMPLES_LEAF)
toolbox.register("attr_min_weight_fraction_leaf", random.choice, MIN_WEIGHT_FRACTION_LEAF)
toolbox.register("attr_max_features", random.choice, MAX_FEATURES)
toolbox.register("attr_max_leaf_nodes", random.choice, MAX_LEAF_NODES)
toolbox.register("attr_min_impurity_decrease", random.choice, MIN_IMPURITY_DECREASE)
toolbox.register("attr_ccp_alpha", random.choice, CCP_ALPHA)

# Structure initializers
# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_criterion, toolbox.attr_splitter, toolbox.attr_max_depth, toolbox.attr_min_samples_split, toolbox.attr_min_samples_leaf, toolbox.attr_min_weight_fraction_leaf, toolbox.attr_max_features, toolbox.attr_max_leaf_nodes, toolbox.attr_min_impurity_decrease, toolbox.attr_ccp_alpha), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("select", tools.selNSGA2)

toolbox.register("select", tools.selTournament, tournsize=5)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", tools.cxUniform, indpb=0.5)


def custom_mutate(individual):
    gene = random.randint(0,9) # Select which parameter to mutate
    if gene == 0:
        individual[0] = toolbox.attr_criterion()
    elif gene == 1:
        individual[1] = toolbox.attr_splitter()
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
        individual[9] = toolbox.attr_ccp_alpha()
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
        print("entered in iscx")
        train_file = os.path.join(data_path, 'ISCX_training.csv')
        test_file = os.path.join(data_path, 'ISCX_Testing.csv')
        X_train, y_train = load_and_preprocess(train_file)
        X_test, y_test = load_and_preprocess(test_file)
        # Scale the features
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
        
    elif dataset_name in ['isot_botnet', 'ctu_final']:  # For ISOT and CTU, we split a single dataset into train and test
        single_file = os.path.join(data_path, f'{dataset_name}.csv')
        X, y = load_and_preprocess(single_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   
        
        # Scale the features
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
        
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
    joblib.dump(clf, 'path_to_your_file'+'.pkl')

    # Print out its metrics
    predictions = clf.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, predictions))
    print("Precision: ", precision_score(y_test, predictions))
    print("Recall: ", recall_score(y_test, predictions))
    print("F1 score: ", f1_score(y_test, predictions))
    print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()

