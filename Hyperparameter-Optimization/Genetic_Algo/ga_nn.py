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
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler

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

# Define individual to params function
def individual_to_params(individual):
    num_hidden, nodes, activation, optimizer, loss = individual
    params = {"num_hidden": num_hidden, "nodes": nodes, "activation": activation, "optimizer": optimizer, "loss": loss}
    return params

# Define model creation function
def createModel(individual):
    params = individual_to_params(individual)
    model = Sequential()
    model.add(Dense(params['nodes'], input_dim=15, activation=params['activation']))
    for _ in range(params['num_hidden']):
        model.add(Dense(params['nodes'], activation=params['activation']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=['accuracy'])
    return model

# Define the evaluation function
def evalModel(individual, X_train, y_train, X_test, y_test, epochs=50):
    model = createModel(individual)
    model.fit(X_train, y_train, epochs=epochs,batch_size=64, verbose=0)
    y_pred = model.predict(X_test)
    
    # Convert probabilities to class labels, you may need to adjust this depending on your problem
    y_pred = [1 if p > 0.5 else 0 for p in y_pred]
    acc=accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return (f1,acc,)

def makeEvalModel(X_train, y_train, X_test, y_test):
    def evalModelWrapper(individual):
        return evalModel(individual, X_train, y_train, X_test, y_test)
    return evalModelWrapper

# Define constants:
POPULATION_SIZE = 10
P_CROSSOVER = 0.7
P_MUTATION = 0.3
NUM_GENERATIONS = 20
HALL_OF_FAME_SIZE = 10

# Genetic Algorithm constants:
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


toolbox = base.Toolbox()

ACTIVATIONS = ['relu', 'sigmoid', 'tanh']
OPTIMIZERS = ['adam', 'sgd', 'rmsprop']
LOSSES = ['binary_crossentropy', 'hinge']
MAX_HIDDEN_LAYERS = 5
MAX_LAYER_NODES = 50

# Attribute generator 
toolbox.register("attr_activation", random.choice, ACTIVATIONS)
toolbox.register("attr_optimizer", random.choice, OPTIMIZERS)
toolbox.register("attr_loss", random.choice, LOSSES)
toolbox.register("attr_num_hidden", random.randint, 1, MAX_HIDDEN_LAYERS)
toolbox.register("attr_nodes", random.randint, 1, MAX_LAYER_NODES)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_num_hidden, toolbox.attr_nodes, toolbox.attr_activation, toolbox.attr_optimizer, toolbox.attr_loss), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("select", tools.selTournament, tournsize=5)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", tools.cxUniform, indpb=0.5)

# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("mate", tools.cxTwoPoint)

def custom_mutate(individual):
    gene = random.randint(0,4) # Select which parameter to mutate
    if gene == 0:
        individual[0] = toolbox.attr_num_hidden()  # number of hidden layers
    elif gene == 1:
        individual[1] = toolbox.attr_nodes()  # nodes per layer
    elif gene == 2:
        individual[2] = toolbox.attr_activation()  # activation function
    elif gene == 3:
        individual[3] = toolbox.attr_optimizer()  # optimizer
    elif gene == 4:
        individual[4] = toolbox.attr_loss()  # loss function
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
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    elif dataset_name in ['isot_botnet', 'ctu_final']:  # For ISOT and CTU, we split a single dataset into train and test
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
            print("ctu selection finished")
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
    model = createModel(best_ind)
    model.fit(X_train, y_train, epochs=50, verbose=0)
    predictions = model.predict(X_test)
    predictions = [round(x[0]) for x in predictions]  # rounding off the prediction values as they might be probabilities

    accuracy = accuracy_score(y_test, predictions)

    # Save the best model
    model.save('kyle_best_nn'+'path_to_file'+'.h5')  # for keras models, .h5 format is used

    # Print out its metrics
    print("Accuracy: ", accuracy_score(y_test, predictions))
    print("Precision: ", precision_score(y_test, predictions))
    print("Recall: ", recall_score(y_test, predictions))
    print("F1 score: ", f1_score(y_test, predictions))
    print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()

