import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

df = pd.read_csv("dataset/3_FinalDatasetCsv.csv")
X = df[["Acc X", "Acc Y", "Acc Z", "gyro_x", "gyro_y", "gyro_z"]]
y = to_categorical(df["label"], num_classes=3)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=df["label"]
)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_int", random.randint, 8, 128)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_nn(individual):
    n1, n2 = individual

    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(n1, input_shape=(6,), activation="relu"))
    model.add(Dense(n2, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_val, y_val, verbose=0)

    return (acc,)

toolbox.register("evaluate", evaluate_nn)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=8, up=128, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga():
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, 
                        stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    print(f"\nBest network: {best_ind}, Accuracy: {hof[0].fitness.values[0]:.4f}")

if __name__ == "__main__":
    run_ga()
