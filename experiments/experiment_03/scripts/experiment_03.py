from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# Vytvoření potřebných adresářů
os.makedirs("../models", exist_ok=True)
os.makedirs("../docs", exist_ok=True)
os.makedirs("../images", exist_ok=True)

print(tf.config.list_physical_devices('GPU'))

# Načtení a předzpracování dat
iris = load_iris()
X = iris.data
y = iris.target

# Normalizace příznaků
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot kódování štítků
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Rozdělení dat: 70 % trénovací, 15 % validační, 15 % testovací
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Definice modelových architektur (upravené topologie)
topologies = [
    [10, 3],
    [20, 10, 3],
    [40, 20, 10, 3],
    [80, 40, 20, 10, 3],
    [160, 80, 40, 20, 10, 3]
]

def build_model(topology):
    # Vytvoření sekvenčního modelu
    model = Sequential()
    model.add(Dense(topology[0], activation='relu', input_shape=(4,)))
    for units in topology[1:-1]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 třídy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Provádění experimentů
history_per_topology = {}
for idx, topology in enumerate(topologies):
    acc_histories = []
    for run in range(10):
        model = build_model(topology)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, verbose=0)
        acc_histories.append(history.history['val_accuracy'][-1])
    history_per_topology[f'Topology_{idx+1}'] = acc_histories

# Vytvoření boxplotu
plt.boxplot(history_per_topology.values(), labels=history_per_topology.keys())
plt.title('Boxplot validační přesnosti jednotlivých topologií')
plt.ylabel('Validační přesnost')
plt.xlabel('Topologie')
plt.grid(True)
plt.savefig('../images/boxplot_val_accuracy.png')

# Najít nejlepší topologii
best_topology_name = max(history_per_topology, key=lambda k: np.median(history_per_topology[k]))
best_topology_index = list(history_per_topology.keys()).index(best_topology_name)
best_topology = topologies[best_topology_index]

# Natrenování finálního modelu
final_model = build_model(best_topology)
final_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, verbose=0)

# Vyhodnocení na testovací sadě
test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Uložení modelu
final_model.save('../models/best_model.h5')

