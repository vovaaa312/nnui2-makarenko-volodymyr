import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from perceptron import Perceptron
import os

# Vytvoření potřebných složek, pokud neexistují
os.makedirs("../models", exist_ok=True)
os.makedirs("../images", exist_ok=True)
os.makedirs("../docs", exist_ok=True)

# Načtení a příprava dat
wine = load_wine()
X = wine.data
y = (wine.target == 1).astype(int)  # Binární klasifikace: třída 1 vs ostatní

# Normalizace vstupních dat
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Rozdělení na trénovací a testovací množinu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Parametry experimentu
n_experiments = 10
learning_rate = 0.1
epochs = 100

all_errors = []
all_accuracies = []
models = []

# Provedení 10 trénovacích běhů
for i in range(n_experiments):
    p = Perceptron(input_size=X.shape[1], learning_rate=learning_rate, epochs=epochs)
    errors = p.train(X_train, y_train)
    acc = p.test(X_test, y_test)

    all_errors.append(errors)
    all_accuracies.append(acc)
    models.append((p.weights.copy(), acc))

    # Uložení výsledků
    np.save(f"../models/errors_run_{i}.npy", errors)
    np.save(f"../models/weights_run_{i}.npy", p.weights)

# Vytvoření boxplotu pro trénovací chyby (posledních 10 epoch)
plt.figure(figsize=(10, 6))
plt.boxplot([e[-10:] for e in all_errors])
plt.title("Trénovací chyba (posledních 10 epoch) pro 10 běhů")
plt.xlabel("Číslo experimentu")
plt.ylabel("Počet chyb")
plt.grid(True)
plt.savefig("images/perceptron_boxplot.png")

# Výpis nejlepšího modelu
best_idx = np.argmax(all_accuracies)
best_weights = models[best_idx][0]
best_accuracy = models[best_idx][1]

print(f"Nejlepší model: běh {best_idx}, přesnost: {best_accuracy}")
print("Váhy nejlepšího modelu:", best_weights)

