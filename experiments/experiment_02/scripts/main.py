import numpy as np
import matplotlib.pyplot as plt
from neuron_network import neuron_network
import os

# Vytvoření potřebných složek v experiment_02
os.makedirs("../models", exist_ok=True)
os.makedirs("../images", exist_ok=True)
os.makedirs("../docs", exist_ok=True)

# Příprava dat
X = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y = np.sin(X)

# Parametry experimentu
hidden_sizes = [1, 2, 4, 8, 16]
input_size = 1
output_size = 1
learning_rate = 0.01
epochs = 1000

loss_histories = []
models = []

# Provedení 5 trénovacích běhů
for i, hidden_size in enumerate(hidden_sizes):
    print(f"Trénování modelu s {hidden_size} neurony ve skryté vrstvě...")
    nn = neuron_network(input_size, hidden_size, output_size, learning_rate)
    loss_history = nn.train(X, y, epochs)

    # Uložení modelu a chyby
    nn.save_model(f"../models/model_hidden_{hidden_size}.npz")
    np.save(f"../models/loss_hidden_{hidden_size}.npy", loss_history)

    loss_histories.append(loss_history)
    models.append(nn)

# Vytvoření boxplotu pro trénovací chyby (posledních 100 epoch)
plt.figure(figsize=(10, 6))
plt.boxplot([loss[-100:] for loss in loss_histories], labels=[str(h) for h in hidden_sizes])
plt.title("Porovnání trénovacích chyb (posledních 100 epoch)")
plt.xlabel("Počet neuronů ve skryté vrstvě")
plt.ylabel("MSE")
plt.grid(True)
plt.savefig("../images/boxplot.png")

# Vyhodnocení nejlepšího modelu
final_losses = [loss[-1] for loss in loss_histories]
best_index = np.argmin(final_losses)
best_model = models[best_index]

print(f"Nejlepší model: {hidden_sizes[best_index]} neuronů, MSE: {final_losses[best_index]}")

# Vytvoření grafu predikce nejlepšího modelu
y_pred = best_model.predict(X)
plt.figure(figsize=(10, 6))
plt.plot(X, y, label="Skutečná funkce (sin)", linewidth=2)
plt.plot(X, y_pred, label=f"Předpověď modelu ({hidden_sizes[best_index]} neuronů)", linestyle='--')
plt.title("Předpověď nejlepšího modelu vs. sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("../images/best_model_prediction.png")

