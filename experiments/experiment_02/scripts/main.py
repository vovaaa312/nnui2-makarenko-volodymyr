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

# Vytvoření dokumentace experimentu
with open("../docs/experiment_02.md", "w", encoding="utf-8") as f:
    f.write("---\n")
    f.write("title: \"Cvičení 4\"\n")
    f.write("author: \"Vaše jméno\"\n")
    f.write("date: \"2025-03-25\"\n")
    f.write("---\n\n")
    f.write("# Záznam do deníku – Trénování neuronové sítě na funkci sin(x)\n\n")
    f.write("## Popis úlohy\n")
    f.write("Cílem experimentu bylo natrénovat jednoduchou plně propojenou neuronovou síť pro aproximaci matematické funkce **sin(x)**. Síť měla jednu skrytou vrstvu s proměnlivým počtem neuronů a lineární výstupní vrstvu.\n\n")
    f.write("## Parametry trénování\n")
    f.write("- **Vstupní vrstva**: 1 neuron\n")
    f.write("- **Výstupní vrstva**: 1 neuron\n")
    f.write("- **Aktivační funkce**: tanh (skrytá vrstva)\n")
    f.write("- **Počet epoch**: 1000\n")
    f.write("- **Learning rate**: 0.01\n")
    f.write("- **Ztrátová funkce**: Mean Squared Error (MSE)\n")
    f.write("- **Trénovací data**: 200 bodů rovnoměrně rozložených na intervalu [-2π, 2π]\n\n")
    f.write("## Experiment: Počet neuronů ve skryté vrstvě\n")
    f.write("Provedl jsem experiment s následujícími velikostmi skryté vrstvy:\n")
    f.write(f"{hidden_sizes}\n")
    f.write("Pro každý model jsem zaznamenal průběh trénovací chyby a finální MSE.\n\n")
    f.write("## Výsledky\n")
    f.write("### Trénovací chyba (MSE) – posledních 100 epoch\n")
    f.write("![Boxplot trénovacích chyb](images/boxplot.png)\n\n")
    f.write(f"### Nejlepší model\n")
    f.write(f"- **Počet neuronů ve skryté vrstvě**: {hidden_sizes[best_index]}\n")
    f.write(f"- **Finální trénovací chyba (MSE)**: {final_losses[best_index]:.4f}\n\n")
    f.write("### Váhy nejlepšího modelu\n")
    f.write("```python\n")
    f.write(f"W1: {models[best_index].W1}\n")
    f.write(f"b1: {models[best_index].b1}\n")
    f.write(f"W2: {models[best_index].W2}\n")
    f.write(f"b2: {models[best_index].b2}\n")
    f.write("```\n")
    f.write("### Porovnání výstupu nejlepšího modelu s funkcí sin(x)\n")
    f.write("![Predikce nejlepšího modelu](images/best_model_prediction.png)\n")