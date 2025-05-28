import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm

def main():
    # Проверка доступности GPU
    if not torch.cuda.is_available():
        raise RuntimeError("GPU není dostupné. Spusťte skript na zařízení s NVIDIA GPU a nainstalovaným CUDA. Zkontrolujte nvidia-smi a instalaci PyTorch s CUDA podporou.")

    start_time = time.strftime("%H:%M:%S %Z on %A, %B %d, %Y")

    # Cesty
    DATA_DIR = 'C:/datasets/GTSRB'
    MODELS_DIR = '../models'
    IMAGES_DIR = '../images'

    # Vytvoření složek покуд неexistují
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Parametry
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.0008
    num_classes = 43
    num_workers = 4

    device = torch.device('cuda')
    print(f"[INFO] Používám zařízení: {device} (GPU: {torch.cuda.get_device_name(0)})")

    # Transformace
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset
    train_data = torchvision.datasets.GTSRB(
        root=DATA_DIR, split='train', download=True, transform=transform)

    test_data = torchvision.datasets.GTSRB(
        root=DATA_DIR, split='test', download=True, transform=transform)

    # Rozdělení train na train/val
    train_size = int(0.85 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

    # DataLoadery
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Definice CNN modelu
    def build_model(topology):
        layers = []
        input_dim = 3
        for out_channels, kernel_size in topology:
            layers.append(nn.Conv2d(input_dim, out_channels, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            input_dim = out_channels

        # Perenášíme model_features na GPU
        model_features = nn.Sequential(*layers).to(device)

        # Spočítáme automaticky velikost
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(device)
            output = model_features(dummy_input)
            flatten_dim = output.shape[1] * output.shape[2] * output.shape[3]

        layers.append(nn.Flatten())
        layers.append(nn.Linear(flatten_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, num_classes))

        # Sbíráme finální model
        model = nn.Sequential(*layers).to(device)

        return model

    # Topologie CNN
    architectures = [
        [(16, 3)],
        [(32, 5), (16, 3)],
        [(64, 3), (32, 5)],
        [(32, 3), (64, 5), (32, 3)],
        [(64, 5), (64, 3), (32, 5)]
    ]

    # Trénink
    val_accuracies = {}
    val_losses = {}

    for idx, topology in enumerate(architectures):
        print(f"\n[INFO] Trénink topologie {idx + 1}: {topology}")
        acc_list = []
        loss_list = []

        start = time.perf_counter()

        for run in range(10):
            print(f"[INFO]   Běh {run + 1}/10")
            model = build_model(topology)
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Tréninková smyčka
            for epoch in range(num_epochs):
                print(f"[INFO]     Epoch {epoch + 1}/{num_epochs}")
                model.train()
                running_loss = 0.0
                for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"[INFO]       Loss: {running_loss / len(train_loader):.4f}")

            # Validace
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validace"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            acc = correct / total
            avg_loss = val_loss / len(val_loader)
            acc_list.append(acc)
            loss_list.append(avg_loss)
            print(f"[INFO]     Val Accuracy: {acc:.4f}, Val Loss: {avg_loss:.4f}")

        end = time.perf_counter()
        print(f"[INFO] Trénink topologie {idx + 1} trval {end - start:.2f} sekund.")

        val_accuracies[f'Topology_{idx + 1}'] = acc_list
        val_losses[f'Topology_{idx + 1}'] = loss_list

    # Boxploty
    plt.figure()
    plt.boxplot(val_accuracies.values(), labels=val_accuracies.keys())
    plt.title('Boxplot validační přesnosti (CNN)')
    plt.ylabel('Validační přesnost')
    plt.grid(True)
    plt.savefig(os.path.join(IMAGES_DIR, 'boxplot_cnn_accuracy.png'))
    plt.close()

    plt.figure()
    plt.boxplot(val_losses.values(), labels=val_losses.keys())
    plt.title('Boxplot validační chyby (loss) (CNN)')
    plt.ylabel('Validační loss')
    plt.grid(True)
    plt.savefig(os.path.join(IMAGES_DIR, 'boxplot_cnn_loss.png'))
    plt.close()

    # Vyber nejlepší topologii
    best_topology_name = max(val_accuracies, key=lambda k: np.median(val_accuracies[k]))
    best_index = int(best_topology_name.split('_')[-1]) - 1
    best_topology = architectures[best_index]

    # Finální trénink nejlepšího modelu
    print(f"\n[INFO] Nejlepší topologie: {best_topology}")
    model = build_model(best_topology)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"[INFO]   Final Epoch {epoch + 1}/{num_epochs}")
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Final Epoch {epoch + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Uložení modelu
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_cnn_model_final.pt'))

    # Vyhodnocení na testovacích datech
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testování"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (CNN)')
    plt.ylabel('Skutečné třídy')
    plt.xlabel('Predikované třídy')
    plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrix_cnn_final.png'))
    plt.close()

    print("\n[INFO] Skript úspěšně dokončen.")



if __name__ == '__main__':
    main()