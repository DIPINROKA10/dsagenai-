"""
train.py - Training Script for Discriminative (CNN) and Generative (CVAE) Models

Trains:
1. SmallCNN classifier on MNIST for 3-5 epochs (>98% target test accuracy).
2. ConditionalVAE on MNIST for 10 epochs (latent_dim=16, conditioned on digit class).

Saves:
- ./checkpoints/classifier.pt
- ./checkpoints/cvae.pt
- ./checkpoints/metrics.json
- ./checkpoints/confusion_matrix.png
"""

import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SmallCNN, ConditionalVAE, cvae_loss_function


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_classifier(model, train_loader, test_loader, device, epochs: int = 4, lr: float = 0.001):
    print(f"\n--- Training Discriminative SmallCNN ({epochs} epochs) ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # Evaluation on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    test_acc = 100.0 * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    return test_acc, all_preds, all_targets


def train_cvae(model, train_loader, test_loader, device, epochs: int = 10, lr: float = 0.001):
    print(f"\n--- Training Generative Conditional VAE ({epochs} epochs) ---")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_kld = 0.0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            # One-hot encode targets
            y_onehot = torch.zeros(data.size(0), 10, device=device)
            y_onehot.scatter_(1, target.unsqueeze(1), 1.0)

            optimizer.zero_grad()
            recon_data, mu, logvar = model(data, y_onehot)
            loss, bce, kld = cvae_loss_function(recon_data, data, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_bce += bce.item() * data.size(0)
            train_kld += kld.item() * data.size(0)
            total_samples += data.size(0)

        epoch_loss = train_loss / total_samples
        print(f"Epoch {epoch}/{epochs} | Total Loss (ELBO): {epoch_loss:.2f} | BCE: {train_bce / total_samples:.2f} | KLD: {train_kld / total_samples:.2f}")

    # Evaluate test reconstruction loss
    model.eval()
    test_bce_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_onehot = torch.zeros(data.size(0), 10, device=device)
            y_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            recon_data, mu, logvar = model(data, y_onehot)
            _, bce, _ = cvae_loss_function(recon_data, data, mu, logvar)
            test_bce_loss += bce.item() * data.size(0)
            total_samples += data.size(0)

    test_bce = test_bce_loss / total_samples
    print(f"CVAE Test Reconstruction Loss (BCE per image): {test_bce:.2f}")
    return test_bce


def plot_and_save_confusion_matrix(all_targets, all_preds, save_path: str):
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("MNIST Test Set Confusion Matrix (CNN Classifier)", color="#f8fafc", fontsize=12, pad=12)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(range(10), color="#cbd5e1")
    ax.set_yticklabels(range(10), color="#cbd5e1")

    # Annotate counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, format(val, "d"),
                ha="center", va="center",
                color="white" if val > thresh else "#0f172a",
                fontsize=8, fontweight="bold"
            )

    ax.set_xlabel("Predicted Digit", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_ylabel("True Digit", color="#94a3b8", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # MNIST transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [0, 1] for CVAE binary cross entropy compatibility
    ])

    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 1. Train SmallCNN
    classifier = SmallCNN(num_classes=10).to(device)
    test_acc, all_preds, all_targets = train_classifier(classifier, train_loader, test_loader, device, epochs=4)
    torch.save(classifier.state_dict(), os.path.join(checkpoint_dir, "classifier.pt"))
    print(f"Saved classifier checkpoint to {checkpoint_dir}/classifier.pt")

    # 2. Train CVAE
    cvae = ConditionalVAE(latent_dim=16, num_classes=10).to(device)
    test_recon_loss = train_cvae(cvae, train_loader, test_loader, device, epochs=10)
    torch.save(cvae.state_dict(), os.path.join(checkpoint_dir, "cvae.pt"))
    print(f"Saved CVAE checkpoint to {checkpoint_dir}/cvae.pt")

    # 3. Save confusion matrix
    cm_path = os.path.join(checkpoint_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(all_targets, all_preds, cm_path)

    # 4. Save metrics.json
    metrics = {
        "classifier_test_accuracy": round(test_acc, 2),
        "cvae_test_recon_loss": round(test_recon_loss, 2),
        "target_accuracy_met": bool(test_acc >= 98.0),
        "epochs_classifier": 4,
        "epochs_cvae": 10,
        "latent_dim": 16,
        "num_classes": 10,
        "device": str(device)
    }
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}: {metrics}")
    print("\nTraining complete! You can now launch the app: streamlit run app.py")


if __name__ == "__main__":
    main()
