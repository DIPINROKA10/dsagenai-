"""
models.py - Neural Network Architectures

Contains:
1. SmallCNN: Discriminative model modeling p(y|x) via a 2-layer convolutional network.
2. ConditionalVAE: Generative model modeling p(x|y, z) and p(z) via a conditional variational autoencoder.
3. Loss functions and helper utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Discriminative CNN Classifier for MNIST digit classification.
    Models p(y|x) - maps input image x to categorical label distribution y.
    
    Architecture:
    - Conv1: 1 -> 32 channels, 3x3 kernel, padding=1 -> ReLU -> MaxPool(2x2) [14x14]
    - Conv2: 32 -> 64 channels, 3x3 kernel, padding=1 -> ReLU -> MaxPool(2x2) [7x7]
    - Dropout(0.25)
    - FC1: 64 * 7 * 7 (3136) -> 128 -> ReLU
    - Dropout(0.5)
    - FC2: 128 -> 10 (logits)
    """

    def __init__(self, num_classes: int = 10):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # -> (batch_size, 32, 14, 14)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # -> (batch_size, 64, 7, 7)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, 3136)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized class probabilities p(y|x) via Softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for MNIST digits.
    Models p(x|y) = \int p(x|y, z) p(z) dz
    
    Conditions both encoder and decoder on digit label y (one-hot encoded vector of size 10).
    Allows targeted generation: draw z ~ N(0, I * tau^2), condition on y, decode to x_hat.
    
    Encoder:
    - [x_flat (784) + y_onehot (10)] -> Linear(794, 512) -> ReLU -> Linear(512, 256) -> ReLU
    - mu: Linear(256, latent_dim)
    - logvar: Linear(256, latent_dim)
    
    Decoder:
    - [z (latent_dim) + y_onehot (10)] -> Linear(latent_dim + 10, 256) -> ReLU -> Linear(256, 512) -> ReLU -> Linear(512, 784) -> Sigmoid
    """

    def __init__(self, latent_dim: int = 16, num_classes: int = 10):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder network
        self.enc_fc1 = nn.Linear(784 + num_classes, 512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder network
        self.dec_fc1 = nn.Linear(latent_dim + num_classes, 256)
        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_fc3 = nn.Linear(512, 784)

    def encode(self, x: torch.Tensor, y_onehot: torch.Tensor):
        """
        x: (batch_size, 784) or (batch_size, 1, 28, 28)
        y_onehot: (batch_size, num_classes)
        """
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        inputs = torch.cat([x, y_onehot], dim=1)
        h = F.relu(self.enc_fc1(inputs))
        h = F.relu(self.enc_fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps * temperature
        temperature: variability scalar (tau > 0). tau=1.0 is standard sampling.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std * temperature

    def decode(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        """
        z: (batch_size, latent_dim)
        y_onehot: (batch_size, num_classes)
        Returns: (batch_size, 1, 28, 28) with values in [0, 1]
        """
        inputs = torch.cat([z, y_onehot], dim=1)
        h = F.relu(self.dec_fc1(inputs))
        h = F.relu(self.dec_fc2(h))
        out_flat = torch.sigmoid(self.dec_fc3(h))
        return out_flat.view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor, temperature: float = 1.0):
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterize(mu, logvar, temperature=temperature)
        recon_x = self.decode(z, y_onehot)
        return recon_x, mu, logvar

    def sample(self, y: int, num_samples: int = 1, temperature: float = 1.0, device: str = "cpu") -> torch.Tensor:
        """
        Sample from prior p(z) ~ N(0, temperature^2 * I) and decode conditioned on label y.
        Returns: (num_samples, 1, 28, 28)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device) * temperature
        y_onehot = torch.zeros(num_samples, self.num_classes, device=device)
        y_onehot[:, y] = 1.0
        with torch.no_grad():
            samples = self.decode(z, y_onehot)
        return samples


def cvae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    """
    Negative Evidence Lower Bound (ELBO) Loss:
    Loss = Reconstruction_Loss (BCE) + beta * KL_Divergence
    """
    if x.dim() == 4:
        x_flat = x.view(x.size(0), -1)
    else:
        x_flat = x

    if recon_x.dim() == 4:
        recon_flat = recon_x.view(recon_x.size(0), -1)
    else:
        recon_flat = recon_x

    # Binary cross entropy reconstruction loss
    bce = F.binary_cross_entropy(recon_flat, x_flat, reduction="sum") / x.size(0)

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    total_loss = bce + beta * kld
    return total_loss, bce, kld
