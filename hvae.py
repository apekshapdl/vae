# Hierarchical VAE in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Reparameterization Trick
# ----------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# ----------------------
# Encoder Networks
# ----------------------
class EncoderZ2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class EncoderZ1(nn.Module):
    def __init__(self, input_dim, latent_dim, z2_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + z2_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x, z2):
        h = torch.cat([x, z2], dim=-1)
        h = F.relu(self.fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# ----------------------
# Decoder Network
# ----------------------
class Decoder(nn.Module):
    def __init__(self, z1_dim, z2_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(z1_dim + z2_dim, 256)
        self.fc_out = nn.Linear(256, output_dim)

    def forward(self, z1, z2):
        h = torch.cat([z1, z2], dim=-1)
        h = F.relu(self.fc1(h))
        return self.fc_out(h)  # Remove sigmoid for real-valued outputs

# ----------------------
# Hierarchical VAE Model
# ----------------------
class HierarchicalVAE(nn.Module):
    def __init__(self, input_dim, z1_dim, z2_dim):
        super().__init__()
        self.encoder_z2 = EncoderZ2(input_dim, z2_dim)
        self.encoder_z1 = EncoderZ1(input_dim, z1_dim, z2_dim)
        self.decoder = Decoder(z1_dim, z2_dim, input_dim)

    def forward(self, x):
        mu_z2, logvar_z2 = self.encoder_z2(x)
        z2 = reparameterize(mu_z2, logvar_z2)

        mu_z1, logvar_z1 = self.encoder_z1(x, z2)
        z1 = reparameterize(mu_z1, logvar_z1)

        x_hat = self.decoder(z1, z2)
        return x_hat, mu_z1, logvar_z1, mu_z2, logvar_z2

# ----------------------
# Loss Function (Hierarchical ELBO)
# ----------------------
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def loss_function(x, x_hat, mu_z1, logvar_z1, mu_z2, logvar_z2):
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kl_z1 = kl_divergence(mu_z1, logvar_z1)
    kl_z2 = kl_divergence(mu_z2, logvar_z2)
    return recon_loss + kl_z1 + kl_z2, recon_loss, kl_z1, kl_z2
