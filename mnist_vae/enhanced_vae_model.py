import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EnhancedVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(EnhancedVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            self._conv_bn_relu(1, 32, 4, 2, 1),  # 28x28 -> 14x14
            self._conv_bn_relu(32, 64, 4, 2, 1),  # 14x14 -> 7x7
            self._conv_bn_relu(64, 128, 3, 2, 1),  # 7x7 -> 4x4
            self._conv_bn_relu(128, 256, 4, 1, 0)  # 4x4 -> 1x1
        )
        self.fc1 = nn.Linear(256, 512)
        self.fc21 = nn.Linear(512, latent_dim)  # Mean of latent variable
        self.fc22 = nn.Linear(512, latent_dim)  # Log variance of latent variable
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, 256)
        self.decoder = nn.Sequential(
            self._conv_transpose_bn_relu(256, 128, 4, 1, 0),  # 1x1 -> 4x4
            self._conv_transpose_bn_relu(128, 64, 3, 2, 1),  # 4x4 -> 7x7
            self._conv_transpose_bn_relu(64, 32, 4, 2, 1),  # 7x7 -> 14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )
        
        self._init_weights()

    def get_latent_dim(self):
        return self.latent_dim

    def _conv_bn_relu(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _conv_transpose_bn_relu(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.fc1(h), 0.2)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.leaky_relu(self.fc3(z), 0.2)
        h = F.leaky_relu(self.fc4(h), 0.2)
        h = h.view(-1, 256, 1, 1)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, kld_weight=1.0):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kld_weight * KLD

def init_model(latent_dim=10, device='cuda'):
    model = EnhancedVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer = init_model(latent_dim=10, device=device)
    print(model)
