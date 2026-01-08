import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicVAE(nn.Module):
    """Basic VAE for Easy task - MFCC features"""
    def __init__(self, input_dim=13, latent_dim=32, hidden_dim=128):
        super(MusicVAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ConvVAE(nn.Module):
    """Convolutional VAE for Medium task - Spectrograms"""
    def __init__(self, latent_dim=32):
        super(ConvVAE, self).__init__()
        # Encoder: 128x128 -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Flatten()  # 64 * 16 * 16 = 16384
        )
        self.fc_mu = nn.Linear(16384, latent_dim)
        self.fc_logvar = nn.Linear(16384, latent_dim)
        
        # Decoder: latent -> 128x128
        self.decoder_input = nn.Linear(latent_dim, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # 128x128
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class HybridConvVAE(nn.Module):
    """Hybrid VAE combining audio spectrograms and text embeddings"""
    def __init__(self, latent_dim=32, text_dim=384):
        super(HybridConvVAE, self).__init__()
        # Audio encoder (same as ConvVAE)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        # Combined encoder
        combined_dim = 16384 + 64
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, audio, text):
        a_feat = self.audio_encoder(audio)
        t_feat = self.text_encoder(text)
        combined = torch.cat([a_feat, t_feat], dim=1)
        return self.fc_mu(combined), self.fc_logvar(combined)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, audio, text):
        mu, logvar = self.encode(audio, text)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ConditionalMusicVAE(nn.Module):
    """Conditional VAE for Hard task - audio + text + labels"""
    def __init__(self, latent_dim=32, text_dim=384, num_classes=6):
        super(ConditionalMusicVAE, self).__init__()
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        # Combined encoder (audio + text + labels)
        combined_dim = 16384 + 64 + num_classes
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)
        
        # Decoder (latent + labels)
        self.decoder_input = nn.Linear(latent_dim + num_classes, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, audio, text, labels_onehot):
        a_feat = self.audio_encoder(audio)
        t_feat = self.text_encoder(text)
        combined = torch.cat([a_feat, t_feat, labels_onehot], dim=1)
        return self.fc_mu(combined), self.fc_logvar(combined)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels_onehot):
        combined = torch.cat([z, labels_onehot], dim=1)
        h = self.decoder_input(combined)
        return self.decoder(h)
    
    def forward(self, audio, text, labels_onehot):
        mu, logvar = self.encode(audio, text, labels_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels_onehot)
        return recon, mu, logvar


class HardMusicCVAE(nn.Module):
    """Enhanced Conditional VAE for Hard task"""
    def __init__(self, latent_dim=32, text_dim=384, num_classes=6):
        super(HardMusicCVAE, self).__init__()
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Out: 16 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Out: 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Out: 64 x 16 x 16
            nn.ReLU(),
            nn.Flatten()  # Total: 64 * 16 * 16 = 16384
        )
        self.combined_in = 16384 + text_dim + num_classes
        self.fc_mu = nn.Linear(self.combined_in, latent_dim)
        self.fc_logvar = nn.Linear(self.combined_in, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + num_classes, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # 128 x 128
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, audio, text, labels_onehot):
        a_feat = self.audio_encoder(audio)  # [batch, 16384]
        combined_enc = torch.cat([a_feat, text, labels_onehot], dim=1)  # [batch, 16774]
        mu, logvar = self.fc_mu(combined_enc), self.fc_logvar(combined_enc)
        z = self.reparameterize(mu, logvar)
        combined_dec = torch.cat([z, labels_onehot], dim=1)  # [batch, 38]
        recon = self.decoder(self.decoder_input(combined_dec))
        return recon, mu, logvar


class SimpleAutoencoder(nn.Module):
    """Simple Autoencoder baseline for Hard task"""
    def __init__(self, latent_dim=32, text_dim=384):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        combined_dim = 16384 + 64
        self.fc_latent = nn.Linear(combined_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, audio, text):
        a_feat = self.audio_encoder(audio)
        t_feat = self.text_encoder(text)
        combined = torch.cat([a_feat, t_feat], dim=1)
        return self.fc_latent(combined)
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, audio, text):
        z = self.encode(audio, text)
        recon = self.decode(z)
        return recon, z

