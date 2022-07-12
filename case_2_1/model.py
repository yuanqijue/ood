import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np
from scipy.stats import norm


class VAE(nn.Module):
    """
    reference https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    """

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)  # output size is out_channels x 2 x 2
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1,
                                   output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=4, padding=0), nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
       Encodes the input by passing through the encoder network
       and returns the latent codes.
       :param input: (Tensor) Input tensor to encoder [N x C x H x W]
       :return: (Tensor) List of latent codes
       """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Any:
        """
         Maps the given latent codes
         onto the image space.
         :param z: (Tensor) [B x D]
         :return: (Tensor) [B x C x H x W]
         """
        result = self.decoder_input(z)
        result = result.view(-1, 265, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        labels = kwargs['labels']

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        bsz = labels.shape[0]
        # features = recons.view(bsz * 2, -1) # using result to calculate
        f1, f2 = torch.split(z, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # batch_size,2,projector

        con_loss = self.contrastive_loss(features, labels, 0.07, 0.07)

        loss = recons_loss + kld_weight * kld_loss + con_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach(),
                'Contrastive_loss': con_loss}  # loss = recons_loss + kld_weight * kld_loss  # return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def contrastive_loss(self, features, labels, temperature, base_temperature):
        batch_size = labels.shape[0]

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        labels = labels.contiguous().view(-1, 1)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)

        # similarity
        features_norm = torch.norm(contrast_feature, dim=1, keepdim=True)
        cov_features_norm = torch.matmul(features_norm, features_norm.T)
        logits = torch.div(anchor_dot_contrast, cov_features_norm)

        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()  # todo why

        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        loss = - (temperature / base_temperature) * (mask * log_prob).sum(1) / mask.sum(1)
        return loss.mean()

    def log_px_z(self, x_hats, logscale, x):
        # todo
        scale = torch.exp(logscale)
        log_px_z_array = []
        for x_hat in x_hats:
            mean = x_hat
            dist = torch.distributions.Normal(mean, scale)
            # measure prob of seeing image under p(x|z)
            log_px_z = dist.log_prob(x)
            log_px_z_array.append(log_px_z.sum(dim=(1, 2, 3)))
        sample_size = len(log_px_z_array)
        log_px_z_array = torch.cat(log_px_z_array, dim=0)
        log_px_z_array = log_px_z_array.view(sample_size, -1)
        log_px_z_array = torch.swapaxes(log_px_z_array, 0, 1)
        return log_px_z_array.numpy()

    def z_x_samples(self, x, num_samples):
        [mu, log_var] = self.encode(x)
        z_samples = []
        log_qz = []

        for m, s in zip(mu, log_var):
            z_vals = [np.random.normal(m[i], np.exp(0.5 * s[i]), num_samples) for i in range(len(m))]
            log_qz_vals = [norm.logpdf(z_vals[i], loc=m[i], scale=np.exp(0.5 * s[i])) for i in range(len(z_vals))]
            z_samples.append(z_vals)
            log_qz.append(log_qz_vals)
        z_samples = np.array(z_samples)
        log_pz = norm.logpdf(z_samples)
        log_pz = log_pz.sum(axis=2)
        log_qz_x = np.array(log_qz)
        log_qz_x = log_qz_x.sum(axis=2)
        return z_samples, log_pz, log_qz_x

    def marginal_likelihood(self, x, num_samples):
        log_scale = nn.Parameter(torch.Tensor([0.0]))
        z_samples, log_pz, log_qz_x = self.z_x_samples(x, num_samples)

        z_samples = torch.tensor(z_samples, dtype=torch.float)
        z_samples = z_samples.view(-1, self.latent_dim)
        x_hat = self.decode(z_samples)
        x_hat = x_hat.view(x.shape[0], num_samples, x.shape[1], x.shape[2], x.shape[3])
        x_hat = torch.swapaxes(x_hat, 0, 1)
        log_pxz = self.log_px_z(x_hat, log_scale, x)

        log_px = log_pxz + log_pz - log_qz_x
        return np.mean(log_px, axis=1)

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
