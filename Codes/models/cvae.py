import torch
from torch import nn
from torch.nn import functional as F

#phoneme conditional variational autoencoder
class VAE(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims,
                 img_size,
                 num_classes=42):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.embed_class = nn.Embedding(num_classes, img_size*img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        modules = []

        self.decoder_input = nn.Linear(latent_dim + 1, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        in_channels += 1
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z, sample=False):
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        if sample:
            for idx, l in enumerate(self.decoder):
                result = l(result)
                if idx == 2:
                    return result
        else:
            result = self.decoder(result)
        
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        y = kwargs['labels'].long()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)
        x = torch.cat([embedded_input, embedded_class], dim = 1)
        x = embedded_input
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y.unsqueeze(1)], dim = 1)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return [loss, recons_loss, -kld_loss]

    def sample(self,
               num_samples,
               current_device,
               y):
        y = y.float()
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        z = torch.cat([z, y.unsqueeze(1)], dim=1)
        samples = self.decode(z, sample=True)
        return samples

    def generate(self, x, **kwargs):

        return self.forward(x, **kwargs)[0]
