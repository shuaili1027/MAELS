import torch
import torch.nn as nn

"""
Architectures based on InfoGAN.
Reference on https://arxiv.org/abs/1606.03657.
"""

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(BaseNetwork):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, 2 * dim_out, kernel_size=3,
                        stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * dim_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * dim_out, dim_out, kernel_size=3,
                        stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out))

        self.init_weights()

    def forward(self, x):
        return x + self.main(x)


class Encoder(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.encoder = nn.Sequential(
            # 64 * 64
            nn.Conv2d(self.params.img_channels, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 32 * 32
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.Dropout(0.5, inplace=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 16 * 16
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.Dropout(0.5, inplace=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # 8 * 8
            nn.Conv2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.Dropout(0.5, inplace=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 4 * 4
        )
        size = self.params.img_size // 16
        self.noise_fc1 = nn.Sequential(
            nn.Linear(128 * size ** 2, 128),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, self.params.noise_dim)
        )

        self.noise_fc2 = nn.Sequential(
            nn.Linear(128 * size ** 2, 128),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, self.params.noise_dim)
        )

        self.init_weights()

    def noise_reparameterize(self, mean, log_var):
        eps = torch.randn(mean.shape).to(self.params.device)
        z = mean + eps * torch.exp(log_var)
        return z

    def forward(self, x):
        out = self.encoder(x)
        mean = self.noise_fc1(out.view(out.shape[0], -1))
        var = self.noise_fc2(out.view(out.shape[0], -1))
        z = self.noise_reparameterize(mean, var)
        return z, mean, var


# i.e. Generator
class Decoder(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.decoder = nn.Sequential(
            # 1 * 1
            nn.ConvTranspose2d(self.params.noise_dim + self.params.label_dim + self.params.varying_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, True),
            # 4 * 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, True),
            # 8 * 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, True),
            # 16 * 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # 32 * 32
            nn.ConvTranspose2d(64, self.params.img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 64 * 64
        )

        self.init_weights()

    def forward(self, z, code):
        x = torch.cat((z, code), dim=-1)
        # x = self.fc(x)
        x = x.view(x.shape[0], -1, 1, 1)
        output = self.decoder(x)
        return output


class D_BACKBONE(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.main = nn.Sequential(
            # 64 * 64
            nn.Conv2d(self.params.img_channels, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 32 * 32
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 16 * 16
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # 8 * 8
            nn.Conv2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 4 * 4
        )

        self.init_weights()

    def forward(self, x):
        out = self.main(x)
        return out


class D_HEAD(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.params = params
        size = self.params.img_size // 16
        self.main = nn.Sequential(nn.Linear(128 * size ** 2, 8),
                                  nn.Dropout(0.5),
                                  nn.BatchNorm1d(8),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Linear(8, 1),
                                  nn.Sigmoid())
        self.init_weights()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.main(x)
        return out

class CODE_HEAD(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.params = params
        size = self.params.img_size // 16
        self.main1 = nn.Sequential(nn.Linear(128 * size ** 2, 16),
                                   nn.Dropout(0.5),
                                   nn.BatchNorm1d(16),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Linear(16, self.params.label_dim),
                                   nn.Softmax())
        self.main2 = nn.Sequential(nn.Linear(128 * size ** 2, 8),
                                   nn.Dropout(0.5),
                                   nn.BatchNorm1d(8),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Linear(8, self.params.varying_dim),
                                   nn.Tanh())
        self.init_weights()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        label = self.main1(x)
        varying = self.main2(x)
        code = torch.cat((label, varying), dim=-1)
        return code