import math
import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    embedding_min_frequency = 1.0
    embedding_max_frequency = 1000.0
    frequencies = torch.exp(
        torch.linspace(
            math.log(embedding_min_frequency),
            math.log(embedding_max_frequency),
            embedding_dim // 2,
            dtype=torch.float32,
            device=timesteps.device
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    emb = timesteps.float()[:, None] * angular_speeds[None, :]
    embeddings = torch.concat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return embeddings


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def swish(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, use_bn=False):
    if use_bn:
        return torch.nn.BatchNorm2d(num_features=in_channels, affine=False)
    else:
        return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-3, affine=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        # not learnable
        self.bn = Normalize(in_channels, use_bn=True)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=1)
        else:
            self.shortcut = torch.nn.Identity()
        

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = swish(x)
        x = self.conv2(x)
        x += residual
        return x
    

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth):
        super().__init__()
        block_in = in_channels
        block_out = out_channels
        self.down = nn.ModuleList()
        for _ in range(block_depth):
            self.down.append(ResidualBlock(block_in,
                                           block_out))
            block_in = block_out
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x, skips = x
        for block in self.down:
            x = block(x)
            skips.append(x)
        x = self.downsample(x)
        return x
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth, size):
        super().__init__()
        block_in = in_channels
        block_out = out_channels
        self.upsample = nn.UpsamplingBilinear2d(size=size)
        self.up = nn.ModuleList()
        for _ in range(block_depth):
            self.up.append(ResidualBlock(block_in,
                                         block_out))
            block_in = block_out * 2  # ch_mult in DDIM
    
    def forward(self, x):
        x, skips = x
        x = self.upsample(x)
        for block in self.up:
            x = torch.concat([x, skips.pop()], dim=1)
            x = block(x)
        return x

"""
(None, 64, 64, 64)
(None, 32, 32, 32)
(None, 64, 16, 16)
(None, 96, 8, 8)
(None, 128, 8, 8)
(None, 128, 8, 8)
(None, 96, 16, 16)
(None, 64, 32, 32)
(None, 32, 64, 64)
(None, 3, 64, 64)
up1.r1: (128 + 96, 96), + 96 from down3.r2.add
up1.r2: (96 + 96, 96), + 96 from down3.r1.add
up2.r1: (96 + 64, 64), + 64 from down2.r2.add
up2.r2: (64 + 64, 64), + 64 from down2.r1.add
up3.r1: (64 + 32, 32), + 32 from down1.r2.add
up3.r2: (32 + 32, 96), + 32 from down1.r1.add
"""
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        image_size = self.config.data.image_size
        in_channels = self.config.model.in_channels
        out_channels = self.config.model.out_channels
        block_depth = self.config.model.num_res_blocks
        self.emb_up = nn.UpsamplingNearest2d(image_size)
        
        # downsampling
        self.conv_in = nn.Conv2d(in_channels,
                                 out_channels[0],
                                 kernel_size=1)
        self.down = nn.ModuleList()
        ch_in = out_channels[0] + self.config.model.embedding_dims
        up_sizes = []
        for ch_out in out_channels[:-1]:
            up_sizes.append(image_size)
            self.down.append(DownBlock(ch_in, ch_out, block_depth))
            ch_in = ch_out
            image_size = image_size >> 1

        # middle
        self.middle = nn.ModuleList()
        for _ in range(block_depth):
            self.middle.append(ResidualBlock(ch_in, out_channels[-1]))
            ch_in = out_channels[-1]
        
        # Upsampling
        up_sizes = up_sizes[::-1]
        self.up = nn.ModuleList()
        ch_in = out_channels[-1]
        for i, ch_out in enumerate(reversed(out_channels[:-1])):
            self.up.append(UpBlock(ch_in + ch_out, ch_out, block_depth, up_sizes[i]))
            ch_in = ch_out

        self.conv_out = nn.Conv2d(out_channels[0], in_channels, kernel_size=1)
        if config.data.normalize:
            self.conv_out = zero_module(self.conv_out)

    def forward(self, noise_images, noise_variances):
        # embed the noise variance instead of time step
        e = sinusoidal_embedding(
            noise_variances, self.config.model.embedding_dims)
        e = self.emb_up(e.view(-1, self.config.model.embedding_dims, 1, 1))

        x = self.conv_in(noise_images)
        x = torch.concat([x, e], dim=1)
        skips = []
        for block in self.down:
            x = block([x, skips])
        
        for block in self.middle:
            x = block(x)

        for block in self.up:
            x = block([x, skips])

        x = self.conv_out(x)
        return x
    
if __name__ == "__main__":
    import yaml
    import sys
    import os
    import argparse
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    # parse config file
    with open("/workspace/gaosf/nano_ddim/configs/keras_oxflower.yml", "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.data.normalize = False
    model = Model(new_config)
    inputs = torch.ones((2, 3, 64, 64))
    diffusion_times = torch.ones(2)
    outputs = model(inputs, diffusion_times)
    print(outputs)
    assert outputs.size() == (2, 3, 64, 64)