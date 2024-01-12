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
    return x * torch.sigmoid(x)


def Normalize(in_channels, use_bn=False):
    if use_bn:
        return torch.nn.BatchNorm2d(num_features=in_channels, affine=False)
    else:
        return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=False)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 emb_channels=None,
                 use_bn=True,
                 attention=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.attention = attention

        self.norm1 = Normalize(in_channels, use_bn=use_bn)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        if emb_channels is not None:
            self.emb_proj = torch.nn.Linear(
                emb_channels, out_channels)
            self.norm2 = Normalize(out_channels, use_bn=use_bn)
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
        if attention:
            self.attn = AttnBlock(out_channels)

    def forward(self, x, emb=None):
        residual = self.shortcut(x)
        x = self.norm1(x)
        if self.norm1.affine:
            x = swish(x)
        x = self.conv1(x)
        if self.emb_channels is not None:
            assert emb is not None
            emb = self.emb_proj(emb)
            x = x + emb[:, :, None, None]
            x = self.norm2(x)
            if self.norm2.affine:
                x = swish(x)
        else:
            x = swish(x)
        x = self.conv2(x)
        x = x + residual
        if self.attention:
            x = self.attn(x)
            x = x + residual
        return x


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 block_depth,
                 use_bn=True,
                 emb_channels=None,
                 attention=False):
        super().__init__()
        block_in = in_channels
        block_out = out_channels
        self.down = nn.ModuleList()
        for _ in range(block_depth):
            self.down.append(ResidualBlock(block_in,
                                           block_out,
                                           use_bn=use_bn,
                                           emb_channels=emb_channels,
                                           attention=attention))
            block_in = block_out
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, emb=None):
        x, skips = x
        for block in self.down:
            x = block(x, emb)
            skips.append(x)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 block_depth,
                 size,
                 use_bn=True,
                 emb_channels=None,
                 attention=False):
        super().__init__()
        block_in = in_channels
        block_out = out_channels
        self.upsample = nn.UpsamplingBilinear2d(size=size)
        self.up = nn.ModuleList()
        for _ in range(block_depth):
            self.up.append(ResidualBlock(block_in,
                                         block_out,
                                         use_bn=use_bn,
                                         emb_channels=emb_channels,
                                         attention=attention))
            block_in = block_out * 2  # ch_mult in DDIM

    def forward(self, x, emb=None):
        x, skips = x
        x = self.upsample(x)
        for block in self.up:
            x = torch.concatenate([x, skips.pop()], dim=1)
            x = block(x, emb)
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
        input_emb_once = config.model.input_emb_once
        attentions = config.model.attentions
        use_bn = config.model.use_bn

        self.input_emb_once = input_emb_once

        self.conv_in = nn.Conv2d(in_channels,
                                 out_channels[0],
                                 kernel_size=1)
        # positional embedding
        if input_emb_once:
            emb_channels = None
            self.emb_up = nn.UpsamplingNearest2d(image_size)
            # concat of image embedding and positional embedding
            ch_in = out_channels[0] + self.config.model.embedding_dims
        else:
            emb_channels = self.config.model.embedding_dims
            self.emb_proj1 = nn.Linear(emb_channels, emb_channels)
            self.emb_proj2 = nn.Linear(emb_channels, emb_channels)
            ch_in = out_channels[0]
        # downsampling
        self.down = nn.ModuleList()
        up_sizes = []
        for ch_out, attention in zip(out_channels[:-1], attentions[:-1]):
            up_sizes.append(image_size)
            self.down.append(DownBlock(
                ch_in, ch_out, block_depth,
                use_bn=use_bn,
                emb_channels=emb_channels,
                attention=attention))
            ch_in = ch_out
            image_size = image_size >> 1

        # middle
        self.middle = nn.ModuleList()
        for _ in range(block_depth):
            self.middle.append(ResidualBlock(
                ch_in, out_channels[-1],
                use_bn=use_bn,
                emb_channels=emb_channels,
                attention=attentions[-1]))
            ch_in = out_channels[-1]

        # Upsampling
        up_sizes = up_sizes[::-1]
        self.up = nn.ModuleList()
        ch_in = out_channels[-1]
        for i, (ch_out, attention) in enumerate(zip(out_channels[-2::-1], attentions[-2::-1])):
            self.up.append(UpBlock(
                ch_in + ch_out, ch_out, block_depth, up_sizes[i],
                use_bn=use_bn,
                emb_channels=emb_channels,
                attention=attention))
            ch_in = ch_out

        self.conv_out = zero_module(nn.Conv2d(out_channels[0], in_channels, kernel_size=1))

    def forward(self, noise_images, emb):
        # image_embedding
        x = self.conv_in(noise_images)
        # embed the noise variance or time step
        e = sinusoidal_embedding(
            emb, self.config.model.embedding_dims)
        if self.input_emb_once:
            e = self.emb_up(e.view(-1, self.config.model.embedding_dims, 1, 1))
            x = torch.concat([x, e], dim=1)
        else:
            e = self.emb_proj1(e)
            e = swish(e)
            e = self.emb_proj2(e)
            e = swish(e)

        skips = []
        for block in self.down:
            x = block([x, skips], e)

        for block in self.middle:
            x = block(x, e)

        for block in self.up:
            x = block([x, skips], e)

        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    import yaml
    import sys
    import os
    import argparse
    from torch.utils.tensorboard import SummaryWriter

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
    with open("/root/workspace/nano_ddim/configs/keras_oxflower.yml", "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    model = Model(new_config)
    writer = SummaryWriter('/root/workspace/nano_ddim/exp/baseline_keras/tensorboard/oxflower/graph')
    writer.add_graph(model, (torch.ones((2, 3, 64, 64)), torch.ones(2)), verbose=False)
    writer.close()
    inputs = torch.ones((2, 3, 64, 64))
    diffusion_times = torch.ones(2)
    outputs = model(inputs, diffusion_times)
    print(outputs)
    assert outputs.size() == (2, 3, 64, 64)