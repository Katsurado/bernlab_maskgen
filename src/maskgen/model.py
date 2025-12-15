import torch

class GlobalResponseNorm(torch.nn.Module):
    def __init__(self, channels, beta=0.0, gamma=0.0):
        super().__init__()
        self.channels = channels
        # IDK why these params won't show up in summary
        self.beta = torch.nn.Parameter(torch.zeros(channels))
        self.gamma = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        b = self.beta.view(1, self.channels, 1, 1)
        g = self.gamma.view(1, self.channels, 1, 1)
        gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
        return g * (x * nx) + b + x
    

class Stochastic_Depth(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.keep_prob = 1-p

    def forward(self, x):
        if self.training and self.keep_prob > 0:
            mask_shape = (x.shape[0], 1, 1, 1)
            mask = x.new_empty(mask_shape).bernoulli_(self.keep_prob)
            mask = mask / self.keep_prob
            x = x * mask
        return x

    def __repr__(self):
        return f"Stochastic_Depth(p = {self.p})"
    
class ConvNextV2_Block(torch.nn.Module):
    def __init__(self, channels, depth, config):
        super().__init__()
        self.channels = channels
        self.depth = depth

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 7, padding =3, groups = channels, bias = False),
            torch.nn.GroupNorm(1, channels),
            torch.nn.Conv2d(channels, 4*channels, 1),
            torch.nn.GELU(),
            GlobalResponseNorm(4*channels),
            torch.nn.Conv2d(4*channels, channels, 1)
        )

        p = self.depth/sum(config['layers']) * config['stochastic_depth']


        self.stochastic_depth = Stochastic_Depth(p)

    def forward(self, x):
        residue = self.block(x)
        residue = self.stochastic_depth(residue)
        return x + residue


class DownSample_Block(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.GroupNorm(1, channels),
            torch.nn.Conv2d(channels, 2*channels, 2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class UpSample_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.GroupNorm(1, in_channels),
            torch.nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x)
    

class Encoder(torch.nn.Module):
    def __init__(self, channels, config) -> None:
        super().__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, channels, 4, stride=4, bias=False),
            torch.nn.GroupNorm(1, channels)
        )

        stage1 = [
            ConvNextV2_Block(channels, i+1, config) 
            for i in range(config['layers'][0])
        ]

        stage2 = [
            ConvNextV2_Block(channels*2, i+1+sum(config['layers'][:1]), config) 
            for i in range(config['layers'][1])
        ]

        stage3 = [
            ConvNextV2_Block(channels*4, i+1+sum(config['layers'][:2]), config) 
            for i in range(config['layers'][2])
        ]

        stage4 = [
            ConvNextV2_Block(channels*8, i+1+sum(config['layers'][:3]), config) 
            for i in range(config['layers'][3])
        ]

        self.stage1 = torch.nn.Sequential(*stage1)
        self.down1 = DownSample_Block(channels)
        self.stage2 = torch.nn.Sequential(*stage2)
        self.down2 = DownSample_Block(2*channels)
        self.stage3 = torch.nn.Sequential(*stage3)
        self.down3 = DownSample_Block(4*channels)
        self.stage4 = torch.nn.Sequential(*stage4)

    def forward(self, x):
        x = self.stem(x)

        stage_1_out = self.stage1(x)
        x = self.down1(stage_1_out)
        stage_2_out = self.stage2(x)
        x = self.down2(stage_2_out)
        stage_3_out = self.stage3(x)
        x = self.down3(stage_3_out)
        stage_4_out = self.stage4(x)

        return [stage_1_out, stage_2_out, stage_3_out, stage_4_out]

        
class Decoder(torch.nn.Module):
    def __init__(self, channels, config) -> None:
        super().__init__()

        C = channels

        # C5 (8C) -> up to 4C, then + C4(4C) -> 8C -> refine at 8C
        self.up1   = UpSample_Block(8*C, 4*C)
        self.ref1  = ConvNextV2_Block(8*C, 0, config)

        # -> up to 2C, + C3(2C) -> 4C -> refine at 4C
        self.up2   = UpSample_Block(8*C, 2*C)   # input will be 8C from ref1
        self.ref2  = ConvNextV2_Block(4*C, 0, config)

        # -> up to C, + C2(C) -> 2C -> refine (1–2 blocks) at 2C
        self.up3   = UpSample_Block(4*C, 1*C)   # input will be 4C from ref2
        self.ref3a = ConvNextV2_Block(2*C, 0, config)
        self.ref3b = ConvNextV2_Block(2*C, 0, config)

        self.predict = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvNextV2_Block(2*C, 0, config),
            torch.nn.Conv2d(2*C, 1, kernel_size=1),
        )

    def forward(self, encoder_outs):
        # first stage cat with last stage, and vice versa
        x = encoder_outs[-1]

        x = self.up1(x)                               # 8C -> 4C (H/32 -> H/16)
        x = torch.cat([x, encoder_outs[2]], dim=1)             # + C4(4C) => 8C
        x = self.ref1(x)

        x = self.up2(x)                               # 8C -> 2C (H/16 -> H/8)
        x = torch.cat([x, encoder_outs[1]], dim=1)             # + C3(2C) => 4C
        x = self.ref2(x)

        x = self.up3(x)                               # 4C -> C (H/8 -> H/4)
        x = torch.cat([x, encoder_outs[0]], dim=1)             # + C2(C) => 2C
        x = self.ref3a(x)
        x = self.ref3b(x)

        return self.predict(x)


class Net(torch.nn.Module):
    def __init__(self, channels, config) -> None:
        super().__init__()
        self.encoder = Encoder(channels, config)
        self.decoder = Decoder(channels, config)

    def forward(self, x):
        encoder_outs = self.encoder(x)
        mask = self.decoder(encoder_outs)
        return mask
    
