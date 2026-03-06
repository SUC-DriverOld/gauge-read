import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtTiny(nn.Module):
    def __init__(self, pretrain=True, input_channels=3):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrain else None
        base_net = convnext_tiny(weights=weights)

        # Keep multimodal compatibility by adapting stem to 4-channel input.
        if input_channels == 4:
            stem = base_net.features[0][0]
            new_stem = nn.Conv2d(
                4,
                stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                bias=(stem.bias is not None),
            )
            with torch.no_grad():
                new_stem.weight[:, :3, :, :] = stem.weight
                new_stem.weight[:, 3:, :, :] = torch.mean(stem.weight, dim=1, keepdim=True)
                if stem.bias is not None:
                    new_stem.bias.copy_(stem.bias)
            base_net.features[0][0] = new_stem

        # ConvNeXt feature hierarchy:
        # features[0]: stem (1/4)
        # features[1]: stage1 blocks (1/4)
        # features[2]: downsample to 1/8
        # features[3]: stage2 blocks (1/8)
        # features[4]: downsample to 1/16
        # features[5]: stage3 blocks (1/16)
        # features[6]: downsample to 1/32
        # features[7]: stage4 blocks (1/32)
        self.stem = base_net.features[0]
        self.stage1 = base_net.features[1]
        self.down2 = base_net.features[2]
        self.stage2 = base_net.features[3]
        self.down3 = base_net.features[4]
        self.stage3 = base_net.features[5]
        self.down4 = base_net.features[6]
        self.stage4 = base_net.features[7]

        # FPN in this project expects C1 at 1/2 resolution.
        # ConvNeXt has no native 1/2 stage, so upsample C2 (1/4) to build C1.
        self.up_c1 = nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        c2 = x

        x = self.down2(x)
        x = self.stage2(x)
        c3 = x

        x = self.down3(x)
        x = self.stage3(x)
        c4 = x

        x = self.down4(x)
        x = self.stage4(x)
        c5 = x

        c1 = self.up_c1(c2)
        return c1, c2, c3, c4, c5
