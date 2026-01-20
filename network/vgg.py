import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VggNet(nn.Module):
    def __init__(self, name="vgg16", pretrain=True, input_channels=3):
        super().__init__()
        if name == "vgg16":
            base_net = models.vgg16(pretrained=False)
        elif name == "vgg16_bn":
            base_net = models.vgg16_bn(pretrained=False)
        else:
            print(" base model is not support !")
        if pretrain:
            print("load the {} weight from ./pretrain".format(name))
            base_net.load_state_dict(model_zoo.load_url(model_urls[name], model_dir="./pretrain"))

        # Modify first layer to accept 4 channels (RGB + Saliency)
        if input_channels == 4:
            original_conv = base_net.features[0]
            new_conv = nn.Conv2d(4, original_conv.out_channels, 
                                 kernel_size=original_conv.kernel_size, 
                                 stride=original_conv.stride, 
                                 padding=original_conv.padding)
            
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight
                # Initialize 4th channel with mean of RGB weights
                new_conv.weight[:, 3:, :, :] = torch.mean(original_conv.weight, dim=1, keepdim=True)
                new_conv.bias = original_conv.bias
                
            base_net.features[0] = new_conv

        if name == "vgg16":
            self.stage1 = nn.Sequential(*[base_net.features[layer] for layer in range(0, 5)])
            self.stage2 = nn.Sequential(*[base_net.features[layer] for layer in range(5, 10)])
            self.stage3 = nn.Sequential(*[base_net.features[layer] for layer in range(10, 17)])
            self.stage4 = nn.Sequential(*[base_net.features[layer] for layer in range(17, 24)])
            self.stage5 = nn.Sequential(*[base_net.features[layer] for layer in range(24, 31)])
        elif name == "vgg16_bn":
            self.stage1 = nn.Sequential(*[base_net.features[layer] for layer in range(0, 7)])
            self.stage2 = nn.Sequential(*[base_net.features[layer] for layer in range(7, 14)])
            self.stage3 = nn.Sequential(*[base_net.features[layer] for layer in range(14, 24)])
            self.stage4 = nn.Sequential(*[base_net.features[layer] for layer in range(24, 34)])
            self.stage5 = nn.Sequential(*[base_net.features[layer] for layer in range(34, 44)])

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)

        return C1, C2, C3, C4, C5


if __name__ == "__main__":
    import torch

    input = torch.randn((4, 3, 512, 512))
    net = VggNet()
    C1, C2, C3, C4, C5 = net(input)
    print(C1.size())
    print(C2.size())
    print(C3.size())
    print(C4.size())
    print(C5.size())
