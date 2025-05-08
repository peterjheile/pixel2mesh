#######
#
#   This serves as a wrapper to load a pretrained CNN. It removes the classifier layhers which are uneeded
#   and returns the Feature extractor model abstracted away from the user. I allow
#   for two different feature models to be used; however, I am going to use Resnet for this application.
#
#######

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights

class FeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', weights='default'):
        super().__init__()

        if backbone == 'resnet18':
            self.backbone = "resnet18"
            resnet_weights = ResNet18_Weights.DEFAULT if weights == 'default' else None
            model = resnet18(weights=resnet_weights)

            self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4


        #I inteded to use this later on for comparing backbones; however
        #I never finsihed the full implemntation for it so It shouldnt work
        elif backbone == 'vgg16':
            self.backbone = "vgg16"
            vgg_weights = VGG16_Weights.DEFAULT if weights == 'default' else None
            model = vgg16(weights=vgg_weights)

            self.block1 = model.features[:5] #[B, 64, 112, 112]
            self.block2 = model.features[5:10]   # [B, 128, 56, 56]
            self.block3 = model.features[10:17] # [B, 256, 28, 28]
            self.block4 = model.features[17:24]  # B, 512, 14, 14]
            self.block5 = model.features[24:] # [B, 512, 7, 7]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        feats = {}

        if self.backbone == 'resnet18':
            x = self.layer0(x)
            x = self.layer1(x)
            feats['layer2'] = self.layer2(x)                # [B, 128, 28, 28]
            feats['layer3'] = self.layer3(feats['layer2'])  # [B, 256, 14, 14]
            feats['layer4'] = self.layer4(feats['layer3'])  # [B, 512, 7, 7]

        elif self.backbone == 'vgg16':
            x = self.block1(x)  # [B, 64, 112, 112]
            x = self.block2(x)  # [B, 128, 56, 56]
            feats['block3'] = self.block3(x)  # [B, 256, 28, 28]
            feats['block4'] = self.block4(feats['block3'])  # [B, 512, 14, 14]
            feats['block5'] = self.block5(feats['block4'])  # [B, 512, 7, 7]

        return feats