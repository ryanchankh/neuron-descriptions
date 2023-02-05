"""Extensions to the models module."""
import collections
import ssl
from typing import Any

import torch
from torch import nn
from torchvision import models

# Workaround for an annoying bug in torchvision...
ssl._create_default_https_context = ssl._create_unverified_context


def __getattr__(name: str) -> Any:
    """Forward to `torchvision.models`."""
    return getattr(models, name)


def alexnet_seq(**kwargs: Any) -> nn.Module:
    """Return sequentialized AlexNet model from torchvision."""
    class AlexNetSeq(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            model = models.alexnet(**kwargs)
            layers = list(
                zip([
                    'conv1',
                    'relu1',
                    'pool1',
                    'conv2',
                    'relu2',
                    'pool2',
                    'conv3',
                    'relu3',
                    'conv4',
                    'relu4',
                    'conv5',
                    'relu5',
                    'pool5',
                ], model.features))
            layers += [('avgpool', model.avgpool), ('flatten', nn.Flatten())]
            layers += zip([
                'dropout6',
                'fc6',
                'relu6',
                'dropout7',
                'fc7',
                'relu7',
                'linear8',
            ], model.classifier)
            self.net = nn.Sequential(collections.OrderedDict(layers))
            
        def multiply_mask(self, layer, mask):
            """Hooks for modifying layer output during forward pass."""
            layer_mask_idx = {
                'conv1': torch.arange(0, 64),
                'conv2': torch.arange(64, 256),
                'conv3': torch.arange(256, 640),
                'conv4': torch.arange(640, 896),
                'conv5': torch.arange(876, 1132)
            }
            layer_mask = torch.index_select(mask, 1, layer_mask_idx[layer])
            layer_mask = layer_mask[:, :, None, None]
            def hook(model, input, output):
                return output * layer_mask
            return hook
        
        def register_hooks_for_masking(self, mask):
            self.hooks = []
            for layer_idx, layer_name in zip([0, 3, 6, 8, 10], ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']):
                layer_hook = self.net[layer_idx].register_forward_hook(self.multiply_mask(layer_name, mask))
                self.hooks.append(layer_hook)
        
        def remove_hooks(self):
            for hook in self.hooks:
                hook.remove()

        def forward(self, x):
            return self.net(x)

    return AlexNetSeq()


def alexnet_seq_old(**kwargs: Any) -> nn.Sequential:
    """Return sequentialized AlexNet model from torchvision."""
    model = models.alexnet(**kwargs)
    layers = list(
        zip([
            'conv1',
            'relu1',
            'pool1',
            'conv2',
            'relu2',
            'pool2',
            'conv3',
            'relu3',
            'conv4',
            'relu4',
            'conv5',
            'relu5',
            'pool5',
        ], model.features))
    layers += [('avgpool', model.avgpool), ('flatten', nn.Flatten())]
    layers += zip([
        'dropout6',
        'fc6',
        'relu6',
        'dropout7',
        'fc7',
        'relu7',
        'linear8',
    ], model.classifier)
    return nn.Sequential(collections.OrderedDict(layers))


def resnet18_seq(**kwargs: Any) -> nn.Sequential:
    """Return sequentialized ResNet-18 model from torchvision."""
    model = models.resnet18(**kwargs)
    return nn.Sequential(
        collections.OrderedDict([
            ('conv1', model.conv1),
            ('bn1', model.bn1),
            ('relu', model.relu),
            ('maxpool', model.maxpool),
            ('layer1', model.layer1),
            ('layer2', model.layer2),
            ('layer3', model.layer3),
            ('layer4', model.layer4),
            ('avgpool', model.avgpool),
            ('flatten', nn.Flatten()),
            ('fc', model.fc),
        ]))


def resnet152_seq(**kwargs: Any) -> nn.Sequential:
    """Return sequentialized ResNet-152 model from torchvision."""
    model = models.resnet152(**kwargs)
    return nn.Sequential(
        collections.OrderedDict([
            ('conv1', model.conv1),
            ('bn1', model.bn1),
            ('relu', model.relu),
            ('maxpool', model.maxpool),
            ('layer1', model.layer1),
            ('layer2', model.layer2),
            ('layer3', model.layer3),
            ('layer4', model.layer4),
            ('avgpool', model.avgpool),
            ('flatten', nn.Flatten()),
            ('fc', model.fc),
        ]))
