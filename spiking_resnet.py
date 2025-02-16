import torch.nn as nn
import torch
from spikingjelly.clock_driven import layer
import torch.nn.functional as F

__all__ = [
    'PreActResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50'
]

thresh = 0.5  # neuronal threshold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        
class batch_norm_2d(nn.Module):
    """DTBN"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm2d1(num_features)

    def forward(self, input):
        y = self.bn(input)
        return y
        
class batch_norm_2d1(nn.Module):
    """DTBN-Zero init"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm2d2(num_features)

    def forward(self, input):
        y = self.bn(input)
        return y


class BatchNorm2d1(torch.nn.BatchNorm2d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm2d2(torch.nn.BatchNorm2d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0)
            nn.init.zeros_(self.bias)



class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, dropout, neuron: callable = None, **kwargs):
        super(PreActBlock, self).__init__()
        whether_bias = True
        self.bn1 = batch_norm_2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=whether_bias)
        self.bn2 = batch_norm_2d1(self.expansion * out_channels)

        self.dropout = layer.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1, bias=whether_bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, bias=whether_bias),
                batch_norm_2d(self.expansion * out_channels)
            )   
        else:
            self.shortcut = nn.Sequential()

        self.relu1 = neuron(**kwargs)
        self.relu2 = neuron(**kwargs)
        
        self.synapse_pre_neurons = [self.bn1]
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)
        out = self.relu2(self.bn2(out))
        out = out + self.shortcut(x)
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes, dropout, neuron: callable = None, **kwargs):
        super(PreActResNet, self).__init__()
        self.num_blocks = num_blocks

        self.data_channels = kwargs.get('c_in', 3)
        self.init_channels = 128
        self.conv1 = nn.Conv2d(self.data_channels, self.init_channels, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn1 = batch_norm_2d(self.init_channels)
        self.layer1 = self._make_layer(block, self.init_channels, num_blocks[0], 1, dropout, neuron, **kwargs)
        self.layer2 = self._make_layer(block, self.init_channels*2, num_blocks[1], 2, dropout, neuron, **kwargs)
        self.layer3 = self._make_layer(block, self.init_channels*4, num_blocks[2], 2, dropout, neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.init_channels * block.expansion, num_classes)

        self.relu1 = neuron(**kwargs)
        self.dropout = layer.Dropout(dropout)
        
        
        self.synapses = []
        self.synapse_next_neurons = []
        for n in [self.layer1]:
            for block in n:
                self.synapses.extend([self.conv1])
                self.synapse_next_neurons.extend(block.synapse_pre_neurons)
                

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout, neuron, **kwargs):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.init_channels, out_channels, stride, dropout, neuron, **kwargs))
            self.init_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

def spiking_resnet18(neuron: callable = None, num_classes=10,  neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet34(neuron: callable = None, num_classes=10,  neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)


def spiking_resnet50(neuron: callable = None, num_classes=10, neuron_dropout=0, **kwargs):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes, neuron_dropout, neuron=neuron, **kwargs)