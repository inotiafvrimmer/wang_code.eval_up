import torch.nn as nn
from spiking_neuron import EnhancedLIFNeuron


class SpikingLinear(nn.Module):
    def __init__(self, in_features, out_features, neuron_params):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.neuron = EnhancedLIFNeuron(**neuron_params)

    def forward(self, x):
        """输入输出"""
        x = self.linear(x)
        spikes, mem = self.neuron(x)
        return spikes, mem


class SpikingConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, neuron_params):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.neuron = EnhancedLIFNeuron(**neuron_params)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """输入输出"""
        batch, time, c, h, w = x.shape

        # 时空处理
        x = x.view(batch * time, c, h, w)
        x = self.conv(x)
        x = self.pool(x)
        _, c_new, h_new, w_new = x.shape

        # 时空维度
        x = x.view(batch, time, c_new, h_new, w_new)

        x = x.flatten(3)

        # 脉冲神经元处理
        spikes, mem = self.neuron(x)
        return spikes.view(batch, time, c_new, h_new, w_new), mem


class SpikingResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels, neuron_params):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.neuron1 = EnhancedLIFNeuron(**neuron_params)
        self.neuron2 = EnhancedLIFNeuron(**neuron_params)

    def forward(self, x):
        identity = x
        x, mem1 = self.neuron1(self.conv1(x))
        x, mem2 = self.neuron2(self.conv2(x))
        return x + identity, (mem1 + mem2) / 2