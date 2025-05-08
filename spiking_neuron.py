import torch
import torch.nn as nn
import numpy as np


class EnhancedLIFNeuron(nn.Module):
    def __init__(self,
                 init_threshold=1.0,
                 threshold_alpha=0.9,
                 threshold_beta=0.1,
                 decay=0.9,
                 grad_surrogate='multi_atan',
                 temporal_window=5,
                 spatial_gamma=0.5):
        """
        动态阈值和时空编码机制
        """
        super().__init__()

        # 动态阈值参数
        self.init_threshold = nn.Parameter(torch.tensor(init_threshold))
        self.alpha = threshold_alpha
        self.beta = threshold_beta

        # 时空编码参数
        self.decay = decay
        self.temporal_conv = nn.Conv1d(1, 1, temporal_window, padding='same')
        self.spatial_gamma = spatial_gamma

        # 梯度替代
        self.grad_surrogate = self._get_surrogate(grad_surrogate)

        # 参数
        self.temporal_kernel = nn.Parameter(torch.ones(temporal_window) / temporal_window)
        self.spatial_attention = nn.Parameter(torch.tensor([1.0]))

        # 状态
        self.register_buffer('mem_potential', None)
        self.register_buffer('current_threshold', None)
        self.register_buffer('spike_history', None)

    def _get_surrogate(self, name):
        surrogates = {
            'atan': lambda x: (x > 0).float() + (0.5 / (1 + (np.pi * x) ** 2)) * (x <= 0).float(),
            'sigmoid': lambda x: torch.sigmoid(5 * x),
            'multi_atan': lambda x: 0.5 * (torch.atan(np.pi * x / 2) / np.pi + 0.5)
        }
        return surrogates[name]

    def _dynamic_threshold(self, spike):
        """动态阈值"""
        # 阈值更新
        if self.spike_history is None:
            self.spike_history = spike
        else:
            self.spike_history = torch.cat([self.spike_history[:, 1:], spike.unsqueeze(1)], dim=1)

        # 计算阈值增量
        avg_spike = torch.mean(self.spike_history.float(), dim=1, keepdim=True)
        threshold_adapt = self.beta * avg_spike

        # 时间衰减阈值
        new_threshold = self.alpha * self.current_threshold + (1 - self.alpha) * self.init_threshold
        return new_threshold + threshold_adapt

    def _temporal_encoding(self, x):
        """时间维度特征编码"""
        # 时间维度卷积 (B, T, C) -> (B, C, T)
        x_t = x.transpose(1, 2)
        convolved = self.temporal_conv(x_t.unsqueeze(1)).squeeze(1)
        return convolved.transpose(1, 2)

    def _spatial_attention(self, x):
        """空间特征增强"""
        spatial_weights = torch.sigmoid(self.spatial_attention * x.mean(dim=1, keepdim=True))
        return x * (1 + self.spatial_gamma * spatial_weights)

    def forward(self, x, init_states=None):
        """增强的前向传播过程"""
        batch_size, time_steps, features = x.shape

        # 初始化状态
        if self.mem_potential is None or batch_size != self.mem_potential.shape[0]:
            device = x.device
            self.mem_potential = torch.zeros(batch_size, features).to(device)
            self.current_threshold = self.init_threshold.expand(batch_size, features).to(device)
            self.spike_history = torch.zeros(batch_size, 3, features).to(device)

        # 时空特征编码
        x = self._temporal_encoding(x)
        x = self._spatial_attention(x)

        spikes = []
        for t in range(time_steps):
            # 膜电位更新
            self.mem_potential = self.decay * self.mem_potential + x[:, t, :]

            # 脉冲生成
            spike = self.grad_surrogate(self.mem_potential - self.current_threshold)

            # 动态阈值调整
            self.current_threshold = self._dynamic_threshold(spike)

            # 重置
            self.mem_potential = self.mem_potential - spike * self.current_threshold.detach()

            spikes.append(spike.unsqueeze(1))

        return torch.cat(spikes, dim=1), self.mem_potential