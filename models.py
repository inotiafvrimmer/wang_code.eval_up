import torch
import torch.nn as nn


class BasicSNN(nn.Module):
    def __init__(self, dataset='mnist'):
        super().__init__()
        self.dataset = dataset.lower()
        self.threshold = 1.0
        self.decay = 0.25

        # 初始化各数据集结构
        if self.dataset == 'mnist':
            self.conv1 = nn.Conv2d(1, 12, 5)
            self.pool = nn.AvgPool2d(2)
            self.conv2 = nn.Conv2d(12, 32, 5)
            self.fc = nn.Linear(32 * 4 * 4, 10)

        elif self.dataset == 'cifar10':
            self._build_cifar_base(out_features=10)

        elif self.dataset == 'cifar100':
            self._build_cifar_base(out_features=100)
            self._enhance_for_cifar100()

        elif self.dataset == 'cifar10-dvs':
            self._build_dvs_structure()
            self._add_temporal_mech()
            
        elif self.dataset == 'dvs-gesture':
            self._build_gesture_structure()
            self._add_gesture_temporal()

        elif self.dataset == 'nmnist':
            self._build_nmnist_structure()

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # 注册膜电位缓存
        buffers = ['mem_conv1', 'mem_conv2', 'mem_conv3',
                   'mem_conv4', 'mem_conv5', 'mem_fc']
        for buf in buffers:
            self.register_buffer(buf, None)

    def _build_cifar_base(self, out_features):
        """CIFAR-10"""
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc = nn.Linear(256 * 4 * 4, out_features)

    def _enhance_for_cifar100(self):
        """CIFAR-100"""
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 100)
        )

    def _build_dvs_structure(self):
        """DVS"""
        self.conv1 = nn.Conv2d(2, 128, 5, stride=2)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(128, 256, 3, groups=4)
        self.conv3 = nn.Conv2d(256, 512, 3, dilation=2)
        self.conv4 = nn.Conv2d(512, 1024, 3)
        self.fc = nn.Linear(1024, 10)

    def _build_gesture_structure(self):
        """DVS-Gesture"""
        self.conv1 = nn.Conv3d(2, 64, (3, 5, 5), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=(1, 1, 1))
        self.t_pool = nn.MaxPool3d((2, 1, 1))  
        self.conv3 = nn.Conv3d(128, 256, (5, 3, 3), dilation=(2, 1, 1))
        self.lstm = nn.LSTM(256 * 6 * 6, 512, bidirectional=True)
        self.fc = nn.Linear(1024, 11)  

    def _build_nmnist_structure(self):
        """N-MNIST"""
        self.conv1 = nn.Conv2d(2, 32, 5)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self.fc = nn.Linear(128 * 3 * 3, 10)


    def forward(self, x):
        device = x.device
        batch_size = x.size(0)

        if self.mem_conv1 is None:
                self.mem_conv1 = torch.zeros(128, device=device)

            # 时空特征提取
            spatial_feat = self._process_dvs(x)

            # 膜电位累积
            mem = self.mem_conv1.repeat(batch_size, 1, 1, 1)
            spikes, mem = self._lif_activation(spatial_feat, mem)
            self.mem_conv1 = mem.detach().mean(dim=0)  

            x = self.pool(spikes)
            x = x.flatten(1)
            fc_out = self.fc(x)

            return [spikes], [mem]

    def evaluate_correlation(self, x, targets):
        """返回平均相关系数 (标量)"""
        with torch.no_grad():
            # 获取模型输出
            spikes, _ = self.forward(x)
            outputs = spikes[:, -1, :]  # (batch, num_classes)

            # 转换为numpy
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # 逐特征计算相关系数
            corr_sum = 0.0
            valid_count = 0
            for i in range(outputs_np.shape[1]):
                for j in range(targets_np.shape[1]):
                    # 计算相关系数
                    x = outputs_np[:, i]
                    y = targets_np[:, j]

                    # 去均值
                    x_centered = x - x.mean()
                    y_centered = y - y.mean()

                    # 协方差计算
                    cov = (x_centered * y_centered).mean()
                    std_x = x_centered.std()
                    std_y = y_centered.std()

                    # 防止除以零
                    if std_x > 1e-8 and std_y > 1e-8:
                        corr = cov / (std_x * std_y)
                        corr_sum += corr
                        valid_count += 1

            return corr_sum / valid_count if valid_count > 0 else 0.0
