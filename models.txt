import torch
import torch.nn as nn


class BasicSNN(nn.Module):
    def __init__(self, dataset='mnist'):
        super().__init__()


    def forward(self, x):
        return spikes, mem_potentials

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