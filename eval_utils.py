import torch
import time
from torch.utils.data import DataLoader
from data_preprocessor import NeuromorphicPreprocessor
from models import BasicSNN


def evaluate_all_datasets(model_params=None, batch_size=128, device=None):
    """综合评估"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        'mnist': 10,
        'cifar10': 10,
        'cifar100': 100,
        'cifar10-dvs': 10,
        'dvs-gesture': 11,
        'nmnist': 10
    }

    results = {}

    for name, num_classes in datasets.items():
        # 数据加载
        loader = _get_test_loader(name, batch_size)

        # 初始化
        model = BasicSNN(dataset=name).to(device)
        if model_params and name in model_params:
            model.load_state_dict(torch.load(model_params[name]))

        # 评估流程
        start_time = time.time()
        acc, total_spikes = _evaluate_model(model, loader, device)
        eval_time = time.time() - start_time

        results[name] = {
            'accuracy': acc,
            'time_sec': round(eval_time, 2),
            'time_steps': model.time_steps,
            'spikes': total_spikes
        }

        # 格式化输出
        print(f"[{name.upper():<12}] Acc: {acc:.2%} | Time: {eval_time:.1f}s | Steps: {model.time_steps} | Spikes: {total_spikes:,}")

    return results


def _get_test_loader(dataset_name, batch_size):
    """测试集加"""
    return NeuromorphicPreprocessor(
        dataset_name=dataset_name,
        encoding='rate' if 'dvs' in dataset_name else 'latency'
    ).get_loader(batch_size=batch_size, split='test')


def _evaluate_model(model, loader, device):
    """评估"""
    model.eval()
    correct = total = 0
    total_spikes = 0

    with torch.no_grad():
        for inputs, labels in loader:
            # 维度处理 (静态数据添加时间步)
            inputs = inputs.to(device)
            if len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(1).repeat(1, model.time_steps, 1, 1, 1)

            # 推理计算
            spikes, _ = model(inputs)
            total_spikes += spikes.sum().item()
            preds = spikes.mean(dim=1).argmax(dim=1)

            # 统计结果
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)

    return (correct / total if total > 0 else 0.0),total_spikes
