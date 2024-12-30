import os
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np


# Load and split for Sine Wave Dataset
def load_and_split(batch_size, tasks=20):
    tasks_train = []
    tasks_test = []

    for task_id in range(tasks):
        # Generate sine wave data with different frequencies
        freq = 0.1 + 0.1 * task_id
        x = torch.linspace(0, 2 * np.pi, 10000).view(-1, 1)
        y = torch.sin(freq * x)

        # Split into train and test sets
        num_train = int(0.8 * len(x))
        train_data = torch.utils.data.TensorDataset(x[:num_train], y[:num_train])
        test_data = torch.utils.data.TensorDataset(x[num_train:], y[num_train:])

        # Use PyTorch's DataLoader for batching
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        tasks_train.append(train_loader)
        tasks_test.append(test_loader)

    return tasks_train, tasks_test


# tasks_train, tasks_test = load_and_split(batch_size=64, tasks=10)
#
# # See the structure of the DataLoader
# for i, task_loader in enumerate(tasks_train):
#     print(f"Task {i + 1} Train Loader: {task_loader.dataset.tensors[0].shape}")
#
# for i, task_loader in enumerate(tasks_test):
#     print(f"Task {i + 1} Test Loader: {task_loader.dataset.tensors[0].shape}")
#
#
# # Visualize all tasks on a single plot
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 8))
# plt.title("Sine Wave Tasks")
# plt.xlabel("X")
# plt.ylabel("Y")
#
# for i, task_loader in enumerate(tasks_train):
#     x, y = task_loader.dataset.tensors
#     plt.plot(x, y, label=f"Task {i + 1}")
#
# plt.legend()
# plt.show()

