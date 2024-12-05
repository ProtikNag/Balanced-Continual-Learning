import os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.utils.data import DataLoader
import torch


def load_cora_data(root="./data", dataset_name="Cora"):
    """
    Load the Cora dataset using PyTorch Geometric.

    Args:
        root (str): Root directory where the dataset will be stored.
        dataset_name (str): Name of the dataset to load (e.g., "Cora").

    Returns:
        torch_geometric.data.Data: Cora dataset object containing node features, edge index, and labels.
    """
    dataset = Planetoid(root=root, name=dataset_name, transform=NormalizeFeatures())
    return dataset[0]  # Single graph data


def split_cora_tasks(data, tasks=3):
    num_classes = data.y.max().item() + 1  # Total number of classes in the dataset
    classes_per_task = num_classes // tasks

    tasks_train = []
    tasks_test = []

    for task_id in range(tasks):
        # Define label range for this task
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task

        # Get indices of nodes belonging to the task's label range
        task_indices = (data.y >= start_class) & (data.y < end_class)
        node_indices = torch.nonzero(task_indices, as_tuple=False).squeeze()

        # Split into train and test indices
        num_train = int(0.8 * len(node_indices))
        train_indices = node_indices[:num_train]
        test_indices = node_indices[num_train:]

        # Create loaders for this task
        train_loader = DataLoader(
            [(data.x[train_indices], data.edge_index, data.y[train_indices])],
            batch_size=len(train_indices),  # Use all nodes in one batch for simplicity
            shuffle=True
        )
        test_loader = DataLoader(
            [(data.x[test_indices], data.edge_index, data.y[test_indices])],
            batch_size=len(test_indices),
            shuffle=False
        )

        tasks_train.append(train_loader)
        tasks_test.append(test_loader)

    return tasks_train, tasks_test