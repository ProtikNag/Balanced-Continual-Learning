import os
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np


def load_and_split(batch_size, tasks=20):
    # Load the COIL-RAG dataset
    dataset = TUDataset(root="./data/TUDataset", name='COIL-RAG', use_node_attr=True)

    # Get total number of classes and calculate classes per task
    num_classes = dataset.num_classes  # Total number of graph classes
    classes_per_task = num_classes // tasks

    tasks_train = []
    tasks_test = []
    task_classes = []  # To store class ranges for each task

    for task_id in range(tasks):
        # Define the range of classes for this task
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task
        task_classes.append((start_class, end_class - 1))  # Store class range

        # Select graphs belonging to the task's class range
        task_indices = [i for i, data in enumerate(dataset) if start_class <= data.y.item() < end_class]
        task_data = [dataset[i] for i in task_indices]

        # Shuffle indices
        np.random.shuffle(task_data)

        # Split into train and test sets
        num_train = int(0.8 * len(task_data))
        train_data = task_data[:num_train]
        test_data = task_data[num_train:]

        # Use PyG's DataLoader for batching
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        tasks_train.append(train_loader)
        tasks_test.append(test_loader)

    return tasks_train, tasks_test, task_classes


# tasks_train, tasks_test, task_classes = load_and_split(batch_size=8, tasks=10)
#
# # Print class ranges for each task
# for task_id, (start_class, end_class) in enumerate(task_classes):
#     print(f"Task {task_id + 1}: Classes {start_class} to {end_class}")
