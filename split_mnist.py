# # bcl/data/split_mnist.py
import os
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_split_mnist_task_indices(dataset, start_class, end_class):
    indices = [i for i, (_, label) in enumerate(dataset) if start_class <= label <= end_class]
    return indices


def cache_indices(file_path, indices):
    with open(file_path, 'wb') as f:
        pickle.dump(indices, f)


def load_cached_indices(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_split_mnist_data(batch_size=64, root='./data', cache_dir='./cache'):
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=get_transform())
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=get_transform())

    tasks_train = []
    tasks_test = []
    for i in range(0, 10, 2):
        # Cache paths
        cache_train_path = os.path.join(cache_dir, f"train_indices_task_{i}_{i + 1}.pkl")
        cache_test_path = os.path.join(cache_dir, f"test_indices_task_{i}_{i + 1}.pkl")

        # Check and load cached indices if available
        if os.path.exists(cache_train_path):
            indices_train = load_cached_indices(cache_train_path)
        else:
            indices_train = get_split_mnist_task_indices(train_dataset, i, i + 1)
            cache_indices(cache_train_path, indices_train)

        task_loader_train = DataLoader(Subset(train_dataset, indices_train), batch_size=batch_size, shuffle=True)
        tasks_train.append(task_loader_train)

        if os.path.exists(cache_test_path):
            indices_test = load_cached_indices(cache_test_path)
        else:
            indices_test = get_split_mnist_task_indices(test_dataset, i, i + 1)
            cache_indices(cache_test_path, indices_test)

        task_loader_test = DataLoader(Subset(test_dataset, indices_test), batch_size=batch_size, shuffle=False)
        tasks_test.append(task_loader_test)

    return tasks_train, tasks_test
