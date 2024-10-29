# bcl/data/split_mnist.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_split_mnist_task_indices(dataset, start_class, end_class):
    indices = [i for i, (_, label) in enumerate(dataset) if start_class <= label <= end_class]
    return indices


def load_split_mnist_data(batch_size=64, root='./data'):
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=get_transform())
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=get_transform())

    tasks_train = []
    tasks_test = []
    for i in range(0, 10, 2):
        indices_train = get_split_mnist_task_indices(train_dataset, i, i + 1)
        task_loader_train = DataLoader(Subset(train_dataset, indices_train), batch_size=batch_size, shuffle=True)
        tasks_train.append(task_loader_train)

        indices_test = get_split_mnist_task_indices(test_dataset, i, i + 1)
        task_loader_test = DataLoader(Subset(test_dataset, indices_test), batch_size=batch_size, shuffle=False)
        tasks_test.append(task_loader_test)

    return tasks_train, tasks_test
