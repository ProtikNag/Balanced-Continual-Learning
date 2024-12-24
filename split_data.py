import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class BatchWrapper:
    def __init__(self, x, y):
        self.x = torch.stack(x).unsqueeze(-1)  # Stack tensors and add feature dimension
        self.y = torch.tensor(y).unsqueeze(-1)  # Combine labels into a single tensor
        self.edge_index = None  # No edges in the sine wave dataset
        self.batch = torch.arange(len(x))  # Dummy batch attribute

    def __iter__(self):
        return iter((self.x, self.y))  # Return only x and y as a tuple


class SineWaveDataset(Dataset):
    def __init__(self, num_samples=1000, frequency=0.1, amplitude=1.0, noise_level=0.15):
        self.num_samples = num_samples
        self.frequency = frequency
        self.amplitude = amplitude
        self.noise_level = noise_level
        self.x, self.y = self.generate_data()

    def generate_data(self):
        x = torch.linspace(0, 1, self.num_samples)
        y = self.amplitude * torch.sin(2 * torch.pi * self.frequency * x)
        noise = self.noise_level * torch.randn_like(y)
        y += noise
        return x, y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_task_datasets(num_tasks=10, num_samples=1000):
    tasks = []
    for task_id in range(num_tasks):
        # Generate random frequency and amplitude for each task
        frequency = np.random.uniform(0.5, 5.0)  # Frequency range [0.5, 5.0]
        amplitude = np.random.uniform(0.5, 2.0)  # Amplitude range [0.5, 2.0]
        dataset = SineWaveDataset(num_samples=num_samples, frequency=frequency, amplitude=amplitude)
        tasks.append(dataset)
    return tasks

def split_and_load_tasks(tasks, batch_size=32):
    train_loaders = []
    test_loaders = []

    for task_id, dataset in enumerate(tasks):
        # Split dataset into train and test sets (80/20 split)
        num_train = int(0.8 * len(dataset))
        train_indices = list(range(num_train))
        test_indices = list(range(num_train, len(dataset)))

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            collate_fn=lambda batch: BatchWrapper(*zip(*batch))
        )
        test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: BatchWrapper(*zip(*batch))
        )

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders