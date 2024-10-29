import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the model - A simple fully connected network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define functions for dataset permutation
def permute_mnist(seed):
    torch.manual_seed(seed)
    idx = torch.randperm(28 * 28)
    return lambda x: x.view(-1)[idx].view(1, 28, 28)

# Load and prepare permuted MNIST datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into 10 tasks with different permutations
tasks = []
task_permutations = [permute_mnist(seed) for seed in range(10)]
for perm in task_permutations:
    permuted_data = datasets.MNIST(root='./data', train=True, download=True,
                                   transform=transforms.Compose([transform, perm]))
    tasks.append(permuted_data)

# DataLoader for training
batch_size = 64
task_loaders = [DataLoader(task, batch_size=batch_size, shuffle=True) for task in tasks]

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training using BCL Algorithm
memory = []  # Memory to store past task samples
memory_size = 200  # Max memory size per task

# Hyperparameters
epochs_per_task = 1
updates_per_player = 20

for task_idx, task_loader in enumerate(task_loaders):
    print(f"Training on Task {task_idx + 1}")

    # Prepare the current task data
    current_task_data = list(task_loader)

    # Player 1 and Player 2 iterations for each batch
    for epoch in range(epochs_per_task):
        for batch_idx, (data, target) in enumerate(current_task_data):
            # Combine current batch with memory samples
            if memory:
                mem_data, mem_target = zip(*memory)
                mem_data = torch.stack(mem_data)
                mem_target = torch.tensor(mem_target)
                combined_data = torch.cat((data, mem_data))
                combined_target = torch.cat((target, mem_target))
            else:
                combined_data, combined_target = data, target

            # Make the data a trainable parameter for Player 1
            delta_x = combined_data.clone().detach().requires_grad_(True)

            # Step 1: Player 1 - Update Input Perturbations to Maximize Cost
            player1_optimizer = optim.SGD([delta_x], lr=0.001)

            for _ in range(updates_per_player):
                model.train()
                player1_optimizer.zero_grad()

                output = model(delta_x)
                loss = criterion(output, combined_target)

                # Player 1 maximizes the generalization cost (gradient ascent)
                (-loss).backward()  # Convert to gradient ascent
                player1_optimizer.step()

            # Step 2: Player 2 - Minimize Forgetting with Parameter Update
            for _ in range(updates_per_player):
                model.train()
                optimizer.zero_grad()

                output = model(combined_data)  # Use the perturbed input from Player 1
                loss = criterion(output, combined_target)

                # Player 2 minimizes forgetting cost (standard gradient descent)
                loss.backward()
                optimizer.step()

            # Evaluate the change in cost for logging or analysis (if needed)
            # Cost evaluations as per the notations provided in your images
            # J_initial = criterion(model(combined_data), combined_target).item()
            # J_after_update = criterion(model(combined_data), combined_target).item()
            # print(f"Change in Cost (Player 2 Update): {J_after_update - J_initial}")

            # Update memory with current batch (FIFO if memory exceeds)
            memory += list(zip(data, target))
            if len(memory) > memory_size:
                memory = memory[-memory_size:]

    print(f"Finished Training Task {task_idx + 1}")

# Evaluation on all tasks
model.eval()
correct, total = 0, 0
for task_idx, task_loader in enumerate(task_loaders):
    for data, target in task_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(f"Accuracy on Task {task_idx + 1}: {100 * correct / total:.2f}%")
