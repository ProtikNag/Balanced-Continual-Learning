import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def remap_labels(targets, task_id):
    """Remaps the labels to match the task-specific output range."""
    start_label = task_id * 2
    end_label = start_label + 1
    remapped_targets = torch.clone(targets)
    remapped_targets[targets == start_label] = 0
    remapped_targets[targets == end_label] = 1
    remapped_targets = remapped_targets[(targets == start_label) | (targets == end_label)]

    return remapped_targets


class BCLModel:
    def __init__(self, model, lr=0.01, epsilon=0.01, k_range=15, x_updates=15, theta_updates=15):
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon  # Perturbation strength for Player 1
        self.k_range = k_range  # Number of interactions between players
        self.x_updates = x_updates  # Updates for Player 1
        self.theta_updates = theta_updates  # Updates for Player 2
        self.task_memory = {}  # Stores data from previous tasks
        self.task_mem_cache = {}  # Cache for quick access
        self.task_count = 0

    # def perturb_inputs(self, inputs, targets):
    #     inputs.requires_grad = True
    #     remapped_targets = remap_labels(targets, self.task_count - 1)
    #
    #     if remapped_targets.size(0) == 0:
    #         return inputs
    #
    #     outputs = self.model(inputs, self.task_count - 1)  # Forward pass
    #     loss = self.criterion(outputs, remapped_targets)
    #
    #     # Gradient ascent to maximize loss (Player 1 strategy)
    #     loss.backward()
    #     perturbed_inputs = inputs + self.epsilon * inputs.grad.sign()  # Gradient ascent step
    #     print(perturbed_inputs, inputs)
    #     inputs.requires_grad = False
    #     return perturbed_inputs

    def update_model(self, inputs, targets, current_task_id):
        # Remap targets for the current task
        remapped_targets = remap_labels(targets, current_task_id - 1)
        valid_indices = (remapped_targets == 0) | (remapped_targets == 1)
        inputs = inputs[valid_indices]
        remapped_targets = remapped_targets[valid_indices]

        loss = None  # Initialize loss to avoid UnboundLocalError

        for kappa in range(self.k_range):
            if self.task_count > 0:
                # Player 1 Strategy: Perturb inputs to simulate discrepancies
                perturbed_inputs = inputs.clone().detach().requires_grad_(True)

                generalization_loss = 0
                forgetting_loss = 0
                for _ in range(self.x_updates):
                    out = self.model(perturbed_inputs, current_task_id - 1)
                    generalization_loss = self.criterion(out, remapped_targets)

                    # Apply gradient ascent for perturbation to maximize generalization cost
                    self.optimizer.zero_grad()
                    generalization_loss.backward(retain_graph=True)
                    perturbed_inputs = perturbed_inputs + self.epsilon * perturbed_inputs.grad.sign()  # Gradient ascent
                    perturbed_inputs = perturbed_inputs.detach().requires_grad_(True)  # Reset for next iteration

                # Player 2 Strategy: Respond to maximized loss by minimizing forgetting
                for _ in range(self.theta_updates):
                    out = self.model(inputs, current_task_id - 1)
                    forgetting_loss = self.criterion(out, remapped_targets)

                    self.optimizer.zero_grad()
                    forgetting_loss.backward()
                    self.optimizer.step()

                if loss is None:
                    loss = generalization_loss + forgetting_loss
                else:
                    loss += generalization_loss + forgetting_loss
            else:
                # Standard training for the first task (no previous task to minimize forgetting)
                out = self.model(inputs, current_task_id - 1)
                loss = self.criterion(out, remapped_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.detach(), out

    def evaluate(self, tasks_test):
        self.model.eval()
        results = []

        with torch.no_grad():

            for task_id, test_loader in enumerate(tasks_test):
                correct = 0
                total = 0

                for data, target in test_loader:
                    data, target = data.to('cpu'), target.to('cpu')
                    outputs = self.model(data, task_id)
                    remapped_target = remap_labels(target, task_id)
                    _, predicted = torch.max(outputs.data, 1)
                    total += remapped_target.size(0)
                    correct += (predicted == remapped_target).sum().item()

                accuracy = 100 * correct / total
                results.append(accuracy)
                print(f'Accuracy on Task {task_id + 1}: {accuracy:.2f}%')

        return results

    def train_task(self, task_loader):
        self.model.train()

        for data, target in task_loader:
            data, target = data.to('cpu'), target.to('cpu')
            loss, _ = self.update_model(data, target, self.task_count + 1)

        self.task_count += 1
        self.task_memory[self.task_count] = task_loader
        x_all, y_all, t_all = [], [], []

        for batch_data, batch_target in task_loader:
            x_all.append(batch_data)
            y_all.append(batch_target)
            t_all.extend([self.task_count] * len(batch_data))

        self.task_mem_cache[self.task_count] = {
            'data': torch.cat(x_all, dim=0),
            'target': torch.cat(y_all, dim=0),
            'task': t_all
        }