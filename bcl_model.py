import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def remap_labels(targets):
    unique_digits = targets.unique()
    remapped_targets = targets.clone()
    remapped_targets[targets == unique_digits[0]] = 0
    remapped_targets[targets == unique_digits[1]] = 1

    return remapped_targets[(targets == unique_digits[0]) | (targets == unique_digits[1])]



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
        self.task_count = 0

    def update_model(self, inputs, targets, current_task_id):
        remapped_targets = remap_labels(targets)
        loss = 0.0

        for kappa in range(self.k_range):
            if self.task_count > 0:
                perturbed_inputs = inputs.clone().detach().requires_grad_(True)

                for _ in range(self.x_updates):
                    out = self.model(perturbed_inputs, current_task_id - 1)
                    gen_loss = self.criterion(out, remapped_targets)

                    # Perturb inputs
                    self.optimizer.zero_grad()
                    gen_loss.backward(retain_graph=True)
                    perturbed_inputs = (perturbed_inputs + self.epsilon * perturbed_inputs.grad.sign()).detach().requires_grad_(True)

                # Update model to minimize forgetting
                for _ in range(self.theta_updates):
                    out = self.model(inputs, current_task_id - 1)
                    forget_loss = self.criterion(out, remapped_targets)

                    self.optimizer.zero_grad()
                    forget_loss.backward()
                    self.optimizer.step()

                loss += gen_loss + forget_loss
            else:
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
                    outputs = self.model(data, task_id)
                    remapped_target = remap_labels(target)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == remapped_target).sum().item()
                accuracy = 100 * correct / total
                results.append(accuracy)
                print(f'Accuracy on Task {task_id + 1}: {accuracy:.2f}%')
        return results

    def train_task(self, task_loader):
        self.model.train()

        for data, target in task_loader:
            loss, _ = self.update_model(data, target, self.task_count + 1)

        self.task_count += 1
        self.task_memory[self.task_count] = task_loader
        x_all, y_all = zip(*[(batch_data, batch_target) for batch_data, batch_target in task_loader])

        # Storing cached data for the current task
        self.task_memory[self.task_count] = {
            'data': torch.cat(x_all, dim=0),
            'target': torch.cat(y_all, dim=0),
            'task': [self.task_count] * len(x_all[0])
        }
