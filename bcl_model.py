import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.autograd.set_detect_anomaly(True)


def remap_labels(targets):
    unique_digits = targets.unique()
    remapped_targets = targets.clone()
    remapped_targets[targets == unique_digits[0]] = 0
    remapped_targets[targets == unique_digits[1]] = 1

    return remapped_targets[(targets == unique_digits[0]) | (targets == unique_digits[1])]


class BCLModel:
    def __init__(self, model, lr=0.001, epsilon=0.001, k_range=100, x_updates=50, theta_updates=50):
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon  # Perturbation strength for Player 1
        self.k_range = k_range  # Number of epochs
        self.x_updates = x_updates  # Updates for Player 1
        self.theta_updates = theta_updates  # Updates for Player 2
        self.task_memory = {}  # Stores data from previous tasks
        self.task_count = 0

    def update_model(self, inputs, targets, current_task_id):
        remapped_targets = remap_labels(targets)
        loss = 0.0
        out = None

        for kappa in range(self.k_range):
            initial_loss = self.criterion(self.model(inputs, current_task_id - 1), remapped_targets)

            if self.task_count > 0:
                perturbed_inputs = inputs.clone().detach().requires_grad_(True)
                initial_loss_1 = self.criterion(self.model(inputs, current_task_id - 1), remapped_targets)

                # Player 1
                gen_loss = None
                for _ in range(self.x_updates):
                    out = self.model(perturbed_inputs, current_task_id - 1)
                    gen_loss = self.criterion(out, remapped_targets)

                    # Perturb inputs
                    self.optimizer.zero_grad()
                    gen_loss.backward(retain_graph=True)
                    perturbed_inputs = perturbed_inputs + self.epsilon * perturbed_inputs.grad.sign()
                    perturbed_inputs = perturbed_inputs.detach().requires_grad_(True)

                # Jk+ζ (θ k ) − Jk (θ k )
                gen_loss = gen_loss - initial_loss_1

                # Player 2
                temp_model = copy.deepcopy(self.model)
                temp_model_optimizer = optim.SGD(temp_model.parameters(), lr=self.epsilon)
                initial_loss_2 = self.criterion(temp_model(inputs, current_task_id - 1), remapped_targets)

                forget_loss = None
                for _ in range(self.theta_updates):
                    out = temp_model(inputs, current_task_id - 1)
                    forget_loss = self.criterion(out, remapped_targets)

                    temp_model_optimizer.zero_grad()
                    forget_loss.backward(retain_graph=True)
                    temp_model_optimizer.step()

                # Jk (θ^(i+ζ) k) − Jk (θ^i k )
                forget_loss = forget_loss - initial_loss_2
                loss += initial_loss + gen_loss + forget_loss
            else:
                loss = self.criterion(self.model(inputs, current_task_id - 1), remapped_targets)

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
        self.task_memory[self.task_count] = task_loader

        for data, target in task_loader:
            loss, _ = self.update_model(data, target, self.task_count + 1)

