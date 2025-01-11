import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

random.seed(42)
torch.autograd.set_detect_anomaly(True)


def remap_labels(target):
    return target % 2


class BCLModel:
    def __init__(self, model, lr=0.01, epsilon=0.01, k_range=50, x_updates=10, theta_updates=10):
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.epsilon = epsilon  # Perturbation strength for Player 1
        self.beta = 0.9
        self.k_range = k_range  # Number of epochs
        self.x_updates = x_updates  # Updates for Player 1
        self.theta_updates = theta_updates  # Updates for Player 2
        self.task_memory = {}  # Stores data from previous tasks
        self.task_count = 0

    @staticmethod
    def normalize_grad(grad, p=2, dim=1, eps=1e-12):
        return grad / grad.norm(p, dim, True).clamp(min=eps).expand_as(grad)

    def update_model(self, data, target):
        target = remap_labels(target)
        out = self.model(data)
        gen_loss, forget_loss = 0, 0
        initial_loss = self.criterion(out, target)

        if self.task_count > 0:
            total_loss = self.beta * self.criterion(self.model(data), target)
            perturbed_input = copy.deepcopy(data)
            perturbed_input.requires_grad = True
            adv_grad = 0
            J_PN_x = self.criterion(self.model(data), target)

            # Player 1
            for _ in range(self.x_updates):
                perturbed_input = perturbed_input + self.epsilon * adv_grad
                adv_grad = torch.autograd.grad(
                    self.criterion(self.model(perturbed_input), target), perturbed_input
                )[0]
                adv_grad = self.normalize_grad(adv_grad)

            # Jk+ζ (θ k ) − Jk (θ k )
            gen_loss = self.criterion(self.model(perturbed_input), target)
            gen_loss = gen_loss - J_PN_x

            # Player 2
            J_P = self.criterion(self.model(data), target)
            temp_model = copy.deepcopy(self.model)
            temp_model_optimizer = optim.SGD(temp_model.parameters(), lr=self.lr)
            J_PN_theta = self.criterion(self.model(data), target)

            for _ in range(self.theta_updates):
                temp_model_optimizer.zero_grad()
                out = temp_model(data)
                forget_loss = self.criterion(out, target)
                forget_loss.backward(retain_graph=True)
                temp_model_optimizer.step()

            # Jk (θ^i k ) - Jk (θ^(i+ζ) k)
            forget_loss = J_PN_theta - self.criterion(temp_model(data), target)
            total_loss += J_P + forget_loss + gen_loss
        else:
            total_loss = self.criterion(self.model(data), target)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), initial_loss, gen_loss, forget_loss, out

    def evaluate(self, tasks_test):
        self.model.eval()
        results = {}

        with torch.no_grad():
            for task_id, test_loader in enumerate(tasks_test):
                correct = 0
                total = 0

                for data, target in test_loader:
                    data, target = data.to(next(self.model.parameters()).device), target.to(
                        next(self.model.parameters()).device)
                    outputs = self.model(data)
                    remapped_target = remap_labels(target)
                    _, predicted = torch.max(outputs.data, 1)

                    total += target.size(0)  # Total number of samples
                    correct += (predicted == remapped_target).sum().item()  # Count correct predictions

                # Calculate accuracy for the task
                accuracy = correct / total if total > 0 else 0.0
                results[f'Task_{task_id + 1}'] = accuracy

        return results

    def train_task(self, task_loader):
        self.model.train()
        self.task_memory[self.task_count] = task_loader

        replay_ratio = 0.25  # Percentage of previous task data to include
        combined_data = []

        if self.task_count > 0:
            for prev_task_id in range(self.task_count):
                prev_loader = self.task_memory[prev_task_id]
                for data, target in prev_loader:
                    combined_data.append((data, target))

            sample_size = int(replay_ratio * len(combined_data))
            if sample_size > 0:
                replay_data = random.sample(combined_data, sample_size)
            else:
                replay_data = []
        else:
            replay_data = []

        full_inputs = []
        full_targets = []

        for data, target in task_loader:
            full_inputs.append(data)
            full_targets.append(target)

        if replay_data:
            for replay_data_item, replay_target_item in replay_data:
                full_inputs.append(replay_data_item)
                full_targets.append(replay_target_item)

        # Concatenate all inputs and targets
        combined_inputs = torch.cat(full_inputs, dim=0)
        combined_targets = torch.cat(full_targets, dim=0)

        initial_loss_list, gen_loss_list, forget_loss_list = [], [], []

        batch_size = 32
        num_samples = len(combined_inputs)

        for epoch in range(self.k_range):
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_inputs = combined_inputs[batch_start:batch_end]
                batch_targets = combined_targets[batch_start:batch_end]

                total_loss, initial_loss, gen_loss, forget_loss, _ = self.update_model(batch_inputs, batch_targets)

                initial_loss_list.append(initial_loss.item())
                if self.task_count > 0:
                    gen_loss_list.append(gen_loss.item())
                    forget_loss_list.append(forget_loss.item())
                else:
                    gen_loss_list.append(gen_loss)
                    forget_loss_list.append(forget_loss)

        self.task_count += 1

        return initial_loss_list, gen_loss_list, forget_loss_list
