import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

random.seed(42)
torch.autograd.set_detect_anomaly(True)


def remap_labels(target):
    remapped_targets = target.clone()

    # Define the mapping for each task
    task_mapping = {
        (0, 1): 0,
        (2, 3): 2,
        (4, 5): 4,
        (6, 7): 6,
        (8, 9): 8
    }

    # Remap each target to [0, 1] range for its task
    for (digit1, digit2), base_label in task_mapping.items():
        remapped_targets[target == digit1] = 0
        remapped_targets[target == digit2] = 1

    return remapped_targets


class BCLModel:
    def __init__(self, model, lr=0.0001, epsilon=0.0001, k_range=10, x_updates=5, theta_updates=5):
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon  # Perturbation strength for Player 1
        self.k_range = k_range  # Number of epochs
        self.x_updates = x_updates  # Updates for Player 1
        self.theta_updates = theta_updates  # Updates for Player 2
        self.task_memory = {}  # Stores data from previous tasks
        self.task_count = 0

    def update_model(self, data, target, task_id):
        target = remap_labels(target)
        loss, gen_loss, forget_loss = 0.0, 0.0, 0.0
        out = self.model(data, task_id)
        initial_loss = self.criterion(out, target)

        if self.task_count > 0:
            perturbed_input = data.clone().detach().requires_grad_(True)
            initial_loss_1 = self.criterion(self.model(data, task_id), target)

            # Player 1
            for _ in range(self.x_updates):
                out = self.model(perturbed_input, task_id)
                gen_loss = self.criterion(out, target)

                # Perturb input
                self.optimizer.zero_grad()
                gen_loss.backward(retain_graph=True)
                perturbed_input = perturbed_input + self.epsilon * perturbed_input.grad.sign()
                perturbed_input = perturbed_input.detach().requires_grad_(True)

            # Jk+ζ (θ k ) − Jk (θ k )
            gen_loss = gen_loss - initial_loss_1

            # Player 2
            temp_model = copy.deepcopy(self.model)
            temp_model_optimizer = optim.SGD(temp_model.parameters(), lr=self.epsilon)
            initial_loss_2 = self.criterion(temp_model(data, task_id), target)

            for _ in range(self.theta_updates):
                out = temp_model(data, task_id)
                forget_loss = self.criterion(out, target)

                temp_model_optimizer.zero_grad()
                forget_loss.backward(retain_graph=True)
                temp_model_optimizer.step()

            # Jk (θ^(i+ζ) k) − Jk (θ^i k )
            forget_loss = initial_loss_2 - forget_loss
            loss += initial_loss + gen_loss + forget_loss.detach()
        else:
            loss = self.criterion(self.model(data, task_id), target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach(), initial_loss, gen_loss, forget_loss, out

    def train_task(self, task_loader):
        self.model.train()
        self.task_memory[self.task_count] = task_loader

        replay_ratio = 0.15  # Percentage of previous task data to include
        combined_data = []

        if self.task_count > 0:
            for prev_task_id in range(self.task_count):
                prev_loader = self.task_memory[prev_task_id]
                for data, target in prev_loader:
                    combined_data.append((data, target, prev_task_id))

            sample_size = int(replay_ratio * len(combined_data))
            if sample_size > 0:
                replay_data = random.sample(combined_data, sample_size)
            else:
                replay_data = []
        else:
            replay_data = []

        full_inputs = []
        full_targets = []
        full_task_ids = []

        for data, target in task_loader:
            task_ids = [self.task_count] * len(data)  # Current task ID for current data
            full_inputs.append(data)
            full_targets.append(target)
            full_task_ids.extend(task_ids)

        if replay_data:
            for replay_data_item, replay_target_item, replay_task_id in replay_data:
                full_inputs.append(replay_data_item)
                full_targets.append(replay_target_item)
                full_task_ids.extend([replay_task_id] * len(replay_data_item))

        # Concatenate all inputs and targets
        combined_inputs = torch.cat(full_inputs, dim=0)
        combined_targets = torch.cat(full_targets, dim=0)

        initial_loss_list, gen_loss_list, forget_loss_list = [], [], []

        for epoch in range(self.k_range):
            for i in range(len(combined_inputs)):
                single_input = combined_inputs[i].unsqueeze(0)
                single_target = combined_targets[i].unsqueeze(0)
                single_task_id = full_task_ids[i]

                total_loss, initial_loss, gen_loss, forget_loss, _ = self.update_model(single_input, single_target,
                                                                                       single_task_id)

                initial_loss_list.append(initial_loss.item())
                if self.task_count > 0:
                    gen_loss_list.append(gen_loss.item())
                    forget_loss_list.append(forget_loss.item())
                else:
                    gen_loss_list.append(gen_loss)
                    forget_loss_list.append(forget_loss)

        self.task_count += 1

        return initial_loss_list, gen_loss_list, forget_loss_list
