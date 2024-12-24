import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

random.seed(42)
torch.autograd.set_detect_anomaly(True)


class BCLModel:
    def __init__(self, model, lr=0.01, epsilon=0.01, k_range=50, x_updates=10, theta_updates=10):
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
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

    def update_model(self, x, edge_index, target, batch):
        out = self.model(x, edge_index, batch)
        gen_loss, forget_loss = 0, 0
        initial_loss = self.criterion(out, target)

        if self.task_count > 0:
            total_loss = self.beta * self.criterion(self.model(x, edge_index, batch), target)
            perturbed_input = x.clone().detach().requires_grad_(True)
            adv_grad = 0
            J_PN_x = self.criterion(self.model(perturbed_input, edge_index, batch), target)

            # Player 1
            for _ in range(self.x_updates):
                perturbed_input = perturbed_input + self.epsilon * adv_grad
                adv_grad = torch.autograd.grad(
                    self.criterion(self.model(perturbed_input, edge_index, batch), target), perturbed_input
                )[0]
                adv_grad = self.normalize_grad(adv_grad)

            # Jk+ζ (θ k ) − Jk (θ k )
            gen_loss = self.criterion(self.model(perturbed_input, edge_index, batch), target)
            gen_loss = gen_loss - J_PN_x

            # Player 2
            J_P = self.criterion(self.model(x, edge_index, batch), target)
            temp_model = copy.deepcopy(self.model)
            temp_model_optimizer = optim.SGD(temp_model.parameters(), lr=self.lr)
            J_PN_theta = self.criterion(self.model(x, edge_index, batch), target)

            for _ in range(self.theta_updates):
                temp_model_optimizer.zero_grad()
                out = temp_model(x, edge_index, batch)
                forget_loss = self.criterion(out, target)
                forget_loss.backward(retain_graph=True)
                temp_model_optimizer.step()

            # Jk (θ^i k ) - Jk (θ^(i+ζ) k)
            forget_loss = J_PN_theta - self.criterion(temp_model(x, edge_index, batch), target)
            total_loss += J_P + forget_loss + gen_loss
        else:
            total_loss = self.criterion(self.model(x, edge_index, batch), target)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), initial_loss, gen_loss, forget_loss, out

    def evaluate(self, tasks_test):
        self.model.eval()
        results = {}

        with torch.no_grad():
            for task_id, test_loader in enumerate(tasks_test):
                total_loss = 0.0
                total_samples = 0

                for batch in test_loader:
                    x, target = batch  # Unpack x and y from BatchWrapper
                    x = x.to(next(self.model.parameters()).device)
                    target = target.to(next(self.model.parameters()).device)

                    outputs = self.model(x)
                    loss = self.criterion(outputs, target)

                    total_loss += loss.item() * len(x)
                    total_samples += len(x)

                # Store average loss for the task
                average_loss = total_loss / total_samples if total_samples > 0 else float('inf')
                results[f'Task_{task_id + 1}'] = average_loss

        return results

    def train_task(self, task_loader):
        self.model.train()
        self.task_memory[self.task_count] = task_loader

        replay_ratio = 0.25  # Percentage of previous task data to include
        combined_batches = []

        if self.task_count > 0:
            for prev_task_id in range(self.task_count):
                prev_loader = self.task_memory[prev_task_id]
                for batch in prev_loader:
                    combined_batches.append(batch)

            sample_size = int(replay_ratio * len(combined_batches))
            if sample_size > 0:
                replay_data = random.sample(combined_batches, sample_size)
            else:
                replay_data = []
        else:
            replay_data = []

        for batch in task_loader:
            combined_batches.append(batch)

        combined_batches.extend(replay_data)

        # Concatenate all inputs and targets
        initial_loss_list, gen_loss_list, forget_loss_list = [], [], []

        for epoch in range(self.k_range):
            for batch in combined_batches:
                x = batch.x
                edge_index = batch.edge_index
                target = batch.y
                batch_attr = batch.batch

                total_loss, initial_loss, gen_loss, forget_loss, _ = self.update_model(
                    x,
                    edge_index,
                    target,
                    batch_attr
                )

                initial_loss_list.append(initial_loss.item())
                if self.task_count > 0:
                    gen_loss_list.append(gen_loss.item())
                    forget_loss_list.append(forget_loss.item())
                else:
                    gen_loss_list.append(gen_loss)
                    forget_loss_list.append(forget_loss)

        self.task_count += 1

        return initial_loss_list, gen_loss_list, forget_loss_list
