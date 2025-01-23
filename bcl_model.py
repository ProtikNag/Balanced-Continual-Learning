import torch
import torch.optim as optim
import torch.nn as nn
import copy
import random

random.seed(42)
torch.autograd.set_detect_anomaly(True)


class BCLModel:
    def __init__(self, model, lr=0.001, epsilon=0.01, k_range=3, x_updates=3, theta_updates=3):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
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

    def update_model(self, batch, task_id):
        input_ids = batch["input_ids"].to(next(self.model.parameters()).device)
        attention_mask = batch["attention_mask"].to(next(self.model.parameters()).device)
        labels = batch["labels"].to(next(self.model.parameters()).device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask, task_id)
        logits = outputs["logits"]

        initial_loss = self.criterion(logits, labels)

        # Initialize gen_loss and forget_loss as tensors
        gen_loss = torch.tensor(0.0, device=input_ids.device)
        forget_loss = torch.tensor(0.0, device=input_ids.device)

        if self.task_count > 0:
            # Create a floating-point copy of input_ids for perturbations
            perturbed_input = input_ids.clone().detach().float().requires_grad_(True)

            # Player 1: Generate adversarial examples
            for _ in range(self.x_updates):
                # Convert perturbed input back to integers before passing to BERT
                perturbed_logits = self.model(
                    perturbed_input.long(), attention_mask, task_id
                )["logits"]
                adv_loss = self.criterion(perturbed_logits, labels)
                adv_grad = torch.autograd.grad(adv_loss, perturbed_input, retain_graph=True)[0]
                adv_grad = self.normalize_grad(adv_grad)
                perturbed_input = perturbed_input + self.epsilon * adv_grad

            # Final adversarial pass: Convert perturbed_input back to integers
            perturbed_logits = self.model(
                perturbed_input.long(), attention_mask, task_id
            )["logits"]
            gen_loss = self.criterion(perturbed_logits, labels) - initial_loss

            # Player 2: Fine-tune with task memory
            temp_model = copy.deepcopy(self.model)
            temp_optimizer = optim.Adam(temp_model.parameters(), lr=self.lr)
            for _ in range(self.theta_updates):
                temp_optimizer.zero_grad()
                temp_logits = temp_model(input_ids, attention_mask, task_id)["logits"]
                forget_loss = self.criterion(temp_logits, labels)
                forget_loss.backward(retain_graph=True)
                temp_optimizer.step()

            forget_loss = initial_loss - forget_loss

        # Total loss
        total_loss = initial_loss + gen_loss + forget_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), initial_loss.item(), gen_loss.item(), forget_loss.item()

    def evaluate(self, test_loader, task_id):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(next(self.model.parameters()).device)
                attention_mask = batch["attention_mask"].to(next(self.model.parameters()).device)
                labels = batch["labels"].to(next(self.model.parameters()).device)

                outputs = self.model(input_ids, attention_mask, task_id)
                logits = outputs["logits"]
                _, predicted = torch.max(logits, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def train_task(self, train_loader, task_id):
        self.model.train()
        self.task_memory[self.task_count] = train_loader

        # Prepare memory replay
        replay_ratio = 0.25
        replay_batches = []

        if self.task_count > 0:
            for prev_task_id in range(self.task_count):
                prev_loader = self.task_memory[prev_task_id]
                for batch in prev_loader:
                    replay_batches.append(batch)

            sample_size = int(replay_ratio * len(replay_batches))
            if sample_size > 0:
                replay_data = random.sample(replay_batches, sample_size)
            else:
                replay_data = []
        else:
            replay_data = []

        # Combine current task and memory data
        combined_batches = list(train_loader) + replay_data

        # Train on combined data
        initial_loss_list, gen_loss_list, forget_loss_list = [], [], []

        for epoch in range(self.k_range):
            for batch in combined_batches:
                total_loss, initial_loss, gen_loss, forget_loss = self.update_model(batch, task_id)

                initial_loss_list.append(initial_loss)
                if self.task_count > 0:
                    gen_loss_list.append(gen_loss)
                    forget_loss_list.append(forget_loss)

        self.task_count += 1

        return initial_loss_list, gen_loss_list, forget_loss_list