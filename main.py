import torch
import csv
from torch.utils.data import DataLoader
from split_data import create_task_datasets, split_and_load_tasks
from bcl_model import BCLModel
from visualize import plot_combined_loss, plot_combined_acc, plot_taskwise_accuracy_progression
from fnn import FeedForwardNN
import random

# Automate task sequence generation
def generate_random_task_sequences(num_tasks, num_sequences=3):
    # Generate `num_sequences` random sequences of task indices
    task_indices = list(range(num_tasks))
    sequences = [random.sample(task_indices, len(task_indices)) for _ in range(num_sequences)]
    return sequences

def log_losses_to_csv(file_name, sequence_id, loss_by_task):
    # Log the losses to a CSV file
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sequence ID", "Task ID", "Average Loss"])
        for task_id, avg_loss in loss_by_task.items():
            writer.writerow([sequence_id, task_id + 1, avg_loss])

def main():
    # Number of tasks and samples per task
    num_tasks = 10
    num_samples = 1000
    batch_size = 16

    # Generate datasets for all tasks
    tasks = create_task_datasets(num_tasks=num_tasks, num_samples=num_samples)

    # Split datasets into train and test loaders
    tasks_train, tasks_test = split_and_load_tasks(tasks, batch_size=batch_size)

    # Generate random sequences of task orders
    task_sequences = generate_random_task_sequences(num_tasks, num_sequences=3)

    # Initialize CSV file for logging losses
    csv_file_name = "sequence_losses.csv"
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sequence ID", "Task ID", "Average Loss"])  # Header row

    for sequence_id, task_order in enumerate(task_sequences, start=1):
        print(f"\nSequence {sequence_id}: {task_order}")
        loss_by_task = {}

        # Initialize model
        input_features = 1  # Sine wave input is a single value (time step)
        hidden_features = 64
        output_classes = 1  # Regression task for sine wave amplitude
        model = FeedForwardNN(
            input_features=input_features,
            hidden_features=hidden_features,
            output_classes=output_classes,
            dropout=0.20
        )
        bcl_model = BCLModel(model)

        for task_id, task_index in enumerate(task_order):
            print(f"\nTraining on Task {task_index + 1}")

            # Get train and test loaders for the current task
            train_task_loader = tasks_train[task_index]
            test_task_loader = tasks_test[task_index]

            # Train the model on the current task
            initial_loss, gen_loss, forget_loss = bcl_model.train_task(train_task_loader)

            # Evaluate the model on the current task
            avg_loss = bcl_model.evaluate(test_task_loader)

            # Print the losses after each step
            print(f"Average Loss: {avg_loss:.4f}")

            # Store average loss for logging
            loss_by_task[task_id] = avg_loss

        # Log losses to CSV after learning all tasks in the sequence
        log_losses_to_csv(csv_file_name, sequence_id, loss_by_task)

        print(f"Losses for Sequence {sequence_id} logged in {csv_file_name}.")

if __name__ == "__main__":
    main()
