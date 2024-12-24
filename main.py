import torch
from torch.utils.data import DataLoader
from split_data import create_task_datasets, split_and_load_tasks
from bcl_model import BCLModel
from visualize import plot_combined_loss, plot_combined_acc, plot_taskwise_accuracy_progression
from fnn import FeedForwardNN
import random

# Automate task sequence generation
def generate_task_sequences(num_tasks):
    # Generate random sequences by shuffling the task indices
    sequences = [
        [7, 0, 1, 2, 3, 4, 5, 6, 8, 9],
        [0, 7, 1, 2, 3, 4, 5, 6, 8, 9],
        [0, 1, 7, 2, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 7, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 3, 7, 4, 5, 6, 8, 9],
        [0, 1, 2, 3, 4, 7, 5, 6, 8, 9],
        [0, 1, 2, 3, 4, 5, 7, 6, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 8, 7, 9],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 7],
    ]
    return sequences

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
    task_sequences = generate_task_sequences(num_tasks)

    for i, task_order in enumerate(task_sequences):
        print(f"\nSequence {i + 1}: {task_order}")
        loss_by_task = {}
        acc_by_task = {}

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

            train_task_loader = tasks_train[task_index]
            test_task_loader = tasks_test[task_index]

            initial_loss, gen_loss, forget_loss = bcl_model.train_task(train_task_loader)
            accuracy_results = bcl_model.evaluate(test_task_loader)

            loss_by_task[task_id] = {
                "initial_loss": initial_loss,
                "gen_loss": gen_loss,
                "forget_loss": forget_loss,
            }
            acc_by_task[task_id] = accuracy_results

        print(acc_by_task)

        plot_combined_loss(loss_by_task, sequence_id=i + 1)
        plot_combined_acc(acc_by_task, sequence_id=i + 1)
        plot_taskwise_accuracy_progression(acc_by_task, sequence_id=i + 1)

if __name__ == "__main__":
    main()