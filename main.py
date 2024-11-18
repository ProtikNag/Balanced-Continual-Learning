from task_specific_mlp import TaskSpecificMLP
from split_mnist import load_split_mnist_data
from bcl_model import BCLModel
from visualize import plot_combined_loss, plot_combined_acc
import random
import torch


def add_noise_to_task(data_loader, noise_level=0.15):
    noisy_data = []
    for data, target in data_loader:
        noise = torch.randn_like(data) * noise_level
        noisy_data.append((data + noise, target))

    return noisy_data


def main():
    # Load data
    tasks_train, tasks_test = load_split_mnist_data(batch_size=32, max_samples_per_task=200)

    # Add noise to task 5
    tasks_train[4] = add_noise_to_task(tasks_train[4], noise_level=0.1)
    tasks_test[4] = add_noise_to_task(tasks_test[4], noise_level=0.1)

    # Generate 5 different random sequences of task orders
    task_sequences = [
        [4, 0, 1, 2, 3],
        [0, 4, 1, 2, 3],
        [0, 1, 4, 2, 3],
        [0, 1, 2, 4, 3],
        [0, 1, 2, 3, 4],
    ]

    for i, task_order in enumerate(task_sequences):
        print(f"\nSequence {i + 1}: {task_order}")
        loss_by_task = {}
        acc_by_task = {}

        # Initialize model
        model = TaskSpecificMLP()
        bcl_model = BCLModel(model)

        for task_id, task_index in enumerate(task_order):
            print(f"\nTraining on Task {task_index + 1}")

            train_task_loader = tasks_train[task_index]
            initial_loss, gen_loss, forget_loss = bcl_model.train_task(train_task_loader)
            accuracy_results = bcl_model.evaluate(tasks_test)

            loss_by_task[task_id] = {
                "initial_loss": initial_loss,
                "gen_loss": gen_loss,
                "forget_loss": forget_loss,
            }
            acc_by_task[task_id] = accuracy_results

        print(acc_by_task)

        plot_combined_loss(loss_by_task, sequence_id=i + 1)
        plot_combined_acc(acc_by_task, sequence_id=i + 1)


if __name__ == "__main__":
    main()
