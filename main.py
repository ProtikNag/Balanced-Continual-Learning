from task_specific_mlp import TaskSpecificMLP
from split_mnist import load_split_mnist_data
from bcl_model import BCLModel
from visualize import plot_combined_loss, plot_combined_acc, plot_taskwise_accuracy_progression
import random
import torch


def add_noise_to_task(data_loader, noise_level=0.20):
    noisy_data = []
    for data, target in data_loader:
        noise = torch.randn_like(data) * noise_level
        noisy_data.append((data + noise, target))

    return noisy_data


def calculate_average_accuracy(acc_by_task, task_order):
    total_accuracy = 0
    for task_id in acc_by_task:
        task_accuracies = [acc_by_task[tid][f'Task_{i + 1}'] for tid, i in enumerate(task_order[:task_id + 1])]
        total_accuracy += sum(task_accuracies) / len(task_accuracies)

    average_accuracy = total_accuracy / len(task_order)
    return average_accuracy


def calculate_average_forgetting(acc_by_task, task_order):
    total_forgetting = 0
    t = len(task_order)
    for i in range(t - 1):
        max_accuracy = max([acc_by_task[j][f'Task_{i + 1}'] for j in range(i + 1)])
        final_accuracy = acc_by_task[t - 1][f'Task_{i + 1}']
        total_forgetting += max_accuracy - final_accuracy
    average_forgetting = total_forgetting / (t - 1)
    return average_forgetting


def main():
    # Load data
    tasks_train, tasks_test = load_split_mnist_data(batch_size=32, max_samples_per_task=200)

    # Add noise to task 5
    tasks_train[4] = add_noise_to_task(tasks_train[4], noise_level=0.20)
    tasks_test[4] = add_noise_to_task(tasks_test[4], noise_level=0.20)

    # Generate 5 different random sequences of task orders
    task_sequences = [
        [4, 0, 1, 2, 3],
        [0, 4, 1, 2, 3],
        [0, 1, 4, 2, 3],
        [0, 1, 2, 4, 3],
        [0, 1, 2, 3, 4],
    ]

    with open("results.txt", "w") as results_file:
        for i, task_order in enumerate(task_sequences):
            print(f"\nSequence {i + 1}: {task_order}")
            results_file.write(f"\nSequence {i + 1}: {task_order}\n")
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

            # Calculate average accuracy and forgetting
            average_accuracy = calculate_average_accuracy(acc_by_task, task_order)
            average_forgetting = calculate_average_forgetting(acc_by_task, task_order)

            results_file.write(f"Average Accuracy: {average_accuracy:.2f}%\n")
            results_file.write(f"Average Forgetting: {average_forgetting:.2f}%\n\n")

            print(f"Average Accuracy: {average_accuracy:.2f}%")
            print(f"Average Forgetting: {average_forgetting:.2f}%")

            # Plot results
            plot_combined_loss(loss_by_task, sequence_id=i + 1)
            plot_combined_acc(acc_by_task, sequence_id=i + 1)
            plot_taskwise_accuracy_progression(acc_by_task, sequence_id=i + 1)


if __name__ == "__main__":
    main()
