from fnn import FNN
from split_data import load_and_split
from bcl_model import BCLModel
from visualize import plot_combined_loss, plot_combined_acc, plot_taskwise_accuracy_progression
import random
import torch
import itertools

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
    # Load data
    num_tasks = 10
    tasks_train, tasks_test = load_and_split(batch_size=64, tasks=num_tasks)

    # Generate random sequences of task orders
    task_sequences = generate_task_sequences(num_tasks)

    for i, task_order in enumerate(task_sequences):
        print(f"\nSequence {i + 1}: {task_order}")
        loss_by_task = {}
        acc_by_task = {}

        # Initialize model
        input_features = 1
        output_classes = 1
        model = FNN(
            input_features=input_features,
            hidden_features=32,
            output_classes=output_classes
        )
        bcl_model = BCLModel(model)

        for task_id, task_index in enumerate(task_order):
            print(f"\nTraining on Task {task_index + 1}")

            train_task_loader = tasks_train[task_index]
            initial_loss, gen_loss, forget_loss = bcl_model.train_task(train_task_loader)
            test_loss_results = bcl_model.evaluate(tasks_test)

            loss_by_task[task_id] = {
                "initial_loss": initial_loss,
                "gen_loss": gen_loss,
                "forget_loss": forget_loss,
            }
            acc_by_task[task_id] = test_loss_results

        print(acc_by_task)

        plot_combined_loss(loss_by_task, sequence_id=i + 1)
        plot_combined_acc(test_loss_results, sequence_id=i + 1)
        plot_taskwise_accuracy_progression(test_loss_results, sequence_id=i + 1)


if __name__ == "__main__":
    main()
