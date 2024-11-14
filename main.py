from task_specific_mlp import TaskSpecificMLP
from split_mnist import load_split_mnist_data
from bcl_model import BCLModel
from visualize import plot_combined_loss
import random


def main():
    # Load data
    tasks_train, tasks_test = load_split_mnist_data(batch_size=32, max_samples_per_task=80)

    # Initialize model
    model = TaskSpecificMLP()
    bcl_model = BCLModel(model)

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

        for task_id, task_index in enumerate(task_order):
            print(f"\nTraining on Task {task_index + 1}")
            task_loader = tasks_train[task_index]
            initial_loss, gen_loss, forget_loss = bcl_model.train_task(task_loader)
            loss_by_task[task_id] = {
                "initial_loss": initial_loss,
                "gen_loss": gen_loss,
                "forget_loss": forget_loss,
            }

            # _ = bcl_model.evaluate(tasks_test)

            # Plot the combined loss for the current sequence
        plot_combined_loss(loss_by_task, sequence_id=i + 1)


if __name__ == "__main__":
    main()
