from task_specific_mlp import TaskSpecificMLP
from split_mnist import load_split_mnist_data
from bcl_model import BCLModel


def main():
    # Load data
    tasks_train, tasks_test = load_split_mnist_data(batch_size=1, max_samples_per_task=10)

    # Initialize model
    model = TaskSpecificMLP()
    bcl_model = BCLModel(model)

    # Train on each task and evaluate
    for task_id, task_loader in enumerate(tasks_train):
        print(f"\nTraining on Task {task_id + 1}")
        bcl_model.train_task(task_loader)


if __name__ == "__main__":
    main()
