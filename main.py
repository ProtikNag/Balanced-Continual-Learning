from task_specific_mlp import TaskSpecificMLP
from split_mnist import load_split_mnist_data
from bcl_model import BCLModel


def main():
    # Load data
    tasks_train, tasks_test = load_split_mnist_data(batch_size=64)

    # Initialize model
    model = TaskSpecificMLP()
    bcl_model = BCLModel(model)

    # Train on each task and evaluate
    all_results = []
    for task_id, task_loader in enumerate(tasks_train):
        print(f"\nTraining on Task {task_id + 1}")
        bcl_model.train_task(task_loader)
        print("Evaluating performance on all tasks seen so far...")
        task_accuracies = bcl_model.evaluate(tasks_test[:task_id + 1])
        all_results.append(task_accuracies)


if __name__ == "__main__":
    main()
