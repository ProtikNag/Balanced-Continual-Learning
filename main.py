from task_specific_mlp import TaskSpecificMLP
from split_mnist import load_split_mnist_data
from bcl_model import BCLModel
from visualize import plot_loss


def main():
    # Load data
    tasks_train, tasks_test = load_split_mnist_data(batch_size=1, max_samples_per_task=10)

    # Initialize model
    model = TaskSpecificMLP()
    bcl_model = BCLModel(model)
    initial_loss, gen_loss, forget_loss = None, None, None
    loss_by_task = {}

    # Train on each task and evaluate
    for task_id, task_loader in enumerate(tasks_train):
        print(f"\nTraining on Task {task_id + 1}")
        initial_loss, gen_loss, forget_loss = bcl_model.train_task(task_loader)
        loss_by_task[task_id] = {
            "initial_loss": initial_loss,
            "gen_loss": gen_loss,
            "forget_loss": forget_loss,
        }

    plot_loss(loss_by_task)


if __name__ == "__main__":
    main()
