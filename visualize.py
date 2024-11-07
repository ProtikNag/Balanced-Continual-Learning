import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs('./figures', exist_ok=True)


def plot_loss(loss_by_task):
    # Define colors for different loss types
    initial_loss_color = 'red'
    gen_loss_color = 'blue'
    forget_loss_color = 'green'

    # Iterate over the tasks and create a separate plot for each
    for task_id, losses in loss_by_task.items():
        plt.figure(figsize=(14, 6))
        iterations = range(len(losses["initial_loss"]))

        # Plot losses for the current task
        plt.plot(iterations, losses["initial_loss"], color=initial_loss_color, label='Initial Loss')
        plt.plot(iterations, losses["gen_loss"], color=gen_loss_color, label='Gen Loss')
        plt.plot(iterations, losses["forget_loss"], color=forget_loss_color, label='Forget Loss')

        # Labels and title
        plt.xlabel('Update Iteration')
        plt.ylabel('Loss Value')
        plt.title(f'Loss Progression for Task {task_id}')
        plt.legend(loc='upper right', fontsize='large')
        plt.grid(True)

        # Save and show the plot for the current task
        plt.savefig(f'./figures/loss_task_{task_id}.png')
        plt.show()
