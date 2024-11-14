import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs('./figures', exist_ok=True)


def plot_combined_loss(loss_by_task, sequence_id):
    # Define colors for different loss types
    initial_loss_color = 'red'
    gen_loss_color = 'blue'
    forget_loss_color = 'green'

    # Calculate total length of iterations
    total_iterations = sum(len(losses["initial_loss"]) for losses in loss_by_task.values())
    current_iteration = 0

    # Create a new figure
    plt.figure(figsize=(22, 12))

    # Iterate over the tasks and plot each on the combined figure
    num_tasks = 5  # Adjust for 5 tasks
    for task_id in range(num_tasks):
        if task_id in loss_by_task:
            losses = loss_by_task[task_id]
            iterations = range(current_iteration, current_iteration + len(losses["initial_loss"]))

            # Plot the losses for the current task
            plt.plot(iterations, losses["initial_loss"], color=initial_loss_color,
                     label='Initial Loss' if task_id == 0 else "")
            plt.plot(iterations, losses["gen_loss"], color=gen_loss_color, label='Gen Loss' if task_id == 0 else "")
            plt.plot(iterations, losses["forget_loss"], color=forget_loss_color,
                     label='Forget Loss' if task_id == 0 else "")

            # Shade the background to indicate task boundaries and label them
            plt.axvspan(current_iteration, current_iteration + len(losses["initial_loss"]) - 1, color='grey', alpha=0.1)
            plt.text(current_iteration + len(losses["initial_loss"]) // 2, plt.ylim()[1] * 0.9, f'Task {task_id + 1}',
                     horizontalalignment='center', fontsize=16, color='black')

            current_iteration += len(losses["initial_loss"])  # Shift iteration index for the next task

    # Labels and title
    plt.xlabel('Update Iteration', fontsize=20)
    plt.ylabel('Loss Value', fontsize=20)
    plt.title(f'Combined Loss Progression for Random Task Sequence {sequence_id}', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.grid(True)

    # Save the combined plot
    plt.savefig(f'./figures/loss_figures/combined_loss_sequence_{sequence_id}.pdf')
