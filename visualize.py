import matplotlib.pyplot as plt
import os
import numpy as np

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


def plot_combined_acc(acc_by_task, sequence_id):
    """
    Plots the accuracy progression for all tasks in a random sequence as a grouped bar graph.

    Args:
        acc_by_task (dict): Dictionary containing accuracy results for each task.
        sequence_id (int): Identifier for the random task sequence.
    """
    # Define the task names and their corresponding colors
    task_names = list(acc_by_task[0].keys())  # e.g., ['Task_1', 'Task_2', 'Task_3', 'Task_4', 'Task_5']
    task_colors = ["blue", "orange", "green", "red", "purple"]

    # Extract the accuracy values for each task across all training stages
    num_stages = len(acc_by_task)
    num_tasks = len(task_names)

    # Create an array for the bar positions
    x = np.arange(num_stages)  # One position per training stage
    bar_width = 0.15  # Width of each bar

    plt.figure(figsize=(12, 8))

    # Plot each task's accuracy progression as a bar group
    for i, task_name in enumerate(task_names):
        # Get accuracies for the current task across all stages
        accuracies = [acc_by_task[stage][task_name] for stage in acc_by_task]
        # Position the bars for this task
        bar_positions = x + i * bar_width
        plt.bar(bar_positions, accuracies, width=bar_width, color=task_colors[i % len(task_colors)], label=task_name)

    # Labels and title
    plt.xlabel('Training Stage', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title(f'Accuracy Progression for Random Task Sequence {sequence_id}', fontsize=16)

    # X-axis ticks and legend
    plt.xticks(x + (num_tasks - 1) * bar_width / 2, [f'Task {i+1}' for i in range(num_stages)], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=12)

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the grouped bar plot
    os.makedirs('./figures/accuracy_figures', exist_ok=True)
    plt.savefig(f'./figures/accuracy_figures/combined_accuracy_sequence_{sequence_id}_bar.pdf')
    plt.close()