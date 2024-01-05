import matplotlib.pyplot as plt
import numpy as np

def plot_result(fed_result):
    rounds = len(fed_result)
    mean_train_loss = []
    mean_val_loss = []
    mean_train_accuracy = []
    mean_val_accuracy = []

    for round_results in fed_result:
        if round_results['loss'] and round_results['accuracy']:
            train_loss_per_round = np.mean([np.mean(epoch[0]) for epoch in round_results['loss']])
            val_loss_per_round = np.mean([np.mean(epoch[1]) for epoch in round_results['loss']])
            train_accuracy_per_round = np.mean([np.mean(epoch[0]) for epoch in round_results['accuracy']])
            val_accuracy_per_round = np.mean([np.mean(epoch[1]) for epoch in round_results['accuracy']])

            mean_train_loss.append(train_loss_per_round)
            mean_val_loss.append(val_loss_per_round)
            mean_train_accuracy.append(train_accuracy_per_round)
            mean_val_accuracy.append(val_accuracy_per_round)

    # Plotting in 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Subplot 1: Loss
    axs[0].plot(range(1, rounds + 1), mean_train_loss, marker='o', color='blue', label='Mean Train Loss')
    axs[0].plot(range(1, rounds + 1), mean_val_loss, marker='o', color='orange', label='Mean Validation Loss')
    axs[0].set_title('Mean Train and Validation Loss')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Mean Loss')
    axs[0].legend()

    # Subplot 2: Accuracy
    axs[1].plot(range(1, rounds + 1), mean_train_accuracy, marker='o', color='green', label='Mean Train Accuracy')
    axs[1].plot(range(1, rounds + 1), mean_val_accuracy, marker='o', color='red', label='Mean Validation Accuracy')
    axs[1].set_title('Mean Train and Validation Accuracy')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Mean Accuracy')
    axs[1].legend()

    # Set integer values on the x-axis
    for ax in axs:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()