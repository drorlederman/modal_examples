import matplotlib.pyplot as plt
import os

def plot_results(epochs, train_losses, val_losses, train_accuracies, val_accuracies, model_path=''):

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    # Save the figure to a file
    plt.savefig(os.path.join(model_path, 'training_plot.png'))
    print(f"Plot saved to {model_path}/training_plot.png")

    plt.show()  # This may not work in a headless environment
