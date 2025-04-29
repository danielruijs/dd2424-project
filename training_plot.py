import json
import matplotlib.pyplot as plt

def main():
    with open('training_history.json', 'r') as f:
        history = json.load(f)

    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1) # Create epoch numbers for x-axis
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()