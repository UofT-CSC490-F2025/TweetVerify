import matplotlib.pyplot as plt


class plotter:
    def __init__(self, image_path):
        self.image_path = image_path

    def plot(self, num_epochs, train_loss, train_acc, val_acc):
        plt.figure()
        plt.plot(range(num_epochs), train_loss)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(self.image_path)
        plt.close()

        plt.figure()
        plt.plot(range(num_epochs), train_acc)
        plt.plot(range(num_epochs), val_acc)
        plt.title("Accuracy over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"])
        plt.savefig(self.image_path)
        plt.close()
