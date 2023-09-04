from matplotlib import pyplot as plt
import os

class Logger:

    def __init__(self):
        self.loss_train = []
        self.loss_val = []

        self.acc_train = []
        self.acc_val = []


    def get_logs(self):
        return self.loss_train, self.loss_val, self.acc_train, self.acc_val


    def restore_logs(self, logs):
        self.loss_train, self.loss_val, self.acc_train, self.acc_val = logs


    def plot(self, hparams, save = False, show = True):
        
        if save:
            loss_path = os.path.join(hparams['model_save_dir'], 'loss.jpg')
            acc_path = os.path.join(hparams['model_save_dir'], 'acc.jpg')

        plt.figure()
        plt.plot(self.acc_train, 'g', label='Training Acc')
        plt.plot(self.acc_val, 'b', label='Validation Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.grid()

        if save:
            plt.savefig(acc_path)
        if show:
            plt.show()

        plt.close()

        plt.figure()
        plt.plot(self.loss_train, 'g', label='Training Loss')
        plt.plot(self.loss_val, 'b', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        if save:
            plt.savefig(loss_path)
        if show:
            plt.show()

        plt.close()