import torch
import os
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = 0

    def __call__(self, val_loss, model, optimizer, epoch, model_name, model_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                # self.save_best_epoch(model_name, model_path)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer, model_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, model_path)
        self.val_loss_min = val_loss

    # def save_best_epoch(self, model_name, model_path):
    #     '''Saves the best epoch number when early stopping occurs'''
    #     file_path = os.path.join(model_path, "best_epoch.txt")
    #     with open(file_path, 'a') as file:
    #         file.write(f"<{model_name}> has the best epoch {self.best_epoch}\n")
