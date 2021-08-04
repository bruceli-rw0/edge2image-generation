import numpy as np

class Metrics:
    def __init__(self, datasize=None):
        self.datasize = datasize

        self.running_loss_G = list()
        self.running_loss_D = list()

        self.epoch_loss_G = list()
        self.epoch_loss_D = list()

    def update(self, loss_G, loss_D):
        self.epoch_loss_G.append(loss_G)
        self.epoch_loss_D.append(loss_D)

    def get_epoch_loss(self):
        return np.sum(self.epoch_loss_G) / self.datasize, np.sum(self.epoch_loss_D) / self.datasize

    def new_epoch(self):
        # save epoch average loss per sample
        self.running_loss_G.append(np.sum(self.epoch_loss_G) / self.datasize)
        self.running_loss_D.append(np.sum(self.epoch_loss_D) / self.datasize)

        # clear epoch loss
        self.epoch_loss_G.clear()
        self.epoch_loss_D.clear()
