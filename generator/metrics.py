import os
import pickle
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
        return np.sum(self.epoch_loss_G), np.sum(self.epoch_loss_D)

    def new_epoch(self):
        # save epoch average loss per sample
        self.running_loss_G.append(np.sum(self.epoch_loss_G))
        self.running_loss_D.append(np.sum(self.epoch_loss_D))

        # clear epoch loss
        self.epoch_loss_G.clear()
        self.epoch_loss_D.clear()

    def save(self, args):
        pickle.dump({
            'loss_G': self.running_loss_G,
            'loss_D': self.running_loss_D,
            'datasize': self.datasize
        }, open(os.path.join(args.root_dir, args.metrics_dir, f'metrics{args.model_id}.pkl'), 'wb'))
