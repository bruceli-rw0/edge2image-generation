from tqdm import tqdm
import torch
from torch.utils import data
from .datasets import CustomDataset
from .models.pix2pix_model import Pix2Pix

def inference(args):
    pass

def train(args):
    pass

def run_model(args):
    dataset = CustomDataset(args)
    print("Length of dataset: ", len(dataset))
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = Pix2Pix(args)
    print(model)

    for i, input in enumerate(tqdm(dataloader)):
        model.set_input(input)
        model.optimize_parameters()
