from tqdm import tqdm
import torch
from torch.utils import data
from .datasets import CustomDataset
from .models.pix2pix_model import Pix2Pix
from .metrics import Metrics
from .visualizer import save_generations

def inference(args, dataloader, model, e) -> None:
    edges = list()
    reals = list()
    fakes = list()
    paths = list()
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            model.set_input(inputs)
            fake = model.forward()
            edges.append((inputs['A'].squeeze().numpy().transpose(1,2,0)+1) / 2)
            reals.append((inputs['B'].squeeze().numpy().transpose(1,2,0)+1) / 2)
            fakes.append((fake.squeeze().numpy().transpose(1,2,0)+1) / 2)
            paths += inputs['A_paths']
        save_generations(edges, reals, fakes, paths, args.eval_result, e)

def train(args, dataloader, model, metrics) -> None:
    model.train()
    for inputs in tqdm(dataloader):
        model.set_input(inputs)
        model.optimize_parameters()
        metrics.update(model.loss_G.detach().numpy(), model.loss_D.detach().numpy())
    loss_G, loss_D = metrics.get_epoch_loss()
    
    print(f"G loss: {loss_G}")
    print(f"D loss: {loss_D}")

def run_model(args) -> None:
    dataset = dict()
    dataloader = dict()

    # load data
    dataset["train"] = CustomDataset(args.train_folder, args.num_train)
    dataloader["train"] = data.DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)
    dataset["eval"] = CustomDataset(args.eval_folder, args.num_eval)
    dataloader["eval"] = data.DataLoader(dataset["eval"], batch_size=1, shuffle=False)

    print(f"Number of training: {len(dataloader['train'])}")
    print(f"Number of evaluation: {len(dataloader['eval'])}")

    # define model
    model = Pix2Pix(args)

    # define metrics for storing running stats
    metrics = dict()
    metrics['train'] = Metrics(len(dataset["train"]))

    for e in tqdm(range(args.n_epochs)):
        if args.do_train:
            train(args, dataloader["train"], model, metrics['train'])
        if args.do_eval:
            inference(args, dataloader["eval"], model, e)
        metrics['train'].new_epoch()
