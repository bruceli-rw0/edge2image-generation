import os
from tqdm.auto import tqdm
from time import gmtime, strftime, localtime

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
        save_generations(args, edges, reals, fakes, paths, str(e))

def train(args, dataloader, model, metrics, e) -> None:
    model.train()
    for inputs in tqdm(dataloader):
        model.set_input(inputs)
        model.optimize_parameters()
        metrics.update(model.loss_G.detach().numpy(), model.loss_D.detach().numpy())
    
    loss_G, loss_D = metrics.get_epoch_loss()
    print(f"Epoch {e}")
    print(f"\t G loss: {loss_G:.4f}", end='')
    print(f"\t D loss: {loss_D:.4f}")
    metrics.new_epoch()
    metrics.save(args)

def run_model(args) -> None:
    # unique identification
    args.model_id = strftime("-%Y-%m-%d-%H-%M-%S", localtime())

    dataset = dict()
    dataloader = dict()

    # load data
    dataset["train"] = CustomDataset(args.train_folder, args.num_train)
    dataset["eval"] = CustomDataset(args.eval_folder, args.num_eval)
    dataloader["train"] = data.DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)
    dataloader["eval"] = data.DataLoader(dataset["eval"], batch_size=1, shuffle=False)

    print(f"Number of training: {len(dataloader['train'])}")
    print(f"Number of evaluation: {len(dataloader['eval'])}")

    # define model
    model = Pix2Pix(args)

    # define metrics for storing running stats
    metrics = dict()
    metrics['train'] = Metrics(len(dataset["train"]))

    if args.do_train:
        if args.checkpoint_dir not in os.listdir(args.root_dir):
            os.mkdir(os.path.join(args.root_dir, args.checkpoint_dir))
        os.mkdir(os.path.join(args.root_dir, args.checkpoint_dir, f'{args.model}{args.model_id}'))

        if args.metrics_dir not in os.listdir(args.root_dir):
            os.mkdir(os.path.join(args.root_dir, args.metrics_dir))

    for e in tqdm(range(1, args.n_epochs+1)):
        if args.do_train:
            train(args, dataloader["train"], model, metrics['train'], e)

            # save epoch checkpoint
            torch.save(
                model.state_dict(), 
                os.path.join(args.root_dir, args.checkpoint_dir, f'{args.model}{args.model_id}', f'state_dict-{e}.pth')
            )

        if args.do_eval:
            inference(args, dataloader["eval"], model, e)

        # model.load_state_dict(
        #     torch.load(
        #         os.path.join(args.checkpoint_dir, f'{args.model}{args.model_id}', f'state_dict-{e}.pth')
        #     ))
