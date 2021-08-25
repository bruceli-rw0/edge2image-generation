import os
from tqdm.auto import tqdm
from time import gmtime, strftime, localtime
import logging
logger = logging.getLogger(__name__)

import torch
from torch.utils import data
from .datasets import CustomDataset
from .models import initialize_model
from .metrics import Metrics
from .util.helper import *
from .util.visualizer import save_generations

def setup_logging(args) -> None:
    if args.log_dir not in os.listdir(args.root_dir):
            os.mkdir(os.path.join(args.root_dir, args.log_dir))
    
    handlers = list()
    handlers.append(logging.FileHandler(
        filename=os.path.join(args.root_dir, args.log_dir, f"log{args.model_id}.txt"), 
        mode='w'
    ))
    handlers.append(logging.StreamHandler())
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers
    )

def inference(args, dataloader, model, e: str) -> None:
    logger.info(f"Begin evaluation - epoch {e} ...")

    edges, reals, fakes, paths = list(), list(), list(), list()
    epoch_iterator = tqdm(dataloader, desc="Iteration", position=0, leave=True)
    model.eval()
    with torch.no_grad():
        for inputs in epoch_iterator:
            fake = model(inputs['A'], inputs['B'])
            edges.append((inputs['A'].squeeze().numpy().transpose(1,2,0)+1) / 2)
            reals.append((inputs['B'].squeeze().numpy().transpose(1,2,0)+1) / 2)
            fakes.append((fake.detach().cpu().squeeze().numpy().transpose(1,2,0)+1) / 2)
            paths += inputs['A_path']
        save_generations(args, edges, reals, fakes, paths, e)
    edges = reals = fakes = paths = None

def train(args, dataloader, model, metrics, e: str) -> None:
    logger.info(f"Begin training - epoch {e} ...")

    epoch_iterator = tqdm(dataloader, desc="Iteration", position=0, leave=True)
    model.train()
    for inputs in epoch_iterator:
        _ = model(inputs['A'], inputs['B'])
        model.optimize_parameters()
        metrics.update(model.loss_G, model.loss_D)
    
    loss_G, loss_D = metrics.get_epoch_loss()
    logger.info(f"Epoch {e}...")
    logger.info(f"Tot. - G loss: {loss_G:.4f} - D loss: {loss_D:.4f}")
    logger.info(f"Avg. - G loss: {loss_G/len(dataloader):.4f} - D loss: {loss_D/len(dataloader):.4f}")
    metrics.new_epoch()
    metrics.save(args)

def run_model(args) -> None:
    # unique identification
    args.model_id = strftime("-%Y-%m-%d-%H-%M-%S", localtime())

    setup_logging(args)
    dataset = dict()
    dataloader = dict()

    # load data
    dataset["train"] = CustomDataset(args.train_folder, args.num_train, args)
    dataset["eval"] = CustomDataset(args.eval_folder, args.num_eval, args)
    dataloader["train"] = data.DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)
    dataloader["eval"] = data.DataLoader(dataset["eval"], batch_size=1, shuffle=False)

    # define model
    model = initialize_model(args)
    model.setup(args)

    # define metrics for storing running stats
    metrics = dict()
    metrics['train'] = Metrics(len(dataset["train"]))

    log_settings(args, logger)
    if args.do_train:
        if args.checkpoints_dir not in os.listdir(args.root_dir):
            os.mkdir(os.path.join(args.root_dir, args.checkpoints_dir))
        os.mkdir(os.path.join(args.root_dir, args.checkpoints_dir, f'{args.model}{args.model_id}'))

        if args.metrics_dir not in os.listdir(args.root_dir):
            os.mkdir(os.path.join(args.root_dir, args.metrics_dir))

    for e in tqdm(range(1, args.n_epochs+1)):
        if args.do_train:
            # train model
            train(args, dataloader["train"], model, metrics['train'], str(e))

            # update learning rates at the end of every epoch
            lr_old, lr_new = model.update_learning_rate()
            logger.info(f'learning rate {lr_old:.7f} -> {lr_new:.7f}')

            # save epoch checkpoint
            if e % args.save_epoch_freq:
                model.save(e)

        if args.do_eval:
            # perform evaluation
            inference(args, dataloader["eval"], model, str(e))
