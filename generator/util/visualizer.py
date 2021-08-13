import os
import matplotlib.pyplot as plt

def save_generations(
    args,
    edges: list, 
    reals: list, 
    fakes: list, 
    paths: list, 
    epoch: str
) -> None:
    if args.results_dir not in os.listdir(args.root_dir, ):
        os.mkdir(os.path.join(args.root_dir, args.results_dir))

    for edge, real, fake, path in zip(edges, reals, fakes, paths):
        fig, axes = plt.subplots(nrows=1, ncols=3)

        axes[0].imshow(edge); axes[0].axis('off'); axes[0].set_title('Edge')
        axes[1].imshow(real); axes[1].axis('off'); axes[1].set_title('Real')
        axes[2].imshow(fake); axes[2].axis('off'); axes[2].set_title('Fake')

        dataset_name = os.path.normpath(path).split(os.sep)[1] + args.model_id
        file_name = os.path.normpath(path).split(os.sep)[-1]
        if dataset_name not in os.listdir(os.path.join(args.root_dir, args.results_dir)):
            os.mkdir(os.path.join(args.root_dir, args.results_dir, dataset_name))
        if epoch not in os.listdir(os.path.join(args.root_dir, args.results_dir, dataset_name)):
            os.mkdir(os.path.join(args.root_dir, args.results_dir, dataset_name, epoch))
        
        fig.savefig(os.path.join(args.root_dir, args.results_dir, dataset_name, epoch, file_name), bbox_inches='tight')
        plt.close()
