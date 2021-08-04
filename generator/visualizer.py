import os
import matplotlib.pyplot as plt

def save_generations(
    edges: list, 
    reals: list, 
    fakes: list, 
    paths: list, 
    save_dir: str,
    epoch: int
) -> None:
    if save_dir not in os.listdir():
        os.mkdir(save_dir)

    for edge, real, fake, path in zip(edges, reals, fakes, paths):
        fig, axes = plt.subplots(nrows=1, ncols=3)

        axes[0].imshow(edge); axes[0].axis('off'); axes[0].set_title('Edge')
        axes[1].imshow(real); axes[1].axis('off'); axes[1].set_title('Real')
        axes[2].imshow(fake); axes[2].axis('off'); axes[2].set_title('Fake')

        dataset_name = os.path.normpath(path).split(os.sep)[1]
        file_name = os.path.normpath(path).split(os.sep)[-1]
        if dataset_name not in os.listdir(save_dir):
            os.mkdir(os.path.join(save_dir, dataset_name))
        if str(epoch) not in os.listdir(os.path.join(save_dir, dataset_name)):
            os.mkdir(os.path.join(save_dir, dataset_name, epoch))
        fig.savefig(os.path.join(save_dir, dataset_name, epoch, file_name), bbox_inches='tight')
