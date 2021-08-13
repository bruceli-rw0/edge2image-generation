import os
import importlib
import warnings
warnings.filterwarnings("ignore")

from .options import BaseOptions
from .trainer import run_model

def main():
    args, _ = BaseOptions().parse(known=True)

    model_filename = "generator.options." + args.model + "_opt"
    modellib = importlib.import_module(model_filename)
    for name, cls in modellib.__dict__.items():
        if name.lower() == (args.model + "options").lower() and issubclass(cls, BaseOptions):
            option_class = cls

    args, parser = option_class().parse()

    run_model(args)

    # save config file
    parser.write_config_file(
        parsed_namespace=parser.parse_known_args(),
        output_file_paths=[os.path.join(
            args.root_dir, 
            args.checkpoints_dir, 
            f'{args.model}{args.model_id}', 
            'config.yaml'
        )],
        exit_after=True
    )

if __name__=="__main__":
    main()
