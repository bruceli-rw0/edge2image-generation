import os
import warnings
warnings.filterwarnings("ignore")
from .options.full_options import FullOptions
from .trainer import run_model

def main():
    args, parser = FullOptions().parse()
    run_model(args)
    parser.write_config_file(
        parsed_namespace=parser.parse_known_args(),
        output_file_paths=[os.path.join(args.root_dir, args.checkpoint_dir, f'{args.model}{args.model_id}', 'config.yaml')],
        exit_after=True
    )

if __name__=="__main__":
    main()
