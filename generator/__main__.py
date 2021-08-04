import argparse
import warnings
warnings.filterwarnings("ignore")
from .options.full_options import FullOptions
from .trainer import run_model

def main():
    args = FullOptions().parse()
    run_model(args)

if __name__=="__main__":
    main()