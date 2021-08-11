import os
import argparse
from glob import glob
from tqdm.auto import tqdm
from PIL import Image
import cv2
import scipy.io as sio
from .hed import HEDModel

def hed(args):
    paths = sorted(glob(os.path.join(args.input_dir, '*.jpg')))
    path_itr = tqdm(paths, desc="Iteration", position=0, leave=True)
    
    hed_model = HEDModel()
    for path in path_itr:
        im = cv2.imread(path)
        hed = hed_model.compute_hed(im)
        filename = os.path.normpath(path).split(os.sep)[-1][:-4]
        # cv2.imwrite(os.path.join(args.output_dir, filename), hed)
        sio.savemat(os.path.join(args.output_dir, f'{filename}.mat'), {'edge_predict': hed})

def crop_and_resize(args):
    paths = sorted(glob(os.path.join(args.input_dir, '*.jpg')))
    path_itr = tqdm(paths, desc="Iteration", position=0, leave=True)

    for path in path_itr:
        im = Image.open(path)
        w, h = im.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        right = (w + crop_size) // 2
        top = (h - crop_size) // 2
        bottom = (h + crop_size) // 2
        im = im.crop((left, top, right, bottom))
        im = im.resize((args.resize_size, args.resize_size))
        filename = os.path.normpath(path).split(os.sep)[-1]
        im.save(os.path.join(args.output_dir, filename))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,  required=True, help='Input image directory')
    parser.add_argument('--output_dir', type=str,  required=True, help='Output image directory')
    parser.add_argument('--task', type=str, required=True, choices=['crop_and_resize', 'hed', 'poshed'])
    parser.add_argument('--resize_size', type=int, help='Output image size for task crop_and_resize')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    globals()[args.task](args)

if __name__=="__main__":
    main()
