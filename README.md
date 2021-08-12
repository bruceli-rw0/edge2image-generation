# Edge2Pic

## Installation
Run the following command to setup the virtual environment and install the required packages.
```bash
make setup
```

Run the following command to download the [Kaggle Landscape dataset](https://www.kaggle.com/arnaud58/landscape-pictures) with edge images.
```bash
make download
```

## Usage
Run the following command to generate edges
```bash
./env/bin/python3 -m image_processing \
    --task hed \
    --input_dir DIRECTORY_TO_IMAGES \
    --output_dir OUTPUT_DIRECTORY
```
Then run the following commands in matlab to perform postprocessing
```matlab
addpath(genpath('toolbox/')); savepath; toolboxCompile;
PostprocessHED(
    PATH_TO_HED_EDGES, 
    OUTPUT_DIRECTORY, 
    1024, 25.0/255.0, 5
)
```

Run the following command to perform a test training and evaluation run.
```bash
make test
```

## Resource
### Models
* [Official PyTorch Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD)

### Datasets
Simple Edges
* Edges2shoes: https://www.kaggle.com/balraj98/edges2shoes-dataset
* Edges2handbags

Landscape
* (4K) Landscape: https://www.kaggle.com/arnaud58/landscape-pictures
* (90K) Aligning Latent and Image Spaces to Connect the Unconnectable: https://github.com/universome/alis
* Flickr Landscape (need web scrap): https://www.flickr.com/groups/landcape/

Other Style
* Abstract painting: https://www.kaggle.com/flash10042/abstract-paintings-dataset

### Latest Matlab Piotr's Computer Vision Matlab Toolbox
* https://github.com/jmbuena/toolbox.badacost.public 

## Reference
* Holistically-Nested Edge Detection: https://github.com/s9xie/hed
* Holistically-Nested Edge Detection with OpenCV and Deep Learning: https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/
