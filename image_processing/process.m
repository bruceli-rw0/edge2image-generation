addpath(genpath('toolbox/')); savepath; toolboxCompile;
PostprocessHED('../datasets/kaggle_landscape/hed', '../datasets/kaggle_landscape/edges', 1024, 25.0/255.0, 5)
