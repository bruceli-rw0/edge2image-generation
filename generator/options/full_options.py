# import argparse
import configargparse
import os
import torch
# from .. import data
from .. import models
# from util import util

class FullOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        # parser.add('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add('--gpu_ids', type=int, default=[], nargs='+', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        # model parameters
        parser.add('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add('--netG', type=str, default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128'], help='specify generator architecture')
        parser.add('--netD', type=str, default='basic', choices=['basic', 'n_layers', 'pixel'], help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add('--norm', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization')
        parser.add('--init_type', type=str, default='normal', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add('--use_dropout', action='store_true', help='use dropout for the generator')
        
        # dataset parameters
        parser.add('--direction', type=str, default='BtoA', help='AtoB or BtoA')
        parser.add('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add('--load_size', type=int, default=286, help='scale images to this size')
        parser.add('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        
        # additional parameters
        parser.add('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add('--suffix', type=str, default='', help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add('--verbose', action='store_true', help='if specified, print more debugging information')

        # visdom and HTML visualization parameters
        parser.add('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add('--display_id', type=int, default=1, help='window id of the web display')
        parser.add('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        
        # network saving and loading parameters
        parser.add('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add('--phase', type=str, default='train', help='train, val, test, etc')
        
        # training parameters
        parser.add('--batch_size', type=int, default=1, help='input batch size')
        parser.add('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. Vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # inference parameters
        # parser.add('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        # parser.add('--phase', type=str, default='test', help='train, val, test, etc')
        
        # Dropout and Batchnorm has different behavior during training and test.
        # parser.add('--eval', action='store_true', help='use eval mode during test time.')
        # parser.add('--num_test', type=int, default=50, help='how many test images to run')

        # rewrite devalue values
        # parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        # parser.set_defaults(load_size=parser.get_default('crop_size'))

        # self.isTrain = True

        #----------------------------------------------------------------------#
        #--------------------------- new parameters ---------------------------#
        #----------------------------------------------------------------------#
        parser.add('--do_train', action='store_true', help='Train model.')
        parser.add('--do_eval', action='store_true', help='Generate images after each training epoch.')
        
        parser.add('--num_train', type=int, default=9, help='Number of images used for training, -1 means use all.')
        parser.add('--num_eval', type=int, default=3, help='Number of images used for evalutation, -1 means use all.')
        parser.add("--train_folder", type=str, default=["datasets/edges2shoes/train"], nargs='+', help='Directory of training data.')
        parser.add('--eval_folder', type=str, default=['datasets/edges2shoes/val'], nargs='+', help='Directory of evaluation data.')
        # parser.add('--eval_result', type=str, default='_results', help='Where to save the inference output')

        parser.add('--root_dir', type=str, default='.')
        parser.add('--checkpoint_dir', type=str, default='_checkpoints')
        parser.add('--metrics_dir', type=str, default='_metrics')
        parser.add('--results_dir', type=str, default='_results')

        parser.add("--save_stats", action='store_true')
        parser.add("--save_log", action='store_true')
        parser.add("--verbose_log", action='store_true')
        # parser.add("--save_model", action='store_true')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """
        Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = configargparse.ArgumentParser()
            parser.add('--config', type=str, required=True, is_config_file=True, help='config file path')
            parser = self.initialize(parser)

        # get the basic options
        args, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = args.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, args.do_train)
        args, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        # dataset_name = opt.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_name)
        # parser = dataset_option_setter(parser, self.do_train)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    # def print_options(self, opt):
    #     """Print and save options

    #     It will print both current options and default values(if different).
    #     It will save options into a text file / [checkpoints_dir] / opt.txt
    #     """
    #     message = ''
    #     message += '----------------- Options ---------------\n'
    #     for k, v in sorted(vars(opt).items()):
    #         comment = ''
    #         default = self.parser.get_default(k)
    #         if v != default:
    #             comment = '\t[default: %s]' % str(default)
    #         message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    #     message += '----------------- End -------------------'
    #     print(message)

    #     # save to the disk
    #     expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    #     util.mkdirs(expr_dir)
    #     file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    #     with open(file_name, 'wt') as opt_file:
    #         opt_file.write(message)
    #         opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        args = self.gather_options()

        # process args.suffix
        if args.suffix:
            suffix = ('_' + args.suffix.format(**vars(args))) if args.suffix != '' else ''
            args.name = args.name + suffix
        # self.print_options(args)

        # set gpu ids
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_ids[0])

        self.args = args
        return self.args, self.parser
