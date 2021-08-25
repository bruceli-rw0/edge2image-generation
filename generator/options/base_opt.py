# import argparse
import configargparse
import torch

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.parser = configargparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        """Define the common options that are used in both training and test."""
        device_choices = ['cpu', 'gpu', 'tpu']
        model_choices = ['pix2pix', 'pix2pixHD']
        norm_choices = ['instance', 'batch', 'none']
        init_choices = ['normal', 'xavier', 'kaiming', 'orthogonal']
        lr_policy_choices = ['linear', 'step', 'plateau', 'cosine']
        preprocess_choices = ['resize_and_crop', 'crop', 'scale_width', 'scale_width_and_crop', 'none']

        # experiment specifics
        self.parser.add('--config', type=str, required=True, is_config_file=True, help='config file path')
        self.parser.add('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add('--device', type=str, default='cpu', help='device for running model', choices=device_choices)
        self.parser.add('--gpu_ids', type=int, default=[0], nargs='+', help='gpu ids: e.g. 0; 0 1 2; 0 2.')
        self.parser.add('--root_dir', type=str, default='.', help='root directory for saving log, metrics, results, and checkpoints')
        self.parser.add('--log_dir', type=str, default='_loggings', help='saves logging here')
        self.parser.add('--metrics_dir', type=str, default='_metrics', help='saves running metrics here')
        self.parser.add('--results_dir', type=str, default='_results', help='saves results here')
        self.parser.add('--checkpoints_dir', type=str, default='_checkpoints', help='saves model checkpoints here')
        self.parser.add('--verbose', action='store_true', help='if specified, print more debugging information')
        
        self.parser.add('--do_train', action='store_true', help='whether train model.')
        self.parser.add('--do_eval', action='store_true', help='whether to generate images after each training epoch.')
        self.parser.add('--model_id', type=str, default='', help='the unique identification of the model')

        self.parser.add('--num_train', type=int, default=-1, help='Number of images used for training, -1 means use all.')
        self.parser.add('--num_eval', type=int, default=-1, help='Number of images used for evalutation, -1 means use all.')
        self.parser.add("--train_folder", type=str, default=["datasets/edges2shoes/train"], nargs='+', help='Directory of training data.')
        self.parser.add('--eval_folder', type=str, default=['datasets/edges2shoes/val'], nargs='+', help='Directory of evaluation data.')
        
        # model parameters
        self.parser.add('--model', type=str, default='pix2pix', help='which model to use.', choices=model_choices)
        self.parser.add('--norm', type=str, default='instance', help='instance or batch normalization', choices=norm_choices)
        self.parser.add('--init_type', type=str, default='normal', help='network initialization', choices=init_choices)
        self.parser.add('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # generator parameters
        self.parser.add('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        self.parser.add('--use_dropout', action='store_true', help='use dropout for the generator')
        # discriminator parameters
        self.parser.add('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        self.parser.add('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        self.parser.add('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        # loading parameters
        self.parser.add('--load_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        self.parser.add('--suffix', type=str, default='', help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # training parameters
        # input data
        self.parser.add('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add('--load_size', type=int, default=256, help='scale images to this size')
        self.parser.add('--crop_size', type=int, default=256, help='then crop to this size')
        self.parser.add('--input_nc', type=int, default=3, help='# input image channels: 3 for RGB and 1 for grayscale')
        self.parser.add('--output_nc', type=int, default=3, help='# output image channels: 3 for RGB and 1 for grayscale')
        # hyperparameters
        self.parser.add('--n_epochs', type=int, default=2, help='number of training epoch')
        self.parser.add('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add('--lr_policy', type=str, default='linear', help='learning rate policy.', choices=lr_policy_choices)
        self.parser.add('--lr_fix_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        self.parser.add('--lr_decay_epochs', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        self.parser.add('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # dataset parameters
        # parser.add('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        self.parser.add('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add('--num_threads', default=4, type=int, help='# threads for loading data')
        self.parser.add('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time', choices=preprocess_choices)
        self.parser.add('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        # network saving and loading parameters
        # self.parser.add('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add('--save_epoch_freq', type=int, default=4, help='frequency of saving checkpoints at the end of epochs')
        # self.parser.add('--save_by_iter', action='store_true', help='whether saves model by iteration')
        self.parser.add('--continue_train', action='store_true', help='continue training: load the latest model')
        # self.parser.add('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        # self.parser.add('--phase', type=str, default='train', help='train, val, test, etc')
        
        # visdom and HTML visualization parameters
        # self.parser.add('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # self.parser.add('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        # self.parser.add('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        # self.parser.add('--display_id', type=int, default=1, help='window id of the web display')
        # self.parser.add('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        # self.parser.add('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        # self.parser.add('--display_port', type=int, default=8097, help='visdom port of the web display')
        # self.parser.add('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        # self.parser.add('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        # self.parser.add('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        
        # inference parameters
        # self.parser.add('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        # parser.add('--phase', type=str, default='test', help='train, val, test, etc')
        
        # Dropout and Batchnorm has different behavior during training and test.
        # parser.add('--eval', action='store_true', help='use eval mode during test time.')
        # parser.add('--num_test', type=int, default=50, help='how many test images to run')

        # rewrite devalue values
        # parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        # parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.initialized = True

    def parse(self, known=False):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        if not self.initialized:  # check if it has been initialized
            self.initialize()
        if known:
            return self.parser.parse_known_args()
        
        args, _ = self.parser.parse_known_args()
        # process args.suffix
        if args.suffix:
            suffix = ('_' + args.suffix.format(**vars(args))) if args.suffix != '' else ''
            args.name = args.name + suffix
        return self.parser.parse_args(), self.parser
