from .base_opt import *

class Pix2PixHDOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        BaseOptions.initialize(self)

        preprocess_choices = ['none', 'resize_and_crop', 'crop', 'scale_width', 'scale_width_and_crop']

        # generator parameters
        self.parser.add('--netG', type=str, default='global', help='selects model to use for netG')
        # local enhancer
        self.parser.add('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        # global generator
        self.parser.add('--n_downsample_global', type=int, default=4, help='number of downsampling layers in global generator') 
        self.parser.add('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

        # discriminator parameters
        self.parser.add('--num_D', type=int, default=3, help='number of discriminators to use')
        self.parser.add('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        self.parser.add('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')

        # for instance-wise features
        self.parser.add('--use_instance_map', action='store_true', help='if specified, do *not* add instance map as input')
        self.parser.add('--use_instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add('--use_label_feat', action='store_true', help='if specified, add encoded label features as input')
        self.parser.add('--feat_num', type=int, default=3, help='vector length for encoded features')
        self.parser.add('--n_clusters', type=int, default=10, help='number of clusters for features')
        self.parser.add('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        # encoder
        self.parser.add('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        self.parser.add('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')

        # experiment specifics
        self.parser.add('--fp16', action='store_true', default=False, help='train with AMP')

        # training parameters
        # input data
        self.parser.add('--label_nc', type=int, default=0, help='# of input label channels')
        self.parser.add('--resize_or_crop', type=str, default='none', help='scaling and cropping of images at load time', choices=preprocess_choices)
