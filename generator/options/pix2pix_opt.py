from .base_opt import *

class Pix2PixOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        BaseOptions.initialize(self)

        netG_choices = ['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128']
        netD_choices = ['basic', 'n_layers', 'pixel']
        objective_choices = ['vanilla', 'lsgan', 'wgangp']

        # generator parameters
        self.parser.add('--netG', type=str, default='unet_256', help='specify generator architecture', choices=netG_choices)

        # discriminator parameters
        self.parser.add('--netD', type=str, default='basic', help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator', choices=netD_choices)

        # training parameters
        self.parser.add('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. Vanilla GAN loss is the cross-entropy objective used in the original GAN paper.', choices=objective_choices)
        self.parser.add('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
