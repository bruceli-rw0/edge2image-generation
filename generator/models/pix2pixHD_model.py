import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from ..util.image_pool import ImagePool
from .base_model import BaseModel
from . import networksHD as networks

class Pix2PixHD(BaseModel, nn.Module):
    """
    This cass implements the pix2pixHD model.
    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        nn.Module.__init__(self)

        if opt.resize_or_crop != 'none' or not opt.do_train: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.do_train
        self.use_features = opt.use_instance_feat or opt.use_label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        #----------------------------- Generator ------------------------------#
        netG_input_nc = input_nc
        if opt.use_instance_map:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(
            netG_input_nc, 
            opt.output_nc, 
            opt.ngf, 
            opt.netG, 
            opt.n_downsample_global, 
            opt.n_blocks_global, 
            opt.n_local_enhancers, 
            opt.n_blocks_local, 
            opt.norm, 
            opt.init_type,
            opt.init_gain,
            self.device,
            self.gpu_ids
        )

        #--------------------------- Discriminator ----------------------------#
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if opt.use_instance_map:
                netD_input_nc += 1
            self.netD = networks.define_D(
                netD_input_nc, 
                opt.ndf, 
                opt.n_layers_D, 
                opt.num_D, 
                opt.norm, 
                opt.init_type,
                opt.init_gain,
                use_sigmoid, 
                not opt.no_ganFeat_loss, 
                self.device,
                self.gpu_ids
            )

        #------------------------------ Encoder -------------------------------#
        if self.gen_features:
            self.netE = networks.define_G(
                opt.output_nc, 
                opt.feat_num, 
                opt.nef, 
                'encoder', 
                opt.n_downsample_E, 
                norm=opt.norm, 
                init_type=opt.init_type,
                init_gain=opt.init_gain,
                device=self.device,
                gpu_ids=self.gpu_ids
            )

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            # ImagePool is currently unused
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            #------------------------ loss functions --------------------------#
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, device=self.device)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.device)

            #-------------------------- optimizers ----------------------------#
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('Only training the local enhancer network (for %d epochs)' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def forward(self, label, image, inst=None, feat=None) -> torch.Tensor:
        # Encode Inputs
        input_label, real_image, inst_map, feat_map = self.encode_input(label, image, inst, feat)  
        self.input_label = input_label.to(self.device)
        self.real_image = real_image.to(self.device)
        self.inst_map = inst_map.to(self.device) if inst_map is not None else None
        self.feat_map = feat_map.to(self.device) if feat_map is not None else None

        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                self.feat_map = self.netE.forward(self.real_image, self.inst_map)
            input_concat = torch.cat((self.input_label, self.feat_map), dim=1)
        else:
            input_concat = self.input_label
        self.fake_image = self.netG.forward(input_concat)
        return self.fake_image

    def optimize_parameters(self) -> None:
        # update discriminator weights
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update generator weights
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def backward_D(self) -> None:
        # Fake
        input = torch.cat((self.input_label, self.fake_image.detach()), dim=1)
        pred_fake_pool = self.netD.forward(self.fake_pool.query(input))
        loss_D_fake = self.criterionGAN(pred_fake_pool, target_is_real=False)

        # Real
        input = torch.cat((self.input_label, self.real_image.detach()), dim=1)
        self.pred_real = self.netD.forward(input)
        loss_D_real = self.criterionGAN(self.pred_real, target_is_real=True)
        
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D = loss_D.item()
        # if self.opt.fp16:
        #     with amp.scale_loss(loss_D, self.optimizer_D) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss_D.backward()
        loss_D.backward()

    def backward_G(self) -> None:
        # Fake the discriminator
        input = torch.cat((self.input_label, self.fake_image), dim=1)
        pred_fake = self.netD.forward(input)
        loss_G_GAN = self.criterionGAN(pred_fake, target_is_real=True)
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], self.pred_real[i][j].detach())
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(self.fake_image, self.real_image)
        
        loss_G = loss_G_GAN + loss_G_GAN_Feat * self.opt.lambda_feat + loss_G_VGG * self.opt.lambda_feat
        self.loss_G = loss_G.item()
        # if self.opt.fp16:
        #     with amp.scale_loss(loss_G, self.optimizer_G) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss_G.backward()
        loss_G.backward()

    def load(self):
        self.load_network(self.netG, 'G', self.opt.load_epoch)
        if self.isTrain:
            self.load_network(self.netD, 'D', self.opt.load_epoch)
        if self.gen_features:
            self.load_network(self.netE, 'E', self.opt.load_epoch)

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    #--------------------------------------------------------------------------#
    #--------------------------- Helper Functions -----------------------------#
    #--------------------------------------------------------------------------#

    def encode_input(self, label_map, real_image=None, inst_map=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data
        else:
            # create one-hot vector for label map 
            N, _, H, W = label_map.size()
            oneHot_size = (N, self.opt.label_nc, H, W)
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if self.opt.use_instance_map:
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features: # USE precomputed feature maps
                feat_map = Variable(feat_map.data)
            if self.opt.label_feat: # add encoded label features as input
                inst_map = label_map

        return input_label, real_image, inst_map, feat_map

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        edge = edge.to(self.device)
        return edge.float()

    #--------------------------------------------------------------------------#

    def inference(self, label):
        label = label.to(self.device)
        self.netG.eval()
        with torch.no_grad():
            return self.netG.forward(label)

    # def inference(self, label, inst, image=None):
    #     # Encode Inputs        
    #     image = Variable(image) if image is not None else None
    #     input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

    #     # Fake Generation
    #     if self.use_features:
    #         if self.opt.use_encoded_image:
    #             # encode the real image to get feature map
    #             feat_map = self.netE.forward(real_image, inst_map)
    #         else:
    #             # sample clusters from precomputed features
    #             feat_map = self.sample_features(inst_map)
    #         input_concat = torch.cat((input_label, feat_map), dim=1)
    #     else:
    #         input_concat = input_label        
           
    #     if torch.__version__.startswith('0.4'):
    #         with torch.no_grad():
    #             fake_image = self.netG.forward(input_concat)
    #     else:
    #         fake_image = self.netG.forward(input_concat)
    #     return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        N, _, H, W = inst_np.size()
        feat_map = torch.FloatTensor(N, self.opt.feat_num, H, W).to(self.device)
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature
