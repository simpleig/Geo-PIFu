import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
import pdb # pdb.set_trace()

class VrnNet(nn.Module):

    def __init__(self, opt, projection_mode='orthogonal'):
        super(VrnNet, self).__init__()

        # ----- init. -----

        self.name = 'vrn'
        self.opt = opt
        self.im_feat_list = []                           # a list of deep voxel features
        self.intermediate_preds_list = []                # a list of estimated occupancy grids
        self.intermediate_3d_gan_pred_fake_gen = []      # a list of 3d gan discriminator output on fake est., for training generator
        self.intermediate_3d_gan_pred_fake_dis = []      # a list of 3d gan discriminator output on fake est., for training discriminator
        self.intermediate_3d_gan_pred_real_dis = None    # a BATCH of 3d gan discriminator output on real gt., for training discriminator
        self.intermediate_render_list = []               # a list of rendered rgb images
        self.intermediate_pseudo_inverseDepth_list = []  # a list of pseudo inverse-depth maps
        self.intermediate_render_discriminator_list = [] # a list of patch-GAN discriminator values of the rendered rgb images

        # ----- generate deep voxels -----
        if True:

            # (BV,3,512,512) | resize                                       | (BV,3,384,384)
            # (BV,3,384,384) | crop                                         | (BV,3,384,256)
            # (BV,3,384,256) | conv2d(k7,s2,p3,i3,o64,b), GN(g32,c64), ReLU | (BV,64,192,128)
            c_len_1 = 64
            self.conv1 = nn.Sequential(nn.Conv2d(3,c_len_1,kernel_size=7,stride=2,padding=3), nn.GroupNorm(32,c_len_1), nn.ReLU(True))        

            # (BV,64,192,128)  | residual_block(i64,o128,GN) | (BV,128,192,128)
            # (BV,128,192,128) | avg_pool2d(k2,s2)           | (BV,128,96,64)
            c_len_2 = 128
            self.conv2 = ConvBlock(c_len_1,c_len_2,self.opt.norm)

            # (BV,128,96,64) | residual_block(i128,o128,GN) | (BV,128,96,64)
            # (BV,128,96,64) | avg_pool2d(k2,s2)            | (BV,128,48,32)
            c_len_3 = 128
            self.conv3 = ConvBlock(c_len_2,c_len_3,self.opt.norm)

            # (BV,128,48,32) | residual_block(i128,o128,GN) | (BV,128,48,32)
            # (BV,128,48,32) | residual_block(i128,o256,GN) | (BV,256,48,32)
            c_len_4 = 128
            self.conv4 = ConvBlock(c_len_3,c_len_4,self.opt.norm)
            c_len_5 = 256
            self.conv5 = ConvBlock(c_len_4,c_len_5,self.opt.norm)

            # (BV,256,48,32) | 4-stack-hour-glass | BCDHW of (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32)
            c_len_deepvoxels = 8
            for hg_module in range(self.opt.vrn_num_modules): # default: 4

                self.add_module('m' + str(hg_module), HourGlass(1, opt.num_hourglass, 256, self.opt.norm, self.opt.upsample_mode))

                self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.opt.norm))
                self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                if self.opt.norm == 'batch':
                    self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
                elif self.opt.norm == 'group':
                    self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
                
                self.add_module("branch_out_3d_unet" + str(hg_module), Unet3D(c_len_in=c_len_deepvoxels, c_len_out=c_len_deepvoxels)) # in-(BV,8,32,48,32), out-(BV,8,32,48,32)

                self.add_module('l' + str(hg_module), nn.Conv2d(256, opt.hourglass_dim, kernel_size=1, stride=1, padding=0))

                if hg_module < self.opt.vrn_num_modules - 1:
                    self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                    self.add_module('al' + str(hg_module), nn.Conv2d(opt.hourglass_dim, 256, kernel_size=1, stride=1, padding=0))

        # ----- occupancy classification from deep voxels -----
        if True:

            # (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32)             | upsampling x4                                | (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128)
            # (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128) | conv3d(i8,o8,k3,s1,nb), BN3d, LeakyReLU(0.2) | (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128)
            # (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128) | conv3d(i8,o1,k1,s1,b), sigmoid               | (BV,1,128,192,128), (BV,1,128,192,128), (BV,1,128,192,128), (BV,1,128,192,128)
            self.conv3d_cls_1 = nn.Sequential(Conv3dSame(c_len_deepvoxels,c_len_deepvoxels,kernel_size=3,bias=False), nn.BatchNorm3d(c_len_deepvoxels,affine=True), nn.LeakyReLU(0.2,True))
            self.conv3d_cls_2 = nn.Sequential(Conv3dSame(c_len_deepvoxels,1,kernel_size=1,bias=True), nn.Sigmoid())

        # ----- view prediction from deep voxels -----
        if self.opt.use_view_pred_loss:

            # for each (B,8,128,192,128)                    | voxels rotation to {front/right/back/left} | (B,8,128,192,128)
            # for each (B,8,128,192,128)                    | visibility estimation                      | (B,1,128,192,128)-visibility_weights, (B,1,192,128)-inverse_depth_map
            # for each (B,8,128,192,128), (B,1,128,192,128) | depth dimension reduction                  | (B,8,192,128)
            # for each (B,8,192,128)                        | RGB image rendering by 2D U-Net            | (B,3,384,256)
            # patch-GAN discriminator

            # visibility estimation
            if True:

                # precompute the depth coords, (B=1,1,128,192,128)
                depth_length = self.opt.vrn_net_input_width  // 2 # 128
                depth_height = self.opt.vrn_net_input_height // 2 # 192
                depth_width  = self.opt.vrn_net_input_width  // 2 # 128 
                depth_coords = torch.arange(-depth_length//2, depth_length//2)[None, None, :, None, None].float() / depth_length # shape: [1, 1, 128, 1, 1], value: {-0.5, ..., 0.4821} as depth coords stored in THE middle channel
                depth_coords = depth_coords.repeat(1, 1, 1, depth_height, depth_width) # shape: [B=1, 1, 128, 192, 128], value: {-0.5, ..., 0.4821} as depth coords stored in EACH middle channel
                depth_coords = torch.flip(depth_coords, [2]) # change format from regular-depth to inverse-depth
                self.register_buffer("depth_coords", depth_coords)

                # (B,8+1(+1),128,192,128) | conv3d(i8+1(+1),o8,k3,s1,nb), BN3d, LeakyReL(0.2) | (B,8,128,192,128)
                self.vis_conv3d_1 = nn.Sequential(Conv3dSame(c_len_deepvoxels+1,c_len_deepvoxels,kernel_size=3,bias=False), nn.BatchNorm3d(c_len_deepvoxels,affine=True), nn.LeakyReLU(0.2,True))

                # (B,8,128,192,128)       | conv3d(i8,o1,k3,s1,b), Softmax(dim=2)             | (B,1,128,192,128)
                self.vis_conv3d_2 = nn.Sequential(Conv3dSame(c_len_deepvoxels,1,kernel_size=3,bias=True), nn.Softmax(dim=2))

            # RGB image rendering by 2D U-Net
            if True:

                # return (B,3,384,256) rgb images
                self.rgb_rendering_unet = rgb_rendering_unet(c_len_in=c_len_deepvoxels, c_len_out=3)

                # L1 RGB pixels reconstruction loss
                self.rgb_rendering_loss = nn.L1Loss(reduction='mean') 

            # patch-GAN discriminator
            if self.opt.use_view_discriminator:

                pass

        # ----- 3D GAN loss on the estimated occupancy -----
        if self.opt.use_3d_gan:

            # for each (BV,  1, 128, 192, 128) | conv3d(k4,s2,  i1,  o8,nb), BN3d, LeakyReLU | (BV,  8, 64, 96, 64)
            # for each (BV,  8,  64,  96,  64) | conv3d(k4,s2,  i8,  o8,nb), BN3d, LeakyReLU | (BV,  8, 32, 48, 32)
            # for each (BV,  8,  32,  48,  32) | conv3d(k4,s2,  i8, o16,nb), BN3d, LeakyReLU | (BV, 16, 16, 24, 16)
            # for each (BV, 16,  16,  24,  16) | conv3d(k4,s2, i16, o32,nb), BN3d, LeakyReLU | (BV, 32,  8, 12,  8)
            # for each (BV, 32,   8,  12,   8) | conv3d(k4,s2, i32, o64,nb), BN3d, LeakyReLU | (BV, 64,  4,  6,  4)
            # for each (BV, 64,   4,   6,   4) | conv3d(k4,s2, i64,o128,nb), BN3d, LeakyReLU | (BV,128,  2,  3,  2)
            # for each (BV,128,   2,   3,   2) | conv3d(k3,s1,i128,  o1, b), Sigmoid         | (BV,  1,  2,  3,  2)
            c_len_1 = 8
            self.gan_3d_dis_conv3d_1 = nn.Sequential(nn.ReplicationPad3d(1),  nn.Conv3d(      1,c_len_1,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm3d(c_len_1,affine=True), nn.LeakyReLU(0.2,True))
            c_len_2 = 8
            self.gan_3d_dis_conv3d_2 = nn.Sequential(nn.ReplicationPad3d(1),  nn.Conv3d(c_len_1,c_len_2,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm3d(c_len_2,affine=True), nn.LeakyReLU(0.2,True))
            c_len_3 = 16
            self.gan_3d_dis_conv3d_3 = nn.Sequential(nn.ReplicationPad3d(1),  nn.Conv3d(c_len_2,c_len_3,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm3d(c_len_3,affine=True), nn.LeakyReLU(0.2,True))
            c_len_4 = 32
            self.gan_3d_dis_conv3d_4 = nn.Sequential(nn.ReplicationPad3d(1),  nn.Conv3d(c_len_3,c_len_4,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm3d(c_len_4,affine=True), nn.LeakyReLU(0.2,True))
            c_len_5 = 64
            self.gan_3d_dis_conv3d_5 = nn.Sequential(nn.ReplicationPad3d(1),  nn.Conv3d(c_len_4,c_len_5,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm3d(c_len_5,affine=True), nn.LeakyReLU(0.2,True))
            c_len_6 = 128
            self.gan_3d_dis_conv3d_6 = nn.Sequential(nn.ReplicationPad3d(1),  nn.Conv3d(c_len_5,c_len_6,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm3d(c_len_6,affine=True), nn.LeakyReLU(0.2,True))
            self.gan_3d_dis_conv3d_7 = nn.Sequential(                        Conv3dSame(c_len_6,      1,kernel_size=3,                   bias= True),                                      nn.Sigmoid()          )

            # GAN loss
            self.gan_3d_loss = GANLoss()

        # weights initialization for conv, fc, batchNorm layers
        init_net(self)

    def get_error(self, labels):
        '''
        Hourglass has its own intermediate supervision scheme
        '''

        # init.
        error_dict = {}

        # accumulate errors from all the latent feature layers
        count = 0
        error = 0.
        error_3d_gan_generator, error_3d_gan_discriminator_fake, error_3d_gan_discriminator_real = 0., 0., 0.
        accuracy_3d_gan_discriminator_fake, accuracy_3d_gan_discriminator_real = 0., 0.
        size_fake = self.intermediate_3d_gan_pred_fake_dis[0].nelement() if self.opt.use_3d_gan else 1.
        size_real = self.intermediate_3d_gan_pred_real_dis.nelement()    if self.opt.use_3d_gan else 1.
        for preds in self.intermediate_preds_list:

            # occupancy CE loss, random baseline is: (1000. * -np.log(0.5) / 2. == 346.574), optimal is: 0.
            w = 0.7
            error += self.opt.weight_occu * (   -w  * torch.mean(   labels  * torch.log(  preds+1e-8)) # preds: (BV,1,128,192,128), labels: (BV,1,128,192,128)
                                             -(1-w) * torch.mean((1-labels) * torch.log(1-preds+1e-8))
                                            ) # R

            # generator loss on fake
            if self.opt.use_3d_gan and count == 0:
                error_3d_gan_generator = self.opt.weight_3d_gan_gen * self.gan_3d_loss(self.intermediate_3d_gan_pred_fake_gen[count], True)

            # discriminator loss on fake
            if self.opt.use_3d_gan and count == 0:
                error_3d_gan_discriminator_fake    = self.gan_3d_loss(self.intermediate_3d_gan_pred_fake_dis[count], False)
                accuracy_3d_gan_discriminator_fake = torch.sum(self.intermediate_3d_gan_pred_fake_dis[count].detach() < 0.5, dtype=torch.float32) / size_fake

            # discriminator loss on real
            if self.opt.use_3d_gan and count == 0:
                error_3d_gan_discriminator_real    = self.gan_3d_loss(self.intermediate_3d_gan_pred_real_dis, True)
                accuracy_3d_gan_discriminator_real = torch.sum(self.intermediate_3d_gan_pred_real_dis >= 0.5, dtype=torch.float32) / size_real

            # update count
            count += 1

        # average loss over different latent feature layers
        error_dict["error"] = error / count
        if self.opt.use_3d_gan:
            error_dict["error_3d_gan_generator"]             = error_3d_gan_generator
            error_dict["error_3d_gan_discriminator_fake"]    = error_3d_gan_discriminator_fake
            error_dict["error_3d_gan_discriminator_real"]    = error_3d_gan_discriminator_real
            error_dict["accuracy_3d_gan_discriminator_fake"] = accuracy_3d_gan_discriminator_fake
            error_dict["accuracy_3d_gan_discriminator_real"] = accuracy_3d_gan_discriminator_real

        return error_dict

    def get_error_view_render(self, target_views):

        # init.
        error = 0.
        len_intermediate_list = len(self.intermediate_render_list)

        # accumulate errors from all the latent feature layers 
        for idx in range(len_intermediate_list):

            # L1 RGB pixels reconstruction loss
            error += self.opt.weight_rgb_recon * self.rgb_rendering_loss(self.intermediate_render_list[idx].contiguous().view(-1).float(), target_views.view(-1))

        # average loss over different latent feature layers
        error /= len_intermediate_list

        return error

    def filter(self, images):

        # (BV,3,512,512) | resize                                       | (BV,3,384,384)
        # (BV,3,384,384) | crop                                         | (BV,3,384,256)
        # (BV,3,384,256) | conv2d(k7,s2,p3,i3,o64,b), GN(g32,c64), ReLU | (BV,64,192,128)
        images = F.interpolate(images, size=self.opt.vrn_net_input_height, mode='bilinear', align_corners=True)
        images = images[:,:,:,images.shape[-1]//2-self.opt.vrn_net_input_width//2:images.shape[-1]//2+self.opt.vrn_net_input_width//2]
        images = self.conv1(images)

        # (BV,64,192,128)  | residual_block(i64,o128,GN) | (BV,128,192,128)
        # (BV,128,192,128) | avg_pool2d(k2,s2)           | (BV,128,96,64)
        images = self.conv2(images)
        images = F.avg_pool2d(images, 2, stride=2)

        # (BV,128,96,64) | residual_block(i128,o128,GN) | (BV,128,96,64)
        # (BV,128,96,64) | avg_pool2d(k2,s2)            | (BV,128,48,32)
        images = self.conv3(images)
        images = F.avg_pool2d(images, 2, stride=2)

        # (BV,128,48,32) | residual_block(i128,o128,GN) | (BV,128,48,32)
        # (BV,128,48,32) | residual_block(i128,o256,GN) | (BV,256,48,32)
        images = self.conv4(images)
        images = self.conv5(images)

        # (BV,256,48,32) | 4-stack-hour-glass | BCDHW of (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32)
        previous = images
        self.im_feat_list = []
        for i in range(self.opt.vrn_num_modules): # default: 4

            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll) # (BV,256,48,32)
            assert(tmp_out.shape[1]%tmp_out.shape[-1] == 0)
            tmp_out = tmp_out.view(tmp_out.shape[0], -1, tmp_out.shape[-1], tmp_out.shape[-2], tmp_out.shape[-1]) # (BV,8,32,48,32)
            tmp_out = self._modules['branch_out_3d_unet'+str(i)](tmp_out) # (BV,8,32,48,32)
            if self.training:

                self.im_feat_list.append(tmp_out)
            else:

                if i == (self.opt.vrn_num_modules-1): self.im_feat_list.append(tmp_out)
            tmp_out = tmp_out.view(tmp_out.shape[0],-1,tmp_out.shape[-2],tmp_out.shape[-1]) # (BV,256,48,32)

            if i < (self.opt.vrn_num_modules-1):
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

    def transform_voxels_to_target(self, voxels_CDHW, target_view_idx):

        # input CDHW ---> front CDHW
        if target_view_idx == 0:

            voxels_transformed = voxels_CDHW

        # input CDHW | 0123 ---> right C,-W,H,D | 0,-3,2,1
        elif target_view_idx == 1:

            voxels_transformed = torch.transpose(voxels_CDHW, dim0=1, dim1=3) # CDHW to CWHD
            voxels_transformed = torch.flip(voxels_transformed, [1])          # CWHD to C,-W,H,D

        # input CDHW | 0123 ---> back C,-D,H,-W | 0,-1,2,-3
        elif target_view_idx == 2:

            voxels_transformed = torch.flip(voxels_CDHW, [1,3]) # CDHW to C,-D,H,-W

        # input CDHW | 0123 ---> left C,W,H,-D | 0,3,2,-1
        elif target_view_idx == 3:

            voxels_transformed = torch.transpose(voxels_CDHW, dim0=1, dim1=3) # CDHW to CWHD
            voxels_transformed = torch.flip(voxels_transformed, [3]) # CWHD to C,W,H,-D

        # sanity check
        else:

            print("Error: undifined target_view_idx {}!".format(target_view_idx.item()))
            pdb.set_trace()

        return voxels_transformed

    def visibility_estimation(self, canonical_tensor):

        # (B,8+1(+1),128,192,128) | conv3d(i8+1(+1),o8,k3,s1,nb), BN3d, LeakyReL(0.2) | (B,8,128,192,128)
        # (B,8,128,192,128)       | conv3d(i8,o1,k3,s1,b), softmax(dim=2)             | (B,1,128,192,128)
        depth_coords_batch = self.depth_coords.repeat(canonical_tensor.shape[0], 1, 1, 1, 1) # e.g. (B=1,1,128,192,128) to (B=5,1,128,192,128)
        canonical_tensor   = torch.cat((depth_coords_batch, canonical_tensor), dim=1)
        visibility_weights = self.vis_conv3d_1(canonical_tensor)
        visibility_weights = self.vis_conv3d_2(visibility_weights)

        # (B,1,128,192,128), (B,1,128,192,128) | pseudo inverse-depth maps | (B, 1, 192, 128)
        inverse_depth_map = (depth_coords_batch * visibility_weights).sum(dim=2)

        # (B,1,128,192,128)-visibility_weights, (B, 1, 192, 128)-inverse_depth_map
        return visibility_weights, inverse_depth_map

    def est_occu(self, occupancy_estimation=True, view_directions=None, view_render=False, view_discriminator=False, prepare_3d_gan=True):

        # init.
        self.intermediate_preds_list                = []
        self.intermediate_3d_gan_pred_fake_gen      = []
        self.intermediate_3d_gan_pred_fake_dis      = []
        self.intermediate_3d_gan_pred_real_dis      = None
        self.intermediate_render_list               = []
        self.intermediate_pseudo_inverseDepth_list  = []
        self.intermediate_render_discriminator_list = []

        # for each level of deep voxels inside the stack-hour-glass networks
        max_count = len(self.im_feat_list) - 1
        count = 0
        for im_feat in self.im_feat_list:

            # upsampling x4 the deep voxels
            # (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32), (BV,8,32,48,32)             | upsampling x4                                | (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128)
            deepVoxels_upsampled = F.interpolate(im_feat, scale_factor=4, mode='trilinear', align_corners=True) # (BV,8,128,192,128)

            # ----- occupancy classification from deep voxels -----
            # (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128) | conv3d(i8,o8,k3,s1,nb), BN3d, LeakyReLU(0.2) | (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128)
            # (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128), (BV,8,128,192,128) | conv3d(i8,o1,k1,s1,b), sigmoid               | (BV,1,128,192,128), (BV,1,128,192,128), (BV,1,128,192,128), (BV,1,128,192,128)
            if occupancy_estimation:

                # conv3d(i8,o8,k3,s1,nb), BN3d, LeakyReLU(0.2)
                pred_final = self.conv3d_cls_1(deepVoxels_upsampled) # (BV,8,128,192,128)

                # conv3d(i8,o1,k1,s1,b), sigmoid
                pred_final = self.conv3d_cls_2(pred_final) # (BV,1,128,192,128)
                self.intermediate_preds_list.append(pred_final)

                # pred_fake and pred_real for 3d-gan loss
                if self.opt.use_3d_gan and count == max_count and prepare_3d_gan: # for generator on fake
                    pred_fake = self.gan_3d_dis_conv3d_1(pred_final)
                    pred_fake = self.gan_3d_dis_conv3d_2(pred_fake)
                    pred_fake = self.gan_3d_dis_conv3d_3(pred_fake)
                    pred_fake = self.gan_3d_dis_conv3d_4(pred_fake)
                    pred_fake = self.gan_3d_dis_conv3d_5(pred_fake)
                    pred_fake = self.gan_3d_dis_conv3d_6(pred_fake)
                    pred_fake = self.gan_3d_dis_conv3d_7(pred_fake)
                    self.intermediate_3d_gan_pred_fake_gen.append(pred_fake)
                if self.opt.use_3d_gan and count == max_count and prepare_3d_gan: # for discriminator on fake, NOTE the .detach() function
                    pred_fake = self.gan_3d_dis_conv3d_1(pred_final.detach())
                    pred_fake = self.gan_3d_dis_conv3d_2(pred_fake)          
                    pred_fake = self.gan_3d_dis_conv3d_3(pred_fake)           
                    pred_fake = self.gan_3d_dis_conv3d_4(pred_fake)          
                    pred_fake = self.gan_3d_dis_conv3d_5(pred_fake)   
                    pred_fake = self.gan_3d_dis_conv3d_6(pred_fake)
                    pred_fake = self.gan_3d_dis_conv3d_7(pred_fake)        
                    self.intermediate_3d_gan_pred_fake_dis.append(pred_fake)
                if self.opt.use_3d_gan and count == 0 and prepare_3d_gan: # for discriminator on real
                    pred_real = self.gan_3d_dis_conv3d_1(self.labels) 
                    pred_real = self.gan_3d_dis_conv3d_2(pred_real)   
                    pred_real = self.gan_3d_dis_conv3d_3(pred_real)   
                    pred_real = self.gan_3d_dis_conv3d_4(pred_real)   
                    pred_real = self.gan_3d_dis_conv3d_5(pred_real)
                    pred_real = self.gan_3d_dis_conv3d_6(pred_real)
                    pred_real = self.gan_3d_dis_conv3d_7(pred_real)
                    self.intermediate_3d_gan_pred_real_dis = pred_real

            # ----- view prediction from deep voxels ----- 
            # for each (B,8,128,192,128)                    | voxels rotation to {front/right/back/left} | (B,8,128,192,128)
            # for each (B,8,128,192,128)                    | visibility estimation                      | (B,1,128,192,128)-visibility_weights, (B,1,192,128)-inverse_depth_map
            # for each (B,8,128,192,128), (B,1,128,192,128) | depth dimension reduction                  | (B,8,192,128)
            # for each (B,8,192,128)                        | RGB image rendering by 2D U-Net            | (B,3,384,256)
            # patch-GAN discriminator
            if view_render:

                # voxels rotation to {front/right/back/left}, output sizes are still (B,8,128,192,128)
                deepVoxels_upsampled_canonical = []
                for batchIdx in range(deepVoxels_upsampled.shape[0]):
                    deepVoxels_upsampled_canonical_tmp = self.transform_voxels_to_target(voxels_CDHW=deepVoxels_upsampled[batchIdx], target_view_idx=view_directions[batchIdx])
                    deepVoxels_upsampled_canonical.append(deepVoxels_upsampled_canonical_tmp[None,:,:,:,:])
                deepVoxels_upsampled_canonical = torch.cat(deepVoxels_upsampled_canonical, dim=0) # (B,8,128,192,128)

                # visibility estimation by 3D U-Net: (B,1,128,192,128)-visibility_weights, (B,1,192,128)-inverse_depth_map
                visibility_weights, inverse_depth_map = self.visibility_estimation(canonical_tensor=deepVoxels_upsampled_canonical)
                self.intermediate_pseudo_inverseDepth_list.append(inverse_depth_map)

                # depth dimension reduction: (B,8,128,192,128) to (B,8,192,128)
                render_rgb = torch.mean(visibility_weights * deepVoxels_upsampled, dim=2) # (B,8,192,128)

                # RGB images (B,3,384,256)-render_rgb rendering by 2D U-Net
                render_rgb = self.rgb_rendering_unet(render_rgb)
                self.intermediate_render_list.append(render_rgb)

                # patch-GAN discriminator
                if view_discriminator:

                    print("Error: code for using self.opt.use_view_discriminator has not been implemented yet!")
                    pdb.set_trace()
                    render_rgb_discriminator = None
                    self.intermediate_render_discriminator_list.append(render_rgb_discriminator)

            # update count
            count += 1

    def get_preds(self):

        return self.intermediate_preds_list[-1] # BCDHW, (BV,1,128,192,128), est. occupancy

    def get_renderings(self):

        # (B,3,384,256)-render_rgb, (B,1,192,128)-inverse_depth_map
        return self.intermediate_render_list[-1], self.intermediate_pseudo_inverseDepth_list[-1]

    def forward(self, images, labels=None, view_directions=None, target_views=None):

        # init
        return_dict = {}
        self.labels = labels

        # compute deep voxels
        self.filter(images=images)

        # estimate occupancy grids, (and render prediction)
        self.est_occu(view_directions=view_directions, view_render=self.opt.use_view_pred_loss, view_discriminator=self.opt.use_view_discriminator)

        # get the estimated_occupancy
        return_dict["pred_occ"] = self.get_preds() # BCDHW, (BV,1,128,192,128), est. occupancy

        # get the rendered_rgb_images & computed_pseudo_inverse_depth_map
        if self.opt.use_view_pred_loss: return_dict["render_rgb"], return_dict["pseudo_inverseDepth"] = self.get_renderings()

        # compute occupancy errors
        error = self.get_error(labels=labels) # R, the mean loss value over all latent feature maps of the stack-hour-glass network
        return_dict.update(error)

        # compute view generation errors
        if self.opt.use_view_pred_loss: return_dict["error_view_render"] = self.get_error_view_render(target_views=target_views)

        # return: estimated mesh voxels, error, (and render_rgb, pseudo_inverseDepth, error_view_render), (and error_3d_gan_generator, error_3d_gan_discriminator_fake, error_3d_gan_discriminator_real)
        return return_dict








