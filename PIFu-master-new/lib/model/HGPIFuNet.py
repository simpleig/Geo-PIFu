import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier, deepVoxelsFusionNetwork, SurfaceClassifier_multiLoss
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
import pdb # pdb.set_trace()

class HGPIFuNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self, opt, projection_mode='orthogonal', error_term=nn.MSELoss()):

        if opt.occupancy_loss_type == 'l1':
            error_term = nn.L1Loss()
        elif opt.occupancy_loss_type == 'mse':
            error_term = nn.MSELoss()
        elif opt.occupancy_loss_type == 'ce':
            error_term = None
        else:
            print("Error: occupancy loss type is not defined {}!".format(opt.occupancy_loss_type))
            pdb.set_trace()

        super(HGPIFuNet, self).__init__(projection_mode=projection_mode, error_term=error_term)

        self.name = 'hgpifu' if opt.deepVoxels_fusion==None else "dvif_"+opt.deepVoxels_fusion

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)

        if opt.deepVoxels_fusion in ["early", "late"]:
            self.opt.mlp_dim[0] += opt.deepVoxels_c_len

        if opt.deepVoxels_fusion == "multiLoss":
            self.surface_classifier = SurfaceClassifier_multiLoss(opt=opt,
                                                                  filter_channels_2d=self.opt.mlp_dim,
                                                                  filter_channels_3d=self.opt.mlp_dim_3d,
                                                                  filter_channels_joint=self.opt.mlp_dim_joint)
        else:
            self.surface_classifier = SurfaceClassifier(
                filter_channels=self.opt.mlp_dim, # default: [257(+len of deepVoxels' features), 1024, 512, 256, 128, 1]
                num_views=self.opt.num_views,
                no_residual=self.opt.no_residual, # default: False
                last_op=nn.Sigmoid(), # output float 0. ~ 1., occupancy
                opt=opt)

        if opt.deepVoxels_fusion == "late":
            self.deepVoxels_fusion_network = deepVoxelsFusionNetwork(c_len_in=opt.deepVoxels_c_len_intoLateFusion, c_len_out=opt.deepVoxels_c_len)

        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        # XYZ-3-direction displacements, for multiRanges_deepVoxels sampling
        if self.opt.multiRanges_deepVoxels:
            displacments = []
            displacments.append([0, 0, 0])
            for x in range(3):
                for y in [-1, 1]:
                    input = [0, 0, 0]
                    input[x] = y * self.opt.displacment
                    displacments.append(input)
            self.register_buffer("displacments", torch.Tensor(displacments))

        # weights initialization for conv, fc, batchNorm layers
        init_net(self)

    def filter(self, images):
        '''
        Filter the input images, store all intermediate features.

        Input
            images: [B * num_views, C, H, W] input images, float -1 ~ 1, RGB

        Output
            im_feat_list: [(B * num_views, opt.hourglass_dim, H/4, W/4), (same_size), (same_size), (same_size)], list length is opt.num_stack, e.g. (2, 256, 128, 128) each entry
            tmpx        :  (B * num_views, 64, H/2, W/2), e.g. (2, 64, 256, 256), detached, thus self.tmpx.requires_grad is False
            normx       :  (B * num_views, 128, H/4, W/4), e.g. (2, 128, 128, 128)
        '''

        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)

        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, transforms=None, labels=None, deepVoxels=None):
        '''
        Given 3D points, query the network predictions for each point. Image features should be pre-computed before this call. store all intermediate features.
        query() function may behave differently during training/testing.

        :param points: (B * num_views, 3, n_in+n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
        :param calibs: (B * num_views, 4, 4) calibration matrix
        :param transforms: default is None
        :param labels: (B, 1, n_in+n_out), float 1.0-inside, 0.0-outside

        :return: [B, Res, n_in+n_out] predictions for each point
        '''

        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms) # (B * num_view, 3, N), points are projected onto XY (-1,1)x(-1,1) plane of the cam coord.
        xy = xyz[:, :2, :] # (B * num_view, 2, N)
        z = xyz[:, 2:3, :] # (B * num_view, 1, N)

        # (B * num_view, N), True-inFoV, False-outFoV
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        # print("should add | among views..."); pdb.set_trace()
        in_img = in_img.view(-1, self.opt.num_views, in_img.shape[1]).float() # (B, num_views, N), 1-inFov, 0-outFov
        in_img, _ = torch.max(in_img, dim=1) # (B, N), 1-inFov, 0-outFov

        # there are two ways to normalize z, I assume PIFU is using the first method
        # z value range length:scale/ortho_ratio/float(self.opt.loadSize // 2)*256 = 2.4049856956795232
        # z_feat value range length: 2.4049856956795232 * 256 / 200 = 3.0783816904697896
        # 1) always make sure that cam-center and world-center depth distance are the same during rendering, so that z will be consistently in the same range for different views
        # 2) otherwise we should use calibs to figure out cam-center and world center distance, and set them to the same value (e.g. centers aligned with zero translation) so that z can stay in the same range for different views
        z_feat = self.normalizer(z, calibs=calibs) # z * 256 / 200, (B * num_view, 1, N), float, roughly -2 ~ 2

        # default is False
        if self.opt.skip_hourglass:

            tmpx_local_feature = self.index(self.tmpx, xy) # torch.nn.functional.grid_sample, in geometry.py

        self.intermediate_preds_list = [] # default length is 4, opt.num_stack
        if self.opt.deepVoxels_fusion == "multiLoss":
        
            for im_feat in self.im_feat_list:

                # 2d features: [(B * num_views, opt.hourglass_dim, n_in+n_out), (B * num_view, 1, n_in+n_out)]
                point_local_feat_list = [self.index(im_feat, xy), z_feat] # torch.nn.functional.grid_sample, in geometry.py
                point_local_feat = torch.cat(point_local_feat_list, 1)

                # 3d features
                if self.opt.multiRanges_deepVoxels:
                    # tri-linear sampling: (BV,3,N) + (B,C=8,D=32,H=48,W=32) ---> (BV,C=56,N)
                    features_3D = self.multiRanges_deepVoxels_sampling(feat=deepVoxels, XYZ=torch.cat([xy, -1. * z], dim=1), displacments=self.displacments)
                else:
                    # tri-linear sampling: (BV,3,N) + (B,C=8,D=32,H=48,W=32) ---> (BV,C=8,N)
                    features_3D = self.index_3d(feat=deepVoxels.transpose(0,1), XYZ=torch.cat([xy, -1. * z], dim=1))
        
                # predict sdf
                pred_sdf_list = self.surface_classifier(feature_2d=point_local_feat, feature_3d=features_3D)
                for pred in pred_sdf_list:
                    pred_visible = in_img[:,None].float() * pred
                    self.intermediate_preds_list.append(pred_visible)

        else:
        
            for im_feat in self.im_feat_list:

                # [(B * num_views, opt.hourglass_dim, n_in+n_out), (B * num_view, 1, n_in+n_out)]
                point_local_feat_list = [self.index(im_feat, xy), z_feat] # torch.nn.functional.grid_sample, in geometry.py

                # deepVoxels' features
                if self.opt.deepVoxels_fusion != None:

                    if self.opt.multiRanges_deepVoxels:
                        # tri-linear sampling: (BV,3,N) + (B,C=8,D=32,H=48,W=32) ---> (BV,C=56,N)
                        features_3D = self.multiRanges_deepVoxels_sampling(feat=deepVoxels, XYZ=torch.cat([xy, -1. * z], dim=1), displacments=self.displacments)
                    else:
                        # tri-linear sampling: (BV,3,N) + (B,C=8,D=32,H=48,W=32) ---> (BV,C=8,N)
                        features_3D = self.index_3d(feat=deepVoxels.transpose(0,1), XYZ=torch.cat([xy, -1. * z], dim=1))

                    # if use late fusion
                    if self.opt.deepVoxels_fusion == "late": features_3D = self.deepVoxels_fusion_network(features_3D)

                    # concatnate with 2D-features and Depth-z
                    point_local_feat_list.insert(0, features_3D)

                # default is False
                if self.opt.skip_hourglass:

                    point_local_feat_list.append(tmpx_local_feature)

                # (B * num_views, opt.hourglass_dim+1, n_in+n_out)
                point_local_feat = torch.cat(point_local_feat_list, 1)

                # out of image plane is always set to 0
                pred = in_img[:,None].float() * self.surface_classifier(point_local_feat) # (B, 1, n_in+n_out) == (B, 1, n_in+n_out) * {(B, 1, n_in+n_out), num_views are canceled out by mean pooling}
                self.intermediate_preds_list.append(pred)

        # at inference time, we only rely on the last layer features from stacked-hour-glass-networks
        self.preds = self.intermediate_preds_list[-1] # (B * num_view, 1, n_in+n_out)

    def get_im_feat(self):
        '''
        Get the image filter
        :return: (BV, 256, 128, 128) image feature after filtering
        '''

        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''

        # accumulate errors from all the latent feature layers 
        error = 0
        for preds in self.intermediate_preds_list:
            
            if self.opt.occupancy_loss_type == "ce":

                # occupancy CE loss, random baseline is: (1000. * -np.log(0.5) / 2. == 346.574), optimal is: 0.
                w = 0.7
                error += 1000. * (   -w  * torch.mean(    self.labels * torch.log(  preds+1e-8)) # preds: (B,1,5000), self.labels: (B,1,5000)
                                  -(1-w) * torch.mean((1-self.labels) * torch.log(1-preds+1e-8))
                                 ) # R
            else:

                error += self.error_term(preds, self.labels) # default is nn.MSELoss()

        # average loss over different latent feature layers
        error /= len(self.intermediate_preds_list)

        return error

    def forward(self, images, points, calibs, transforms=None, labels=None, deepVoxels=None):
        """
        input
            images    : (B * num_views, C, W, H) RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            points    : (B * num_views, 3, n_in+n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
            calibs    : (B * num_views, 4, 4) calibration matrix
            labels    : (B, 1, n_in+n_out), float 1.0-inside, 0.0-outside
            transforms: default is None

        return
            res  : (B==2, 1, n_in + n_out) occupancy estimation of "points", float 0. ~ 1.
            error: R, occupancy loss
        """

        # Phase 1: get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels, deepVoxels=deepVoxels)

        # Phase 3: get the prediction
        res = self.get_preds() # return self.preds, (B, 1, n_in+n_out)
        
        # Phase 4: get the error, default is nn.MSELoss()
        error = self.get_error() # R

        return res, error








