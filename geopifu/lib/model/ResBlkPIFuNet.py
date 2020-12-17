import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
import functools
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from ..net_util import *
import pdb

class ResBlkPIFuNet(BasePIFuNet):
    def __init__(self, opt,
                 projection_mode='orthogonal'):
        if opt.color_loss_type == 'l1':
            error_term = nn.L1Loss()
        elif opt.color_loss_type == 'mse':
            error_term = nn.MSELoss()

        super(ResBlkPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'respifu'
        self.opt = opt

        norm_type = get_norm_layer(norm_type=opt.norm_color) # default: nn.InstanceNorm2d without {affine, tracked statistics}
        self.image_filter = ResnetFilter(opt, norm_layer=norm_type)

        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim_color, # default: 513, 1024, 512, 256, 128, 3
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual, # default: False
            last_op=nn.Tanh(), # output float -1 ~ 1, RGB colors
            opt=self.opt)

        self.normalizer = DepthNormalizer(opt)

        # weights initialization for conv, fc, batchNorm layers
        init_net(self)

    def filter(self, images):
        '''
        Filter the input images store all intermediate features.

        Input:
            images: (BV, 3, 512, 512) input images

        output:
            self.im_feat: (BV, 256, 128, 128)
        '''

        self.im_feat = self.image_filter(images)

    def attach(self, im_feat):

        self.im_feat = torch.cat([im_feat, self.im_feat], 1)

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point. Image features should be pre-computed before this call. store all intermediate features.
        query() function may behave differently during training/testing.

        Input
            points: (B * num_views, 3, 5000), near surface points, loat XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
            calibs: (B * num_views, 4, 4) calibration matrix
            transforms: default is None
            labels: (B, 3, 5000), gt-RGB color, -1 ~ 1 float
        
        Output
            self.pred: (B, 3, 5000) RGB predictions for each point, float -1 ~ 1
        '''

        if labels is not None:
            self.labels = labels # (B, 3, 5000), gt-RGB color, -1 ~ 1 float

        # (B * num_view, 3, N), points are projected onto XY (-1,1)x(-1,1) plane of the cam coord.
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :] # (B * num_view, 2, N)
        z = xyz[:, 2:3, :] # (B * num_view, 1, N)

        # (B * num_view, 1, N)
        z_feat = self.normalizer(z)

        # [(B * num_views, 512, 5000), (B * num_view, 1, 5000)]
        point_local_feat_list = [self.index(self.im_feat, xy), z_feat]

        # (B * num_views, 512+1, 5000)
        point_local_feat = torch.cat(point_local_feat_list, 1)

        # (B, 3, 5000), num_views are canceled out by mean pooling, float -1 ~ 1. RGB rolor
        self.preds = self.surface_classifier(point_local_feat)

    def forward(self, images, im_feat, points, calibs, transforms=None, labels=None):
        """
        input
            images: (BV, 3, 512, 512)
            im_feat: (BV, 256, 128, 128) from the stacked-hour-glass filter
        """

        # extract self.im_feat, (BV, 256, 128, 128), not the input im_feat
        self.filter(images)

        # concat the input im_feat with self.im_feat and get the new self.im_feat: (BV, 512, 128, 128)
        self.attach(im_feat)

        # extract self.preds: (B, 3, 5000), num_views are canceled out by mean pooling, float -1 ~ 1. RGB rolor
        self.query(points, calibs, transforms, labels)

        # return self.preds, (B, 3, 5000)
        res = self.get_preds()

        # get the error, default is nn.L1Loss()
        error = self.get_error() # R

        return res, error

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, last)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if last:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetFilter(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, opt, input_nc=3, output_nc=256, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer, default: 64
            norm_layer          -- normalization layer, default is nn.InstanceNorm2d without {affine, tracked statistics}
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert (n_blocks >= 0)
        super(ResnetFilter, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers, i in {0, 1}
            mult = 2 ** i # in {1, 2}
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling # default: 4
        for i in range(n_blocks):  # add ResNet blocks, default: 6

            # the last resnet block
            if i == n_blocks - 1:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias, last=True)] # input/output dimensions same

            # the previous resnet blocks
            else:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)] # input/output dimensions same

        if opt.use_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""

        return self.model(input)












