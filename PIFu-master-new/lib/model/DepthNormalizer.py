import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNormalizer(nn.Module):
    def __init__(self, opt):
        super(DepthNormalizer, self).__init__()
        self.opt = opt

    def forward(self, z, calibs=None, index_feat=None):
        '''
        Normalize z_feature
        :param z_feat: (B * num_views, 1, N) depth value for z in the image coordinate system
        :return:
        '''
        z_feat = z * (self.opt.loadSize // 2) / self.opt.z_size # z * 256 / 200

        return z_feat
