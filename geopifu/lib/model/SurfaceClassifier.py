import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb # pdb.set_trace()

class deepVoxelsFusionNetwork(nn.Module):

    def __init__(self, c_len_in, c_len_out):

        super(deepVoxelsFusionNetwork, self).__init__()

        # (BV,  8, N) | nn.Conv1d( i8,o16,b), LeakyReLU | (BV, 16, N)
        # (BV, 16, N) | nn.Conv1d(i16,o32,b), LeakyReLU | (BV, 32, N)
        # (BV, 32, N) | nn.Conv1d(i32,o64,b), LeakyReLU | (BV, 64, N)
        self.fusion_fc_1 = nn.Conv1d(  c_len_in, c_len_in*2, 1)
        self.fusion_fc_2 = nn.Conv1d(c_len_in*2, c_len_in*4, 1)
        self.fusion_fc_3 = nn.Conv1d(c_len_in*4,  c_len_out, 1)

    def forward(self, feature):

        # (BV,  8, N) | nn.Conv1d( i8,o16,b), LeakyReLU | (BV, 16, N)
        feature = self.fusion_fc_1(feature)
        feature = F.leaky_relu(feature)
        # (BV, 16, N) | nn.Conv1d(i16,o32,b), LeakyReLU | (BV, 32, N)
        feature = self.fusion_fc_2(feature)
        feature = F.leaky_relu(feature)
        # (BV, 32, N) | nn.Conv1d(i32,o64,b), LeakyReLU | (BV, 64, N)
        feature = self.fusion_fc_3(feature)
        feature = F.leaky_relu(feature)

        return feature

class SurfaceClassifier_multiLoss(nn.Module):

    def __init__(self, opt, filter_channels_2d, filter_channels_3d, filter_channels_joint):

        super(SurfaceClassifier_multiLoss, self).__init__()

        # ----- 2d features branch -----
        self.filters_2d = []
        for idx in range(0, len(filter_channels_2d)-1):
            if idx == 0:
                self.filters_2d.append(  nn.Conv1d(                      filter_channels_2d[idx],filter_channels_2d[idx+1],1)  )
            else:
                self.filters_2d.append(  nn.Conv1d(filter_channels_2d[0]+filter_channels_2d[idx],filter_channels_2d[idx+1],1)  )
            self.add_module("features_2d_conv%d"%(idx), self.filters_2d[idx])

        # ----- 3d features branch -----
        self.filters_3d = []
        for idx in range(0, len(filter_channels_3d)-1):
            if idx == 0:
                self.filters_3d.append(  nn.Conv1d(                      filter_channels_3d[idx],filter_channels_3d[idx+1],1)  )
            else:
                self.filters_3d.append(  nn.Conv1d(filter_channels_3d[0]+filter_channels_3d[idx],filter_channels_3d[idx+1],1)  )
            self.add_module("features_3d_conv%d"%(idx), self.filters_3d[idx])

        # ----- fused features branch -----
        filter_channels_joint[0] = filter_channels_2d[0]  + filter_channels_3d[0]
        filter_channels_fused    = filter_channels_2d[-2] + filter_channels_3d[-2]
        self.filters_joint = []
        for idx in range(0, len(filter_channels_joint)-1):
            if idx == 0:
                self.filters_joint.append(  nn.Conv1d(filter_channels_joint[0]+     filter_channels_fused,filter_channels_joint[idx+1],1)  )
            else:
                self.filters_joint.append(  nn.Conv1d(filter_channels_joint[0]+filter_channels_joint[idx],filter_channels_joint[idx+1],1)  )
            self.add_module("features_joint_conv%d"%(idx), self.filters_joint[idx])

        # ----- the last layer for (0., 1.) sdf pred -----
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, feature_2d, feature_3d):

        # init.
        pred_sdf = []

        # ----- 2d features branch -----
        feature_2d_skip = feature_2d
        feature_2d_pass = feature_2d
        for idx in range(len(self.filters_2d)):

            if (idx == len(self.filters_2d)-1) and (not self.training):
                continue

            feature_2d_pass = self._modules["features_2d_conv%d"%(idx)](feature_2d_pass if idx==0 else torch.cat([feature_2d_pass,feature_2d_skip],1))
            if idx != len(self.filters_2d)-1:
                feature_2d_pass = F.leaky_relu(feature_2d_pass)
                if idx == len(self.filters_2d)-2:
                    feature_2d_fuse = feature_2d_pass
            else:
                pred_sdf_2d = self.sigmoid_layer(feature_2d_pass)
                pred_sdf.append(pred_sdf_2d)

        # ----- 3d features branch -----
        feature_3d_skip = feature_3d
        feature_3d_pass = feature_3d
        for idx in range(len(self.filters_3d)):

            if (idx == len(self.filters_3d)-1) and (not self.training):
                continue

            feature_3d_pass = self._modules["features_3d_conv%d"%(idx)](feature_3d_pass if idx==0 else torch.cat([feature_3d_pass,feature_3d_skip],1))
            if idx != len(self.filters_3d)-1:
                feature_3d_pass = F.leaky_relu(feature_3d_pass)
                if idx == len(self.filters_3d)-2:
                    feature_3d_fuse = feature_3d_pass
            else:
                pred_sdf_3d = self.sigmoid_layer(feature_3d_pass)
                pred_sdf.append(pred_sdf_3d)

        # ----- fused features branch -----
        feature_joint_skip = torch.cat([feature_2d_skip,feature_3d_skip],1)
        feature_joint_pass = torch.cat([feature_2d_fuse,feature_3d_fuse],1)
        for idx in range(len(self.filters_joint)):

            feature_joint_pass = self._modules["features_joint_conv%d"%(idx)](torch.cat([feature_joint_pass,feature_joint_skip],1))
            if idx != len(self.filters_joint)-1:
                feature_joint_pass = F.leaky_relu(feature_joint_pass)
            else:
                pred_sdf_joint = self.sigmoid_layer(feature_joint_pass)
                pred_sdf.append(pred_sdf_joint)

        # return
        return pred_sdf

class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None, opt=None):
        """
        input
            filter_channels: default is [257, 1024, 512, 256, 128, 1]
            no_residual    : default is False
        """
        super(SurfaceClassifier, self).__init__()

        self.filters = [] # length is filter_channels-1, default is 5   
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels # default: [257, 1024, 512, 256, 128, 1]
        self.last_op = last_op
        self.opt = opt

        # default is False
        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:

            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''
        Input
            feature: (B * num_views, opt.hourglass_dim+1, n_in+n_out)
        
        Return
            y: (B, 1, n_in+n_out), num_views are canceled out by mean pooling
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):

            # default is False
            if self.no_residual:
                # y = f(y)
                y = self._modules["conv%d"%(i)](y) if len(self.opt.gpu_ids) > 1 else f(y)
            else:
                # y = f(
                #     y if i == 0
                #     else torch.cat([y, tmpy], 1)
                # ) # with skip connections from feature
                y = self._modules["conv%d"%(i)]( y if i==0 else torch.cat([y, tmpy],1) ) if len(self.opt.gpu_ids) > 1 else f( y if i == 0 else torch.cat([y, tmpy],1) )

            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y
