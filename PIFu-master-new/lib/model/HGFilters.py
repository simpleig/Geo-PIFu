import torch
import torch.nn as nn
import torch.nn.functional as F
from ..net_util import *
import pdb # pdb.set_trace()

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch',upsample_mode="bicubic"):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules # default: 1
        self.depth = depth # default: 2
        self.features = num_features # default: 256
        self.norm = norm # default: group
        self.upsample_mode = upsample_mode # default: bicubic

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1: # default: 2
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.

        if self.upsample_mode == "bicubic":
            up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        elif self.upsample_mode == "nearest":
            up2 = F.interpolate(low3, scale_factor=2, mode='nearest')
        else:
            print("Error: undefined self.upsample_mode {}!".format(self.upsample_mode))

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):
    def __init__(self, opt):
        super(HGFilter, self).__init__()
        self.num_modules = opt.num_stack # default: 4

        self.opt = opt

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        if self.opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.opt.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt.norm)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.opt.norm)
        self.conv4 = ConvBlock(128, 256, self.opt.norm)

        # Stacking part
        for hg_module in range(self.num_modules): # default: 4
            self.add_module('m' + str(hg_module), HourGlass(1, opt.num_hourglass, 256, self.opt.norm, self.opt.upsample_mode))

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.opt.norm))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
                
            self.add_module('l' + str(hg_module), nn.Conv2d(256, opt.hourglass_dim, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt.hourglass_dim, 256, kernel_size=1, stride=1, padding=0))

            if hg_module == (self.num_modules-1) and self.opt.recover_dim:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt.hourglass_dim, 256, kernel_size=1, stride=1, padding=0))

        # recover stack-hour-glass output feature dimensions from BVx256x128x128 to BVx256x512x512
        if self.opt.recover_dim:
            self.recover_dim_match_fea_1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
            self.recover_dim_conv_1      = ConvBlock(256, 256, self.opt.norm)
            self.recover_dim_match_fea_2 = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0)
            self.recover_dim_conv_2      = ConvBlock(256, 256, self.opt.norm)
            
    def forward(self, x):
        '''
        Filter the input images, store all intermediate features.

        Input
            x: [B * num_views, C, H, W] input images, float -1 ~ 1, RGB

        Output
            outputs:       [(B * num_views, opt.hourglass_dim, H/4, W/4), (same_size), (same_size), (same_size)], list length is opt.num_stack
            tmpx.detach():  (B * num_views, 64, H/2, W/2)
            normx:          (B * num_views, 128, H/4, W/4)
        '''

        raw_x = x
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.opt.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules): # default: 4

            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            # recover stack-hour-glass output feature dimensions from BVx256x128x128 to BVx256x512x512
            if i == (self.num_modules-1) and self.opt.recover_dim:

                # merge features
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                fea_upsampled = previous + ll + tmp_out_ # (BV,256,128,128)

                # upsampling: (BV,256,128,128) to (BV,256,256,256)
                if self.opt.upsample_mode == "bicubic":

                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='bicubic', align_corners=True) # (BV,256,256,256)
                elif self.opt.upsample_mode == "nearest":
                    
                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='nearest') # (BV,256,256,256)
                else:
                    
                    print("Error: undefined self.upsample_mode {} when self.opt.recover_dim {}!".format(self.opt.upsample_mode,self.opt.recover_dim))
                fea_upsampled = fea_upsampled + self.recover_dim_match_fea_1(tmpx)
                fea_upsampled = self.recover_dim_conv_1(fea_upsampled) # (BV,256,256,256)

                # upsampling: (BV,256,256,256) to (BV,256,512,512)
                if self.opt.upsample_mode == "bicubic":

                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='bicubic', align_corners=True) # (BV,256,512,512)
                elif self.opt.upsample_mode == "nearest":
                    
                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='nearest') # (BV,256,512,512)
                else:
                    
                    print("Error: undefined self.upsample_mode {} when self.opt.recover_dim {}!".format(self.opt.upsample_mode,self.opt.recover_dim))
                fea_upsampled = fea_upsampled + self.recover_dim_match_fea_2(raw_x)
                fea_upsampled = self.recover_dim_conv_2(fea_upsampled) # (BV,256,512,512)

                outputs.append(fea_upsampled)

        return outputs, tmpx.detach(), normx




