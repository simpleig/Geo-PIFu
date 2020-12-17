import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import functools

import numpy as np
from .mesh_util import *
from .sample_util import *
from .geometry import index
import cv2
from PIL import Image
from tqdm import tqdm
import pdb # pdb.set_trace()

def reshape_multiview_tensors(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def gen_mesh(opt, net, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

def gen_mesh_color(opt, netG, netC, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    netG.filter(image_tensor)
    netC.filter(image_tensor)
    netC.attach(netG.get_im_feat())

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction(
            netG, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)

        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)

        color = np.zeros(verts.shape)
        interval = opt.num_sample_color
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def calc_error(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)

            res, error = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)

            IOU, prec, recall = compute_acc(res, label_tensor)

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

def calc_error_color(opt, netG, netC, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_color_arr = []

        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            color_sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)

            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)

            rgb_tensor = data['rgbs'].to(device=cuda).unsqueeze(0)

            netG.filter(image_tensor)
            _, errorC = netC.forward(image_tensor, netG.get_im_feat(), color_sample_tensor, calib_tensor, labels=rgb_tensor)

            # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
            #       .format(idx, num_tests, errorG.item(), errorC.item()))
            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal., kinda like std

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    # define the initialization rules for different layers
    def init_func(m):

        # name of one layer
        classname = m.__class__.__name__

        # init. regular conv, fc layers
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            # init the weights
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            # init the bias (if have it)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # init. batchNorm layers
        elif classname.find('BatchNorm2d') != -1:

            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    # multi-GPU settings, default: single-GPU training/test
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)
    
    # return is not necessary, not used anywhere
    return net


def imageSpaceRotation(xy, rot):
    '''
    args:
        xy: (B, 2, N) input
        rot: (B, 2) x,y axis rotation angles

    rotation center will be always image center (other rotation center can be represented by additional z translation)
    '''
    disp = rot.unsqueeze(2).sin().expand_as(xy)
    return (disp * xy).sum(dim=1)


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, 32)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv2dSame(nn.Module):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

        # self.weight = self.net[1].weight
        # self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)

class rgb_rendering_unet(nn.Module):

    def __init__(self, c_len_in, c_len_out, opt=None):

        super(rgb_rendering_unet, self).__init__()

        # (B,    8,192,128) |   conv2d(    i8, o8,k3,s1,nb), BN, LeakyReLU(0.2), Dp(0.1) | (B, 8,192,128) -----> skip-0: (B, 8,192,128)
        # (B,    8,192,128) |   conv2d(    i8,o16,k4,s2,nb), BN, LeakyReLU(0.2), Dp(0.1) | (B,16, 96, 64) -----> skip-1: (B,16, 96, 64)
        # (B,   16, 96, 64) |   conv2d(   i16,o16,k3,s1, b),     LeakyReLU(0.2), Dp(0.1) | (B,16, 96, 64)
        # (B,   16, 96, 64) |   conv2d(   i16,o32,k4,s2, b),     LeakyReLU(0.2), Dp(0.1) | (B,32, 48, 32)
        c_len_1 = c_len_in
        self.rendering_enc_conv2d_1 = nn.Sequential(Conv2dSame(c_len_in,c_len_1,kernel_size=3,bias=False), nn.BatchNorm2d(c_len_1,affine=True), nn.LeakyReLU(0.2,True), nn.Dropout2d(0.1,False))
        c_len_2 = c_len_1 * 2
        self.rendering_enc_conv2d_2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c_len_1,c_len_2,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm2d(c_len_2,affine=True), nn.LeakyReLU(0.2,True), nn.Dropout2d(0.1,False))
        c_len_3 = c_len_2
        self.rendering_enc_conv2d_3 = nn.Sequential(Conv2dSame(c_len_2,c_len_3,kernel_size=3,bias=True), nn.LeakyReLU(0.2,True), nn.Dropout2d(0.1,False))
        c_len_4 = c_len_3 * 2
        self.rendering_enc_conv2d_4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c_len_3,c_len_4,kernel_size=4,padding=0,stride=2,bias=True), nn.LeakyReLU(0.2,True), nn.Dropout2d(0.1,False))

        # (B,   32, 48, 32) | deconv2d(   i32,o16,k4,s2, b),               ReLU, Dp(0.1) | (B,16, 96, 64)
        # (B,   16, 96, 64) |   conv2d(   i16,o16,k3,s1, b),               ReLU, Dp(0.1) | (B,16, 96, 64)
        # (B,16+16, 96, 64) | deconv2d(i16+16, o8,k4,s2,nb), BN,           ReLU, Dp(0.1) | (B, 8,192,128) <----- skip-1: (B,16, 96, 64)
        # (B,    8,192,128) |   conv2d(    i8, o8,k3,s1,nb), BN,           ReLU, Dp(0.1) | (B, 8,192,128)
        # (B,  8+8,192,128) | deconv2d(  i8+8, o3,k4,s2,nb), BN,           ReLU, Dp(0.1) | (B, 3,384,256) <----- skip-0: (B, 8,192,128)
        # (B,    3,384,256) |   conv2d(    i3, o3,k3,s1, b), Tanh                        | (B, 3,384,256)
        self.rendering_dec_conv2d_1 = nn.Sequential(nn.ConvTranspose2d(c_len_4,c_len_3,kernel_size=4,stride=2,padding=1,bias=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_2 = nn.Sequential(Conv2dSame(c_len_3,c_len_2,kernel_size=3,bias=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_3 = nn.Sequential(nn.ConvTranspose2d(c_len_2*2,c_len_1,kernel_size=4,stride=2,padding=1,bias=False), nn.BatchNorm2d(c_len_1,affine=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_4 = nn.Sequential(Conv2dSame(c_len_1,c_len_1,kernel_size=3,bias=False), nn.BatchNorm2d(c_len_1,affine=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_5 = nn.Sequential(nn.ConvTranspose2d(c_len_1*2,c_len_out,kernel_size=4,stride=2,padding=1,bias=False), nn.BatchNorm2d(c_len_out,affine=True), nn.ReLU(True), nn.Dropout2d(0.1,False))
        self.rendering_dec_conv2d_6 = nn.Sequential(Conv2dSame(c_len_out,c_len_out,kernel_size=3,bias=True), nn.Tanh())

    def forward(self, x):

        # init.
        skip_list = []

        # encoder
        x = self.rendering_enc_conv2d_1(x); skip_list.append(x)             # (B,    8,192,128) |   conv2d(    i8, o8,k3,s1,nb), BN, LeakyReLU(0.2), Dp(0.1) | (B, 8,192,128) -----> skip-0: (B, 8,192,128)
        x = self.rendering_enc_conv2d_2(x); skip_list.append(x)             # (B,    8,192,128) |   conv2d(    i8,o16,k4,s2,nb), BN, LeakyReLU(0.2), Dp(0.1) | (B,16, 96, 64) -----> skip-1: (B,16, 96, 64)
        x = self.rendering_enc_conv2d_3(x)                                  # (B,   16, 96, 64) |   conv2d(   i16,o16,k3,s1, b),     LeakyReLU(0.2), Dp(0.1) | (B,16, 96, 64)
        x = self.rendering_enc_conv2d_4(x); skip_list.append(x)             # (B,   16, 96, 64) |   conv2d(   i16,o32,k4,s2, b),     LeakyReLU(0.2), Dp(0.1) | (B,32, 48, 32)

        # decoder
        x = self.rendering_dec_conv2d_1(x)                                  # (B,   32, 48, 32) | deconv2d(   i32,o16,k4,s2, b),               ReLU, Dp(0.1) | (B,16, 96, 64)
        x = self.rendering_dec_conv2d_2(x)                                  # (B,   16, 96, 64) |   conv2d(   i16,o16,k3,s1, b),               ReLU, Dp(0.1) | (B,16, 96, 64)
        x = self.rendering_dec_conv2d_3(torch.cat([skip_list[1],x], dim=1)) # (B,16+16, 96, 64) | deconv2d(i16+16, o8,k4,s2,nb), BN,           ReLU, Dp(0.1) | (B, 8,192,128) <----- skip-1: (B,16, 96, 64)
        x = self.rendering_dec_conv2d_4(x)                                  # (B,    8,192,128) |   conv2d(    i8, o8,k3,s1,nb), BN,           ReLU, Dp(0.1) | (B, 8,192,128)
        x = self.rendering_dec_conv2d_5(torch.cat([skip_list[0],x], dim=1)) # (B,  8+8,192,128) | deconv2d(  i8+8, o3,k4,s2,nb), BN,           ReLU, Dp(0.1) | (B, 3,384,256) <----- skip-0: (B, 8,192,128)
        x = self.rendering_dec_conv2d_6(x)                                  # (B,    3,384,256) |   conv2d(    i3, o3,k3,s1, b), Tanh                        | (B, 3,384,256)

        return x

class Conv3dSame(nn.Module):
    '''3D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReplicationPad3d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb, ka, kb)),
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

    def forward(self, x):

        return self.net(x)

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.eps = 1e-9

    def __call__(self, input, target_is_real):
        if target_is_real:
            # if True, then the loss encourages input to increase
            return -1.*torch.mean(torch.log(input + self.eps))
        else:
            # if False, then the loss encourages input to decrease
            return -1.*torch.mean(torch.log(1 - input + self.eps))

class Unet3D(nn.Module):

    def __init__(self, c_len_in, c_len_out, opt=None):

        super(Unet3D, self).__init__()

        # (BV,8,32,48,32) | conv3d(k3,s1,i8,o8,nb), BN3d, LearkyReLU(0.2) | (BV,8,32,48,32) ------> skip-0: (BV,8,32,48,32)
        c_len_1 = 8
        self.conv3d_pre_process = nn.Sequential(Conv3dSame(c_len_in,c_len_1,kernel_size=3,bias=False), nn.BatchNorm3d(c_len_1,affine=True), nn.LeakyReLU(0.2,True))

        # (BV,8,32,48,32) | conv3d(k4,s2,i8,o16,nb), BN3d, LeakyReLU(0.2) | (BV,16,16,24,16) ------> skip-1: (BV,16,16,24,16)
        c_len_2 = 16
        self.conv3d_enc_1 = nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(c_len_1,c_len_2,kernel_size=4,padding=0,stride=2,bias=False), nn.BatchNorm3d(c_len_2,affine=True), nn.LeakyReLU(0.2,True))

        # (BV,16,16,24,16) | conv3d(k4,s2,i16,o32,b), LeakyReLU(0.2) | (BV,32,8,12,8)
        c_len_3 = 32
        self.conv3d_enc_2 = nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(c_len_2,c_len_3,kernel_size=4,padding=0,stride=2,bias=True), nn.LeakyReLU(0.2,True))

        # (BV,32,8,12,8) | DeConv3d(k4,s2,i32,o16,b), ReLU | (BV,16,16,24,16)
        self.deconv3d_dec_2 = nn.Sequential(nn.ConvTranspose3d(c_len_3,c_len_2,kernel_size=4,stride=2,padding=1,bias=True), nn.ReLU(True))

        # (BV,16+16,16,24,16) | DeConv3d(k4,s2,i32,o8,nb), BN3d, ReLU | (BV,8,32,48,32) <------ skip-1: (BV,16,16,24,16)
        self.deconv3d_dec_1 = nn.Sequential(nn.ConvTranspose3d(c_len_2*2,c_len_1,kernel_size=4,stride=2,padding=1,bias=False), nn.BatchNorm3d(c_len_1,affine=True), nn.ReLU(True))

        # (BV,8+8,32,48,32) | Conv3d(k3,s1,i16,o8,nb), BN3d, ReLU | (BV,8,32,48,32) <------ skip-0: (BV,8,32,48,32)
        self.conv3d_final_process = nn.Sequential(Conv3dSame(c_len_1*2,c_len_out,kernel_size=3,bias=False), nn.BatchNorm3d(c_len_out,affine=True), nn.ReLU(True))

    def forward(self, x):
        """
        e.g. in-(BV,8,32,48,32), out-(BV,8,32,48,32)
        """

        skip_encoder_list = []

        # (BV,8,32,48,32) | conv3d(k3,s1,i8,o8,nb), BN3d, LearkyReLU(0.2) | (BV,8,32,48,32) ------> skip-0: (BV,8,32,48,32)
        x = self.conv3d_pre_process(x)
        skip_encoder_list.append(x)

        # (BV,8,32,48,32) | conv3d(k4,s2,i8,o16,nb), BN3d, LeakyReLU(0.2) | (BV,16,16,24,16) ------> skip-1: (BV,16,16,24,16)
        x = self.conv3d_enc_1(x)
        skip_encoder_list.append(x)

        # (BV,16,16,24,16) | conv3d(k4,s2,i16,o32,b), LeakyReLU(0.2) | (BV,32,8,12,8)
        x = self.conv3d_enc_2(x)

        # (BV,32,8,12,8) | DeConv3d(k4,s2,i32,o16,b), ReLU | (BV,16,16,24,16)
        x = self.deconv3d_dec_2(x)

        # (BV,16+16,16,24,16) | DeConv3d(k4,s2,i32,o8,nb), BN3d, ReLU | (BV,8,32,48,32) <------ skip-1: (BV,16,16,24,16)
        x = torch.cat([skip_encoder_list[1], x], dim=1)
        x = self.deconv3d_dec_1(x)

        # (BV,8+8,32,48,32) | Conv3d(k3,s1,i16,o8,nb), BN3d, ReLU | (BV,8,32,48,32) <------ skip-0: (BV,8,32,48,32)
        x = torch.cat([skip_encoder_list[0], x], dim=1)
        x = self.conv3d_final_process(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3
  












