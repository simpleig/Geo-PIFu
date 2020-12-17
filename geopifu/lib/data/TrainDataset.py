from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging

import pdb # pdb.set_trace()
import glob
import sys
import json
import copy

SENSEI_DATA_DIR          = "/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c"
this_file_path_abs       = os.path.dirname(__file__)
target_dir_path_relative = os.path.join(this_file_path_abs, '../../..')
target_dir_path_abs      = os.path.abspath(target_dir_path_relative)
sys.path.insert(0, target_dir_path_abs)
from Constants import consts

log = logging.getLogger('trimesh')
log.setLevel(40)

def load_trimesh(root_dir):
    """
    XYZ direction
        X-right, Y-up, Z-outwards. All meshes face outwards along +Z.
    return
        a dict of ALL meshes, indexed by mesh-names
    """
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_100k.obj' % sub_name))

    return meshs


def load_trimesh_iccv(meshPathsList):
    """
    XYZ direction
        X-right, Y-down, Z-inwards. All meshes face inwards along +Z.
    return
        a dict of ALL meshes, indexed by mesh-names
    """

    meshs = {}
    count = 0
    for meshPath in meshPathsList:
        print("Loading mesh %d/%d..." % (count, len(meshPathsList)))
        count += 1
        meshs[meshPath] = trimesh.load(meshPath)

    print("Sizes of meshs dict: %.3f MB." % (sys.getsizeof(meshs)/1024.)); pdb.set_trace()

    return meshs


def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255 # (5000, 1) Red-inside
    g = (prob < 0.5).reshape([-1, 1]) * 255 # (5000, 1) green-outside
    b = np.zeros(r.shape) # (5000, 1)

    to_save = np.concatenate([points, r, g, b], axis=-1) # (5000, 6)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

def save_volume(volume, fname, dim_h, dim_w, voxel_size):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = volume.shape[0], volume.shape[1], volume.shape[2]
    with open(fname, 'w') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if volume[xx, yy, zz] > 0:
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))

class TrainDatasetICCV(Dataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train', allow_aug=True):

        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')
        self.meshDirSearch = opt.meshDirSearch

        self.B_MIN = np.array([-consts.real_w/2., -consts.real_h/2., -consts.real_w/2.])
        self.B_MAX = np.array([ consts.real_w/2.,  consts.real_h/2.,  consts.real_w/2.])

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize
        self.allow_aug = allow_aug

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout # inside/outside 3d-sampling-points for occupancy query, default: 5000
        self.num_sample_color = self.opt.num_sample_color

        self.yaw_list = list(range(0,360,1))
        self.pitch_list = [0]
        self.subjects = self.get_subjects() # a list of mesh paths for training or test

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # PIL to tensor, for the target view if use view rendering losses
        if opt.use_view_pred_loss: 
            self.to_tensor_target_view = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        # a dict of ALL meshes, indexed by mesh-names
        # self.mesh_dic = load_trimesh_iccv(self.subjects)
        # self.mesh_dic = {}
        self.training_inds, self.testing_inds = self.get_training_test_indices(args=self.opt, shuffle=False)

        # record epoch idx for offline query sampling
        self.epochIdx = 0

    def get_subjects(self):
        """
        Return
            a list of mesh paths for training or test
        """

        personClothesDirs = glob.glob("%s/DeepHumanDataset/dataset/results_*" % (self.meshDirSearch)) # 202
        assert(len(personClothesDirs) == 202)

        meshPaths = [] # 6795
        for dirEach in personClothesDirs:
            meshPathsTmp = glob.glob("%s/*" % (dirEach))
            meshPathsTmp = [p+"/mesh.obj" for p in meshPathsTmp]
            meshPaths.extend(meshPathsTmp)
        meshNum = len(meshPaths) # 6795
        assert(meshNum == 6795)

        meshPaths_train = meshPaths[:int(np.ceil(self.opt.trainingDataRatio * meshNum))]  # 5436 unique mesh paths
        meshPaths_test  = meshPaths[int(np.floor(self.opt.trainingDataRatio * meshNum)):] # 1359 unique mesh paths

        return meshPaths_train if self.is_train else meshPaths_test

    def get_training_test_indices(self, args, shuffle=False):

        # sanity check for args.totalNumFrame
        totalNumFrameTrue = len(glob.glob(args.datasetDir+"/config/*.json"))
        assert((args.totalNumFrame == totalNumFrameTrue) or (args.totalNumFrame == totalNumFrameTrue+len(consts.black_list_images)//4))

        max_idx = args.totalNumFrame # total data number: N*M'*4 = 6795*4*4 = 108720
        indices = np.asarray(range(max_idx))
        assert(len(indices)%4 == 0)

        testing_flag = (indices >= args.trainingDataRatio*max_idx)
        testing_inds = indices[testing_flag] # 21744 testing indices: array of [86976, ..., 108719]
        testing_inds = testing_inds.tolist()
        if shuffle: np.random.shuffle(testing_inds)
        assert(len(testing_inds) % 4 == 0)

        training_inds = indices[np.logical_not(testing_flag)] # 86976 training indices: array of [0, ..., 86975]
        training_inds = training_inds.tolist()
        if shuffle: np.random.shuffle(training_inds)
        assert(len(training_inds) % 4 == 0)

        return training_inds, testing_inds

    def __len__(self):

        # return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list) # (number-of-training/test-meshes * 360-degree-renderings) := namely, the-number-of-training-views
        return len(self.training_inds) if self.is_train else len(self.testing_inds)

    def rotateY_by_view(self, view_id):
        """
        input
            view_id: 0-front, 1-right, 2-back, 3-left
        """

        rotAngByViews = [0, -90., -180., -270.]
        angle = np.radians(rotAngByViews[view_id])
        ry = np.array([ [ np.cos(angle), 0., np.sin(angle)],
                        [            0., 1.,            0.],
                        [-np.sin(angle), 0., np.cos(angle)] ]) # (3,3)
        ry = np.transpose(ry)

        return ry

    def get_render_iccv(self, args, num_views, view_id_list, random_sample=False, dataConfig=None, view_id=None):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] for 3x512x512 images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] for 1x512x512 masks
        '''

        # init.
        render_list = []
        calib_list = []
        extrinsic_list = []
        mask_list = []
        
        # get a list of view ids, (if multi-view enabled)
        assert(num_views <= len(view_id_list))
        view_ids = [view_id_list[(len(view_id_list)//num_views*offset) % len(view_id_list)] for offset in range(num_views)]
        if num_views > 1 and random_sample: view_ids = np.random.choice(view_id_list, num_views, replace=False)

        # for each view
        for vid in view_ids:

            # ----- load mask Part-0 -----
            if True:

                # set path
                mask_path = "%s/maskImage/%06d.jpg" % (args.datasetDir, vid)
                if not os.path.exists(mask_path):
                    print("Can not find %s!!!" % (mask_path))
                    pdb.set_trace()

                # {read, discretize} data, values only within {0., 1.}
                mask_data = np.round((cv2.imread(mask_path)[:,:,0]).astype(np.float32)/255.) # (1536, 1024)
                mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32) # (1536, 1536)
                mask_data_padded[:,mask_data_padded.shape[0]//2-min(mask_data.shape)//2:mask_data_padded.shape[0]//2+min(mask_data.shape)//2] = mask_data # (1536, 1536)

                # NN resize to (512, 512)
                mask_data_padded = cv2.resize(mask_data_padded, (self.opt.loadSize,self.opt.loadSize), interpolation=cv2.INTER_NEAREST)
                mask_data_padded = Image.fromarray(mask_data_padded)

            # ----- load image Part-0 -----
            if True:

                # set paths
                image_path = '%s/rgbImage/%06d.jpg' % (args.datasetDir, vid)
                if not os.path.exists(image_path):
                    print("Can not find %s!!!" % (image_path))
                    pdb.set_trace()

                # read data BGR -> RGB, np.uint8
                image = cv2.imread(image_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
                image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8) # (1536, 1536, 3)
                image_padded[:,image_padded.shape[0]//2-min(image.shape[:2])//2:image_padded.shape[0]//2+min(image.shape[:2])//2,:] = image # (1536, 1536, 3)

                # resize to (512, 512, 3), np.uint8
                image_padded = cv2.resize(image_padded, (self.opt.loadSize, self.opt.loadSize))
                image_padded = Image.fromarray(image_padded)

            # ----- load calib and extrinsic Part-0 -----
            if True:

                # intrinsic matrix: ortho. proj. cam. model
                trans_intrinsic = np.identity(4) # trans intrinsic
                scale_intrinsic = np.identity(4) # ortho. proj. focal length
                scale_intrinsic[0,0] =  1./consts.h_normalize_half # const ==  2.
                scale_intrinsic[1,1] =  1./consts.h_normalize_half # const ==  2.
                scale_intrinsic[2,2] = -1./consts.h_normalize_half # const == -2.
                
                # extrinsic: model to cam R|t
                extrinsic        = np.identity(4)
                # randomRot        = np.array(dataConfig["randomRot"], np.float32) # by random R
                viewRot          = self.rotateY_by_view(view_id=view_id) # by view direction R
                # extrinsic[:3,:3] = np.dot(viewRot, randomRot)
                extrinsic[:3,:3] = viewRot

            # ----- training data augmentation -----
            if self.is_train and self.allow_aug:

                # Pad images
                pad_size         = int(0.1 * self.load_size)
                image_padded     = ImageOps.expand(image_padded, pad_size, fill=0)
                mask_data_padded = ImageOps.expand(mask_data_padded, pad_size, fill=0)
                w, h   = image_padded.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    image_padded     = transforms.RandomHorizontalFlip(p=1.0)(image_padded)
                    mask_data_padded = transforms.RandomHorizontalFlip(p=1.0)(mask_data_padded)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    image_padded     = image_padded.resize((w, h), Image.BILINEAR)
                    mask_data_padded = mask_data_padded.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                image_padded = image_padded.crop((x1, y1, x1 + tw, y1 + th))
                mask_data_padded = mask_data_padded.crop((x1, y1, x1 + tw, y1 + th))

                # color space augmentation
                image_padded = self.aug_trans(image_padded)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    image_padded = image_padded.filter(blur)

            # ----- load mask Part-1 -----
            if True:

                # convert to (1, 512, 512) tensors, float, 1-fg, 0-bg
                mask_data_padded = transforms.ToTensor()(mask_data_padded).float() # 1. inside, 0. outside
                mask_list.append(mask_data_padded)

            # ----- load image Part-1 -----
            if True:

                # convert to (3, 512, 512) tensors, RGB, float, -1 ~ 1. note that bg is 0 not -1.
                image_padded = self.to_tensor(image_padded) # (3, 512, 512), float -1 ~ 1
                image_padded = mask_data_padded.expand_as(image_padded) * image_padded
                render_list.append(image_padded)

            # ----- load calib and extrinsic Part-1 -----
            if True:

                # obtain the final calib and save calib/extrinsic
                intrinsic = np.matmul(trans_intrinsic, scale_intrinsic)
                calib     = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
                extrinsic = torch.Tensor(extrinsic).float()
                calib_list.append(calib)         # save calib
                extrinsic_list.append(extrinsic) # save extrinsic

        return {'img'      : torch.stack(render_list   , dim=0),
                'calib'    : torch.stack(calib_list    , dim=0), # model will be transformed into a XY-plane-center-aligned-2x2x2-volume of the cam. coord.
                'extrinsic': torch.stack(extrinsic_list, dim=0),
                'mask'     : torch.stack(mask_list     , dim=0)
               }

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] for 3x512x512 images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] for 1x512x512 masks
        '''

        # init.
        render_list = []
        calib_list = []
        extrinsic_list = []
        mask_list = []

        # pitch-idx
        pitch = self.pitch_list[pid] # default: 0

        # yaw-idx (list, if multi-view enabled)
        view_ids = [self.yaw_list[(yid + len(self.yaw_list)//num_views*offset) % len(self.yaw_list)] for offset in range(num_views)]
        if num_views >1 and random_sample: view_ids = np.random.choice(self.yaw_list, num_views, replace=False)
        
        # for each yaw-idx
        for vid in view_ids:

            # paths
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            param_path  = os.path.join(self.PARAM , subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            mask_path   = os.path.join(self.MASK  , subject, '%d_%d_%02d.png' % (vid, pitch, 0))

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center') # T_world_cam, model centroid should be aligned with the cam center (thus be the same for the whole dataset assuming object centroids are also aligned with the 3d volume center), so that after converting to the cam-coord., depth-z can exist in a consistent range for different objects/cameras.
            # model rotation
            R = param.item().get('R')           # R_cam_world

            # cam. extrinsic
            translate = -np.matmul(R, center).reshape(3, 1) # T_cam_world = -R_cam_world*T_world_cam
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0) # model will be transformed into a XY-plane-center-aligned-256x256x256-volume of the cam. coord.

            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio # scale/ortho_ratio should be the same for the whole dataset, e.g. 2.4049856956795232
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio # ideally should also have -1 for depth-z, otherwise the cam coord is not strictly right-hand-coord.

            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)

            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            # rgb image and it mask
            render = Image.open(render_path).convert('RGB')
            mask   = Image.open(mask_path).convert('L')

            # training data augmentation, doesn't seems to make sense, I don't think I need this
            if self.is_train:

                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float() # 1. inside, 0. outside
            mask_list.append(mask)

            render = self.to_tensor(render) # cHW, float -1 ~ 1
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {'img'      : torch.stack(render_list   , dim=0),
                'calib'    : torch.stack(calib_list    , dim=0), # model will be transformed into a XY-plane-center-aligned-2x2x2-volume of the cam. coord.
                'extrinsic': torch.stack(extrinsic_list, dim=0),
                'mask'     : torch.stack(mask_list     , dim=0)
               }

    def voxelization_normalization(self,verts,useMean=True,useScaling=True):
        """
        normalize the mesh into H [-0.5,0.5]*(1-margin), W/D [-0.333,0.333]*(1-margin)
        """

        vertsVoxelNorm = copy.deepcopy(verts)
        vertsMean, scaleMin = None, None

        if useMean:
            vertsMean = np.mean(vertsVoxelNorm,axis=0,keepdims=True) # (1, 3)
            vertsVoxelNorm -= vertsMean

        xyzMin = np.min(vertsVoxelNorm, axis=0); assert(np.all(xyzMin < 0))
        xyzMax = np.max(vertsVoxelNorm, axis=0); assert(np.all(xyzMax > 0))

        if useScaling:
            scaleArr = np.array([consts.threshWD/abs(xyzMin[0]), consts.threshH/abs(xyzMin[1]),consts.threshWD/abs(xyzMin[2]), consts.threshWD/xyzMax[0], consts.threshH/xyzMax[1], consts.threshWD/xyzMax[2]])
            scaleMin = np.min(scaleArr)
            vertsVoxelNorm *= scaleMin

        return vertsVoxelNorm, vertsMean, scaleMin

    def select_sampling_method_iccv_offline(self,index):
        """
        Path examples of the offline sampled query points 
            occu_sigma3.5_pts5k/088046_ep000_inPts.npy            0.0573 MB, np.float64, (2500, 3)
            occu_sigma3.5_pts5k/088046_ep000_outPts.npy           0.0573 MB, np.float64, (2500, 3)
        """

        # get inside and outside points
        inside_points_path  = "%s/%s/%06d_ep%03d_inPts.npy"  % (self.opt.datasetDir, self.opt.sampleType,index,self.epochIdx)
        outside_points_path = "%s/%s/%06d_ep%03d_outPts.npy" % (self.opt.datasetDir, self.opt.sampleType,index,self.epochIdx)
        assert(os.path.exists(inside_points_path) and os.path.exists(outside_points_path))
        inside_points    = np.load(inside_points_path)  # (N_in , 3), np.float64 
        outside_points   = np.load(outside_points_path) # (N_out, 3), np.float64
        num_sample_inout = inside_points.shape[0] + outside_points.shape[0]
        assert(num_sample_inout == self.num_sample_inout)

        # get samples and labels: {1-inside, 0-outside}
        samples = np.concatenate([inside_points, outside_points], 0).T # (3, n_in + n_out)
        labels  = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1) # (1, n_in + n_out)
        samples = torch.Tensor(samples).float() # convert np.array to torch.Tensor 
        labels  = torch.Tensor(labels).float()

        return {'samples': samples, # (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                'labels' : labels   # (1, n_in + n_out), 1.0-inside, 0.0-outside
               }

    def select_sampling_method_iccv_online(self,dataConfig):
        
        # read mesh
        # mesh = trimesh.load(dataConfig["meshPath"])
        meshPathTmp = dataConfig["meshPath"]
        if SENSEI_DATA_DIR in meshPathTmp:
            backward_offset = 2 + len(self.opt.datasetDir.split('/')[-2]) + len(self.opt.datasetDir.split('/')[-1])
            meshPathTmp     = meshPathTmp.replace(SENSEI_DATA_DIR, self.opt.datasetDir[:-backward_offset])
        assert(os.path.exists(meshPathTmp))
        mesh = trimesh.load(meshPathTmp)

        # normalize into volumes of X~[+-0.333], Y~[+-0.5], Z~[+-0.333]
        randomRot           = np.array(dataConfig["randomRot"], np.float32) # by random R
        mesh.vertex_normals = np.dot(mesh.vertex_normals, np.transpose(randomRot))
        mesh.face_normals   = np.dot(mesh.face_normals  , np.transpose(randomRot))
        mesh.vertices, _, _ = self.voxelization_normalization(np.dot(mesh.vertices,np.transpose(randomRot)))

        # uniformly sample points on mesh surface
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout) # (N1,3)

        # add gausian noise to surface points
        sample_points = surface_points + np.random.normal(scale=1.*self.opt.sigma/consts.dim_h, size=surface_points.shape) # (N1, 3)

        # uniformly sample inside the 128x192x128 volume, surface-points : volume-points ~ 16:1
        # volume_dims = np.array([[1.*consts.dim_w/consts.dim_h, 1.0, 1.*consts.dim_w/consts.dim_h]])
        # random_points = (np.random.rand(self.num_sample_inout//4, 3) - 0.5) * volume_dims # (N2, 3)
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout//4, 3) * length + self.B_MIN # (N2, 3)
        sample_points = np.concatenate([sample_points, random_points], 0) # (N1+N2, 3)
        np.random.shuffle(sample_points) # (N1+N2, 3)

        # determine {1, 0} occupancy ground-truth
        inside = mesh.contains(sample_points)
        inside_points  = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        # constrain (n_in + n_out) <= self.num_sample_inout
        nin = inside_points.shape[0]
        # inside_points  =  inside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else inside_points
        # outside_points = outside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else outside_points[:(self.num_sample_inout - nin)]
        if nin > self.num_sample_inout//2:
            inside_points  =  inside_points[:self.num_sample_inout//2]
            outside_points = outside_points[:self.num_sample_inout//2]
        else:
            inside_points  = inside_points
            outside_points = outside_points[:(self.num_sample_inout - nin)]

        # get samples and labels: {1-inside, 0-outside}
        samples = np.concatenate([inside_points, outside_points], 0).T # (3, n_in + n_out)
        labels  = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1) # (1, n_in + n_out)
        samples = torch.Tensor(samples).float() # convert np.array to torch.Tensor 
        labels  = torch.Tensor(labels).float()
        del mesh

        return {'samples': samples, # (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                'labels' : labels   # (1, n_in + n_out), 1.0-inside, 0.0-outside
               }

    def select_sampling_method(self, subject):

        # # fix random seeds for test, not sure why yet
        # if not self.is_train:
        #     random.seed(1991)
        #     np.random.seed(1991)
        #     torch.manual_seed(1991)
        
        mesh = self.mesh_dic[subject] # get the mesh with name of subject
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout) # uniformly sample 3d-points on the mesh surface
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)

        # add random points within image space
        length = self.B_MAX - self.B_MIN # np.array([256, 256, 256])
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN # uniform sampling within a 256x256x256 volume, surface-points : volume-points ~ 16 : 1
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        # determine {1, 0} occupancy ground-truth
        inside = mesh.contains(sample_points)
        inside_points  = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        # constrain (n_in + n_out) <= self.num_sample_inout
        nin = inside_points.shape[0]
        inside_points  =  inside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else inside_points
        outside_points = outside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else outside_points[:(self.num_sample_inout - nin)]

        # get samples and labels: {1-inside, 0-outside}
        samples = np.concatenate([inside_points, outside_points], 0).T # (3, n_in + n_out)
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1) # (1, n_in + n_out)
        # save_samples_truncted_prob('out.ply', samples.T, labels.T)
        # exit()
        samples = torch.Tensor(samples).float() # convert np.array to torch.Tensor 
        labels = torch.Tensor(labels).float()
        del mesh

        return {'samples': samples, # (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                'labels' : labels   # (1, n_in + n_out), 1.0-inside, 0.0-outside
               }

    def get_color_sampling(self, subject, yid, pid=0):
        
        yaw = self.yaw_list[yid]
        pitch = self.pitch_list[pid]

        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        
        # Segmentation mask for the uv render.
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0 # [H, W] bool, True-inside, False-outside

        # UV render. each pixel is the color of the point.
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0 # [H, W, 3] 0 ~ 1 float

        # Normal render. each pixel is the surface normal of the point.
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0 # [H, W, 3] -1 ~ 1 float

        # Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]
        
        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))        # (512^2, 1), bool
        uv_render = uv_render.reshape((-1, 3)) # (512^2, 3), 0 ~ 1 float
        uv_normal = uv_normal.reshape((-1, 3)) # (512^2, 3), -1 ~ 1 float
        uv_pos = uv_pos.reshape((-1, 3))       # (512^2, 3), XYZ coords

        # not that those are surface points, not just vertices, otherwise UV maps are sparse
        surface_colors = uv_render[uv_mask] # 0 ~ 1 float
        surface_normal = uv_normal[uv_mask] # -1 ~ 1 float
        surface_points = uv_pos[uv_mask] # XYZ coords of those mesh surface points in ??? scale

        if self.num_sample_color: # default: 0
            assert(self.num_sample_color <= surface_points.shape[0]-1) # e.g. 5000 vs 158847-1
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color) # minor bug, should be just "surface_points.shape[0]"
            surface_colors = surface_colors[sample_list].T # (3, num_sample_color), 0 ~ 1 float
            surface_normal = surface_normal[sample_list].T # (3, num_sample_color), -1 ~ 1 float
            surface_points = surface_points[sample_list].T # (3, num_sample_color), XYZ coords

        # Sample "num_sample_color" query 3d-points, along the normal directions with Gaussian offset
        normal  = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal # (3, num_sample_color), XYZ coords

        # Normalized to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

        return {'color_samples': samples, # (3, num_sample_color), XYZ coords
                'rgbs': rgbs_color        # (3, num_sample_color), -1 ~ 1 float
               }

    def get_mesh_voxels(self,idx,view_id_list):

        # init.
        dict_to_return = {}

        # ----- mesh voxels -----
        if True:

            # mesh voxels path
            mesh_volume_path = "%s/meshVoxels/%06d.npy" % (self.opt.datasetDir, idx)

            # sanity check
            if not os.path.exists(mesh_volume_path):
                print("Can not find %s!!!" % (mesh_volume_path))
                pdb.set_trace()

            # load data
            mesh_volume = np.load(mesh_volume_path) # dtype('bool'), WHD-(128, 192, 128)

            # WHD -> DHW (as required by tensorflow)
            mesh_volume = np.transpose(mesh_volume, (2, 1, 0))
            
            # convert dtype
            mesh_volume = mesh_volume.astype(np.float32)        # (128,192,128)
            mesh_volume = torch.from_numpy(mesh_volume)[None,:] # (1,128,192,128)

            # add to dict
            dict_to_return["meshVoxels"] = mesh_volume # (C-1, D-128, H-192, W-128), 1-inside, 0-outsie

        # ----- select a direction for the target view -----
        if self.opt.use_view_pred_loss:

            # sample a frandom view direction from {front, right, back, left}
            view_directions = np.random.choice([0,1,2,3], 1, replace=False, p=self.opt.view_probs_front_right_back_left)[0]
            vid             = view_id_list[view_directions]

            # add to dict
            dict_to_return["view_directions"] = torch.tensor(view_directions, dtype=torch.int) # an integer, {0, 1, 2, 3} maps to {front, right, back, left}

        # ----- load mask of the target view -----
        if self.opt.use_view_pred_loss:

            # set path
            mask_path = "%s/maskImage/%06d.jpg" % (self.opt.datasetDir, vid)
            if not os.path.exists(mask_path):
                print("Target view: can not find %s!!!" % (mask_path))
                pdb.set_trace()

            # {read, discretize} data, values only within {0., 1.}, is float, not int.
            mask_data = np.round((cv2.imread(mask_path)[:,:,0]).astype(np.float32)/255.) # (1536, 1024)

            # NN resize to (H-384, W-256)
            mask_data = cv2.resize(mask_data, (self.opt.vrn_net_input_width, self.opt.vrn_net_input_height), interpolation=cv2.INTER_NEAREST)
            # note that Image.fromarray:
            # 1) will keep float32 values, thus we need to divide mask_data by 255. and make sure mask_data is np.float32
            # 2) will normalize uint8 into (0.,1.) range, thus we DO NOT divide image by 255. and make sure image is np.uint8
            # 3) has some requirements for input shapes
            mask_data = Image.fromarray(mask_data)

            # convert to (1, H-384, W-256) tensors, float, 1-fg, 0-bg
            mask_data = transforms.ToTensor()(mask_data).float() # 1. inside, 0. outside

            # add to dict
            dict_to_return["target_mask"] = mask_data # (C-1, H-384, W-256) masks, float 1.0-inside, 0.0-outside

        # ----- load image of the target view -----
        if self.opt.use_view_pred_loss:

            # set paths
            image_path = '%s/rgbImage/%06d.jpg' % (self.opt.datasetDir, vid)
            if not os.path.exists(image_path):
                print("Can not find %s!!!" % (image_path))
                pdb.set_trace()

            # read data BGR -> RGB, np.uint8
            image = cv2.imread(image_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}

            # resize to (H-384, W-256, C-3), np.uint8
            image = cv2.resize(image, (self.opt.vrn_net_input_width, self.opt.vrn_net_input_height))
            image = Image.fromarray(image)

            # convert to (C-3, H-384, W-256) tensors, RGB, float, -1 ~ 1. note that bg is 0 not -1.
            image = self.to_tensor_target_view(image) # (C-3, H-384, W-256), float -1 ~ 1
            image = mask_data.expand_as(image) * image

            # add to dict
            dict_to_return["target_view"] = image # (C-3, H-384, W-256) RGB images, float -1. ~ 1., bg is all ZEROS not -1.

        return dict_to_return

    def get_deepVoxels(self,idx):

        # init.
        dict_to_return = {}

        # ----- deepVoxels -----

        # set path
        deepVoxels_path = "%s/%06d_deepVoxels.npy" % (self.opt.deepVoxelsDir, idx)
        if not os.path.exists(deepVoxels_path):
            print("DeepVoxels: can not find %s!!!" % (deepVoxels_path))
            pdb.set_trace()

        # load npy, (C=8,W=32,H=48,D=32), C-XYZ, np.float32, only positive values
        deepVoxels_data = np.load(deepVoxels_path)

        # (C=8,W=32,H=48,D=32) to (C=8,D=32,H=48,W=32)
        deepVoxels_data = np.transpose(deepVoxels_data, (0,3,2,1))
        dict_to_return["deepVoxels"] = torch.from_numpy(deepVoxels_data)

        return dict_to_return

    def get_item(self, index):
        """
        data structures wrt index
            [pitch-idx, yaw-idx, mesh-idx] of sizes (len(self.pitch_list), len(self.yaw_list), len(self.subject))
        """

        # init.
        visualCheck_0 = False
        visualCheck_1 = False

        # ----- determine {volume_id, view_front_id, view_right_id, view_back_id, view_left_id} -----

        if not self.is_train: index += len(self.training_inds)

        volume_id  = index // 4 * 4
        view_id    = index - volume_id
        front_id   = volume_id + view_id
        right_id   = volume_id + (view_id+1) % 4
        back_id    = volume_id + (view_id+2) % 4
        left_id    = volume_id + (view_id+3) % 4
        index_list = [front_id, right_id, back_id, left_id]
        index_list_names = ["front", "right", "back", "left"]

        # ----- load "name", "index", "mesh_path", "b_min", "b_max", "in_black_list" -----

        # read config and get gt mesh path
        config_path = "%s/config/%06d.json" % (self.opt.datasetDir, index)
        with open(config_path) as f: dataConfig = json.load(f)
        meshPath = dataConfig["meshPath"]
        meshName = meshPath.split("/")[-3] + "+" + meshPath.split("/")[-2]

        res = {"name"         : meshName,
               "index"        : index,
               "mesh_path"    : meshPath,
               "b_min"        : self.B_MIN,
               "b_max"        : self.B_MAX,
               "in_black_list": ("%06d"%index) in consts.black_list_images
              }

        # ----- load "img", "calib", "extrinsic", "mask" -----

        """
        render_data
            'img'      : [num_views, C, H, W] RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            'calib'    : [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask'     : [num_views, 1, H, W] for 1x512x512 masks, float 1.0-inside, 0.0-outside
        """
        render_data = self.get_render_iccv(args=self.opt, num_views=self.num_views, view_id_list=index_list, random_sample=self.opt.random_multiview, dataConfig=dataConfig,view_id=view_id)
        res.update(render_data)

        # ----- load "samples", "labels" if needed -----

        if self.opt.num_sample_inout: # default: 5000

            """
            sample_data is a dict.
                "samples" : (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                "labels"  : (1, n_in + n_out), float 1.0-inside, 0.0-outside
            """
            # sample_data = self.select_sampling_method_iccv_online(dataConfig) if self.opt.online_sampling else self.select_sampling_method_iccv_offline()
            sample_data = self.select_sampling_method_iccv_offline(index) if (self.is_train and (not self.opt.online_sampling)) else self.select_sampling_method_iccv_online(dataConfig)
            res.update(sample_data)
        
            # check "calib" by projecting "samplings" onto "img"
            if visualCheck_0:
                print("visualCheck_0: see if 'samples' can be properly projected to the 'img' by 'calib'...")

                # image RGB
                img = np.uint8((np.transpose(res['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0) # HWC, BGR, float 0 ~ 255, de-normalized by mean 0.5 and std 0.5
                cv2.imwrite("./sample_images/%06d_img.png"%(index), img)

                # mask
                mask = np.uint8(res['mask'][0,0].numpy() * 255.0) # (512, 512), 255 inside, 0 outside
                cv2.imwrite("./sample_images/%06d_mask.png"%(index), mask)

                # orthographic projection
                rot   = res['calib'][0,:3, :3]  # R_norm(-1,1)Cam_model, assuming that ortho. proj.: cam_f==256, c_x==c_y==256, img.shape==(512,512,3)
                trans = res['calib'][0,:3, 3:4] # T_norm(-1,1)Cam_model
                pts = torch.addmm(trans, rot, res['samples'][:, res['labels'][0] > 0.5])  # (3, 2500)
                pts = 0.5 * (pts.numpy().T + 1.0) * 512 # (2500,3), ortho. proj.: cam_f==256, c_x==c_y==256, img.shape==(512,512,3)
                imgProj = cv2.UMat(img)
                for p in pts: cv2.circle(imgProj, (p[0], p[1]), 2, (0,255,0), -1)
                cv2.imwrite("./sample_images/%06d_img_ptsProj.png"%(index), imgProj.get())     

                # save points in 3d
                samples_roted = torch.addmm(trans, rot, res['samples']) # (3, N)
                samples_roted[2,:] *= -1
                save_samples_truncted_prob("./sample_images/%06d_samples.ply"%(index), samples_roted.T, res["labels"].T)

        # ----- load "color_samples", "rgbs" -----

        if self.num_sample_color: # default: 0

            """
            color_data
                "color_samples" : (3, num_sample_color), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                "rgbs"          : (3, num_sample_color), -1 ~ 1 float
            """
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)

        # ----- load "meshVoxels" -----

        if self.opt.load_single_view_meshVoxels:

            # "meshVoxels"     : (C-1, D-128, H-192, W-128), 1-inside, 0-outsie
            # "view_directions": an integer, {0, 1, 2, 3} maps to {front, right, back, left}
            # "target_mask"    : (C-1, H-384, W-256) masks, float 1.0-inside, 0.0-outside
            # "target_view"    : (C-3, H-384, W-256) RGB images, float -1. ~ 1., bg is all ZEROS not -1. 
            meshVoxels_data = self.get_mesh_voxels(idx=index,view_id_list=index_list)
            res.update(meshVoxels_data)

            # check if the loaded mesh voxels are correct
            if visualCheck_1:

                print("visualCheck_1: check if the loaded mesh voxels are correct...")

                # ----- input view, namely front view -----
                if True:

                    # img
                    img = np.uint8((np.transpose(res['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0) # HWC, BGR, float 0 ~ 255, de-normalized by mean 0.5 and std 0.5
                    cv2.imwrite("./sample_images/%06d_img_input.png"%(index), img)

                    # mesh voxels
                    meshVoxels_check = res["meshVoxels"][0].numpy() # DHW
                    meshVoxels_check = np.transpose(meshVoxels_check, (2,1,0)) # WHD, XYZ
                    save_volume(meshVoxels_check, fname="./sample_images/%06d_meshVoxels_input.obj"%(index), dim_h=consts.dim_h, dim_w=consts.dim_w, voxel_size=consts.voxel_size)

                # ----- target view, a random view from {front, right, back, left} -----
                if self.opt.use_view_pred_loss:

                    # init.
                    target_view_name = index_list_names[res["view_directions"]]

                    # target img
                    img = np.uint8((np.transpose(res['target_view'].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0) # HWC, BGR, float 0 ~ 255, de-normalized by mean 0.5 and std 0.5
                    cv2.imwrite("./sample_images/%06d_img_target_%s.png"%(index,target_view_name), img)

                    # target mesh voxels (this is to verify voxel transformation functions)
                    meshVoxels_target = self.transform_voxels_to_target(voxels_CDHW=res["meshVoxels"], target_view_idx=res["view_directions"])
                    meshVoxels_target = np.transpose(meshVoxels_target[0].numpy(), (2,1,0)) # WHD, XYZ
                    save_volume(meshVoxels_target, fname="./sample_images/%06d_meshVoxels_target_%s.obj"%(index,target_view_name), dim_h=consts.dim_h, dim_w=consts.dim_w, voxel_size=consts.voxel_size)

                pdb.set_trace()

        # ----- load "deepVoxels" -----

        if self.opt.deepVoxels_fusion != None:

            # "deepVoxels": (C=8,D=32,H=48,W=32), np.float32, only positive values
            deepVoxels_data = self.get_deepVoxels(idx=index)
            res.update(deepVoxels_data)

        # return a data point of dict. structure
        return res

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

    def get_item_ori(self, index):
        """
        data structures wrt index
            [pitch-idx, yaw-idx, mesh-idx] of sizes (len(self.pitch_list), len(self.yaw_list), len(self.subject))
        """

        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)  # mesh-idx
        tmp = index // len(self.subjects) 
        yid = tmp % len(self.yaw_list)    # yaw-idx
        pid = tmp // len(self.yaw_list)   # pitch-idx

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid] # name of a SINGLE mesh, e.g. e.g. ["rp_dennis_posed_004"]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid, random_sample=self.opt.random_multiview)
        res.update(render_data)

        if self.opt.num_sample_inout: # default: 5000
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)
        
        # img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        # rot = render_data['calib'][0,:3, :3]
        # trans = render_data['calib'][0,:3, 3:4]
        # pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
        # pts = 0.5 * (pts.numpy().T + 1.0) * 512
        # for p in pts:
        #     img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        if self.num_sample_color: # default: 0
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        return res
        # except Exception as e:
        #     print(e)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)

class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):

        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')

        self.B_MIN = np.array([-128, -28, -128])
        self.B_MAX = np.array([128, 228, 128])

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout # inside/outside 3d-sampling-points for occupancy query, default: 5000
        self.num_sample_color = self.opt.num_sample_color

        self.yaw_list = list(range(0,360,1))
        self.pitch_list = [0]
        self.subjects = self.get_subjects() # a sorted list of {training OR test} mesh folder names, e.g. [rp_dennis_posed_004]

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        self.mesh_dic = load_trimesh(self.OBJ) # a dict of ALL meshes, indexed by mesh-names

    def get_subjects(self):
        """
        Return
            a sorted list of {training OR test} mesh folder names, e.g. [rp_dennis_posed_004]
        """

        all_subjects = os.listdir(self.RENDER) # a list of ALL mesh folder names, e.g. [rp_dennis_posed_004]
        var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str) # a list of test mesh folder names, e.g. [rp_dennis_posed_004], can also be Empty
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects))) # a sorted list of training mesh folder names, e.g. [rp_dennis_posed_004]
        else:
            return sorted(list(var_subjects)) # a sorted list of test mesh folder names, e.g. [rp_dennis_posed_004]

    def __len__(self):

        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list) # (number-of-training/test-meshes * 360-degree-renderings) := namely, the-number-of-training-views

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] for 3x512x512 images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] for 1x512x512 masks
        '''

        # init.
        render_list = []
        calib_list = []
        extrinsic_list = []
        mask_list = []

        # pitch-idx
        pitch = self.pitch_list[pid] # default: 0

        # yaw-idx (list, if multi-view enabled)
        view_ids = [self.yaw_list[(yid + len(self.yaw_list)//num_views*offset) % len(self.yaw_list)] for offset in range(num_views)]
        if random_sample: view_ids = np.random.choice(self.yaw_list, num_views, replace=False)
        
        # for each yaw-idx
        for vid in view_ids:

            # paths
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            param_path  = os.path.join(self.PARAM , subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            mask_path   = os.path.join(self.MASK  , subject, '%d_%d_%02d.png' % (vid, pitch, 0))

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center') # T_world_cam, model centroid should be aligned with the cam center (thus be the same for the whole dataset assuming object centroids are also aligned with the 3d volume center), so that after converting to the cam-coord., depth-z can exist in a consistent range for different objects/cameras.
            # model rotation
            R = param.item().get('R')           # R_cam_world

            # cam. extrinsic
            translate = -np.matmul(R, center).reshape(3, 1) # T_cam_world = -R_cam_world*T_world_cam
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0) # model will be transformed into a XY-plane-center-aligned-256x256x256-volume of the cam. coord.

            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio # scale/ortho_ratio should be the same for the whole dataset, e.g. 2.4049856956795232
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio # ideally should also have -1 for depth-z, otherwise the cam coord is not strictly right-hand-coord.

            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)

            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            # rgb image and it mask
            render = Image.open(render_path).convert('RGB')
            mask   = Image.open(mask_path).convert('L')

            # training data augmentation, doesn't seems to make sense, I don't think I need this
            if self.is_train:

                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float() # 1. inside, 0. outside, (1, 512, 512)
            mask_list.append(mask)

            render = self.to_tensor(render) # cHW, float -1 ~ 1, (3, 512, 512)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {'img'      : torch.stack(render_list   , dim=0),
                'calib'    : torch.stack(calib_list    , dim=0), # model will be transformed into a XY-plane-center-aligned-2x2x2-volume of the cam. coord.
                'extrinsic': torch.stack(extrinsic_list, dim=0),
                'mask'     : torch.stack(mask_list     , dim=0)
               }

    def select_sampling_method(self, subject):

        # # fix random seeds for test, not sure why yet
        # if not self.is_train:
        #     random.seed(1991)
        #     np.random.seed(1991)
        #     torch.manual_seed(1991)
        
        mesh = self.mesh_dic[subject] # get the mesh with name of subject
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout) # uniformly sample 3d-points on the mesh surface
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)

        # add random points within image space
        length = self.B_MAX - self.B_MIN # np.array([256, 256, 256])
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN # uniform sampling within a 256x256x256 volume, surface-points : volume-points ~ 16 : 1
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        # determine {1, 0} occupancy ground-truth
        inside = mesh.contains(sample_points)
        inside_points  = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        # constrain (n_in + n_out) <= self.num_sample_inout
        nin = inside_points.shape[0]
        inside_points  =  inside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else inside_points
        outside_points = outside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else outside_points[:(self.num_sample_inout - nin)]

        # get samples and labels: {1-inside, 0-outside}
        samples = np.concatenate([inside_points, outside_points], 0).T # (3, n_in + n_out)
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1) # (1, n_in + n_out)
        # save_samples_truncted_prob('out.ply', samples.T, labels.T)
        # exit()
        samples = torch.Tensor(samples).float() # convert np.array to torch.Tensor 
        labels = torch.Tensor(labels).float()
        del mesh

        return {'samples': samples, # (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                'labels' : labels   # (1, n_in + n_out), 1.0-inside, 0.0-outside
               }

    def get_color_sampling(self, subject, yid, pid=0):
        
        yaw = self.yaw_list[yid]
        pitch = self.pitch_list[pid]

        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        
        # Segmentation mask for the uv render.
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0 # [H, W] bool, True-inside, False-outside

        # UV render. each pixel is the color of the point.
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0 # [H, W, 3] 0 ~ 1 float

        # Normal render. each pixel is the surface normal of the point.
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0 # [H, W, 3] -1 ~ 1 float

        # Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]
        
        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))        # (512^2, 1), bool
        uv_render = uv_render.reshape((-1, 3)) # (512^2, 3), 0 ~ 1 float
        uv_normal = uv_normal.reshape((-1, 3)) # (512^2, 3), -1 ~ 1 float
        uv_pos = uv_pos.reshape((-1, 3))       # (512^2, 3), XYZ coords

        # not that those are surface points, not just vertices, otherwise UV maps are sparse
        surface_colors = uv_render[uv_mask] # 0 ~ 1 float
        surface_normal = uv_normal[uv_mask] # -1 ~ 1 float
        surface_points = uv_pos[uv_mask] # XYZ coords of those mesh surface points in ??? scale

        if self.num_sample_color: # default: 0
            assert(self.num_sample_color <= surface_points.shape[0]-1) # e.g. 5000 vs 158847-1
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color) # minor bug, should be just "surface_points.shape[0]"
            surface_colors = surface_colors[sample_list].T # (3, num_sample_color), 0 ~ 1 float
            surface_normal = surface_normal[sample_list].T # (3, num_sample_color), -1 ~ 1 float
            surface_points = surface_points[sample_list].T # (3, num_sample_color), XYZ coords

        # Sample "num_sample_color" query 3d-points, along the normal directions with Gaussian offset
        normal  = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal # (3, num_sample_color), XYZ coords

        # Normalized to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

        return {'color_samples': samples, # (3, num_sample_color), XYZ coords
                'rgbs': rgbs_color        # (3, num_sample_color), -1 ~ 1 float
               }

    def get_item(self, index):
        """
        data structures wrt index
            [pitch-idx, yaw-idx, mesh-idx] of sizes (len(self.pitch_list), len(self.yaw_list), len(self.subject))
        """

        # init.
        visualCheck_0 = False

        # ----- determine [pitch-idx, yaw-idx, mesh-idx] -----

        sid = index % len(self.subjects)  # mesh-idx
        tmp = index // len(self.subjects) 
        yid = tmp % len(self.yaw_list)    # yaw-idx
        pid = tmp // len(self.yaw_list)   # pitch-idx

        # ----- load "name", "mesh_path", "sid", "yid", "pid", "b_min", "b_max" -----

        subject = self.subjects[sid] # name of a SINGLE mesh, e.g. e.g. ["rp_dennis_posed_004"]
        res = {'name': subject,
               'index': index,
               'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
               'sid': sid,
               'yid': yid,
               'pid': pid,
               'b_min': self.B_MIN,
               'b_max': self.B_MAX,
              }

        # ----- load "img", "calib", "extrinsic", "mask" -----

        """
        render_data
            'img'      : [num_views, C, W, H] RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            'calib'    : [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask'     : [num_views, 1, W, H] for 1x512x512 masks
        """
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid, random_sample=self.opt.random_multiview)
        res.update(render_data)

        # ----- load "samples", "labels" if needed -----

        if self.opt.num_sample_inout: # default: 5000

            """
            sample_data is a dict.
                "samples" : (3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                "labels"  : (1, n_in + n_out), float 1.0-inside, 0.0-outside
            """
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)
        
        # check "calib" by projecting "samplings" onto "img"
        if visualCheck_0:
            print("visualCheck_0: see if 'samples' can be properly projected to the 'img' by 'calib'...")
            img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0) # HWC, RGB, float 0 ~ 255, de-normalized by mean 0.5 and std 0.5
            rot   = render_data['calib'][0,:3, :3]  # R_norm(-1,1)Cam_model, assuming that ortho. proj.: cam_f==256, c_x==c_y==256, img.shape==(512,512,3)
            trans = render_data['calib'][0,:3, 3:4] # T_norm(-1,1)Cam_model
            pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # (3, 2500)
            pts = 0.5 * (pts.numpy().T + 1.0) * 512 # (2500,3), ortho. proj.: cam_f==256, c_x==c_y==256, img.shape==(512,512,3)
            imgProj = cv2.UMat(img)
            for p in pts: cv2.circle(imgProj, (p[0], p[1]), 2, (0,255,0), -1)
            cv2.imwrite("./sample_images/%06d_img.png"%(index), img)
            cv2.imwrite("./sample_images/%06d_img_ptsProj.png"%(index), imgProj.get())            

        # ----- load "color_samples", "rgbs" -----

        if self.num_sample_color: # default: 0

            """
            color_data
                "color_samples" : (3, num_sample_color), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
                "rgbs"          : (3, num_sample_color), -1 ~ 1 float
            """
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)

        # return a data point of dict. structure
        return res

    def get_item_ori(self, index):
        """
        data structures wrt index
            [pitch-idx, yaw-idx, mesh-idx] of sizes (len(self.pitch_list), len(self.yaw_list), len(self.subject))
        """

        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)  # mesh-idx
        tmp = index // len(self.subjects) 
        yid = tmp % len(self.yaw_list)    # yaw-idx
        pid = tmp // len(self.yaw_list)   # pitch-idx

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid] # name of a SINGLE mesh, e.g. e.g. ["rp_dennis_posed_004"]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid, random_sample=self.opt.random_multiview)
        res.update(render_data)

        if self.opt.num_sample_inout: # default: 5000
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)
        
        # img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        # rot = render_data['calib'][0,:3, :3]
        # trans = render_data['calib'][0,:3, 3:4]
        # pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
        # pts = 0.5 * (pts.numpy().T + 1.0) * 512
        # for p in pts:
        #     img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        if self.num_sample_color: # default: 0
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        return res
        # except Exception as e:
        #     print(e)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)






