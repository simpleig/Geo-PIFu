import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

import copy
import pdb # pdb.set_trace()
this_file_path_abs       = os.path.dirname(__file__)
target_dir_path_relative = os.path.join(this_file_path_abs, '../..')
target_dir_path_abs      = os.path.abspath(target_dir_path_relative)
sys.path.insert(0, target_dir_path_abs)
from Constants import consts
target_dir_path_relative = os.path.join(this_file_path_abs, '../../DataUtil')
target_dir_path_abs      = os.path.abspath(target_dir_path_relative)
sys.path.insert(0, target_dir_path_abs)
import ObjIO

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSize', type=int, default="1")
    parser.add_argument('--first_channel_len_of_rgb_N_sem_input', type=int, default="8")
    parser.add_argument('--bottleWidth', type=int, default="4")

    args = parser.parse_args()

    return args

def get_training_test_indices(args, shuffle):

    # sanity check for args.totalNumFrame
    assert(os.path.exists(args.datasetDir))
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

def compute_split_range(testing_inds, args):
    """
    determine split range, for multi-process running
    """

    dataNum = len(testing_inds)
    splitLen = int(np.ceil(1.*dataNum/args.splitNum))
    splitRange = [args.splitIdx*splitLen, min((args.splitIdx+1)*splitLen, dataNum)]

    testIdxList = []
    for eachTestIdx in testing_inds[splitRange[0]:splitRange[1]]:
        if ("%06d"%(eachTestIdx)) in consts.black_list_images: continue
        print("checking %06d-%06d-%06d..." % (testing_inds[splitRange[0]], eachTestIdx, testing_inds[splitRange[1]-1]+1))

        # check existance
        configPath = "%s/config/%06d.json" % (args.datasetDir, eachTestIdx)
        maskPath  = "%s/maskImage/%06d.jpg" % (args.datasetDir, eachTestIdx)
        rgbImagePath = "%s/rgbImage/%06d.jpg"% (args.datasetDir, eachTestIdx)
        meshVoxelsPath = "%s/meshVoxels/%06d.npy" % (args.datasetDir, eachTestIdx)
        assert(os.path.exists(configPath)) # config file
        assert(os.path.exists(maskPath)) # mask
        assert(os.path.exists(rgbImagePath)) # rgb image
        assert(os.path.exists(meshVoxelsPath)) # gt mesh voxels path

        # save idx
        testIdxList.append([configPath,maskPath,rgbImagePath,meshVoxelsPath])

        # early break if need visual demo
        if len(testIdxList) == args.visual_demo_mesh:
            break

    return testIdxList

def use_provided_idx(args):

    testIdxList = []
    for eachTestIdx in args.give_idx:
        if ("%06d"%(eachTestIdx)) in consts.black_list_images: continue
        print("checking %06d..." % (eachTestIdx))

        # check existance
        configPath = "%s/config/%06d.json" % (args.datasetDir, eachTestIdx)
        maskPath  = "%s/maskImage/%06d.jpg" % (args.datasetDir, eachTestIdx)
        rgbImagePath = "%s/rgbImage/%06d.jpg"% (args.datasetDir, eachTestIdx)
        meshVoxelsPath = "%s/meshVoxels/%06d.npy" % (args.datasetDir, eachTestIdx)
        assert(os.path.exists(configPath)) # config file
        assert(os.path.exists(maskPath)) # mask
        assert(os.path.exists(rgbImagePath)) # rgb image
        assert(os.path.exists(meshVoxelsPath)) # gt mesh voxels path

        # save idx
        testIdxList.append([configPath,maskPath,rgbImagePath,meshVoxelsPath])

    return testIdxList

def inverseRotateY(points,angle):
    """
    Rotate the points by a specified angle., LEFT hand rotation
    """

    angle = np.radians(angle)
    ry = np.array([ [ np.cos(angle), 0., np.sin(angle)],
                    [            0., 1.,            0.],
                    [-np.sin(angle), 0., np.cos(angle)] ]) # (3,3)
    return np.dot(points, ry) # (N,3)

def voxelization_normalization(verts,useMean=True,useScaling=True):
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

def read_and_canonize_gt_mesh(preFix,args,withTexture=False):

    # get config path
    config_path = "%s/config/%06d.json" % (args.datasetDir, preFix)

    # read config and get gt mesh path & randomRot
    with open(config_path) as f: dataConfig = json.load(f)
    gtMeshPath = dataConfig["meshPath"]
    randomRot = np.array(dataConfig["randomRot"], np.float32)

    # load gt mesh
    gtMeshPath = "%s/../DeepHumanDataset/dataset/%s/%s/mesh.obj" % (args.datasetDir, gtMeshPath.split("/")[-3], gtMeshPath.split("/")[-2])
    gtMesh = ObjIO.load_obj_data(gtMeshPath)

    # voxel-based canonization for the gt mesh
    gtMesh["vn"] = np.dot(gtMesh["vn"],np.transpose(randomRot))
    # vertsZeroMean, meshNormMean, _ = voxelization_normalization(gtMesh["v"],useScaling=False) # we want to determine scaling factor, after applying Rot jittering so that the mesh fits better into WHD
    # vertsCanonized, _, meshNormScale = voxelization_normalization(np.dot(vertsZeroMean,np.transpose(randomRot)),useMean=False)
    # gtMesh["v"] = vertsCanonized
    gtMesh["v"], _, _ = voxelization_normalization(np.dot(gtMesh["v"],np.transpose(randomRot)))

    # determine {front,right,back,left} by preFix
    volume_id = preFix // 4 * 4
    view_id = preFix - volume_id # {0,1,2,3} map to {front,right,back,left}

    # rotate the gt mesh by {front,right,back,left} view
    rotAngByViews = [0, -90., -180., -270.]
    gtMesh["vn"] = inverseRotateY(points=gtMesh["vn"],angle=rotAngByViews[view_id]) # normal of the mesh
    gtMesh["v"]  = inverseRotateY(points=gtMesh["v"],angle=rotAngByViews[view_id]) # vertex of the mesh

    # del. texture of the gt mesh
    if not withTexture: del gtMesh["vc"]

    # return the gt mesh
    return gtMesh

class Evaluator:

    def __init__(self, opt, projection_mode='orthogonal'):

        # init.
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # set cuda
        assert(torch.cuda.is_available())
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netV = VrnNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netV.name)

        # load well-trained weights
        if opt.load_netV_checkpoint_path is not None:
            print('loading for net V ...', opt.load_netV_checkpoint_path)
            assert(os.path.exists(opt.load_netV_checkpoint_path))
            if opt.load_from_multi_GPU_shape    : netV.load_state_dict(self.load_from_multi_GPU(path=opt.load_netV_checkpoint_path), strict= not opt.partial_load)
            if not opt.load_from_multi_GPU_shape: netV.load_state_dict(torch.load(opt.load_netV_checkpoint_path, map_location=cuda), strict= not opt.partial_load)

        self.cuda = cuda
        self.netV = netV

        # show parameter sizes and FLOP count
        parameter_size_V_enc_dec  = sum([param.nelement() for param in netV.parameters()])
        parameter_size_V_only_enc = 0.
        for name, param in netV.named_parameters():
            if "cls" not in name:
                parameter_size_V_only_enc += param.nelement()
        flop_count                = 0.
        print("Model computation cost: parameter_size_V_enc_dec({}), parameter_size_V_only_enc({}), flop_count({})...".format(parameter_size_V_enc_dec, parameter_size_V_only_enc, flop_count))

    def load_from_multi_GPU(self, path):

        # original saved file with DataParallel
        state_dict = torch.load(path)

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        return new_state_dict

    def load_image(self, image_path, mask_path, configPath, visual_demo_flag, meshVoxels_path):

        # init.
        data_return = {}

        # ----- mesh name -----
        if True:

            with open(configPath) as f: dataConfig = json.load(f)
            meshPath = dataConfig["meshPath"]
            meshName = meshPath.split("/")[-3] + "+" + meshPath.split("/")[-2]

            # push into dict.
            data_return["name"] = meshName

        # ----- load mask -----
        if True:

            # {read, discretize} data, values only within {0., 1.}
            mask_data = np.round((cv2.imread(mask_path)[:,:,0]).astype(np.float32)/255.) # (1536, 1024)
            mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32) # (1536, 1536)
            mask_data_padded[:,mask_data_padded.shape[0]//2-min(mask_data.shape)//2:mask_data_padded.shape[0]//2+min(mask_data.shape)//2] = mask_data # (1536, 1536)

            # NN resize to (512, 512)
            mask_data_padded = cv2.resize(mask_data_padded, (self.opt.loadSize,self.opt.loadSize), interpolation=cv2.INTER_NEAREST)
            mask_data_padded = Image.fromarray(mask_data_padded)

            # convert to (1, 512, 512) tensors, float, 1-fg, 0-bg
            mask_data_padded = transforms.ToTensor()(mask_data_padded).float() # 1. inside, 0. outside

            # push into dict.
            data_return["mask"] = mask_data_padded.unsqueeze(0) # (1, 1, 512, 512), 1-fg, 0-bg

        # ----- load image -----
        if True:

            # read data BGR -> RGB, np.uint8
            image = cv2.imread(image_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8) # (1536, 1536, 3)
            image_padded[:,image_padded.shape[0]//2-min(image.shape[:2])//2:image_padded.shape[0]//2+min(image.shape[:2])//2,:] = image # (1536, 1536, 3)

            # resize to (512, 512, 3), np.uint8
            image_padded = cv2.resize(image_padded, (self.opt.loadSize, self.opt.loadSize))
            image_padded = Image.fromarray(image_padded)

            # convert to (3, 512, 512) tensors, RGB, float, -1 ~ 1. note that bg is 0 not -1.
            image_padded = self.to_tensor(image_padded) # (3, 512, 512), float -1 ~ 1
            image_padded = mask_data_padded.expand_as(image_padded) * image_padded

            # push into dict.
            data_return["img"] = image_padded.unsqueeze(0) # BCHW, (1,3,512,512), float -1 ~ 1, bg are all ZEROS, not -1.

        # ----- load gt meshVoxels -----
        if visual_demo_flag:

            # load data
            mesh_volume = np.load(meshVoxels_path) # dtype('bool'), WHD-(128, 192, 128)

            # WHD -> DHW (as required by tensorflow)
            mesh_volume = np.transpose(mesh_volume, (2, 1, 0))
            
            # convert dtype
            mesh_volume = mesh_volume.astype(np.float32)        # (128,192,128)
            mesh_volume = torch.from_numpy(mesh_volume)[None,:] # (1,128,192,128)

            # push into dict.
            data_return["meshVoxels"] = mesh_volume # (C=1, D=128, H=192, W=128), float 1.0-inside, 0.0-outside

        return data_return

    def eval(self, data, save_path=None, visual_demo_flag=False):
        '''
        Evaluate a data point
        '''

        # init.
        opt = self.opt
        save_path_png, save_path_gt_obj = None, None

        # marching cube for mesh generation
        with torch.no_grad():

            # set to eval mode
            self.netV.eval()
            if self.opt.name in self.opt.must_run_in_train_modes: self.netV.train()
            
            # generate and save the mesh
            save_path_png, save_path_gt_obj = gen_mesh_vrn(opt, self.netV, self.cuda, data, save_path, also_generate_mesh_from_gt_voxels=visual_demo_flag)

        return save_path_png, save_path_gt_obj

    def prepare_deepVoxels(self, data, save_path, save_deepVoxels=True, save_meshVoxels=False):

        # marching cube for mesh generation
        with torch.no_grad():

            # set to eval mode
            self.netV.eval()
            if self.opt.name in self.opt.must_run_in_train_modes: self.netV.train()

            # retrieve and reshape the data for one frame (or multi-view)
            image_tensor = data['img'].to(device=self.cuda) # (V=1, C=3, H=512, W=512), RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            image_tensor = image_tensor.view(-1, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1]) # (V=1, C=3, H=512, W=512)
            
            # get deepVoxels, use hour-glass networks to extract image features
            if save_deepVoxels:
                self.netV.filter(image_tensor)
                deepVoxels = self.netV.im_feat_list[-1] # torch.float32, (B=1,C=8,D=32,H=48,W=32), etc. deepVoxels
                deepVoxels = deepVoxels[0].detach().cpu().numpy() # (C=8,D=32,H=48,W=32)
                deepVoxels = np.transpose(deepVoxels, (0,3,2,1)) # (C=8,W=32,H=48,D=32), C-XYZ, np.float32, has both posi/neg values
                np.save(save_path, deepVoxels) # 1.6M

            # get est. occu.
            if save_meshVoxels:
                self.netV.est_occu()
                pred_occ = self.netV.get_preds() # torch.float32, BCDHW, (B,1,128,192,128), est. occupancy
                pred_occ = pred_occ[0,0].detach().cpu().numpy() # DHW
                pred_occ = np.transpose(pred_occ, (2,1,0)) # (W=128,H=192,D=128), XYZ, np.float32, 0. ~ 1.
                save_path = save_path.replace("_deepVoxels.npy", "_meshVoxels.npy")
                np.save(save_path, pred_occ) # 13M

def main(args):
    """
    For each test image will save the following items for example, about 2.6 MB
 
        088046_meshRefined.obj            2.5 MB
        088046_meshRefined.png             52 KB

    In total, (21744 test images) * (2.6 MB) / 1024. = 55 G

    For each train image will save the following items for example, about 1.6 MB

        000000_deepVoxels.npy             1.6 MB, (C=8,W=32,H=48,D=32), C-XYZ, np.float32, has both posi/neg values

    In total, (86976+21744 train+test images) * (1.6 MB) / 1024. = 170 G
    """

    # init.
    visual_demo_flag = (args.visual_demo_mesh > 0) or (args.give_idx[0] != None)

    # create dirs for saving results and debug info.
    if not os.path.exists(args.resultsDir): os.makedirs(args.resultsDir)

    # get training/test data indices
    if args.give_idx[0] is None: training_inds, testing_inds = get_training_test_indices(args=args,shuffle=args.shuffle_train_test_ids)
    if args.give_idx[0] is None: testIdxList = compute_split_range(testing_inds=testing_inds if args.dataType=="test" else training_inds+testing_inds, args=args)
    if args.give_idx[0] is not None: testIdxList = use_provided_idx(args=args)

    # init the Pytorch network
    evaluator = Evaluator(args)
    
    # start inference
    frameIdx = [0, 0, 0]
    frameIdx[0] = int( testIdxList[0][-1].split("/")[-1].split(".")[0])
    frameIdx[2] = int(testIdxList[-1][-1].split("/")[-1].split(".")[0])+1
    count = 0
    timeStart = time.time()
    for pathList in testIdxList:

        """
        load data
            'name'           : meshName,
            'img'            : image_padded.unsqueeze(0),      # BCHW, (1,3,512,512), float -1 ~ 1, bg are all ZEROS, not -1.
            'mask'           : mask_data_padded.unsqueeze(0), # (1, 1, 512, 512), 1-fg, 0-bg
            "meshVoxels"     : (C-1, D-128, H-192, W-128), 1-inside, 0-outsie
            "view_directions": an integer, {0, 1, 2, 3} maps to {front, right, back, left}
            "target_mask"    : (C-1, H-384, W-256) masks, float 1.0-inside, 0.0-outside
            "target_view"    : (C-3, H-384, W-256) RGB images, float -1. ~ 1., bg is all ZEROS not -1. 
        """
        data = evaluator.load_image(image_path=pathList[2], mask_path=pathList[1], configPath=pathList[0], visual_demo_flag=visual_demo_flag, meshVoxels_path=pathList[3])

        # forward pass and save results
        frameIdx[1] = int(pathList[2].split("/")[-1].split(".")[0])
        save_path_obj = "%s/%06d_meshRefined.obj" % (args.resultsDir, frameIdx[1]) # save the reconstructed mesh obj
        save_path_npy = "%s/%06d_deepVoxels.npy" % (args.resultsDir, frameIdx[1]) # save the estimated deepVoxels npy
        if args.dataType == "test" : save_path_png, save_path_gt_obj = evaluator.eval(data, save_path=save_path_obj, visual_demo_flag=visual_demo_flag)
        if args.dataType == "train": evaluator.prepare_deepVoxels(data=data, save_path=save_path_npy, save_deepVoxels=True, save_meshVoxels=False)

        # compute timing info
        count += 1
        hrsPassed = (time.time()-timeStart) / 3600.
        hrsEachIter = hrsPassed / count
        numItersRemain = len(testIdxList) - count
        hrsRemain = numItersRemain * hrsEachIter # hours that remain
        minsRemain = hrsRemain * 60. # minutes that remain

        # log
        expName = args.resultsDir.split("/")[-1]
        print("Exp %s, GPU-%d, split-%02d/%02d | inference: %06d-%06d-%06d | remains %.3f h(s) ......" % (expName,args.gpu_id,args.splitIdx,args.splitNum,frameIdx[0],frameIdx[1],frameIdx[2],hrsRemain))

        # visual sanity check
        if visual_demo_flag:

            print("saving demo for visualCheck...")

            # the input 512x512x3 image
            os.system("cp %s ./sample_images/" % (save_path_png))

            # the estimated mesh .obj
            os.system("cp %s ./sample_images/" % (save_path_obj))

            # the gt high-resolution mesh .obj
            gtMesh = read_and_canonize_gt_mesh(preFix=frameIdx[1],args=args,withTexture=True)
            ObjIO.save_obj_data_color(gtMesh, "./sample_images/%06d_meshGT.obj" % (frameIdx[1]))

            # the gt low-resolution mesh. obj
            os.system("mv %s ./sample_images/" % (save_path_gt_obj))

if __name__ == '__main__':

    # parse args.
    args = BaseOptions().parse()

    # main function
    main(args=args)













