import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
# import torch
# from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
# import torchvision.transforms as transforms
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

import trimesh
import logging
log = logging.getLogger('trimesh')
log.setLevel(40)

# global consts
B_MIN = np.array([-consts.real_w/2., -consts.real_h/2., -consts.real_w/2.])
B_MAX = np.array([ consts.real_w/2.,  consts.real_h/2.,  consts.real_w/2.])

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
        assert(os.path.exists(configPath)) # config file
        
        # save idx
        testIdxList.append([configPath])

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

def main(args):
    """
    For each frame will save the following items for example, about 0.115 MB
 
        occu_sigma3.5_pts5k/088046_ep000_inPts.npy            0.0573 MB, np.float64, (2500, 3)
        occu_sigma3.5_pts5k/088046_ep000_outPts.npy           0.0573 MB, np.float64, (2500, 3)

    In total, (86976 frames) * (15 epochs) * (0.115 MB) / 1024. = 146.5 G
    """

    # init.
    visualCheck_0 = False

    # create dirs for saving query pts
    # saveQueryDir = "%s/%s" % (args.datasetDir, args.sampleType)
    saveQueryDir = "./%s_split%02d_%02d" % (args.sampleType, args.splitNum, args.splitIdx)
    if not os.path.exists(saveQueryDir): os.makedirs(saveQueryDir)

    # get training/test data indices
    training_inds, testing_inds = get_training_test_indices(args=args,shuffle=False)
    trainIdxList = compute_split_range(testing_inds=training_inds,args=args)
    
    # start query pts sampling
    frameIdx = [0, 0, 0]
    frameIdx[0] = int( trainIdxList[0][0].split("/")[-1].split(".")[0])
    frameIdx[2] = int(trainIdxList[-1][0].split("/")[-1].split(".")[0])+1
    count = 0
    previousMeshPath, mesh, meshVN, meshFN, meshV, randomRot = None, None, None, None, None, np.zeros((3,3))
    timeStart = time.time()
    t0 = time.time()
    for pathList in trainIdxList:
        frameIdx[1] = int(pathList[0].split("/")[-1].split(".")[0])

        # load config
        config_path = pathList[0]
        with open(config_path) as f: dataConfig = json.load(f)

        # only when it's a different mesh: load a mesh
        if dataConfig["meshPath"] != previousMeshPath:
            mesh = trimesh.load(dataConfig["meshPath"])
            meshVN = copy.deepcopy(mesh.vertex_normals) 
            meshFN = copy.deepcopy(mesh.face_normals)   
            meshV  = copy.deepcopy(mesh.vertices)       # np.float64
            previousMeshPath = dataConfig["meshPath"]

        # normalize into volumes of X~[+-0.333], Y~[+-0.5], Z~[+-0.333]
        randomRot           = np.array(dataConfig["randomRot"], np.float32) # by random R
        mesh.vertex_normals = np.dot(mesh.vertex_normals, np.transpose(randomRot))
        mesh.face_normals   = np.dot(mesh.face_normals  , np.transpose(randomRot))
        mesh.vertices, _, _ = voxelization_normalization(np.dot(mesh.vertices,np.transpose(randomRot)))
        t_load_mesh = time.time() - t0

        # for each epoch, sample and save query pts
        t_sample_pts, t_save_pts, t_move_files = 0., 0., 0.
        for epochId in range(args.epoch_range[0], args.epoch_range[1]):

            # uniformly sample points on mesh surface
            t1 = time.time()
            surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * args.num_sample_inout) # (N1,3)

            # add gausian noise to surface points
            sample_points = surface_points + np.random.normal(scale=1.*args.sigma/consts.dim_h, size=surface_points.shape) # (N1, 3)

            # uniformly sample inside the 128x192x128 volume, surface-points : volume-points ~ 16:1
            length = B_MAX - B_MIN
            random_points = np.random.rand(args.num_sample_inout//4, 3) * length + B_MIN # (N2, 3)
            sample_points = np.concatenate([sample_points, random_points], 0) # (N1+N2, 3)
            np.random.shuffle(sample_points) # (N1+N2, 3)

            # determine {1, 0} occupancy ground-truth
            inside = mesh.contains(sample_points)
            inside_points  = sample_points[inside]                 # np.float64
            outside_points = sample_points[np.logical_not(inside)] # np.float64

            # constrain (n_in + n_out) <= self.num_sample_inout
            nin = inside_points.shape[0]
            # inside_points  =  inside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else inside_points
            # outside_points = outside_points[:self.num_sample_inout//2] if nin > self.num_sample_inout//2 else outside_points[:(self.num_sample_inout - nin)]
            if nin > args.num_sample_inout//2:
                inside_points  =  inside_points[:args.num_sample_inout//2]
                outside_points = outside_points[:args.num_sample_inout//2]
            else:
                inside_points  = inside_points
                if outside_points.shape[0] < (args.num_sample_inout - nin):
                    print("Error: outside_points.shape[0] {} < (args.num_sample_inout - nin) {}!".format(outside_points.shape[0], (args.num_sample_inout - nin)))
                    pdb.set_trace()
                outside_points = outside_points[:(args.num_sample_inout - nin)]
            t_sample_pts += (time.time() - t1)

            # save query pts
            t1 = time.time()
            inside_path  = "%s/%06d_ep%03d_inPts.npy"  % (saveQueryDir, frameIdx[1], epochId)
            outside_path = "%s/%06d_ep%03d_outPts.npy" % (saveQueryDir, frameIdx[1], epochId)
            # inside_path  = "./sample_images/%06d_ep%03d_inPts.npy"  % (frameIdx[1], epochId)
            # outside_path = "./sample_images/%06d_ep%03d_outPts.npy" % (frameIdx[1], epochId)
            np.save(inside_path , inside_points)
            np.save(outside_path, outside_points)
            t_save_pts += (time.time() - t1)

            # # move query pts
            # t3 = time.time()
            # inside_path_new  = "%s/%06d_ep%03d_inPts.npy"  % (saveQueryDir, frameIdx[1], epochId)
            # outside_path_new = "%s/%06d_ep%03d_outPts.npy" % (saveQueryDir, frameIdx[1], epochId)
            # os.system("mv %s %s" % (inside_path , inside_path_new ))
            # os.system("mv %s %s" % (outside_path, outside_path_new))
            # t_move_files += (time.time() - t3)

        # visual sanity check
        t2 = time.time()
        if visualCheck_0:

            print("visualCheck_0: see if query samples are inside the volume...")

            # inside pts, red
            samples = np.concatenate([inside_points], 0).T # (3, n_in)
            labels  = np.concatenate([np.ones((1, inside_points.shape[0]))], 1) # (1, n_in)
            save_samples_truncted_prob("./sample_images/%06d_shape_samples_inside.ply"%(frameIdx[1]), samples.T, labels.T)

            # outside pts, green
            samples = np.concatenate([outside_points], 0).T # (3, n_in)
            labels  = np.concatenate([np.zeros((1, outside_points.shape[0]))], 1) # (1, n_in)
            save_samples_truncted_prob("./sample_images/%06d_shape_samples_outside.ply"%(frameIdx[1]), samples.T, labels.T)

            # normalized gt mesh
            gtMesh = {"v": mesh.vertices, "vn": mesh.vertex_normals, "vc": mesh.visual.vertex_colors, "f": mesh.faces, "fn": mesh.face_normals}
            ObjIO.save_obj_data_color(gtMesh, "./sample_images/%06d_meshGT.obj" % (frameIdx[1]))

            pdb.set_trace()

        # un-normalize the mesh
        mesh.vertex_normals = meshVN
        mesh.face_normals   = meshFN
        mesh.vertices       = meshV

        # compute timing info
        count += 1
        hrsPassed = (time.time()-timeStart) / 3600.
        hrsEachIter = hrsPassed / count
        numItersRemain = len(trainIdxList) - count
        hrsRemain = numItersRemain * hrsEachIter # hours that remain
        minsRemain = hrsRemain * 60. # minutes that remain

        # log
        expName = args.sampleType
        t_unnorm_mesh = time.time() - t2
        print("Exp %s, split-%02d/%02d | inference: %06d-%06d-%06d | load_mesh:%.3f, sample_pts:%.3f, save_pts:%.3f, move_files:%.3f, unnorm_mesh:%.3f | remains %.3f m(s) ......" % (expName,args.splitIdx,args.splitNum,frameIdx[0],frameIdx[1],frameIdx[2],t_load_mesh,t_sample_pts,t_save_pts,t_move_files,t_unnorm_mesh,minsRemain))
        t0 = time.time()

if __name__ == '__main__':

    # parse args.
    args = BaseOptions().parse()

    # main function
    main(args=args)













