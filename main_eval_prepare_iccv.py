import pdb # pdb.set_trace()
import argparse
import os
import numpy as np
import time
import DataUtil.ObjIO as ObjIO
from Constants import consts
import glob
import DataUtil.VoxelizerUtil as voxel_util
import json
import copy
from subprocess import call
from MyCamera import ProjectPointsOrthogonal
from MyRenderer import ColoredRenderer
import cv2 as cv
import DataUtil.CommonUtil as util
import torch
import sys
this_file_path_abs       = os.path.dirname(__file__)
target_dir_path_relative = os.path.join(this_file_path_abs, 'pyTorchChamferDistance/chamfer_distance')
target_dir_path_abs      = os.path.abspath(target_dir_path_relative)
sys.path.insert(0, target_dir_path_abs)
from chamfer_distance import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--totalNumFrame', type=int, default="108720", help="total data number: N*M'*4 = 6795*4*4 = 108720")
    parser.add_argument('--trainingDataRatio', type=float, default="0.8")
    parser.add_argument('--datasetDir', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data/humanRender")
    parser.add_argument('--resultsDir', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data/humanRender/deepHumanResults/expName")
    parser.add_argument('--splitNum', type=int, default="8", help="for multi-process running")
    parser.add_argument('--splitIdx', type=int, default="0", help="{0, ..., splitNum-1}")
    parser.add_argument('--compute_vn', action='store_true', help="e.g. pifu doesn't compute 'vn' when saving the mesh, thus we compute it now")
    parser.add_argument('--only_compute_additional_metrics', action='store_true', help="e.g. a patch-fix to add additional metrics for previously evaluated exps.")

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

    meshRefinedPath_list = []
    for eachTestIdx in testing_inds[splitRange[0]:splitRange[1]]:
        if ("%06d"%(eachTestIdx)) in consts.black_list_images: continue
        print("checking %06d-%06d-%06d..." % (testing_inds[splitRange[0]], eachTestIdx, testing_inds[splitRange[1]-1]+1))

        # check existance
        meshRefinedPath = "%s/%06d_meshRefined.obj" % (args.resultsDir,eachTestIdx)
        assert(os.path.exists(meshRefinedPath))
        
        # save path
        meshRefinedPath_list.append(meshRefinedPath)

    return meshRefinedPath_list

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

def inverseRotateY(points,angle):
    """
    Rotate the points by a specified angle., LEFT hand rotation
    """

    angle = np.radians(angle)
    ry = np.array([ [ np.cos(angle), 0., np.sin(angle)],
                    [            0., 1.,            0.],
                    [-np.sin(angle), 0., np.cos(angle)] ]) # (3,3)
    return np.dot(points, ry) # (N,3)

def read_and_canonize_gt_mesh(args,preFix,withTexture=False):

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

def get_canonized_gt_mesh_voxels(args,preFix):

    # get config path
    print("get canonized, gt mesh voxels...")
    config_path = "%s/config/%06d.json" % (args.datasetDir,preFix)

    # read config and get gt mesh path & randomRot
    with open(config_path) as f: dataConfig = json.load(f)
    gtMeshPath = dataConfig["meshPath"]
    randomRot = np.array(dataConfig["randomRot"], np.float32)

    # load gt mesh
    gtMesh = ObjIO.load_obj_data(gtMeshPath)

    # voxel-based canonization for the gt mesh
    gtMesh["vn"] = np.dot(gtMesh["vn"],np.transpose(randomRot))
    # vertsZeroMean, meshNormMean, _ = voxelization_normalization(gtMesh["v"],useScaling=False) # we want to determine scaling factor, after applying Rot jittering so that the mesh fits better into WHD
    # vertsCanonized, _, meshNormScale = voxelization_normalization(np.dot(vertsZeroMean,np.transpose(randomRot)),useMean=False)
    # gtMesh["v"] = vertsCanonized
    gtMesh["v"], _, _ = voxelization_normalization(np.dot(gtMesh["v"],np.transpose(randomRot)))
    del gtMesh["vc"]

    # determine {front,right,back,left} by preFix
    volume_id = preFix // 4 * 4
    view_id = preFix - volume_id # {0,1,2,3} map to {front,right,back,left}

    # rotate the gt mesh by {front,right,back,left} view
    rotAngByViews = [0, -90., -180., -270.]
    gtMesh["vn"] = inverseRotateY(points=gtMesh["vn"],angle=rotAngByViews[view_id]) # normal of the mesh
    gtMesh["v"]  = inverseRotateY(points=gtMesh["v"],angle=rotAngByViews[view_id]) # vertex of the mesh

    # save into .obj
    gtMeshPathNew = gtMeshPath.replace("mesh.obj","mesh_normalized.obj")
    ObjIO.save_obj_data_binary(gtMesh, gtMeshPathNew)
    assert(os.path.exists(gtMeshPathNew))

    # voxelization, XYZ (128,192,128) voxels (not DHW, but WHD), 1 inside, 0 outside
    voxels = voxel_util.voxelize_2(gtMeshPathNew,consts.dim_h,consts.dim_w,consts.voxelizer_path)
    voxels = voxel_util.binary_fill_from_corner_3D(voxels)
    call(["rm", gtMeshPathNew])
    assert(not os.path.exists(gtMeshPathNew))

    return voxels

def render_front_normals(args,mesh,rn):

    # init.
    rn.set(f=mesh['f'], bgcolor=np.zeros(3))
    
    # pifu doesn't compute "vn" when saving the mesh, thus we compute it now
    if args.compute_vn: mesh['vn'] = util.calc_normal(mesh) # normals from marchingCube are only slightly diff. from opendr's

    # front
    ptsToRender = mesh["v"]
    colorToRender = mesh["vn"]*np.array([1.,-1.,-1.]) # change normal format from "regular-depth-outwards" to "rgb-normals" 
    rn.set(v=ptsToRender, vc=(colorToRender+1.)/2.)
    normal_front = np.float32(np.copy(rn.r))
    visMap = rn.visibility_image
    fg_front = np.asarray(visMap != consts.constBackground, np.float32).reshape(visMap.shape)

    # [0,1] -> [-1,1]
    normal_front = 2.*normal_front - 1.

    # unit normalization
    normal_front /= np.linalg.norm(normal_front, ord=2, axis=2, keepdims=True)

    # reset bg color
    normal_front *= fg_front[:,:,None]

    return [normal_front], [fg_front]

def get_front_normals_n_mask(args,preFix):

    # the multi-view data is saved as [id(n): {obj-i,view-0}, id(n+1): {obj-i,view-1}, id(n+2): {obj-i,view-2}, id(n+3): {obj-i,view-3}, ...]
    volume_id = preFix // 4 * 4
    view_id = preFix - volume_id
    front_id = volume_id + view_id

    # ----- load masks -----

    # set paths
    mask_front_path = '%s/maskImage/%06d.jpg' % (args.datasetDir, front_id)

    # sanity check
    if not os.path.exists(mask_front_path):
        print("Can not find %s!!!" % (mask_front_path))
        pdb.set_trace()

    # {read, discretize} data, values only within {0., 1.}
    mask_front = np.round((cv.imread(mask_front_path)[:,:,0]).astype(np.float32)/255.) # (H, W)

    # NN resize to (2H,2W)
    mask_front = cv.resize(mask_front, (2*consts.dim_w,2*consts.dim_h), interpolation=cv.INTER_NEAREST)

    # ----- load normals -----

    # set paths
    normal_front_path = '%s/normalRGB/%06d.jpg' % (args.datasetDir, front_id)

    # sanity check
    if not os.path.exists(normal_front_path):
        print("Can not find %s!!!" % (normal_front_path))
        pdb.set_trace()

    # read data BGR -> RGB
    normal_front = cv.imread(normal_front_path)[:,:,::-1]

    # convert dtype
    normal_front = normal_front.astype(np.float32)/255.

    # scale value ranges [0., 1.] -> [-1., 1.]
    normal_front = 2.*normal_front - 1. # (2H,2W,3)

    # resize to (2H,2W,3)
    normal_front = cv.resize(normal_front, (2*consts.dim_w,2*consts.dim_h))

    # unit normalization
    normal_front /= np.linalg.norm(normal_front, ord=2, axis=2, keepdims=True)

    # reset bg color
    normal_front *= mask_front[:,:,None]

    return [normal_front], [mask_front]

def compute_normal_errors(nml_refi, nml_gt, msk):

    # init.
    msk_sum = np.sum(msk)

    # ----- cos. dis in (0, 2) -----
    cos_diff_map_refi = msk*(1-np.sum(nml_refi*nml_gt, axis=-1, keepdims=True))
    cos_error2 = (np.sum(cos_diff_map_refi) / msk_sum).astype(np.float32)

    # ----- l2 dis in (0, 4) -----
    l2_diff_map_refi = msk*np.linalg.norm(nml_refi-nml_gt, axis=-1, keepdims=True)
    l2_error2 = (np.sum(l2_diff_map_refi) / msk_sum).astype(np.float32)

    return cos_error2, l2_error2

def cos_n_l2_normal_dis(nml_1_list, nml_2_list, gtMask_list):

    # init.
    normal_errors = list() # [cos-dis, l2-dis] normal errors for the front view

    # for each view
    assert(len(nml_1_list) == 1)
    for idx in range(len(nml_1_list)):
        cos_error2, l2_error2 = compute_normal_errors(nml_1_list[idx], nml_2_list[idx], gtMask_list[idx][:,:,None].astype(np.float32))
        normal_errors.append([cos_error2, l2_error2])

    return normal_errors

def compute_point_based_metrics(args,estMeshPath,preFix,chamfer_dist,scale):

    # init.
    chamfer_dis, gtV_2_estM_dis, estV_2_gtM_dis, estMesh = None, None, None, None

    # load canonized est mesh
    estMesh = ObjIO.load_obj_data(estMeshPath)

    # load canonized gt mesh
    gtMesh = read_and_canonize_gt_mesh(args=args,preFix=preFix,withTexture=False)
    visualCheck = False
    if visualCheck:
        print("visualCheck inside compute_point_based_metrics: see if the EST and the GT meshes can align well...")
        ObjIO.save_obj_data_color(estMesh, "./examples/%06d_meshEST_for_pointDis.obj" % (preFix))
        ObjIO.save_obj_data_color( gtMesh,  "./examples/%06d_meshGT_for_pointDis.obj" % (preFix))
        pdb.set_trace()

    # compute gt vertex to est mesh distance
    estMesh_v = torch.from_numpy((estMesh["v"][None,:,:]).astype(np.float32)).cuda().contiguous() # e.g. (46918, 3), torch.float32
    gtMesh_v  = torch.from_numpy( (gtMesh["v"][None,:,:]).astype(np.float32)).cuda().contiguous() # e.g. (93182, 3), torch.float32
    dist_left2right, dist_right2left = chamfer_dist(gtMesh_v, estMesh_v)
    gtV_2_estM_dis = torch.mean(dist_left2right).item()

    # compute est vertex to gt mesh distance
    estV_2_gtM_dis = torch.mean(dist_right2left).item()

    # compute chamfer distance
    chamfer_dis = (gtV_2_estM_dis + estV_2_gtM_dis) / 2.

    # return
    return chamfer_dis*scale, estV_2_gtM_dis*scale, estMesh

def compute_n_save_normal_erros(args,estMeshPath,rn,preFix,estMesh=None):

    # load canonized est mesh
    if estMesh == None:
        estMesh = ObjIO.load_obj_data(estMeshPath)

    # render {front} normals for canonized est mesh, [-1,1] in RGB coord.
    est_normals_list, est_mask_list = render_front_normals(args=args,mesh=estMesh,rn=rn)

    # load {front} normals & mask for canonized est mesh, [-1,1] in RGB coord. and {0.,1.} masks
    gt_normals_list, gt_mask_list = get_front_normals_n_mask(args=args,preFix=preFix)

    # visual check for est/gt normals
    visualCheck = False
    if visualCheck:
        print("Visual check: est/gt normals...")
        firstRow  = np.concatenate(((est_normals_list[0]+1.)/2., (est_normals_list[1]+1.)/2., (est_normals_list[2]+1.)/2., (est_normals_list[3]+1.)/2.), axis=1)
        secondRow = np.concatenate((est_mask_list[0],est_mask_list[1],est_mask_list[2],est_mask_list[3]), axis=1)
        secondRow = np.concatenate((secondRow[:,:,None],secondRow[:,:,None],secondRow[:,:,None]), axis=2)
        firstRow  = firstRow * secondRow
        thirdRow  = np.concatenate(( (gt_normals_list[0]+1.)/2.,  (gt_normals_list[1]+1.)/2.,  (gt_normals_list[2]+1.)/2.,  (gt_normals_list[3]+1.)/2.), axis=1)
        fourthRow = np.concatenate((gt_mask_list[0],gt_mask_list[1],gt_mask_list[2],gt_mask_list[3]), axis=1)
        fourthRow = np.concatenate((fourthRow[:,:,None],fourthRow[:,:,None],fourthRow[:,:,None]), axis=2)
        thirdRow  = thirdRow * fourthRow
        fullImage = (np.concatenate((firstRow,secondRow,thirdRow,fourthRow), axis=0)*255.).astype(np.uint8)[:,:,::-1]
        cv.imwrite("./examples/%06d_evalPrepare_normals.png"%(preFix), fullImage)
        pdb.set_trace()

    # compute normal errors
    normal_errors = cos_n_l2_normal_dis(est_normals_list, gt_normals_list, gt_mask_list)

    return normal_errors

def main(args):
    """
    for each mesh, render 4-view-normals and get mesh-voxels, also compute and save {3D-IoU, Normal errors}
    """

    # flags for visual sanity check
    visualCheck_0 = False

    # init.
    rn = ColoredRenderer()
    rn.camera = ProjectPointsOrthogonal(rt=np.array([0,0,0]), t=np.array([0,0,2]), f=np.array([consts.dim_h*2,consts.dim_h*2]), c=np.array([consts.dim_w,consts.dim_h]), k=np.zeros(5))
    rn.frustum = {'near': 0.5, 'far': 25, 'height': consts.dim_h*2, 'width': consts.dim_w*2}
    chamfer_dist = ChamferDistance()

    # get training/test data indices
    training_inds, testing_inds = get_training_test_indices(args=args,shuffle=False)
    meshRefinedPath_list = compute_split_range(testing_inds=testing_inds,args=args)

    # for each mesh, render 4-view-normals and get mesh-voxels, also compute and save {3D-IoU, Normal errors}
    frameIdx = [0, 0, 0]
    frameIdx[0] = int( meshRefinedPath_list[0].split("/")[-1].split("_meshRefined")[0])
    frameIdx[2] = int(meshRefinedPath_list[-1].split("/")[-1].split("_meshRefined")[0])+1
    count = 0
    timeStart = time.time()
    for meshPath in meshRefinedPath_list:

        # init.
        frameIdx[1] = int(meshPath.split("/")[-1].split("_meshRefined")[0])
        evalMetricsPath      = "%s/%06d_evalMetrics.json" % (args.resultsDir, frameIdx[1])
        evalMetricsPath_Next = "%s/%06d_evalMetrics.json" % (args.resultsDir, frameIdx[1]+1)
        evalMetricsPath_additional      = "%s/%06d_evalMetrics_additional.json" % (args.resultsDir, frameIdx[1])
        evalMetricsPath_Next_additional = "%s/%06d_evalMetrics_additional.json" % (args.resultsDir, frameIdx[1]+1)
        if os.path.exists(evalMetricsPath) and os.path.exists(evalMetricsPath_Next) and os.path.exists(evalMetricsPath_additional) and os.path.exists(evalMetricsPath_Next_additional):
            continue

        # ----- compute point based distance metrics -----
        if True:

            # note that the losses have been multiplied by "scale"
            chamfer_dis, estV_2_gtM_dis, estMesh = compute_point_based_metrics(args=args,estMeshPath=meshPath,preFix=frameIdx[1],chamfer_dist=chamfer_dist,scale=10000.)

        # ----- save the additional eval metrics into .json of args.resultsDir dir -----
        if True:

            evalMetrics_additional = {"chamfer_dis"   : chamfer_dis,
                                      "estV_2_gtM_dis": estV_2_gtM_dis}
            with open(evalMetricsPath_additional, "w") as outfile:
                json.dump(evalMetrics_additional, outfile)
            if visualCheck_0:
                print("check eval metrics additional json results...")
                print(evalMetrics_additional)
                os.system("cp %s ./examples/%06d_evalPrepare_metrics_additional.json" % (evalMetricsPath_additional,frameIdx[1]))
                pdb.set_trace()

        # ----- render front-view-normal & compute normal errors of [cos-dis, l2-dis] -----
        if not args.only_compute_additional_metrics:

            normal_errors = compute_n_save_normal_erros(args=args,estMeshPath=meshPath,rn=rn,preFix=frameIdx[1],estMesh=estMesh)
            assert(normal_errors.shape[0] == 1 and normal_errors.shape[1] == 2)

        # ----- save the eval metrics into .json of args.resultsDir dir -----
        if not args.only_compute_additional_metrics:

            evalMetrics = {"norm_cos_dis_ft": np.array([normal_errors[0][0]]).tolist(),
                           "norm_l2_dis_ft":  np.array([normal_errors[0][1]]).tolist()}
            with open(evalMetricsPath, 'w') as outfile:
                json.dump(evalMetrics, outfile)
            visualCheck = False
            if visualCheck:
                print("check eval metrics json results...")
                print(evalMetrics)
                os.system("cp %s ./examples/%06d_evalPrepare_metrics.json" % (evalMetricsPath,frameIdx[1]))
                pdb.set_trace()

        # compute timing info
        count += 1
        hrsPassed = (time.time()-timeStart) / 3600.
        hrsEachIter = hrsPassed / count
        numItersRemain = len(meshRefinedPath_list) - count
        hrsRemain = numItersRemain * hrsEachIter # hours that remain
        minsRemain = hrsRemain * 60. # minutes that remain

        # log
        expName = args.resultsDir.split("/")[-1]
        print("Exp. %s inference: split %d/%d | frameIdx %06d-%06d-%06d | remains %.3f m(s) ......" % (expName,args.splitIdx,args.splitNum,frameIdx[0],frameIdx[1],frameIdx[2],minsRemain))


if __name__ == '__main__':

    # parse args.
    args = parse_args()

    # main function
    main(args=args)




