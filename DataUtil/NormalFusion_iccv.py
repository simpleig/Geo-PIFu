from __future__ import division, print_function
import numpy as np
import math
import scipy
import argparse
import scipy.io as sio
import scipy.signal as sis
import cv2 as cv

from skimage import measure

import CommonUtil as util
import ObjIO
import pdb # pdb.set_trace()
import VoxelizerUtil as voxel_util
import os
import copy
import json

dim_h = 192
dim_w = 128
voxel_size = 1.0 / dim_h
H_NORMALIZE_HALF = 0.5
meshNormMargin = 0.15
threshH = H_NORMALIZE_HALF * (1-meshNormMargin)
threshWD = H_NORMALIZE_HALF * dim_w/dim_h * (1-meshNormMargin)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--volume_file', type=str, required=True, help='path to meshVoxels file (.npy)')
    parser.add_argument('--normal_file', type=str, required=True, help='path to normal map (uint8 .png)')
    parser.add_argument('--datasetDir', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data/humanRender")
    parser.add_argument('--visual_demo_flag', type=int, default="0")

    args = parser.parse_args()

    return args

def load_volume(vol_dir):

    # sanity check
    if not os.path.exists(vol_dir):
        print("Error: can not find %s!" % (vol_dir))
        pdb.set_trace()

    # read meshVoxels
    vol = np.load(vol_dir) # WHD, integers within [0,255]
    vol = vol.astype(np.float32)/255. # WHD, continuous values within (0,1)
    assert(vol.shape == (dim_w,dim_h,dim_w))

    return vol

def extract_orig_mesh(vol):
    assert len(vol.shape) == 3
    vertices, simplices, normals, _ = measure.marching_cubes_lewiner(vol, level=0.5) # https://scikit-image.org/docs/dev/api/skimage.measure.html#marching-cubes-lewiner
    vertices = vertices*2.0
    mesh = dict()
    mesh['v'] = vertices
    mesh['f'] = simplices
    mesh['f'] = mesh['f'][:, (1, 0, 2)] # to ensure that normals computed by opendr are facing outwards wrt. the mesh
    mesh['vn'] = util.calc_normal(mesh) # normals from marchingCube are only slightly diff. from opendr's

    return mesh, normals

def upsample_mesh(mesh):
    new_mesh = dict()
    orig_v_num = len(mesh['v'])
    # find out edges (without repetition)
    edges = np.vstack([np.hstack([mesh['f'][:, 0:1], mesh['f'][:, 1:2]]),
                       np.hstack([mesh['f'][:, 1:2], mesh['f'][:, 2:3]]),
                       np.hstack([mesh['f'][:, 2:3], mesh['f'][:, 0:1]])])

    # constructs a new edge array without repetition
    edges_wo_rep = list()
    edges_dict = dict()
    for e in edges:
        if e[0] > e[1]:
            e[0], e[1] = e[1], e[0]
        k1 = str(e[0]) + '_' + str(e[1])
        k2 = str(e[1]) + '_' + str(e[0])
        if k1 not in edges_dict and k2 not in edges_dict:
            edges_dict[k1] = len(edges_wo_rep) + orig_v_num
            edges_dict[k2] = len(edges_wo_rep) + orig_v_num
            edges_wo_rep.append(e)
    edges_wo_rep = np.asarray(edges_wo_rep)

    # upsamples point cloud by dividing edges
    new_v = np.copy(mesh['v'])
    new_v_middle = (new_v[edges_wo_rep[:, 0], :] +
                    new_v[edges_wo_rep[:, 1], :]) / 2.0
    new_v = np.vstack([new_v, new_v_middle])
    new_mesh['v'] = new_v

    # upsamples point normals
    if 'vn' in mesh and mesh['vn'] is not None and len(mesh['vn'].shape) > 0:
        new_n = np.copy(mesh['vn'])
        new_n_middle = (new_n[edges_wo_rep[:, 0], :] +
                        new_n[edges_wo_rep[:, 1], :]) / 2.0
        new_n = np.vstack([new_n, new_n_middle])
        new_mesh['vn'] = new_n

    # divides triangles
    new_f = list()
    for f in mesh['f']:
        v1, v2, v3 = f[0], f[1], f[2]
        v4 = edges_dict[str(v1) + '_' + str(v2)]
        v5 = edges_dict[str(v2) + '_' + str(v3)]
        v6 = edges_dict[str(v3) + '_' + str(v1)]
        new_f.append(np.array([v1, v4, v6]))
        new_f.append(np.array([v4, v2, v5]))
        new_f.append(np.array([v4, v5, v6]))
        new_f.append(np.array([v6, v5, v3]))

    new_mesh['f'] = np.asarray(new_f)
    return new_mesh

def proj_frontal_mask(vol):
    mask0 = np.max(vol, axis=2) # (W,H,D) -> (W,H), continuous values (0,1)
    mask0 = cv.resize(mask0, (mask0.shape[1] * 2, mask0.shape[0] * 2)) # (2*W, 2*H), continuous values (0,1)
    mask0[np.where(mask0 > 0.5)]  = 1.0 # fg set to 1, (2*W, 2*H)
    mask0[np.where(mask0 <= 0.5)] = 0.0 # bg set to 0, (2*W, 2*H)
    kernel = np.ones((3, 3), np.float32)
    mask0 = cv.erode(mask0, kernel, iterations=2) # erode enforces 0 >> 1, to make fg slimmer, to ignore boundry normals later
    return mask0

def proj_frontal_depth(vol):
    vol_bin = np.zeros_like(vol) # (W,H,D)
    vol_bin[np.where(vol>0.5)] = 1.0
    d_array = np.asarray(range(128), dtype=np.float32)*2 # [  0.,   2.,   4. ..., 254.] of size (128,)
    depthRange = np.max(d_array)
    d_array = d_array[::-1] # cuz our meshVoxels are not front/back flipped
    d_array = np.reshape(d_array, (1, 1, -1)) # (1,1,128)

    vol_dpt_0 = vol_bin * d_array # (W,H,D==128)
    dpt0 = np.max(vol_dpt_0, axis=2) # get front surface, but in reverse-depth format (larger depth means closer to camera), of size (W,H)
    dpt0 = cv.resize(dpt0, (dpt0.shape[1] * 2, dpt0.shape[0] * 2)) # reverse-dront-detph of size (2W,2H)
    return dpt0, depthRange

def load_normal_map(nml_map_dir, mask0):
    """
    normal-of-inverse-depth-outwards  not-inverse-depth-inwards  not-inverse-depth-outwards (combined-pre-two) RGB-normal-coords (combined-all)
    X                                  1                         -1                         -1                  1                -1 
    Y                                  1                         -1                         -1                 -1                 1
    Z                                 -1                         -1                          1                 -1                -1
    """

    # ----- load the masked front normal in "RGB-normal-coords" -----

    normal_front = cv.imread(nml_map_dir)[:,:,::-1] # (2H,2W,XYZ), front normals in "RGB-normal-coords", values in [0, 255]
    normal_front = normal_front.astype(np.float32)/255. # (2H,2W,XYZ), front normals in "RGB-normal-coords", values in (0, 1)
    normal_front = 2.*normal_front - 1. # (2H,2W,XYZ), front normals in "RGB-normal-coords", values in (-1, 1)
    assert(normal_front.shape == (2*dim_h,2*dim_w,3))
    normal_front = np.transpose(normal_front, (1, 0, 2)) # (2H,2W,XYZ) to (2W,2H,XYZ)
    normal_front *= np.expand_dims(mask0, axis=-1) # (2W,2H,XYZ) * (2W,2H,1) == (2W,2H,XYZ), bg set to XYZ of 000

    # ----- change normals from "RGB-normal-coords" to "inverse-depth-inwards" format -----

    normal_front[:,:,0] *= 1 # "RGB-normal-coords" -> "not-inverse-depth-outwards"
    normal_front[:,:,1] *= -1
    normal_front[:,:,2] *= -1

    normal_front[:,:,0] *= 1 # "not-inverse-depth-outwards" -> "inverse-depth-inwards"
    normal_front[:,:,1] *= 1
    normal_front[:,:,2] *= -1

    return normal_front

def smooth_surface_normal(mesh, lamb=0.5, preFix=None):
    nml_neighbor = np.zeros_like(mesh['vn'])
    for f in mesh['f']:
        n0, n1, n2 = mesh['vn'][f[0]], mesh['vn'][f[1]], mesh['vn'][f[2]]
        nml_neighbor[f[0]] += n1 + n2
        nml_neighbor[f[1]] += n2 + n0
        nml_neighbor[f[2]] += n0 + n1
    # nml_neighbor_len2 = np.sum(np.square(nml_neighbor),axis=1, keepdims=True)
    # nml_neighbor /= np.sqrt(nml_neighbor_len2)  # normalize normals

    # nml_neighbor_norm = np.sqrt(np.sum(np.square(nml_neighbor),axis=1, keepdims=True))
    # nml_neighbor_normed= nml_neighbor / nml_neighbor_norm

    nml_neighbor_norm = np.sqrt(np.sum(np.square(nml_neighbor),axis=1, keepdims=True)) # (N,1)
    invalid_normals_flag = (nml_neighbor_norm == 0.)[:,0] # (N,)
    valid_normals_flag = np.logical_not(invalid_normals_flag) # (N,)

    nml_neighbor[invalid_normals_flag] = mesh['vn'][invalid_normals_flag]
    nml_neighbor[valid_normals_flag] = nml_neighbor[valid_normals_flag] / nml_neighbor_norm[valid_normals_flag]

    mesh['vn'] = mesh['vn'] * (1-lamb) + nml_neighbor * lamb

def assigned_normal(mesh, dpt0, nml0, msk0):
    """
    # Wowww?! No optimization at all, this is simply using the refined-normals to update mesh-computed-normals
    """

    kernel = np.ones((3, 3), np.float32)
    msk0 = cv.erode(msk0, kernel, iterations=2)
    wmap = cv.GaussianBlur(msk0, (43, 43), 0)*2.0-1.0
    x_max, y_max = nml0.shape[0]-1, nml0.shape[1]-1
    for vi in range(len(mesh['v'])):
        v = mesh['v'][vi]
        n = mesh['vn'][vi]
        if v[0] < 0 or v[0] >= x_max or v[1] < 0 or v[1] >= y_max: # should be always False due to the spacing=(1,1,1) of marching cube
            continue
        pixel_x = int(round(v[0]))
        pixel_y = int(round(v[1]))

        if msk0[pixel_x, pixel_y] > 0 \
                and wmap[pixel_x, pixel_y] > 0 \
                and abs(dpt0[pixel_x, pixel_y] - v[2]) < 2: # the last statement is kinda like a simple visibility check, to find the visible vertices by depth alignment
            w = wmap[pixel_x, pixel_y]
            normalsFused = nml0[pixel_x, pixel_y, :] * w + mesh['vn'][vi] * (1-w) # Wowww?! No optimization at all, this is simply using the refined-normals to update mesh-computed-normals
            # mesh['vn'][vi] = nml0[pixel_x, pixel_y, :]
            mesh['vn'][vi] = normalsFused / np.linalg.norm(normalsFused) # unit normalization

def mesh_canonization(mesh):
    """
    translate & rescale the mesh from [[0,2W),[0,2H),[0,2D)] ---> [(-0.33,0.33),(-0.5,0.5),(-0.33,0.33)]
    """

    # deep copy
    meshCpy = dict()
    meshCpy['v']  = copy.deepcopy(mesh['v'])
    meshCpy['f']  = copy.deepcopy(mesh['f'])
    meshCpy['vn'] = copy.deepcopy(mesh['vn'])

    # translate
    meshCpy['v'][:,0] -= dim_w # X, from [0,2W) to [-W,W)
    meshCpy['v'][:,1] -= dim_h # Y, from [0,2H) to [-H,H)
    meshCpy['v'][:,2] -= dim_w # Z, from [0,2D) to [-D,D)

    # rescale
    meshCpy['v'][:,0] /= (2*dim_h) # X, from [-W,W) to (-0.33,0.33)
    meshCpy['v'][:,1] /= (2*dim_h) # Y, from [-H,H) to (-0.5,0.5)
    meshCpy['v'][:,2] /= (2*dim_h) # Z, from [-D,D) to (-0.33,0.33)

    return meshCpy

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
        scaleArr = np.array([threshWD/abs(xyzMin[0]), threshH/abs(xyzMin[1]), threshWD/abs(xyzMin[2]), threshWD/xyzMax[0], threshH/xyzMax[1], threshWD/xyzMax[2]])
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
    vertsZeroMean, meshNormMean, _ = voxelization_normalization(gtMesh["v"],useScaling=False) # we want to determine scaling factor, after applying Rot jittering so that the mesh fits better into WHD
    vertsCanonized, _, meshNormScale = voxelization_normalization(np.dot(vertsZeroMean,np.transpose(randomRot)),useMean=False)
    gtMesh["v"] = vertsCanonized

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

def main(args):

    # ----- load meshVoxels and normals -----

    preFix = int(args.volume_file.split("/")[-1].split("_meshVoxels")[0])
    vol = load_volume(args.volume_file) # WHD of continuous values within (0., 1.)
    mask0 = proj_frontal_mask(vol) # front mask of size (2*W, 2*H), 1-fg and 0-bg, slimmer after erode for ignoring the boundry
    dpt0, depthRange = proj_frontal_depth(vol) # reverse-front-detph of size (2W,2H), values in range [0., 254.]
    nml0 = load_normal_map(args.normal_file, mask0) # masked front normals of size (2W,2H,3), in inverse-depth-inwards format, values in XYZ and of range [-1,1]
    if args.visual_demo_flag:
        voxel_util.save_volume(vol > 0.5, "./examples/%06d_meshRef_meshVoxels.obj"%(preFix), dim_h, dim_w, voxel_size)
        voxel_util.save_volume_doubleIdx(vol > 0.5, "./examples/%06d_meshRef_meshVoxelsDoubleIdx.obj"%(preFix)) # this is aligned with the mesh of spacing=(1,1,1)
        frontMaskErosion = (np.transpose(mask0, (1,0))*255.).astype(np.uint8)
        frontMaskErosion = np.concatenate((frontMaskErosion[:,:,None],frontMaskErosion[:,:,None],frontMaskErosion[:,:,None]), axis=2)
        frontDepthInverse = (np.transpose(dpt0, (1,0))).astype(np.uint8)
        frontDepthInverse = np.concatenate((frontDepthInverse[:,:,None],frontDepthInverse[:,:,None],frontDepthInverse[:,:,None]), axis=2)
        nml0_ = copy.deepcopy(nml0)
        nml0_[:,:,2] *= -1 # change format from inverse-depth-inwards to regular-depth-outwards
        nml0_[:,:,1] *= -1 # change format from regular-depth-outwards to rgb-normals
        nml0_[:,:,2] *= -1
        nml0_ = (nml0_+1.)/2.*255.
        nml0_ *= np.expand_dims(mask0, axis=-1)
        frontNormal = (np.transpose(nml0_, (1,0,2))).astype(np.uint8)[:,:,::-1]
        oneRow = np.concatenate((frontMaskErosion, frontDepthInverse, frontNormal), axis=1)
        assert(oneRow.shape == (2*dim_h,6*dim_w,3))
        cv.imwrite("./examples/%06d_meshRef_frontMaskDepthNormal.png"%(preFix), oneRow)

    # ----- marching cube to get mesh from meshVoxels -----

    # run marching cube
    mesh, normals_by_marchingCube = extract_orig_mesh(vol)

    # save the raw mesh by marching cube
    mesh_ = dict()
    mesh_['v'] = np.copy(mesh['v'])
    mesh_['f'] = np.copy(mesh['f'])
    mesh_['vn'] = np.copy(mesh['vn'])
    meshCoarsePath = args.volume_file.replace("_meshVoxels.npy","_meshCorase.obj")
    ObjIO.save_obj_data_binary(mesh_canonization(mesh_), meshCoarsePath) # 2.5 MB
    if args.visual_demo_flag:

        os.system("cp %s ./examples/%06d_meshCoarse.obj" % (meshCoarsePath, preFix))

    # simply smooth the normals using normals of the connected vertices
    nml_smooth_iter_num = 2
    for _ in range(nml_smooth_iter_num): smooth_surface_normal(mesh=mesh_, lamb=0.5, preFix=preFix)

    # upsample the mesh by edge-division
    mesh_upsampled = upsample_mesh(mesh_)
    if args.visual_demo_flag:
        meshUpsampledPath = "./examples/%06d_meshUpsampled.obj" % (preFix)
        ObjIO.save_obj_data_binary(mesh_canonization(mesh_upsampled), meshUpsampledPath) # 10.5 MB

    # get the normal-refined mesh, Wowww?! No optimization at all, this is simply using the refined-normals to update mesh-computed-normals
    dpt0 = depthRange - dpt0 # change "inverse-depth" to "not-inverse-depth"
    nml0[:,:,2] *= -1 # change "inverse-depth-inwards" to "not-inverse-depth-outwards" formats
    assigned_normal(mesh_upsampled, dpt0, nml0, mask0)

    # save normal-refined mesh
    meshRefinedPath = args.volume_file.replace("_meshVoxels.npy","_meshRefined.obj")
    ObjIO.save_obj_data_binary(mesh_canonization(mesh_upsampled), meshRefinedPath) # 10.8 MB
    if args.visual_demo_flag:
        gtMesh = read_and_canonize_gt_mesh(preFix=preFix,args=args,withTexture=False)
        ObjIO.save_obj_data_binary(gtMesh, "./examples/%06d_meshGT.obj" % (preFix))
        os.system("cp %s ./examples/%06d_meshRefined.obj" % (meshRefinedPath, preFix)) # 10.8 MB

if __name__ == '__main__':

    # parse args.
    args = parse_args()

    # main function
    main(args=args)
