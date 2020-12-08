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
import copy

dim_h = 192
dim_w = 128
hb_ratio = dim_w / dim_h
voxel_size = 1.0 / dim_h


def load_volume(vol_dir):
    vol = sio.loadmat(vol_dir)
    vol = vol['mesh_volume'] # (D,H,W) of continuous value within (0., 1.)
    vol = np.transpose(vol, (2, 1, 0)) # to (W,H,D)
    print('volume loaded. volume.shape:', vol.shape)
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
    print('mesh[v] =', type(mesh['v']), mesh['v'].shape)
    print('mesh[vn] =', type(mesh['vn']), mesh['vn'].shape)
    print('mesh[f] =', type(mesh['f']), mesh['f'].shape)

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
    d_array = np.reshape(d_array, (1, 1, -1)) # (1,1,128)

    vol_dpt_0 = vol_bin * d_array # (W,H,D==128)
    dpt0 = np.max(vol_dpt_0, axis=2) # get front surface, but in reverse-depth format (larger depth means closer to camera), of size (W,H)
    dpt0 = cv.resize(dpt0, (dpt0.shape[1] * 2, dpt0.shape[0] * 2)) # reverse-dront-detph of size (2W,2H)
    return dpt0


def load_normal_map(nml_map_dir, mask0):
    """
    normal-of-inverse-depth-outwards  not-inverse-depth-inwards  not-inverse-depth-outwards (combined-pre-two) RGB-normal-coords (combined-all)
    X                                  1                         -1                         -1                  1                -1 
    Y                                  1                         -1                         -1                 -1                 1
    Z                                 -1                         -1                          1                 -1                -1
    """

    nml0 = cv.imread(args.normal_file, cv.IMREAD_UNCHANGED) # front normals in inverse-depth-outwards format, values in [0, 2*32767.5]
    nml0 = np.float32(nml0) / 32767.5 - 1.0 # (2H,2W,XYZ) ,value range [-1,1]
    nml0 = np.transpose(nml0, (1, 0, 2)) # (2H,2W,XYZ) to (2W,2H,XYZ)
    nml0 *= np.expand_dims(mask0, axis=-1) # (2W,2H,XYZ) * (2W,2H,1) == (2W,2H,XYZ), bg set to XYZ of 000
    nml0 *= -1.0 # change normals from inverse-depth-outwards to inverse-depth-inwards formats
    return nml0


def smooth_surface_normal(mesh, lamb=0.5):
    nml_neighbor = np.zeros_like(mesh['vn'])
    for f in mesh['f']:
        n0, n1, n2 = mesh['vn'][f[0]], mesh['vn'][f[1]], mesh['vn'][f[2]]
        nml_neighbor[f[0]] += n1 + n2
        nml_neighbor[f[1]] += n2 + n0
        nml_neighbor[f[2]] += n0 + n1
    nml_neighbor_len2 = np.sum(np.square(nml_neighbor),axis=1, keepdims=True)
    nml_neighbor /= np.sqrt(nml_neighbor_len2)  # normalize normals
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
            mesh['vn'][vi] = nml0[pixel_x, pixel_y, :] * w + mesh['vn'][vi] * (1-w) # Wowww?! No optimization at all, this is simply using the refined-normals to update mesh-computed-normals
            # mesh['vn'][vi] = nml0[pixel_x, pixel_y, :]
            mesh['vn'][vi] /= np.linalg.norm(mesh['vn'][vi]) # unit normalization


def main(vol_dir, nml_map_dir):
    vol = load_volume(vol_dir) # WHD of continuous values within (0., 1.), format of front/back flipped
    mask0 = proj_frontal_mask(vol) # front mask of size (2*W, 2*H), 1-fg and 0-bg, slimmer after erode for ignoring the boundry
    dpt0 = proj_frontal_depth(vol) # reverse-front-detph of size (2W,2H), values in range [0., 254.]
    nml0 = load_normal_map(nml_map_dir, mask0) # masked front normals of size (2W,2H,3), in inverse-depth-inwards format, values in XYZ and of range [-1,1]
    visualCheck = False
    if visualCheck:
        voxel_util.save_volume(vol > 0.5, "/home/code-base/exps/deepHumanBaseline/humanModeling/examples/visualCheck_meshVoxels.obj", dim_h, dim_w, voxel_size)
        voxel_util.save_volume_doubleIdx(vol > 0.5, "/home/code-base/exps/deepHumanBaseline/humanModeling/examples/visualCheck_meshVoxelsDoubleIdx.obj") # this is aligned with the mesh of spacing=(1,1,1)
        voxel_util.save_volume(vol[:,:,::-1] > 0.5, "/home/code-base/exps/deepHumanBaseline/humanModeling/examples/visualCheck_meshVoxels_rectified.obj", dim_h, dim_w, voxel_size)
        cv.imwrite("/home/code-base/exps/deepHumanBaseline/humanModeling/examples/visualCheck_frontMaskErosion.png", (np.transpose(mask0, (1,0))*255.).astype(np.uint8))
        cv.imwrite("/home/code-base/exps/deepHumanBaseline/humanModeling/examples/visualCheck_frontDepthInverse.png", (np.transpose(dpt0, (1,0))).astype(np.uint8))
        nml0_ = copy.deepcopy(nml0)
        nml0_[:,:,2] *= -1 # change format from inverse-depth-inwards to regular-depth-outwards
        nml0_[:,:,1] *= -1 # change format from regular-depth-outwards to rgb-normals
        nml0_[:,:,2] *= -1
        nml0_ = (nml0_+1.)/2.*255.
        cv.imwrite("/home/code-base/exps/deepHumanBaseline/humanModeling/examples/visualCheck_frontNormal.png", (np.transpose(nml0_, (1,0,2))).astype(np.uint8)[:,:,::-1])

    mesh, normals_by_marchingCube = extract_orig_mesh(vol)

    # dpt0 = cv.resize(dpt0, (dpt0.shape[1]*2, dpt0.shape[0]*2))
    # mask0 = cv.resize(mask0, (mask0.shape[1]*2, mask0.shape[0]*2))

    mesh_ = dict()      # extract_hd_mesh(vol)
    mesh_['v'] = np.copy(mesh['v'])
    mesh_['f'] = np.copy(mesh['f'])
    mesh_['vn'] = np.copy(mesh['vn'])
    ObjIO.save_obj_data_binary(mesh_, vol_dir[:-4] + '_out.obj')
    visualCheck = False
    if visualCheck:
        mesh_1 = dict()
        mesh_1['v'] = np.copy(mesh['v'])
        mesh_1['f'] = np.copy(mesh['f'])
        mesh_1['vn'] = np.copy(normals_by_marchingCube) # normals from marchingCube are only slightly diff. from opendr's
        ObjIO.save_obj_data_binary(mesh_1, vol_dir[:-4] + '_out_normalsMarchingCube.obj')

    # visually check if normal fusion is really useful: there are visual diff. due to lighting, but no additional geometric details
    visualCheck = False
    if visualCheck:

        mesh_1 = dict()
        mesh_1['v'] = np.copy(mesh['v'])
        mesh_1['f'] = np.copy(mesh['f'])
        mesh_1['vn'] = np.copy(mesh['vn'])
        nml_smooth_iter_num = 2
        for _ in range(nml_smooth_iter_num):
            smooth_surface_normal(mesh_1)
        mesh_upsampled_1 = upsample_mesh(mesh_1)
        util.rotate_model_in_place(mesh_upsampled_1, 0, 0, np.pi) # rotate around z-axis, we don't need this
        util.flip_axis_in_place(mesh_upsampled_1, -1, 1, 1) # flip left/right, we don't need this
        ObjIO.save_obj_data_binary(mesh_upsampled_1, vol_dir[:-4] + '_out_detailed_noFusion.obj')

        mesh_2 = dict()
        mesh_2['v'] = np.copy(mesh['v'])
        mesh_2['f'] = np.copy(mesh['f'])
        mesh_2['vn'] = np.copy(mesh['vn'])
        mesh_upsampled_2 = upsample_mesh(mesh_2)
        util.rotate_model_in_place(mesh_upsampled_2, 0, 0, np.pi) # rotate around z-axis, we don't need this
        util.flip_axis_in_place(mesh_upsampled_2, -1, 1, 1) # flip left/right, we don't need this
        ObjIO.save_obj_data_binary(mesh_upsampled_2, vol_dir[:-4] + '_out_detailed_noSmooth_noFusion.obj')

    # simply smooth the normals using normals of the connected vertices
    nml_smooth_iter_num = 2
    for _ in range(nml_smooth_iter_num):
        smooth_surface_normal(mesh_)

    # upsample the mesh by edge-division
    mesh_upsampled = upsample_mesh(mesh_)

    # Wowww?! No optimization at all, this is simply using the refined-normals to update mesh-computed-normals
    assigned_normal(mesh_upsampled, dpt0, nml0, mask0)

    util.rotate_model_in_place(mesh_upsampled, 0, 0, np.pi) # rotate around z-axis, we don't need this
    ObjIO.save_obj_data_binary(mesh_upsampled, vol_dir[:-4] + '_out_detailed_beforeLRF.obj')
    util.flip_axis_in_place(mesh_upsampled, -1, 1, 1) # flip left/right, we don't need this
    ObjIO.save_obj_data_binary(mesh_upsampled, vol_dir[:-4] + '_out_detailed.obj')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_file', type=str, required=True,
                        help='path to volume file (.mat)')
    parser.add_argument('--normal_file', type=str, required=True,
                        help='path to normal map (16bit .png)')
    args = parser.parse_args()
    if not args.volume_file.endswith('.mat'):
        print('Invalid volume file!!!')
        raise RuntimeError
    if not args.normal_file.endswith('.png'):
        print('Invalid normal map file!!!')
        raise RuntimeError
    main(args.volume_file, args.normal_file)
