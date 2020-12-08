from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure


def reconstruction_iccv(net, cuda, calib_tensor, resolution_x, resolution_y, resolution_z, b_min, b_max, use_octree=False, num_samples=700000, transform=None, deepVoxels=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor, (num_views, 4, 4)
    :param resolution: resolution of the grid cell, 256
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration, True
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''

    # First we create a grid by resolution and transforming matrix for grid coordinates to real world xyz
    # coords: WHD, XYZ, voxel-space converted to mesh-coords, (3, 256, 256, 256)
    # mat   : 4x4, {XYZ-scaling, trans} matrix from voxel-space to mesh-coords, by left Mul. with voxel-space idx tensor
    coords, mat = create_grid(resolution_x, resolution_y, resolution_z, b_min, b_max, transform=transform)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points  = np.expand_dims(points, axis=0)                   # (1,         3, num_samples)
        points  = np.repeat(points, net.num_views, axis=0)         # (num_views, 3, num_samples)
        samples = torch.from_numpy(points).to(device=cuda).float() # (num_views, 3, num_samples)
        net.query(points=samples, calibs=calib_tensor, deepVoxels=deepVoxels) # calib_tensor is (num_views, 4, 4)
        pred = net.get_preds()[0][0]                               # (num_samples,)
        return pred.detach().cpu().numpy()   

    # Then we evaluate the grid, use_octree default: True
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples) # XYZ, WHD, (256, 256, 256), float 0. ~ 1. for occupancy
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples) # XYZ, WHD, (256, 256, 256), float 0. ~ 1. for occupancy

    # Finally we do marching cubes
    try:

        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4] # (3,N), convert verts from voxel-space into mesh-coords
        verts = verts.T # (N,3)

        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1


def reconstruction(net, cuda, calib_tensor, resolution, b_min, b_max, use_octree=False, num_samples=10000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor, (num_views, 4, 4)
    :param resolution: resolution of the grid cell, 256
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration, True
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''

    # First we create a grid by resolution and transforming matrix for grid coordinates to real world xyz
    # coords: WHD, XYZ, voxel-space converted to mesh-coords, (3, 256, 256, 256)
    # mat   : 4x4, {XYZ-scaling, trans} matrix from voxel-space to mesh-coords, by left Mul. with voxel-space idx tensor
    coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max, transform=transform)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points  = np.expand_dims(points, axis=0)                   # (1,         3, num_samples)
        points  = np.repeat(points, net.num_views, axis=0)         # (num_views, 3, num_samples)
        samples = torch.from_numpy(points).to(device=cuda).float() # (num_views, 3, num_samples)
        net.query(samples, calib_tensor)                           # calib_tensor is (num_views, 4, 4)
        pred = net.get_preds()[0][0]                               # (num_samples,)
        return pred.detach().cpu().numpy()   

    # Then we evaluate the grid, use_octree default: True
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples) # XYZ, WHD, (256, 256, 256), float 0. ~ 1. for occupancy
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples) # XYZ, WHD, (256, 256, 256), float 0. ~ 1. for occupancy

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4] # (3,N), convert verts from voxel-space into mesh-coords
        verts = verts.T # (N,3)

        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    """
    input
        mesh_path: XXX.obj
        verts    : (N, 3) in the mesh-coords.
        faces    : (N, 3), order not switched yet
        colors   : (N, 3), RGB, float 0 ~ 1
    """

    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1])) # switch the order, so that the computed normals later can face outwards from the mesh
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
