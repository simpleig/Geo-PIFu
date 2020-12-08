import torch
import numpy as np
from .mesh_util import *
from .sample_util import *
from .geometry import *
import cv2
from PIL import Image
from tqdm import tqdm
import pdb # pdb.set_trace()

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

def verts_canonization(verts, dim_w, dim_h):
    """
    translate & rescale the verts from [[0,2W),[0,2H),[0,2D)] ---> [(-0.33,0.33),(-0.5,0.5),(-0.33,0.33)]
    """

    # translate
    verts[:,0] -= dim_w # X, from [0,2W) to [-W,W)
    verts[:,1] -= dim_h # Y, from [0,2H) to [-H,H)
    verts[:,2] -= dim_w # Z, from [0,2D) to [-D,D)

    # rescale
    verts[:,0] /= (2.*dim_h) # X, from [-W,W) to (-0.33,0.33)
    verts[:,1] /= (2.*dim_h) # Y, from [-H,H) to (-0.5,0.5)
    verts[:,2] /= (2.*dim_h) # Z, from [-D,D) to (-0.33,0.33)

    return verts

# generate proj-color-mesh for a data point (can be multi-view input)
def gen_mesh_vrn(opt, net, cuda, data, save_path, also_generate_mesh_from_gt_voxels=True):

    # init,
    save_path_png, save_path_gt_obj = None, None
    visualCheck_0 = False

    # retrieve and reshape the data for one frame (or multi-view)
    image_tensor = data['img'].to(device=cuda) # (V, C-3, H-512, W-512), RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
    image_tensor = image_tensor.view(-1, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1]) # (V, C-3, H-512, W-512)
    
    # use hour-glass networks to extract image features
    net.filter(image_tensor)

    try:

        # ----- save the single-view/multi-view input image(s) of this data point -----
        if True:

            save_img_path = save_path[:-4] + '.png'
            save_img_list = []
            for v in range(image_tensor.shape[0]):
                save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 # RGB -> BGR, (3,512,512), [0, 255]
                save_img_list.append(save_img)
            save_img = np.concatenate(save_img_list, axis=1)
            Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path) # BGR -> RGB, cuz "PIL save RGB-array into proper-color.png"

            # update return path
            save_path_png = save_img_path

        # ----- save est. mesh with proj-color -----
        if True:

            # get est. occu.
            net.est_occu(prepare_3d_gan=False)
            pred_occ = net.get_preds() # BCDHW, (V,1,128,192,128), est. occupancy
            pred_occ = pred_occ[0,0].detach().cpu().numpy() # DHW
            pred_occ = np.transpose(pred_occ, (2,1,0)) # WHD, XYZ
            if visualCheck_0:
                print("visualCheck_0: check the est voxels...")
                save_volume(pred_occ>0.5, fname="./sample_images/%s_est_mesh_voxels.obj"%(save_path[:-4].split("/")[-1]), dim_h=192, dim_w=128, voxel_size=1./192.)
                pdb.set_trace()

            # est, marching cube
            vol = pred_occ # WHD, XYZ
            verts, faces, normals, _ = measure.marching_cubes_lewiner(vol, level=0.5) # https://scikit-image.org/docs/dev/api/skimage.measure.html#marching-cubes-lewiner
            verts = verts*2.0 # this is only to match the verts_canonization function
            verts = verts_canonization(verts=verts,dim_w=pred_occ.shape[0],dim_h=pred_occ.shape[1])
            verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float() # (1, 3, N)
            xyz_tensor = verts_tensor * 2.0 # （1, 3, N） Tensor of xyz coordinates in the image plane of (-1,1) zone and of the-first-view
            uv = xyz_tensor[:, :2, :] # (1, 2, N) for xy, float -1 ~ 1
            color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T # (N, 3), RGB, float -1 ~ 1
            color = color * 0.5 + 0.5 # (N, 3), RGB, float 0 ~ 1
            save_obj_mesh_with_color(save_path, verts, faces, color)

        # ----- save marching cube mesh from gt. low-resolution mesh voxels -----
        if also_generate_mesh_from_gt_voxels:

            # get gt. occu.
            meshVoxels_tensor = data['meshVoxels'].to(device=cuda) # (1, D-128, H-192, W-128), float 1.0-inside, 0.0-outside
            meshVoxels_tensor = meshVoxels_tensor.view(-1, 1, meshVoxels_tensor.shape[-3], meshVoxels_tensor.shape[-2], meshVoxels_tensor.shape[-1]) # (V, 1, D-128, H-192, W-128)
            gt_occ = meshVoxels_tensor[0,0].detach().cpu().numpy() # DHW
            gt_occ = np.transpose(gt_occ, (2,1,0)) # WHD, XYZ

            # gt, marching cube
            save_path = save_path[:-4] + '_GT_lowRes.obj'
            vol = gt_occ # WHD, XYZ
            verts, faces, normals, _ = measure.marching_cubes_lewiner(vol, level=0.5) # https://scikit-image.org/docs/dev/api/skimage.measure.html#marching-cubes-lewiner
            verts = verts*2.0 # this is only to match the verts_canonization function
            verts = verts_canonization(verts=verts,dim_w=pred_occ.shape[0],dim_h=pred_occ.shape[1])
            verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float() # (1, 3, N)
            xyz_tensor = verts_tensor * 2.0 # （1, 3, N） Tensor of xyz coordinates in the image plane of (-1,1) zone and of the-first-view
            uv = xyz_tensor[:, :2, :] # (1, 2, N) for xy, float -1 ~ 1
            color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T # (N, 3), RGB, float -1 ~ 1
            color = color * 0.5 + 0.5 # (N, 3), RGB, float 0 ~ 1
            save_obj_mesh_with_color(save_path, verts, faces, color)

            # update return path
            save_path_gt_obj = save_path

    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

    # return paths
    return save_path_png, save_path_gt_obj

# generate proj-color-mesh for a data point (can be multi-view input)
def gen_mesh_iccv(opt, net, cuda, data, save_path, use_octree=True):

    image_tensor      = data['img'].to(device=cuda)   # (num_views, 3, 512, 512)
    calib_tensor      = data['calib'].to(device=cuda) # (num_views, 4, 4)
    deepVoxels_tensor = torch.zeros([1], dtype=torch.int32).to(device=cuda) # small dummy tensors
    if opt.deepVoxels_fusion != None: deepVoxels_tensor = data["deepVoxels"].to(device=cuda)[None,:] # (B=1,C=8,D=32,H=48,W=32), np.float32, all >= 0.

    # use hour-glass networks to extract image features
    net.filter(image_tensor)

    # the volume of query space
    b_min = data['b_min']
    b_max = data['b_max']

    try:

        # ----- save the single-view/multi-view input image(s) of this data point -----

        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 # RGB -> BGR, (3,512,512), [0, 255]
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path) # BGR -> RGB, cuz "PIL save RGB-array into proper-color.png"

        # ----- save mesh with proj-color -----

        # verts: (N, 3) in the mesh-coords, the same coord as data["samples"], N << 256*256*256
        # faces: (N, 3)
        verts, faces, _, _ = reconstruction_iccv(net, cuda, calib_tensor, opt.resolution_x, opt.resolution_y, opt.resolution_z, b_min, b_max, use_octree=use_octree, deepVoxels=deepVoxels_tensor)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float() # (1, N, 3)
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1]) # （1, 3, N） Tensor of xyz coordinates in the image plane of (-1,1) zone and of the-first-view
        uv = xyz_tensor[:, :2, :] # (1, 2, N) for xy, float -1 ~ 1
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T # (N, 3), RGB, float -1 ~ 1
        color = color * 0.5 + 0.5 # (N, 3), RGB, float 0 ~ 1
        save_obj_mesh_with_color(save_path, verts, faces, color)

    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

# generate proj-color-mesh for a data point (can be multi-view input)
def gen_mesh(opt, net, cuda, data, save_path, use_octree=True):

    image_tensor = data['img'].to(device=cuda)   # (num_views, 3, 512, 512)
    calib_tensor = data['calib'].to(device=cuda) # (num_views, 4, 4)

    # use hour-glass networks to extract image features
    net.filter(image_tensor)

    # the volume of query space
    b_min = data['b_min']
    b_max = data['b_max']

    try:

        # ----- save the single-view/multi-view input image(s) of this data point -----

        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 # RGB -> BGR, (3,512,512), [0, 255]
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path) # BGR -> RGB, cuz "PIL save RGB-array into proper-color.png"

        # ----- save mesh with proj-color -----

        # verts: (N, 3) in the mesh-coords, the same coord as data["samples"], N << 256*256*256
        # faces: (N, 3)
        verts, faces, _, _ = reconstruction(net, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float() # (1, N, 3)
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1]) # （1, 3, N） Tensor of xyz coordinates in the image plane of (-1,1) zone and of the-first-view
        uv = xyz_tensor[:, :2, :] # (1, 2, N) for xy, float -1 ~ 1
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T # (N, 3), RGB, float -1 ~ 1
        color = color * 0.5 + 0.5 # (N, 3), RGB, float 0 ~ 1
        save_obj_mesh_with_color(save_path, verts, faces, color)

    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

# generate est-color-mesh for a data point (can be multi-view input)
def gen_mesh_color_iccv(opt, netG, netC, cuda, data, save_path, use_octree=True):

    image_tensor = data['img'].to(device=cuda)   # (num_views, 3, 512, 512)
    calib_tensor = data['calib'].to(device=cuda) # (num_views, 4, 4)

    # use hour-glass networks to extract image features
    netG.filter(image_tensor)
    netC.filter(image_tensor)
    netC.attach(netG.get_im_feat())

    # the volume of query space
    b_min = data['b_min']
    b_max = data['b_max']

    try:

        # ----- save the single-view/multi-view input image(s) of this data point -----
        
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 # RGB -> BGR, (3,512,512), [0, 255]
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path) # BGR -> RGB, cuz "PIL save RGB-array into proper-color.png"

        # ----- save mesh with est-color -----

        # verts: (N, 3) in the mesh-coords, the same coord as data["samples"], N << 256*256*256
        # faces: (N, 3)
        verts, faces, _, _ = reconstruction_iccv(netG, cuda, calib_tensor, opt.resolution_x, opt.resolution_y, opt.resolution_z, b_min, b_max, use_octree=use_octree)
        
        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float() # (1,         3, N)
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)             # (num_views, 3, N)
        color = np.zeros(verts.shape) # (N, 3)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5 # RGB (3, interval), float 0. ~ 1.
            color[left:right] = rgb.T # RGB (interval, 3), float 0. ~ 1.
        save_obj_mesh_with_color(save_path, verts, faces, color)

    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

# generate est-color-mesh for a data point (can be multi-view input)
def gen_mesh_color(opt, netG, netC, cuda, data, save_path, use_octree=True):

    image_tensor = data['img'].to(device=cuda)   # (num_views, 3, 512, 512)
    calib_tensor = data['calib'].to(device=cuda) # (num_views, 4, 4)

    # use hour-glass networks to extract image features
    netG.filter(image_tensor)
    netC.filter(image_tensor)
    netC.attach(netG.get_im_feat())

    # the volume of query space
    b_min = data['b_min']
    b_max = data['b_max']

    try:

        # ----- save the single-view/multi-view input image(s) of this data point -----
        
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 # RGB -> BGR, (3,512,512), [0, 255]
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path) # BGR -> RGB, cuz "PIL save RGB-array into proper-color.png"

        # ----- save mesh with est-color -----

        # verts: (N, 3) in the mesh-coords, the same coord as data["samples"], N << 256*256*256
        # faces: (N, 3)
        verts, faces, _, _ = reconstruction(netG, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)

        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float() # (1,         3, N)
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)             # (num_views, 3, N)
        color = np.zeros(verts.shape) # (N, 3)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5 # RGB (3, interval), float 0. ~ 1.
            color[left:right] = rgb.T # RGB (interval, 3), float 0. ~ 1.

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
    """
    input
        res         : (1, 1, n_in + n_out), res[0] are estimated occupancy probs for the query points
        label_tensor: (1, 1, n_in + n_out), float 1.0-inside, 0.0-outside
    
    return
        IOU, precision, and recall
    """

    # compute {IOU, precision, recall} based on the current query 3D points
    with torch.no_grad():

        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()
        
        union    = union.sum().float()
        if union == 0:
            union = 1

        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1

        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1

        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt

def calc_error_vrn_occu(opt, net, cuda, dataset, num_tests):
    """
    return
        avg. {error, IoU, precision, recall} computed among num_test frames, each frame has e.g. 5000 query points for evaluation.
    """

    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_total_arr, erorr_arr, error_view_render_arr, error_3d_gan_generator_arr, error_3d_gan_discriminator_fake_arr, error_3d_gan_discriminator_real_arr, IOU_arr, prec_arr, recall_arr = [], [], [], [], [], [], [], [], []
        accuracy_3d_gan_discriminator_fake_arr, accuracy_3d_gan_discriminator_real_arr = [], []
        for idx in tqdm(range(num_tests)):

            # retrieve data for one frame (or multi-view)
            data              = dataset[idx * len(dataset) // num_tests]
            image_tensor      = data['img'].to(device=cuda)        # (V, C-3, H-512, W-512), RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            meshVoxels_tensor = data['meshVoxels'].to(device=cuda) # (1, D-128, H-192, W-128), float 1.0-inside, 0.0-outside
            viewDirectionIdx_tensor = torch.zeros([1], dtype=torch.int32).to(device=cuda) # small dummy tensors
            target_view_tensor      = torch.zeros([1], dtype=torch.int32).to(device=cuda) # small dummy tensors
            if opt.use_view_pred_loss: viewDirectionIdx_tensor = data["view_directions"].to(device=cuda) # integers, {0, 1, 2, 3} maps to {front, right, back, left}
            if opt.use_view_pred_loss: target_view_tensor      = data['target_view'].to(device=cuda)     # (C-3, H-384, W-256) target-view RGB images, float -1. ~ 1., bg is all ZEROS not -1. 

            # reshape the data
            image_tensor            =            image_tensor.view(-1,         image_tensor.shape[-3],       image_tensor.shape[-2],       image_tensor.shape[-1]) # (V,      C-3, H-512, W-512)
            meshVoxels_tensor       =       meshVoxels_tensor.view(-1, 1, meshVoxels_tensor.shape[-3],  meshVoxels_tensor.shape[-2],  meshVoxels_tensor.shape[-1]) # (1, 1, D-128, H-192, W-128)
            if opt.use_view_pred_loss: viewDirectionIdx_tensor = viewDirectionIdx_tensor.view(-1                                                                                            ) # (1,)
            if opt.use_view_pred_loss: target_view_tensor      =      target_view_tensor.view(-1,   target_view_tensor.shape[-3], target_view_tensor.shape[-2], target_view_tensor.shape[-1]) # (1, C-3, H-384, W-256)

            # forward pass
            # if opt.use_view_pred_loss: pred_occ, error, render_rgb, pseudo_inverseDepth, error_view_render = net.forward(images=image_tensor, labels=meshVoxels_tensor, view_directions=viewDirectionIdx_tensor, target_views=target_view_tensor)
            # else: pred_occ, error = net.forward(images=image_tensor, labels=meshVoxels_tensor, view_directions=viewDirectionIdx_tensor, target_views=target_view_tensor)
            forward_return_dict = net.forward(images=image_tensor, labels=meshVoxels_tensor, view_directions=viewDirectionIdx_tensor, target_views=target_view_tensor)
            pred_occ                           = forward_return_dict["pred_occ"]
            error                              = forward_return_dict["error"].mean().item()
            error_view_render                  = forward_return_dict["error_view_render"].mean().item()                  if opt.use_view_pred_loss else 0.
            error_3d_gan_generator             = forward_return_dict["error_3d_gan_generator"].mean().item()             if opt.use_3d_gan         else 0.
            error_3d_gan_discriminator_fake    = forward_return_dict["error_3d_gan_discriminator_fake"].mean().item()    if opt.use_3d_gan         else 0.
            error_3d_gan_discriminator_real    = forward_return_dict["error_3d_gan_discriminator_real"].mean().item()    if opt.use_3d_gan         else 0.
            accuracy_3d_gan_discriminator_fake = forward_return_dict["accuracy_3d_gan_discriminator_fake"].mean().item() if opt.use_3d_gan         else 0.
            accuracy_3d_gan_discriminator_real = forward_return_dict["accuracy_3d_gan_discriminator_real"].mean().item() if opt.use_3d_gan         else 0.
            error_total                        = error + error_view_render + error_3d_gan_generator

            # compute errors {IOU, prec, recall} based on the current set of query 3D points
            IOU, prec, recall = compute_acc(pred_occ, meshVoxels_tensor) # R, R, R

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            error_total_arr.append(error_total)
            erorr_arr.append(error)
            error_view_render_arr.append(error_view_render)
            error_3d_gan_generator_arr.append(error_3d_gan_generator)
            error_3d_gan_discriminator_fake_arr.append(error_3d_gan_discriminator_fake)
            accuracy_3d_gan_discriminator_fake_arr.append(accuracy_3d_gan_discriminator_fake)
            error_3d_gan_discriminator_real_arr.append(error_3d_gan_discriminator_real)
            accuracy_3d_gan_discriminator_real_arr.append(accuracy_3d_gan_discriminator_real)
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(error_total_arr), np.average(erorr_arr), np.average(error_view_render_arr), np.average(error_3d_gan_generator_arr), np.average(error_3d_gan_discriminator_fake_arr), np.average(accuracy_3d_gan_discriminator_fake_arr), np.average(error_3d_gan_discriminator_real_arr), np.average(accuracy_3d_gan_discriminator_real_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

def calc_error(opt, net, cuda, dataset, num_tests):
    """
    return
        avg. {error, IoU, precision, recall} computed among num_test frames, each frame has e.g. 5000 query points for evaluation.
    """
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):

            # retrieve data for one frame (or multi-view)
            data = dataset[idx * len(dataset) // num_tests]
            image_tensor  = data['img'].to(device=cuda)                  # (num_views, C, W, H) for 3x512x512 images, float -1. ~ 1.
            calib_tensor  = data['calib'].to(device=cuda)                # (num_views, 4, 4) calibration matrix
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0) # (1, 3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views) # (num_views, 3, n_in + n_out)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)   # (1, 1, n_in + n_out), float 1.0-inside, 0.0-outside
            deepVoxels_tensor = torch.zeros([label_tensor.shape[0]], dtype=torch.int32).to(device=cuda) # small dummy tensors
            if opt.deepVoxels_fusion != None: deepVoxels_tensor = data["deepVoxels"].to(device=cuda)[None,:] # (B=1,C=8,D=32,H=48,W=32), np.float32, all >= 0.

            # forward pass
            res, error = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, deepVoxels=deepVoxels_tensor) # (1, 1, n_in + n_out), R
            if len(opt.gpu_ids) > 1: error = error.mean()

            # compute errors {IOU, prec, recall} based on the current set of query 3D points
            IOU, prec, recall = compute_acc(res, label_tensor) # R, R, R

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

            # retrieve data for one frame (or multi-view)
            data = dataset[idx * len(dataset) // num_tests]
            image_tensor        = data['img'].to(device=cuda)
            calib_tensor        = data['calib'].to(device=cuda)
            color_sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)
            rgb_tensor          = data['rgbs'].to(device=cuda).unsqueeze(0)

            # forward pass
            netG.filter(image_tensor)
            _, errorC = netC.forward(image_tensor, netG.get_im_feat(), color_sample_tensor, calib_tensor, labels=rgb_tensor)

            # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
            #       .format(idx, num_tests, errorG.item(), errorC.item()))
            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)





