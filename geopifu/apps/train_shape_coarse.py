import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index

import pdb # pdb.set_trace()
from torch import nn
this_file_path_abs       = os.path.dirname(__file__)
target_dir_path_relative = os.path.join(this_file_path_abs, '../..')
target_dir_path_abs      = os.path.abspath(target_dir_path_relative)
sys.path.insert(0, target_dir_path_abs)
from Constants import consts

# get options
opt = BaseOptions().parse()

def load_from_multi_GPU(path):

    # original saved file with DataParallel
    state_dict = torch.load(path)

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

def train(opt):

    # ----- init. -----

    # visual checks
    visualCheck_0=False

    # set GPU idx
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids if len(opt.gpu_ids) > 1 else opt.gpu_id
    cuda = torch.device('cuda')
    if len(opt.gpu_ids) > 1: assert(torch.cuda.device_count() > 1)

    # make dir to save weights
    os.makedirs(opt.checkpoints_path, exist_ok=True) # exist_ok=True: will NOT make a new dir if already exist
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)

    # make dir to save visualizations
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    # save args.
    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile: outfile.write(json.dumps(vars(opt), indent=2))

    # ----- create train/test dataloaders -----

    train_dataset   = TrainDatasetICCV(opt, phase='train')
    test_dataset    = TrainDatasetICCV(opt, phase='test')
    projection_mode = train_dataset.projection_mode # default: 'orthogonal'

    # train dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches, num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('train data sizes: ', len(train_dataset)) # 360, (number-of-training-meshes * 360-degree-renderings) := namely, the-number-of-training-views
    print('train data iters for each epoch: ', len(train_data_loader)) # ceil[train-data-sizes / batch_size]

    # test dataloader: batch size should be 1 and use all the points for evaluation
    # test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data sizes: ', len(test_dataset)) # 360, (number-of-test-meshes * 360-degree-renderings) := namely, the-number-of-training-views
    # print('test data iters for each epoch: ', len(test_data_loader)) # ceil[test-data-sizes / 1]

    # ----- build networks -----

    # {create, deploy} networks to the specified GPU
    netV = VrnNet(opt, projection_mode)
    print('Using Network: ', netV.name)
    if len(opt.gpu_ids) > 1: netV = nn.DataParallel(netV)
    netV.to(cuda)
    
    # define the optimizer
    if opt.use_3d_gan:

        # init.
        discriminator_parameters, generator_parameters = [], []

        # classify weights
        for name, param in netV.named_parameters():
            if "gan_3d_dis" in name:
                discriminator_parameters.append(param)
            else:
                generator_parameters.append(param)

        # separate optimizers
        optimizer_dis = torch.optim.Adam(discriminator_parameters, lr=opt.learning_rate_3d_gan, betas=(0.5, 0.999)        )
        optimizerV    = torch.optim.RMSprop( generator_parameters, lr=opt.learning_rate       , momentum=0, weight_decay=0)
        lr_dis = opt.learning_rate_3d_gan
        lr     = opt.learning_rate
    else:

        optimizerV = torch.optim.RMSprop(netV.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
        lr = opt.learning_rate

    # ----- load pre-trained weights if provided -----

    # load well-trained weights
    if opt.load_netV_checkpoint_path is not None:
        print('loading for net V ...', opt.load_netV_checkpoint_path)
        assert(os.path.exists(opt.load_netV_checkpoint_path))
        if opt.load_from_multi_GPU_shape    : netV.load_state_dict(load_from_multi_GPU(path=opt.load_netV_checkpoint_path), strict= not opt.partial_load)
        if not opt.load_from_multi_GPU_shape: netV.load_state_dict(torch.load(opt.load_netV_checkpoint_path, map_location=cuda), strict= not opt.partial_load)

    # load mid-training weights 
    if opt.continue_train:
        model_path = '%s/%s/netV_epoch_%d_%d' % (opt.checkpoints_path, opt.resume_name, opt.resume_epoch, opt.resume_iter)
        print('Resuming from ', model_path)
        assert(os.path.exists(model_path))
        netV.load_state_dict(torch.load(model_path, map_location=cuda))

        # change lr
        for epoch in range(0, opt.resume_epoch+1):
            lr_dis = adjust_learning_rate(optimizer_dis, epoch, lr_dis, opt.schedule, opt.gamma)
            lr     = adjust_learning_rate(   optimizerV, epoch,     lr, opt.schedule, opt.gamma)

    # ----- enter the training loop -----

    print("entering the training loop...")
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch+1,0) # usually: 0
    for epoch in range(start_epoch, opt.num_epoch):
        netV.train() # set to training mode (e.g. enable dropout, BN update)
        epoch_start_time = time.time()
        iter_data_time   = time.time()

        # start an epoch of training
        for train_idx, train_data in enumerate(train_data_loader): # 17396 iters for each epoch
            iter_start_time = time.time()

            # get a training batch
            image_tensor            = train_data['img'].to(device=cuda)                                            # (B,   V,   C-3, H-512, W-512), RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            meshVoxels_tensor       = train_data['meshVoxels'].to(device=cuda)                                     # (B, C-1, D-128, H-192, W-128), float 1.0-inside, 0.0-outside
            viewDirectionIdx_tensor = torch.zeros([meshVoxels_tensor.shape[0]], dtype=torch.int32).to(device=cuda) # small dummy tensors
            target_view_tensor      = torch.zeros([meshVoxels_tensor.shape[0]], dtype=torch.int32).to(device=cuda) # small dummy tensors
            if opt.use_view_pred_loss: viewDirectionIdx_tensor = train_data["view_directions"].to(device=cuda)     # (B,), integers, {0, 1, 2, 3} maps to {front, right, back, left}
            if opt.use_view_pred_loss: target_view_tensor      = train_data['target_view'].to(device=cuda)         # (B, C-3, H-384, W-256) target-view RGB images, float -1. ~ 1., bg is all ZEROS not -1. 

            # visualCheck_0: verify that img input is rgb, and mesh voxels are aligned with the image
            if visualCheck_0:

                print("visualCheck_0: verify that img input is rgb, and mesh voxels are aligned with the image...")

                for bIdx in range(image_tensor.shape[0]):
                    print("{}/{}...".format(bIdx, image_tensor.shape[0]))

                    # RGB img
                    img_BGR = ((np.transpose(image_tensor[bIdx,0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)*255.).astype(np.uint8)[:,:,::-1] # RGB to BGR, (512,512,3), [0, 255]
                    img_RGB = img_BGR[:,:,::-1]
                    cv2.imwrite("./sample_images/%s_%03d_img_input_by_cv2.png"%(opt.name,bIdx), img_BGR)          # cv2 save BGR-array into proper-color.png
                    Image.fromarray(img_RGB).save("./sample_images/%s_%03d_img_input_by_PIL.png"%(opt.name,bIdx)) # PIL save RGB-array into proper-color.png

                    # mesh voxels
                    meshVoxels_check = meshVoxels_tensor[bIdx,0].detach().cpu().numpy() # DHW
                    meshVoxels_check = np.transpose(meshVoxels_check, (2,1,0)) # WHD, XYZ
                    save_volume(meshVoxels_check, fname="./sample_images/%s_%03d_meshVoxels.obj"%(opt.name,bIdx), dim_h=consts.dim_h, dim_w=consts.dim_w, voxel_size=consts.voxel_size)

                pdb.set_trace()

            # reshape tensors for multi-view settings
            image_tensor = image_tensor.view(-1, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1]) # (BV, C-3, H-512, W-512)

            # network forward pass: (B,C=1,D=128,H=192,W=128)-pred_occ, R-error, (B,C=3,H=384,W=256)-render_rgb, (B,C=1,H=192,W=128)-pseudo_inverseDepth, R-error_view_render
            # if opt.use_view_pred_loss: pred_occ, error, render_rgb, pseudo_inverseDepth, error_view_render = netV.forward(images=image_tensor, labels=meshVoxels_tensor, view_directions=viewDirectionIdx_tensor, target_views=target_view_tensor)
            # else: pred_occ, error = netV.forward(images=image_tensor, labels=meshVoxels_tensor, view_directions=viewDirectionIdx_tensor, target_views=target_view_tensor)
            forward_return_dict = netV.forward(images=image_tensor, labels=meshVoxels_tensor, view_directions=viewDirectionIdx_tensor, target_views=target_view_tensor)
            pred_occ = forward_return_dict["pred_occ"]
            error    = forward_return_dict["error"]
            if opt.use_view_pred_loss:
                render_rgb          = forward_return_dict["render_rgb"]
                pseudo_inverseDepth = forward_return_dict["pseudo_inverseDepth"]
                error_view_render   = forward_return_dict["error_view_render"]
            accuracy_3d_gan_discriminator_fake, accuracy_3d_gan_discriminator_real = 0., 0.
            if opt.use_3d_gan:
                error_3d_gan_generator             = forward_return_dict["error_3d_gan_generator"]
                error_3d_gan_discriminator_fake    = forward_return_dict["error_3d_gan_discriminator_fake"]
                error_3d_gan_discriminator_real    = forward_return_dict["error_3d_gan_discriminator_real"]
                accuracy_3d_gan_discriminator_fake = forward_return_dict["accuracy_3d_gan_discriminator_fake"].mean().item()
                accuracy_3d_gan_discriminator_real = forward_return_dict["accuracy_3d_gan_discriminator_real"].mean().item()

            # compute gradients and update weights for 3d gan discriminator
            error_3d_gan_discriminator_fake_mean, error_3d_gan_discriminator_real_mean = torch.tensor(0.).to(cuda), torch.tensor(0.).to(cuda)
            if opt.use_3d_gan and ((accuracy_3d_gan_discriminator_fake<opt.discriminator_accuracy_update_threshold) or (accuracy_3d_gan_discriminator_real<opt.discriminator_accuracy_update_threshold)):
                optimizer_dis.zero_grad()
                error_3d_gan_discriminator_fake_mean = error_3d_gan_discriminator_fake.mean() if accuracy_3d_gan_discriminator_fake<opt.discriminator_accuracy_update_threshold else torch.tensor(0.).to(cuda)
                error_3d_gan_discriminator_real_mean = error_3d_gan_discriminator_real.mean() if accuracy_3d_gan_discriminator_real<opt.discriminator_accuracy_update_threshold else torch.tensor(0.).to(cuda)
                error_3d_gan_discriminator           = error_3d_gan_discriminator_fake_mean + error_3d_gan_discriminator_real_mean
                error_3d_gan_discriminator.backward()
                optimizer_dis.step()

            # compute gradients and update weights for the main network
            optimizerV.zero_grad()
            error_mean                  = error.mean()
            error_view_render_mean      = error_view_render.mean()      if opt.use_view_pred_loss else torch.tensor(0.).to(cuda)
            error_3d_gan_generator_mean = error_3d_gan_generator.mean() if opt.use_3d_gan         else torch.tensor(0.).to(cuda)
            error_total                 = error_mean + error_view_render_mean + error_3d_gan_generator_mean
            error_total.backward()
            optimizerV.step()

            # timming
            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (iter_net_time - epoch_start_time) # remaining sec(s) for this epoch

            # log for every opt.freq_plot iters
            if (train_idx == len(train_data_loader)-1) or (train_idx % opt.freq_plot == 0):
                gpu_in_use   = opt.gpu_ids if len(opt.gpu_ids)>1 else opt.gpu_id
                error_in_use = "Errs-Gen: %.6f = Occu: %.6f + Render: %.6f + Gen: %.6f | Dis3d-Fake: %.6f(%.3f), Dis3d-Real: %.6f(%.3f)"%\
                              (error_total.item(), error_mean.item(), error_view_render_mean.item(), error_3d_gan_generator_mean.item(),
                               error_3d_gan_discriminator_fake_mean.item(), accuracy_3d_gan_discriminator_fake, error_3d_gan_discriminator_real_mean.item(), accuracy_3d_gan_discriminator_real)
                print('Name: {}, GPU-{} | Epoch: {}/{} | {}/{} | {} | LR: {:.06f} | Sigma: {:.02f} | dataT: {:.05f} | netT: {:.05f} | ETA: {:02d}:{:02d}'.format(
                      opt.name, gpu_in_use, epoch, opt.num_epoch, train_idx, len(train_data_loader), error_in_use, lr, opt.sigma,
                      iter_start_time -  iter_data_time, # dataloading time
                      iter_net_time   - iter_start_time, # network training time
                      int(eta // 60),             # remaining min(s)
                      int(eta - 60 * (eta // 60)) # left-over sec(s)
                     ))

            # save weights for every opt.freq_save iters
            if (train_idx == len(train_data_loader)-1) or (train_idx % opt.freq_save == 0 and train_idx != 0):
                # torch.save(netV.state_dict(), '%s/%s/netV_latest'   % (opt.checkpoints_path, opt.name))
                torch.save(netV.state_dict(), '%s/%s/netV_epoch_%d_%d' % (opt.checkpoints_path, opt.name, epoch, train_idx))

            # save gt and est meshVoxels.obj for every opt.freq_save_ply iters
            if (train_idx == len(train_data_loader)-1) or (train_idx % opt.freq_save_ply) == 0:
                view_directions_names = ["front", "right", "back", "left"]

                # save gt meshVoxels
                meshVoxels_check = meshVoxels_tensor[0,0].detach().cpu().numpy() # DHW
                meshVoxels_check = np.transpose(meshVoxels_check, (2,1,0)) # WHD, XYZ
                save_volume(meshVoxels_check, fname='%s/%s/gt_%d_%d_meshVoxels.obj'%(opt.results_path,opt.name,epoch,train_idx), dim_h=consts.dim_h, dim_w=consts.dim_w, voxel_size=consts.voxel_size)

                # save est. meshVoxels
                meshVoxels_check = pred_occ[0,0].detach().cpu().numpy() # DHW
                meshVoxels_check = np.transpose(meshVoxels_check, (2,1,0)) # WHD, XYZ
                meshVoxels_check = meshVoxels_check > 0.5 # discretization
                save_volume(meshVoxels_check, fname='%s/%s/est_%d_%d_meshVoxels.obj'%(opt.results_path,opt.name,epoch,train_idx), dim_h=consts.dim_h, dim_w=consts.dim_w, voxel_size=consts.voxel_size)

                # .png (with augmentation)
                save_path = '%s/%s/gt_%d_%d.png' % (opt.results_path, opt.name, epoch, train_idx)
                image_tensor_reshaped = image_tensor.view(-1, opt.num_views, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1]) # (B, V, C, H, W)
                img_BGR = ((np.transpose(image_tensor_reshaped[0,0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)*255.).astype(np.uint8)[:,:,::-1] # RGB to BGR, (512,512,3), [0, 255]
                cv2.imwrite(save_path, img_BGR)          # cv2 save BGR-array into proper-color.png

                # save pseudo-inverse-depth-map
                if opt.use_view_pred_loss:
                    save_path = '%s/%s/est_%d_%d_depth_%s.png'%(opt.results_path,opt.name,epoch,train_idx,view_directions_names[viewDirectionIdx_tensor[0]])
                    pseudo_inverseDepth_check     = np.transpose(pseudo_inverseDepth[0].detach().cpu().numpy(), (1,2,0)) # (H=192,W=128,C=1), values in (-0.5, 0.5)
                    pseudo_inverseDepth_check_min = np.min(pseudo_inverseDepth_check)
                    pseudo_inverseDepth_check_max = np.max(pseudo_inverseDepth_check)
                    pseudo_inverseDepth_check     = ((pseudo_inverseDepth_check-pseudo_inverseDepth_check_min)/(pseudo_inverseDepth_check_max-pseudo_inverseDepth_check_min)*255.).astype(np.uint8) # (H=192,W=128,C=1), values in (0., 255.)
                    cv2.imwrite(save_path, pseudo_inverseDepth_check)

                # save rendered-target-view .png
                if opt.use_view_pred_loss:
                    save_path = '%s/%s/est_%d_%d_target_%s.png' % (opt.results_path, opt.name, epoch, train_idx, view_directions_names[viewDirectionIdx_tensor[0]])
                    img_BGR = ((np.transpose(render_rgb[0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)*255.).astype(np.uint8)[:,:,::-1] # RGB to BGR, (384,256,3), [0, 255]
                    cv2.imwrite(save_path, img_BGR)          # cv2 save BGR-array into proper-color.png

                # save gt-target-view .png
                if opt.use_view_pred_loss:
                    save_path = '%s/%s/gt_%d_%d_target_%s.png' % (opt.results_path, opt.name, epoch, train_idx, view_directions_names[viewDirectionIdx_tensor[0]])
                    img_BGR = ((np.transpose(target_view_tensor[0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)*255.).astype(np.uint8)[:,:,::-1] # RGB to BGR, (384,256,3), [0, 255]
                    cv2.imwrite(save_path, img_BGR)          # cv2 save BGR-array into proper-color.png

            # for recording dataloading time
            iter_data_time = time.time()

        # (lr * opt.gamma) at epoch indices defined in opt.schedule
        if opt.use_3d_gan:
            lr_dis = adjust_learning_rate(optimizer_dis, epoch, lr_dis, opt.schedule, opt.gamma)
        lr         = adjust_learning_rate(   optimizerV, epoch,     lr, opt.schedule, opt.gamma)

        # evaluate the model after each training epoch
        with torch.no_grad():
            netV.eval() # set to test mode (e.g. disable dropout, BN does't update)
            if opt.name in opt.must_run_in_train_modes: netV.train()

            # save metrics
            metrics_path = os.path.join(opt.results_path, opt.name, 'metrics.txt')
            if epoch == start_epoch:
                with open(metrics_path, 'w') as outfile:
                    outfile.write("Metrics\n\n")

            # quantitative eval. for {vrn_occupancy_loss_type, IOU, prec, recall} metrics
            if not opt.no_num_eval:
                test_losses = {}

                # compute metrics for 100 test frames
                print('calc error (test) ...')
                total_error, occu_error, render_error, error_3d_gan_generator, error_3d_gan_discriminator_fake, accuracy_3d_gan_discriminator_fake, error_3d_gan_discriminator_real, accuracy_3d_gan_discriminator_real, IoU, prec, recall = calc_error_vrn_occu(opt, netV, cuda, test_dataset, num_tests=100)
                text_show_0 = 'Epoch-{} | eval  test Err: {:06f} = Occu: {:06f} + Render: {:06f} + Gen: {:06f} | Dis3d-Fake: {:06f}({:03f}), Dis3d-Real: {:06f}({:03f}) | IOU: {:06f} prec: {:06f} recall: {:06f}'.format(epoch, total_error, occu_error, render_error, error_3d_gan_generator, error_3d_gan_discriminator_fake, accuracy_3d_gan_discriminator_fake, error_3d_gan_discriminator_real, accuracy_3d_gan_discriminator_real, IoU, prec, recall)
                print(text_show_0)

                # compute metrics for 100 train frames
                print('calc error (train) ...')
                train_dataset.allow_aug = False # switch-off training data aug.
                total_error, occu_error, render_error, error_3d_gan_generator, error_3d_gan_discriminator_fake, accuracy_3d_gan_discriminator_fake, error_3d_gan_discriminator_real, accuracy_3d_gan_discriminator_real, IoU, prec, recall = calc_error_vrn_occu(opt, netV, cuda, train_dataset, num_tests=100)
                train_dataset.allow_aug = True  # switch-on  training data aug.
                text_show_1 = 'Epoch-{} | eval train Err: {:06f} = Occu: {:06f} + Render: {:06f} + Gen: {:06f} | Dis3d-Fake: {:06f}({:03f}), Dis3d-Real: {:06f}({:03f}) | IOU: {:06f} prec: {:06f} recall: {:06f}'.format(epoch, total_error, occu_error, render_error, error_3d_gan_generator, error_3d_gan_discriminator_fake, accuracy_3d_gan_discriminator_fake, error_3d_gan_discriminator_real, accuracy_3d_gan_discriminator_real, IoU, prec, recall)
                print(text_show_1)
                with open(metrics_path, 'a') as outfile:
                    outfile.write(text_show_0+"  ||  "+text_show_1+"\n")

            # qualitative eval. by generating meshes
            if not opt.no_gen_mesh:

                # generate meshes for opt.num_gen_mesh_test test frames
                print('generate mesh (test) ...')
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    test_data = random.choice(test_dataset) # get a random item from all the test items
                    save_path = '%s/%s/test_eval_epoch%d_%d_%s.obj' % (opt.results_path, opt.name, epoch, test_data["index"], test_data['name'])
                    gen_mesh_vrn(opt, netV.module if len(opt.gpu_ids)>1 else netV, cuda, test_data, save_path)

                # generate meshes for opt.num_gen_mesh_test train frames
                print('generate mesh (train) ...')
                train_dataset.allow_aug = False # switch-off training data aug.
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    train_data = random.choice(train_dataset) # get a random item from all the test items
                    save_path = '%s/%s/train_eval_epoch%d_%d_%s.obj' % (opt.results_path, opt.name, epoch, train_data["index"], train_data['name'])
                    gen_mesh_vrn(opt, netV.module if len(opt.gpu_ids)>1 else netV, cuda, train_data, save_path)
                train_dataset.allow_aug = True  # switch-on  training data aug.

if __name__ == '__main__':

    train(opt)









