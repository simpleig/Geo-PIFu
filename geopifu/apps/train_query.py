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

# get options
opt = BaseOptions().parse()

def train(opt, visualCheck_0=False):

    # ----- init. -----

    # set GPU idx
    if len(opt.gpu_ids) > 1: os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    cuda = torch.device('cuda') if len(opt.gpu_ids) > 1 else torch.device('cuda:%d' % opt.gpu_id)
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
    netG = HGPIFuNet(opt, projection_mode)
    print('Using Network: ', netG.name)
    if len(opt.gpu_ids) > 1: netG = nn.DataParallel(netG)
    netG.to(cuda)
    
    # define the optimizer
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    lr = opt.learning_rate

    # ----- load pre-trained weights if provided -----

    # load well-trained weights
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        assert(os.path.exists(opt.load_netG_checkpoint_path))
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda), strict= not opt.partial_load)

    # load mid-training weights 
    if opt.continue_train:
        model_path = '%s/%s/netG_epoch_%d_%d' % (opt.checkpoints_path, opt.resume_name, opt.resume_epoch, opt.resume_iter)
        print('Resuming from ', model_path)
        assert(os.path.exists(model_path))
        netG.load_state_dict(torch.load(model_path, map_location=cuda))

        # change lr
        for epoch in range(0, opt.resume_epoch+1):
            lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

    # ----- enter the training loop -----

    print("entering the training loop...")
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch+1,0) # usually: 0
    for epoch in range(start_epoch, opt.num_epoch):
        netG.train() # set to training mode (e.g. enable dropout, BN update)
        epoch_start_time = time.time()
        iter_data_time = time.time()

        # start an epoch of training
        for train_idx, train_data in enumerate(train_data_loader): # 14496 iters for each epoch
            iter_start_time = time.time()

            # get a training batch
            image_tensor      = train_data['img'].to(device=cuda)     # (B==2, num_views, C, W, H) RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
            calib_tensor      = train_data['calib'].to(device=cuda)   # (B==2, num_views, 4, 4) calibration matrix
            sample_tensor     = train_data['samples'].to(device=cuda) # (B==2, 3, n_in + n_out), float XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
            label_tensor      = train_data['labels'].to(device=cuda)  # (B==2, 1, n_in + n_out), float 1.0-inside, 0.0-outside
            extrinsic_tensor  = train_data['extrinsic'] # (B, num_views, 4, 4) extrinsic matrix
            deepVoxels_tensor = torch.zeros([label_tensor.shape[0]], dtype=torch.int32).to(device=cuda) # small dummy tensors
            if opt.deepVoxels_fusion != None: deepVoxels_tensor = train_data["deepVoxels"].to(device=cuda) # (B,C=8,D=32,H=48,W=32), np.float32, all >= 0.

            # visual check to show that img input is RGB, not BGR
            if visualCheck_0:
                print("Debug: verify that img input is rgb...")
                img_BGR = ((np.transpose(image_tensor[0,0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)*255.).astype(np.uint8)[:,:,::-1] # RGB to BGR, (512,512,3), [0, 255]
                img_RGB = img_BGR[:,:,::-1]
                pdb.set_trace()
                cv2.imwrite("./sample_images/%s_img_input_by_cv2.png"%(opt.name), img_BGR)          # cv2 save BGR-array into proper-color.png
                Image.fromarray(img_RGB).save("./sample_images/%s_img_input_by_PIL.png"%(opt.name)) # PIL save RGB-array into proper-color.png

            # reshape tensors for multi-view settings
            image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)        # (B * num_views, C, W, H)， (B * num_views, 4, 4)
            if opt.num_views > 1: sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views) # (B * num_views, 3, n_in + n_out)
            
            # network forward pass
            res, error = netG.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, deepVoxels=deepVoxels_tensor) # (B, 1, 5000), R

            # compute gradients and update weights
            optimizerG.zero_grad()
            if len(opt.gpu_ids) > 1: error = error.mean()
            error.backward()
            optimizerG.step()

            # timming
            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (iter_net_time - epoch_start_time) # remaining sec(s) for this epoch

            # log for every opt.freq_plot iters, 10 iters
            if (train_idx == len(train_data_loader)-1) or (train_idx % opt.freq_plot == 0):
                gpu_in_use = opt.gpu_ids if len(opt.gpu_ids)>1 else opt.gpu_id
                print('Name: {}, GPU-{} | Epoch: {}/{} | {}/{} | Err: {:.06f} | LR: {:.06f} | Sigma: {:.02f} | dataT: {:.05f} | netT: {:.05f} | ETA: {:02d}:{:02d}'.format(
                      opt.name, gpu_in_use, epoch, opt.num_epoch, train_idx, len(train_data_loader), error.item(), lr, opt.sigma,
                      iter_start_time -  iter_data_time, # dataloading time
                      iter_net_time   - iter_start_time, # network training time
                      int(eta // 60),             # remaining min(s)
                      int(eta - 60 * (eta // 60)) # left-over sec(s)
                                                                                                                                                                            )
                     )

            # save weights for every opt.freq_save iters, 50 iters
            if (train_idx == len(train_data_loader)-1) or (train_idx % opt.freq_save == 0 and train_idx != 0):
                # torch.save(netG.state_dict(), '%s/%s/netG_latest'   % (opt.checkpoints_path, opt.name))
                torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d_%d' % (opt.checkpoints_path, opt.name, epoch, train_idx))

            # save query points into .ply (red-inside, green-outside) for every opt.freq_save_ply iters, 100 iters
            if (train_idx == len(train_data_loader)-1) or (train_idx % opt.freq_save_ply) == 0:

                # .ply (rotated to align with the view)
                save_path = '%s/%s/pred_%d_%d.ply' % (opt.results_path, opt.name, epoch, train_idx)
                res_one_frame = res[0].cpu() # (1, 5000)
                # points = sample_tensor[0].transpose(0, 1).cpu() # (n_in + n_out, 3)
                # save_samples_truncted_prob(save_path, points.detach().numpy(), res_one_frame.detach().numpy())
                rot   = extrinsic_tensor[0, 0, :3,  :3] # (3, 3)
                trans = extrinsic_tensor[0, 0, :3, 3:4] # (3, 1)
                samples_roted = torch.addmm(trans, rot, sample_tensor[0].cpu()) # (3, N)
                save_samples_truncted_prob(save_path, samples_roted.T.detach().numpy(), res_one_frame.T.detach().numpy())

                # .png (with augmentation)
                save_path = '%s/%s/pred_%d_%d.png' % (opt.results_path, opt.name, epoch, train_idx)
                image_tensor_reshaped = image_tensor.view(label_tensor.shape[0], -1, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1]) # (B==2, num_views, C, W, H)
                img_BGR = ((np.transpose(image_tensor_reshaped[0,0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)*255.).astype(np.uint8)[:,:,::-1] # RGB to BGR, (512,512,3), [0, 255]
                cv2.imwrite(save_path, img_BGR)          # cv2 save BGR-array into proper-color.png

            # for recording dataloading time
            iter_data_time = time.time()

        # update epoch idx of the training dataset
        train_dataset.epochIdx = (train_dataset.epochIdx+1) % opt.epoch_offline_len

        # (lr * opt.gamma) at epoch indices defined in opt.schedule
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

        # evaluate the model after each training epoch
        with torch.no_grad():
            netG.eval() # set to test mode (e.g. disable dropout, BN does't update)

            # save metrics
            metrics_path = os.path.join(opt.results_path, opt.name, 'metrics.txt')
            if epoch == start_epoch:
                with open(metrics_path, 'w') as outfile:
                    outfile.write("Metrics\n\n")

            # quantitative eval. for {MSE, IOU, prec, recall} metrics
            if not opt.no_num_eval:
                test_losses = {}

                # compute metrics for 100 test frames
                print('calc error (test) ...')
                test_errors = calc_error(opt, netG, cuda, test_dataset, num_tests=50) # avg. {error, IoU, precision, recall} computed among 100 frames, each frame has e.g. 5000 query points for evaluation.
                text_show_0 = 'Epoch-{} | eval  test MSE: {:06f} IOU: {:06f} prec: {:06f} recall: {:06f}'.format(epoch, *test_errors)
                print(text_show_0)
                MSE, IOU, prec, recall = test_errors
                test_losses['MSE(test)'] = MSE
                test_losses['IOU(test)'] = IOU
                test_losses['prec(test)'] = prec
                test_losses['recall(test)'] = recall

                # compute metrics for 100 train frames
                print('calc error (train) ...')
                train_dataset.allow_aug = False # switch-off training data aug.
                train_errors = calc_error(opt, netG, cuda, train_dataset, num_tests=50)
                train_dataset.allow_aug = True  # switch-on  training data aug.
                text_show_1 = 'Epoch-{} | eval train MSE: {:06f} IOU: {:06f} prec: {:06f} recall: {:06f}'.format(epoch, *train_errors)
                print(text_show_1)
                MSE, IOU, prec, recall = train_errors
                test_losses['MSE(train)'] = MSE
                test_losses['IOU(train)'] = IOU
                test_losses['prec(train)'] = prec
                test_losses['recall(train)'] = recall
                with open(metrics_path, 'a') as outfile:
                    outfile.write(text_show_0+"  ||  "+text_show_1+"\n")

            # qualitative eval. by generating meshes
            if not opt.no_gen_mesh:

                # generate meshes for opt.num_gen_mesh_test test frames
                print('generate mesh (test) ...')
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    test_data = random.choice(test_dataset) # get a random item from all the test items
                    save_path = '%s/%s/test_eval_epoch%d_%d_%s.obj' % (opt.results_path, opt.name, epoch, test_data["index"], test_data['name'])
                    gen_mesh_iccv(opt, netG.module if len(opt.gpu_ids)>1 else netG, cuda, test_data, save_path)

                # generate meshes for opt.num_gen_mesh_test train frames
                print('generate mesh (train) ...')
                train_dataset.allow_aug = False # switch-off training data aug.
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    train_data = random.choice(train_dataset) # get a random item from all the test items
                    save_path = '%s/%s/train_eval_epoch%d_%d_%s.obj' % (opt.results_path, opt.name, epoch, train_data["index"], train_data['name'])
                    gen_mesh_iccv(opt, netG.module if len(opt.gpu_ids)>1 else netG, cuda, train_data, save_path)
                train_dataset.allow_aug = True  # switch-on  training data aug.

if __name__ == '__main__':

    train(opt)









