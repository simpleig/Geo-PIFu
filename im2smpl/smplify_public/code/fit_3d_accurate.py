# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Fit a SMPL model to the image given the pose estimation and segmentation mask
    (Modified from the code of Simplify)
"""

import logging
import os
from time import time
import argparse

import cv2
import numpy as np
import chumpy as ch

from opendr.camera import ProjectPoints
import pdb

from lib.robustifiers import GMOf
from smpl_webuser.serialization import load_model
from smpl_webuser.lbs import global_rigid_transformation
from smpl_webuser.verts import verts_decorated
from lib.max_mixture_prior import MaxMixtureCompletePrior
import pdb # pdb.set_trace()

flength = 5000.

_LOGGER = logging.getLogger(__name__)

# Mapping from LSP joints to SMPL joints.
# 0 Right ankle  8
# 1 Right knee   5
# 2 Right hip    2
# 3 Left hip     1
# 4 Left knee    4
# 5 Left ankle   7
# 6 Right wrist  21
# 7 Right elbow  19
# 8 Right shoulder 17
# 9 Left shoulder  16
# 10 Left elbow    18
# 11 Left wrist    20
# 12 Neck           -
# 13 Head top       added

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', type=str, required=True,
                        help='path to image file')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='output directory')
    return parser.parse_args()


# --------------------Camera estimation --------------------
def guess_init(model, focal_length, j2d, init_pose):
    """Initialize the camera translation via triangle similarity, by using the torso
    joints        .
    :param model: SMPL model
    :param focal_length: camera focal length (kept fixed)
    :param j2d: 14x2 array of CNN joints
    :param init_pose: 72D vector of pose parameters used for initialization (kept fixed)
    :returns: 3D vector corresponding to the estimated camera translation
    """
    cids = np.arange(0, 12)
    # map from LSP to SMPL joints
    j2d_here = j2d[cids]
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    opt_pose = ch.array(init_pose)
    _, A_global = global_rigid_transformation(opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])
    Jtr = Jtr[smpl_ids].r

    # 9 is L shoulder, 3 is L hip
    # 8 is R shoulder, 2 is R hip
    diff3d = np.array([Jtr[9] - Jtr[3], Jtr[8] - Jtr[2]])
    mean_height3d = np.mean(np.sqrt(np.sum(diff3d**2, axis=1)))

    diff2d = np.array([j2d_here[9] - j2d_here[3], j2d_here[8] - j2d_here[2]])
    mean_height2d = np.mean(np.sqrt(np.sum(diff2d**2, axis=1)))

    est_d = focal_length * (mean_height3d / mean_height2d)
    init_t = np.array([0., 0., est_d])
    return init_t


def initialize_camera(model,
                      j2d,
                      img,
                      init_pose,
                      flength=flength):
    """Initialize camera translation and body orientation
    :param model: SMPL model
    :param j2d: 14x2 array of CNN joints
    :param img: h x w x 3 image
    :param init_pose: 72D vector of pose parameters used for initialization
    :param flength: camera focal length (kept fixed)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on
                     both the estimated one and its flip)
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing the estimated camera,
              a boolean deciding if both the optimized body orientation and its flip should be
              considered, 3D vector for the body orientation
    """
    # optimize camera translation and body orientation based on torso joints
    # LSP torso ids:
    # 2=right hip, 3=left hip, 8=right shoulder, 9=left shoulder
    torso_cids = [2, 3, 8, 9]
    # corresponding SMPL torso ids
    torso_smpl_ids = [2, 1, 17, 16]

    center = np.array([img.shape[1] / 2, img.shape[0] / 2])
    rt = ch.zeros(3)    # initial camera rotation
    init_t = guess_init(model, flength, j2d, init_pose)
    t = ch.array(init_t)    # initial camera translation

    opt_pose = ch.array(init_pose)
    _, A_global = global_rigid_transformation(opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])

    # initialize the camera and project SMPL joints
    cam = ProjectPoints(f=np.array([flength, flength]), rt=rt, t=t, k=np.zeros(5), c=center)
    cam.v = Jtr     # project SMPL joints

    # optimize for camera translation and body orientation
    free_variables = [cam.t, opt_pose[:3]]
    ch.minimize(
        # data term defined over torso joints...
        {'cam': j2d[torso_cids] - cam[torso_smpl_ids],
         # ...plus a regularizer for the camera translation
         'cam_t': 1e2 * (cam.t[2] - init_t[2])},
        x0=free_variables, method='dogleg', callback=None,
        options={'maxiter': 100, 'e_3': .0001, 'disp': 0})

    return cam, opt_pose[:3].r


# --------------------Core optimization --------------------
def optimize_on_joints(j2d,
                       model,
                       cam,
                       img,
                       prior,
                       init_pose,
                       init_shape,
                       n_betas=10,
                       conf=None):
    """Fit the model to the given set of joints, given the estimated camera
    :param j2d: 14x2 array of CNN joints
    :param model: SMPL model
    :param cam: estimated camera
    :param img: h x w x 3 image
    :param prior: mixture of gaussians pose prior
    :param init_pose: 72D vector, pose prediction results provided by HMR
    :param init_shape: 10D vector, shape prediction results provided by HMR
    :param n_betas: number of shape coefficients considered during optimization
    :param conf: 14D vector storing the confidence values from the CNN
    :returns: a tuple containing the optimized model, its joints projected on image space, the
              camera translation
    """
    # define the mapping LSP joints -> SMPL joints
    # cids are joints ids for LSP:
    cids = range(12) + [13]
    # joint ids for SMPL
    # SMPL does not have a joint for head, instead we use a vertex for the head
    # and append it later.
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    # the vertex id for the joint corresponding to the head
    head_id = 411

    # weights assigned to each joint during optimization;
    # the definition of hips in SMPL and LSP is significantly different so set
    # their weights to zero
    base_weights = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    # initialize the shape to the mean shape in the SMPL training set
    betas = ch.array(init_shape)

    # instantiate the model:
    # verts_decorated allows us to define how many
    # shape coefficients (directions) we want to consider (here, n_betas)
    sv = verts_decorated(
        trans=ch.zeros(3),
        pose=ch.array(init_pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :n_betas],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)

    # make the SMPL joints depend on betas
    Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                       for i in range(len(betas))])
    J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(
        model.v_template.r)

    # get joint positions as a function of model pose, betas and trans
    (_, A_global) = global_rigid_transformation(
        sv.pose, J_onbetas, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

    # add the head joint, corresponding to a vertex...
    Jtr = ch.vstack((Jtr, sv[head_id]))

    # ... and add the joint id to the list
    smpl_ids.append(len(Jtr) - 1)

    # update the weights using confidence values
    weights = base_weights * conf[cids] if conf is not None else base_weights

    # project SMPL joints on the image plane using the estimated camera
    cam.v = Jtr

    # data term: distance between observed and estimated joints in 2D
    obj_j2d = lambda w, sigma: (
        w * weights.reshape((-1, 1)) * GMOf((j2d[cids] - cam[smpl_ids]), sigma))

    # mixture of gaussians pose prior
    pprior = lambda w: w * prior(sv.pose)
    # joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    alpha = 10
    my_exp = lambda x: alpha * ch.exp(x)
    obj_angle = lambda w: w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[
                                             58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])

    # weight configuration used in the paper, with joints + confidence values from the CNN
    # (all the weights used in the code were obtained via grid search, see the paper for more details)
    # the first list contains the weights for the pose priors,
    # the second list contains the weights for the shape prior
    opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                      [1e2, 5 * 1e1, 1e1, .5 * 1e1])

    # run the optimization in 4 stages, progressively decreasing the
    # weights for the priors
    for stage, (w, wbetas) in enumerate(opt_weights):
        _LOGGER.info('stage %01d', stage)
        objs = {}
        objs['j2d'] = obj_j2d(1., 100)
        objs['pose'] = pprior(w)
        objs['pose_exp'] = obj_angle(0.317 * w)
        objs['betas'] = wbetas * betas

        ch.minimize(objs, x0=[sv.betas, sv.pose],
                    method='dogleg', callback=None,
                    options={'maxiter': 100,
                             'e_3': .0001,
                             'disp': 0})

    return sv, cam.r, cam.t.r


def optimize_on_joints_and_silhouette(j2d,
                                      sil,
                                      model,
                                      cam,
                                      img,
                                      prior,
                                      init_pose,
                                      init_shape,
                                      n_betas=10,
                                      conf=None):
    """Fit the model to the given set of joints, given the estimated camera
    :param j2d: 14x2 array of CNN joints
    :param sil: h x w silhouette with soft boundaries (np.float32, range(-1, 1))
    :param model: SMPL model
    :param cam: estimated camera
    :param img: h x w x 3 image
    :param prior: mixture of gaussians pose prior
    :param init_pose: 72D vector, pose prediction results provided by HMR
    :param init_shape: 10D vector, shape prediction results provided by HMR
    :param n_betas: number of shape coefficients considered during optimization
    :param conf: 14D vector storing the confidence values from the CNN
    :returns: a tuple containing the optimized model, its joints projected on image space, the
              camera translation
    """
    # define the mapping LSP joints -> SMPL joints
    cids = range(12) + [13]
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]
    head_id = 411

    # weights assigned to each joint during optimization;
    # the definition of hips in SMPL and LSP is significantly different so set
    # their weights to zero
    base_weights = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    betas = ch.array(init_shape)

    # instantiate the model:
    sv = verts_decorated(
        trans=ch.zeros(3),
        pose=ch.array(init_pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=betas,
        shapedirs=model.shapedirs[:, :, :n_betas],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs)

    # make the SMPL joints depend on betas
    Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                       for i in range(len(betas))])
    J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(
        model.v_template.r)

    # get joint positions as a function of model pose, betas and trans
    (_, A_global) = global_rigid_transformation(
        sv.pose, J_onbetas, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

    # add the head joint
    Jtr = ch.vstack((Jtr, sv[head_id]))
    smpl_ids.append(len(Jtr) - 1)

    # update the weights using confidence values
    weights = base_weights * conf[cids] if conf is not None else base_weights

    # project SMPL joints and vertex on the image plane using the estimated camera
    cam.v = ch.vstack([Jtr, sv])

    # obtain a gradient map of the soft silhouette
    grad_x = cv2.Sobel(sil, cv2.CV_32FC1, 1, 0) * 0.125
    grad_y = cv2.Sobel(sil, cv2.CV_32FC1, 0, 1) * 0.125

    # data term #1: distance between observed and estimated joints in 2D
    obj_j2d = lambda w, sigma: (
        w * weights.reshape((-1, 1)) * GMOf((j2d[cids] - cam[smpl_ids]), sigma)
    )

    # data term #2: distance between the observed and projected boundaries
    obj_s2d = lambda w, sigma, flag, target_pose : (
        w * flag * GMOf((target_pose - cam[len(Jtr):(len(Jtr)+6890)]), sigma)
    )

    # mixture of gaussians pose prior
    pprior = lambda w: w * prior(sv.pose)
    # joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    alpha = 10
    my_exp = lambda x: alpha * ch.exp(x)
    obj_angle = lambda w: w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[58]),
                                              my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])


    # run the optimization in 4 stages, progressively decreasing the
    # weights for the priors
    print('****** Optimization on joints')
    curr_pose = sv.pose.r
    opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                      [1e2, 5 * 1e1, 1e1, .5 * 1e1])
    for stage, (w, wbetas) in enumerate(opt_weights):
        _LOGGER.info('stage %01d', stage)
        objs = {}
        objs['j2d'] = obj_j2d(1., 100)
        objs['pose'] = pprior(w)
        objs['pose_exp'] = obj_angle(0.317 * w)
        objs['betas'] = wbetas * betas
        objs['thetas'] = wbetas * (sv.pose - curr_pose) # constrain theta changes

        ch.minimize(objs, x0=[sv.betas, sv.pose],
                    method='dogleg', callback=None,
                    options={'maxiter': 100, 'e_3': .001, 'disp': 0})
    curr_pose = sv.pose.r
    # cam.v = ch.vstack([Jtr, sv.r])

    # run the optimization in 2 stages, progressively decreasing the
    # weights for the priors
    print('****** Optimization on silhouette and joints')
    opt_weights = zip([57.4, 4.78], [2e2, 1e2])
    for stage, (w, wbetas) in enumerate(opt_weights):
        _LOGGER.info('stage %01d', stage)
        # find the boundary vertices and estimate their expected location
        smpl_vs = cam.r[len(Jtr):, :]
        boundary_flag = np.zeros((smpl_vs.shape[0], 1))
        expected_pos = np.zeros((smpl_vs.shape[0], 2))
        for vi, v in enumerate(smpl_vs):
            r, c = int(v[1]), int(v[0])
            if r < 0 or r >= sil.shape[0] or c < 0 or c >= sil.shape[1]:
                continue
            sil_v = sil[r, c]
            grad = np.array([grad_x[r, c], grad_y[r, c]])
            grad_n = np.linalg.norm(grad)
            if grad_n > 1e-1 and sil_v < 0.4:   # vertex on or out of the boundaries
                boundary_flag[vi] = 1.0
                step = (grad/grad_n) * (sil_v/grad_n)
                expected_pos[vi] = np.array([c - step[0], r - step[1]])

        # run optimization
        objs = {}
        objs['j2d'] = obj_j2d(1., 100)
        objs['s2d'] = obj_s2d(5., 100, boundary_flag, expected_pos)
        objs['pose'] = pprior(w)
        objs['pose_exp'] = obj_angle(0.317 * w)
        objs['betas'] = wbetas * betas  # constrain beta changes
        objs['thetas'] = wbetas * (sv.pose - curr_pose) # constrain theta changes
        ch.minimize(objs, x0=[sv.betas, sv.pose],
                    method='dogleg', callback=None,
                    options={'maxiter': 100, 'e_3': .001, 'disp': 0})

    return sv, cam.r, cam.t.r


def run_single_fit(img,
                   j2d,
                   conf,
                   seg,
                   model,
                   init_pose,
                   init_shape,
                   n_betas=10,
                   flength=flength):
    """Run the fit for one specific image.
    :param img: h x w x 3 image
    :param j2d: 14x2 array of CNN joints
    :param conf: 14D vector storing the confidence values from the CNN
    :param seg: h x w soft silhouette
    :param model: SMPL model
    :param init_pose: 72D vector, pose prediction results provided by HMR
    :param init_shape: 10D vector, shape prediction results provided by HMR
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (kept fixed during optimization)
    :returns: a tuple containing camera/model parameters
    """

    # create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    # init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))

    # estimate the camera parameters
    print('**** Run camera optimization')
    (cam, body_orient) = initialize_camera(model, j2d, img,
                                           init_pose, flength=flength)

    # fit
    # print('**** Run body model optimization (on joints)')
    # (sv, opt_j2d, t) = optimize_on_joints(j2d, model, cam, img, prior,
    #                                       init_pose, init_shape,
    #                                       n_betas=n_betas, conf=conf)
    print('**** Run body model optimization (on silhouette and joints)')
    (sv, opt_j2d, t) = optimize_on_joints_and_silhouette(j2d, seg, model, cam, img, prior,
                                                         init_pose, init_shape,
                                                         n_betas=n_betas, conf=conf)

    # return fit parameters
    params = {'cam_t': cam.t.r,
              'f': cam.f.r,
              'pose': sv.pose.r,
              'betas': sv.betas.r}

    return params


def proj_smpl_onto_img(img, smpl, pose, shape, cam_f, cam_t):
    if isinstance(cam_f, float):
        cam_f = np.array([cam_f, cam_f])

    smpl.pose[:] = pose
    smpl.betas[:] = shape
    center = np.array([img.shape[1] / 2, img.shape[0] / 2])
    cam = ProjectPoints(
        f=cam_f, rt=ch.zeros(3), t=cam_t, k=np.zeros(5), c=center)

    cam.v = smpl.r
    for v in cam.r:
        r = int(round(v[1]))
        c = int(round(v[0]))
        if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
            img[r, c, :] = np.asarray([255, 255, 255])

    return img


def main(img_dir, joint_dir, joint_scores_dir, seg_dir, smpl_param_dir, smpl_dir, out_dir):
    print('** Read image from ' + img_dir)
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    j_indices = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9]

    print('** Read joints from ' + joint_dir)
    j2d = np.zeros((len(j_indices), 2))
    with open(joint_dir, 'r') as fp:
        lines = fp.readlines()
        line = lines[0]
        line = line.split('\t')
        if line[-1] == '\n' or line[-1] == '':
            line = line[:-1]    # ignore useless ending character
        pixels = line[1:]
        for i, ji in enumerate(j_indices):
            j2d[i, 0] = int(pixels[ji*2])
            j2d[i, 1] = int(pixels[ji*2+1])

    print('** Read joint confidence from ' + joint_scores_dir)
    conf = np.zeros((len(j_indices),))
    with open(joint_scores_dir, 'r') as fp:
        lines = fp.readlines()
        line = lines[0]
        line = line.split('\t')
        if line[-1] == '\n' or line[-1] == '':
            line = line[:-1]    # ignore useless ending character
        scores = line[1:]
        for i, ji in enumerate(j_indices):
            conf[i] = float(scores[ji])

    seg = cv2.imread(seg_dir)
    if len(seg.shape) == 3:
        seg = cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)
    seg = np.array(seg, np.float32)/255.0
    seg = cv2.blur(seg, (15, 15))*2.0 - 1.25

    print('** Read initial SMPL params from ' + smpl_param_dir)
    init_pose = np.zeros(72, dtype=np.float32)
    init_shape = np.zeros(10, dtype=np.float32)
    with open(smpl_param_dir, 'r') as fp:
        lines = fp.readlines()
        pose_line, shape_line = lines[0], lines[1]

        pose_line = pose_line.split(' ')
        if pose_line[-1] == '\n' or pose_line[-1] == '':
            pose_line = pose_line[:-1]    # ignore useless ending characters
        for ti, theta_str in enumerate(pose_line):
            init_pose[ti] = float(theta_str)

        shape_line = shape_line.split(' ')
        if shape_line[-1] == '\n' or shape_line[-1] == '':
            shape_line = shape_line[:-1]    # ignore useless ending characters
        for bi, beta_str in enumerate(shape_line):
            init_shape[bi] = float(beta_str)

    print('** Load SMPL model from ' + smpl_dir)
    model = load_model(smpl_dir)
    # img = proj_smpl_onto_img(img, model, init_pose, init_shape,
    #                          5000., np.array([-0.12224903, -0.03651915, 22.0691487]))
    # cv2.imshow('img', img)
    # cv2.waitKey()

    print('** Run fitting')
    params = run_single_fit(img, j2d, conf, seg, model, init_pose, init_shape)

    img_dir, img_name = os.path.split(img_fname)
    out_fname = os.path.join(out_dir, img_name+'.final.txt')
    print('** Saving to ' + out_fname)
    with open(out_fname, 'w') as fp:
        for v in params['cam_t']:
            fp.write('%f ' % v)
        fp.write('\n')
        for v in params['f']:
            fp.write('%f ' % v)
        fp.write('\n')
        for v in params['pose']:
            fp.write('%f ' % v)
        fp.write('\n')
        for v in params['betas']:
            fp.write('%f ' % v)
    img = proj_smpl_onto_img(img, model, params['pose'],
                             params['betas'], params['f'], params['cam_t'])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, img_name+'.smpl_proj.png'), img)
    cv2.imwrite(os.path.join(out_dir, img_name+'.soft_mask.png'),
                np.uint8(seg*127+127))
    # cv2.imshow('img', img)
    # cv2.waitKey()
    with open(os.path.join(out_dir, img_name+'.smpl.obj'), 'w') as fp:
        for v in model.r:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in model.f+1:
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )



if __name__ == '__main__':
    args = parse_args()
    img_fname = args.img_file
    out_dir = args.out_dir
    img_fname = os.path.abspath(img_fname)
    out_dir = os.path.abspath(out_dir)
    main(img_fname,
         img_fname+'.joints.txt',
         img_fname+'.joint_scores.txt',
         img_fname+'.segment.png',
         img_fname+'.smpl_param.txt',
         'models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
         out_dir)
