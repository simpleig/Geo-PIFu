"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

An example script that shows how to use the fit parameter and pose SMPL.
Also visualizes the detected joints.
"""

if __name__ == '__main__':
    from os.path import join
    import cPickle as pickle
    import numpy as np
    from glob import glob
    import matplotlib.pyplot as plt

    from smpl_webuser.serialization import load_model
    from opendr.camera import ProjectPoints

    from render_model import render_model

    import argparse
    parser = argparse.ArgumentParser(description='run SMPLify on LSP dataset')
    parser.add_argument(
        'base_dir',
        default='/scratch1/projects/smplify_public/',
        nargs='?',
        help="Directory that contains results/, i.e."
        "the directory you untared human_eva_results.tar.gz")
    parser.add_argument(
        '-seq',
        default='S1_Walking',
        nargs='?',
        help="Human Eva sequence name: S{1,2,3}_{Walking,Box} ")
    args = parser.parse_args()

    base_dir = args.base_dir
    model_dir = join(base_dir, 'code', 'models')

    seq = args.seq + '_1_C1'

    data_dir = join(base_dir, 'results/human_eva', seq)

    results_path = join(data_dir, 'all_results.pkl')
    joints_path = join(data_dir, 'est_joints.npz')

    if 'S1' in seq:
        model_path = join(model_dir, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    else:
        model_path = join(model_dir, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    # Load everything:
    # SMPL model
    model = load_model(model_path)
    # detected joints
    est = np.load(joints_path)['est_joints']

    # SMPL parameters + camera
    print('opening %s' % results_path)
    with open(results_path, 'r') as f:
        res = pickle.load(f)
    poses = res['poses']
    betas = res['betas']

    # Camera rotation is always at identity,
    # The rotation of the body is encoded by the first 3 bits of poses.
    cam_ts = res['cam_ts']
    focal_length = res['focal_length']
    principal_pt = res['principal_pt']

    # Setup camera:
    cam = ProjectPoints(
        f=focal_length, rt=np.zeros(3), k=np.zeros(5), c=principal_pt)

    h = 480
    w = 640

    # Corresponding ids of detected and SMPL joints (except head)
    lsp_ids = range(0, 12)
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    plt.ion()

    for i, (joints, pose, beta,
            cam_t) in enumerate(zip(est.T, poses, betas, cam_ts)):

        joints = joints[lsp_ids, :]

        # Pose the model:
        model.pose[:] = pose
        # Shape the model: the model requires 10 dimensional betas, 
        # pad it with 0 if length of beta is less than 10
        model.betas[:] = np.hstack((beta, np.zeros(10 - len(beta))))

        # Set camera location.
        cam.t = cam_t
        # make it project SMPL joints in 3D.
        cam.v = model.J_transformed[smpl_ids]
    
        # projected SMPL joints in image coordinate.
        smpl_joints = cam.r

        # Render this.
        res_im = (render_model(model.r, model.f, w, h, cam) *
                  255.).astype('uint8')
        plt.show()
        plt.subplot(121)
        plt.imshow(np.ones((h, w, 3)))
        plt.scatter(smpl_joints[:, 0], smpl_joints[:, 1], c='w')
        plt.scatter(joints[:, 0], joints[:, 1], c=joints[:, 2])
        plt.axis('off')
        plt.subplot(122)
        plt.cla()
        plt.imshow(res_im)
        plt.axis('off')
        plt.show()
        plt.pause(1)
        raw_input('Press any key to continue...')

    plt.off()
