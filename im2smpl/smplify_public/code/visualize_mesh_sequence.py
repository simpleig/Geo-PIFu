#!/usr/local/bin/ipython --gui=wx
"""
Copyright 2016 Max Plank Society, Federica Bogo, Angjoo Kanazawa.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

This script visualizes the mesh (face & vertices) contained in the provided
meshes.hdf5 files
Requires vtkpython and mayavi:
sudo apt-get install libvtk5-dev python-vtk
pip install mayavi

"""
from argparse import ArgumentParser
import h5py
from itertools import count
from mayavi import mlab
import numpy as np


def main(hdf5_path):

    with h5py.File(hdf5_path, 'r') as f:
        all_verts = np.array(f.get('all_verts'))
        faces = np.array(f.get('faces'))

    fig = mlab.figure(1, bgcolor=(1, 1, 1))

    @mlab.animate(delay=1000, ui=True)
    def animation():
        for i in count():
            frame = i % all_verts.shape[2]
            verts = all_verts[:, :, frame].T
            mlab.clf()
            mlab.triangular_mesh(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                faces,
                color=(.9, .7, .7))
            fig.scene.z_minus_view()
            mlab.view(azimuth=180)
            mlab.title('mesh %d' % i, size=0.5, height=0, color=(0, 0, 0))
            yield

    a = animation()
    mlab.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Visuzalize mesh.hdf5 files')
    parser.add_argument(
        'path',
        type=str,
        default='../results/lsp/meshes.hdf5',
        nargs='?',
        help='Path to meshes.hdf5')
    args = parser.parse_args()
    main(args.path)
