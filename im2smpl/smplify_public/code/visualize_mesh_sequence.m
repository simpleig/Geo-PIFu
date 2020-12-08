%Copyright 2016 Max Plank Society. All rights reserved.
%This software is provided for research purposes only.
%By using this software you agree to the terms of the SMPLify license here:
%     http://smplify.is.tue.mpg.de/license


function visualize_mesh_sequence( hdf5_path )
% VISUALIZE_MESH_SEQUENCE Visualize sequence of meshes in meshes.hdf5
%   INPUTS:
%        - verts: array of size num_verts x 3 x num_frames
%        - faces: array of size num_faces x 3
%
% Copyright (c) by Gerard Pons-Moll 2015

    verts = h5read(hdf5_path, '/all_verts');
    verts = permute(verts, [2 3 1]);
    faces = h5read(hdf5_path, '/faces')';

    figure(1); clf;
    ha=gca;
    v=verts(:,:,1);
    p=patch('parent',ha,'Vertices',v,'faces',faces+1);
    % Set visualization parameters
    xyzmax=max(max(verts),[],3);
    xyzmin=min(min(verts),[],3);
    axis equal off;
    axis([xyzmin(1) xyzmax(1) xyzmin(2) xyzmax(2) xyzmin(3) xyzmax(3)]);
    camlight
    set(p, 'ambientStrength', 0.4);
    set(p, 'diffuseStrength', 0.4);
    set(p, 'FaceLighting', 'phong');
    set(p, 'edgeLighting', 'phong');
    set(p,'FaceColor',[0.7,0.78,1],'EdgeColor','white')
    cameratoolbar('setmode', 'orbit');
    cameratoolbar('setcoordsys', 'y');
    set(ha, 'cameraupvector', [0 -1 0]);
    view(0, -90);


    cameratoolbar('show');

    for frame=1:size(verts,3)
        set(p,'Vertices',verts(:,:,frame),'faces',faces+1)
        pause(1)
    end
end
