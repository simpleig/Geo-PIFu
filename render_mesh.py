import os
import numpy as np
from opendr.camera import ProjectPoints
# from opendr.renderer import ColoredRenderer
from MyCamera import ProjectPointsOrthogonal
from MyRenderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
import cv2 as cv
import pdb # pdb.set_trace()
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from DataUtil.ObjIO import load_obj_data, save_obj_data, save_obj_data_binary
import glob
import time
import argparse
import DataUtil.VoxelizerUtil as voxel_util
from subprocess import call
import scipy
import copy
import sys
sys.path.insert(0, './im2smpl/smplify_public/code')
from smpl_webuser.serialization import load_model
import json

SMPL_MALE_PKL_PATH = './im2smpl/smplify_public/code/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
SMPL_FEMALE_PKL_PATH = './im2smpl/smplify_public/code/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
assert(  os.path.exists(SMPL_MALE_PKL_PATH) and os.path.exists(SMPL_FEMALE_PKL_PATH)  )
FACES_ATLAS_IDX_U_V_NPY_PATH = './UVTextureConverter/input/facesAtlas_Idx_U_V.npy'
assert(  os.path.exists(FACES_ATLAS_IDX_U_V_NPY_PATH)  )
SIGMA_SQUARE = 0.05*0.05 # for matching between voxels and SMPL vertices
KNN_K = 4
NUM_VIEW_RENDERING = 4
VOXEL_H = 192
VOXEL_W = 128
H_NORMALIZE_HALF = 0.5
VOXEL_SIZE = 2.*H_NORMALIZE_HALF/VOXEL_H
VOXELIZER_PATH = './voxelizer/build/bin'
BACKGROUND_NAMES = ["bedroom", 
                    "bridge",
                    "church_outdoor", 
                    "classroom",
                    "conference_room",
                    "dining_room",
                    "kitchen", 
                    "living_room", 
                    "restaurant", 
                    "tower"]
SUPPORTED_ADDITIONAL_TYPES = ["smplSemVoxels"]

class render_mesh(object):

    def __init__(self,w=128*2,h=192*2,f=5000.,near=0.5,far=25,saveDir=None,meshNormMargin=0.15):
        
        self.meshNormMargin = meshNormMargin # margin when normalizing mesh into H [-0.5,0.5]*(1-margin), W/D [-0.333,0.333]*(1-margin)
        self.threshH = H_NORMALIZE_HALF * (1-self.meshNormMargin)
        self.threshWD = H_NORMALIZE_HALF * VOXEL_W/VOXEL_H * (1-self.meshNormMargin)

        self.w = w
        self.h = h
        self.f = f

        self.near = near
        self.far = far

        self.colors = {'pink': [.7, .7, .9],
                       'neutral': [.9, .9, .8],
                       'capsule': [.7, .75, .5],
                       'yellow': [.5, .7, .75],
                       'white': [1, 1, 1],
                       'lightWhite': [.7, .7, .7]}

        self.constBackground = 4294967295

        self.rn = None

        self.vertsRaw, self.verts, self.faces, self.vertsColor, self.vertsNormalSourcePose, self.vertsNormal = None, None, None, None, None, None
        self.meshNormMean, self.meshNormScale = None, None # the mean and scaling factor for voxel-region-based normalization: self.vertsRaw -> self.verts

        # load SMPL rest shape
        self.smpl_vtx_std = voxel_util.get_smpl_std_vertex() # the original normalization with distortion
        self.smpl_vtx_std_keepRatio = voxel_util.get_smpl_std_vertex_keepRatio() # normalize into [0,1] and keep the DHW ratio

        # mesh vertex semantic labels, obtained by matching with SMPL rest shape
        self.mesh_vertex_label, self.mesh_vertex_label_keepRatio = None, None
        self.vertsSmpl, self.vertsSmplSourcePose, self.facesSmpl = None, None, None

        # SMPL vertices and joints, parsed from SMPL parameters txt, not the directly provided SMPL obj
        self.vertsFromSmplParamsSourcePose, self.jointsFromSmplParamsSourcePose, self.jointsFromSmplParams = None, None, None

        self.rgbImage = None
        self.visMap, self.barycentricMap, self.maskImage = None, None, None
        self.normalMap, self.normalRGB = None, None
        self.meshVoxels, self.maskFromVoxels = None, None
        self.meshSem, self.meshSemKeepRatio = None, None
        self.skeleton2D, self.skeleton3D = None, None
        self.smplSem, self.smplSemKeepRatio = None, None
        self.smplIUV = None
        self.smplSemVoxels = None

        self.saveDir = saveDir

        self.quickDemo = None
        self.eachMeshWithBgNum = None
        self.newMeshToRead = None

        # load facesAtlas_Idx_U_V.npy
        self.facesAtlas_Idx_U_V = np.load(FACES_ATLAS_IDX_U_V_NPY_PATH)

        # for visually checking the rendered data
        self.visual_check_counter = 0

        # for reproducing the rendering process of each mesh
        self.meshPath = None
        self.bgImgPath = None
        self.dataPrefix = None
        self.randomRot = None # for each mesh, we add a random (3,3) rotation matrix
        self.skyLightRotDegrees = [-60, 60, 180] # Sky: {left,right,back} lightings

        # for using the saved config file
        self.loadedConfig = None
        self.addionalType = None

    def render_colored_mesh_with_background_use_config(self,quickDemo=False,loadedConfig=None,addionalType=None):

        # init. args.
        self.meshPath = loadedConfig["meshPath"]
        self.bgImgPath = loadedConfig["bgImgPath"]
        self.quickDemo = quickDemo
        self.dataPrefix = loadedConfig["dataPrefix"]
        self.eachMeshWithBgNum = loadedConfig["args"]["eachMeshWithBgNum"]
        self.loadedConfig = loadedConfig
        self.addionalType = addionalType
        if addionalType not in SUPPORTED_ADDITIONAL_TYPES:
            print("addionalType: {} is not supported yet!".format(addionalType))
            pdb.set_trace()
        self.newMeshToRead = True if (self.quickDemo or (self.dataPrefix%(self.eachMeshWithBgNum*NUM_VIEW_RENDERING) == 0)) else False

        # rendering by addionalType
        if addionalType == "smplSemVoxels":

            # load mesh, apply rot, get {mean,scaling}
            # load smpl, apply rot, normalize it with {mean, scaling}
            self.set_v_f_vc_bgcolor(meshPath=loadedConfig["meshPath"])

            # get and save smplSemVoxels for {FRONT,RIGHT,BACK,LEFT}
            def get_N_save_smplSemVoxels(meshPath=None,info="FRONT"):

                # get smplSemVoxels
                self.smplSemVoxels = self.semanticVoxelizationSMPL(meshPath=meshPath,info=info)

                # get prefix
                prefixDict = {"FRONT": 0, "RIGHT": 1, "BACK":2, "LEFT": 3}
                assert(info in prefixDict.keys())
                prefix = int(self.dataPrefix)+int(prefixDict[info])

                # create dir and save smplSemVoxels.obj and .npy
                dirName = self.create_dir(self.saveDir+"/smplSemVoxels")
                np.save("%s/%06d.npy"%(dirName,prefix), (self.smplSemVoxels*255).astype(np.uint8)) # 9.4 MB, (XYZC of WHDC)

                # # visual consistency check with other rendering items
                # voxel_util.save_v_volume(self.smplSemVoxels, "%s/%06d.obj"%(dirName,prefix), VOXEL_H, VOXEL_W, VOXEL_SIZE)
                # voxel_util.save_volume(np.load("%s/meshVoxels/%06d.npy"%(self.saveDir,prefix)), "%s/%06d_meshVoxels.obj"%(dirName,prefix), VOXEL_H, VOXEL_W, VOXEL_SIZE)
                # os.system("cp %s/skeleton3D/%06d.obj %s/%06d_skeleton3D.obj" % (self.saveDir,prefix,dirName,prefix))
                # os.system("cp %s/rgbImage/%06d.jpg %s/%06d_RGB.jpg" % (self.saveDir,prefix,dirName,prefix))
                
            # get and save for FRONT
            get_N_save_smplSemVoxels(meshPath=self.meshPath,info="FRONT")

            # change idx and save for RIGHT
            get_N_save_smplSemVoxels(meshPath=self.meshPath,info="RIGHT")

            # change idx and save for BACK
            get_N_save_smplSemVoxels(meshPath=self.meshPath,info="BACK")

            # change idx and save for LEFT
            get_N_save_smplSemVoxels(meshPath=self.meshPath,info="LEFT")

    def render_colored_mesh_with_background(self,meshPath,bgImgPath,quickDemo=False,dataPrefix=None,eachMeshWithBgNum=None):

        # init. args.
        self.meshPath = meshPath
        self.bgImgPath = bgImgPath
        self.quickDemo = quickDemo
        self.dataPrefix = dataPrefix
        self.eachMeshWithBgNum = eachMeshWithBgNum
        self.newMeshToRead = True if (self.quickDemo or (self.dataPrefix%(self.eachMeshWithBgNum*NUM_VIEW_RENDERING) == 0)) else False

        # init. render
        self.rn = ColoredRenderer()

        # set cam
        self.rn.camera = self.init_cam(w=self.w,
                                       h=self.h,
                                       rt=np.array([0,0,0]),
                                       t=np.array([0,0,2]),
                                       f=self.f)

        # set rendering frustum region
        self.rn.frustum = {'near': self.near,
                           'far': self.far,
                           'height': self.h,
                           'width': self.w}

        # set background image
        self.rn.background_image = self.load_image(imgPath=bgImgPath)

        # set {v, f, vc, bgcolor}
        self.set_v_f_vc_bgcolor(meshPath=meshPath)

        # set sky lighting directions
        self.set_sky_lighting_directions()

        #----- FRONT view rendering -----
        self.start_rendering(accuRotDegree=0, info="FRONT", dataPrefix=dataPrefix, meshPath=meshPath)

        #----- RIGHT view rendering -----
        self.start_rendering(accuRotDegree=-90, info="RIGHT", dataPrefix=dataPrefix, meshPath=meshPath)

        #----- BACK view rendering -----
        self.start_rendering(accuRotDegree=-90, info="BACK", dataPrefix=dataPrefix, meshPath=meshPath)

        #----- LEFT view rendering -----
        self.start_rendering(accuRotDegree=-90, info="LEFT", dataPrefix=dataPrefix, meshPath=meshPath)
        
    def set_sky_lighting_directions(self):

        self.skyLightRotDegrees[0] = -np.random.randint(3., high=90+1) # left
        self.skyLightRotDegrees[1] = np.random.randint(3., high=90+1) # right

        # range constraint, to avoid over-bright
        leftAngleThresh = -40
        rightAngleThresh = 40
        if (self.skyLightRotDegrees[0] >= leftAngleThresh) and (self.skyLightRotDegrees[1] <= rightAngleThresh):

            if abs(self.skyLightRotDegrees[0]) >= self.skyLightRotDegrees[1]:
                self.skyLightRotDegrees[1] = None
            else:
                self.skyLightRotDegrees[0] = None

        else:

            if self.skyLightRotDegrees[0] >= leftAngleThresh:
                self.skyLightRotDegrees[0] = None

            if self.skyLightRotDegrees[1] <= rightAngleThresh:
                self.skyLightRotDegrees[1] = None

        assert(self.skyLightRotDegrees[0] or self.skyLightRotDegrees[1])

    def quick_demo(self,info="Viewpoint?",meshPath="mesh_normalized.obj",smplPath="smpl_normalized.obj"):

        # init. args.
        demoDir = "./examplesRendering/results"
        if info == "FRONT":
            if not os.path.exists(demoDir):
                os.makedirs(demoDir)
            else:
                os.system("rm -r %s/*"%(demoDir))
        assert( os.path.exists(demoDir) )

        # save rgbImage
        cv.imwrite("%s/%s_rgb.jpg"%(demoDir,info), (self.rgbImage*255).astype(np.uint8)[:,:,::-1])

        # save normalRGB
        cv.imwrite("%s/%s_normal.jpg"%(demoDir,info), (self.normalRGB*255).astype(np.uint8)[:,:,::-1])

        # save maskImage
        cv.imwrite("%s/%s_mask.jpg"%(demoDir,info), (self.maskImage*255).astype(np.uint8))

        # save mesh_normalized.obj, only for FRONT
        if info == "FRONT":
            assert(  os.path.exists(meshPath)  )
            call(["mv", meshPath, "%s/%s_mesh_normalized.obj"%(demoDir,info)])

        # save meshVoxels as .obj & .npy
        voxel_util.save_volume(self.meshVoxels, "%s/%s_meshVoxels.obj"%(demoDir,info), VOXEL_H, VOXEL_W, VOXEL_SIZE)
        np.save("%s/%s_meshVoxels.npy"%(demoDir,info), self.meshVoxels) # 3.1 MB

        # save maskFromVoxels
        colNew = int((self.f/self.maskFromVoxels.shape[0])*self.maskFromVoxels.shape[1])
        rowNew = self.f
        maskFromVoxelsResized = cv.resize((self.maskFromVoxels*255).astype(np.uint8), (colNew,rowNew), interpolation=cv.INTER_NEAREST)
        cv.imwrite("%s/%s_maskFromVoxelsResized.jpg"%(demoDir,info), maskFromVoxelsResized)
        cv.imwrite("%s/%s_maskFromVoxels.jpg"%(demoDir,info), (self.maskFromVoxels*255).astype(np.uint8))

        # save mesh semantic segmentation mask (with KeepRatio)
        cv.imwrite("%s/%s_meshSem.jpg"%(demoDir,info), (self.meshSem*255).astype(np.uint8)[:,:,::-1])
        cv.imwrite("%s/%s_meshSemKeepRatio.jpg"%(demoDir,info), (self.meshSemKeepRatio*255).astype(np.uint8)[:,:,::-1])

        # save 3D skeletons
        save_obj_data_binary({"v": self.skeleton3D}, "%s/%s_skeleton3D.obj"%(demoDir,info))
        np.save("%s/%s_skeleton3D.npy"%(demoDir,info), self.skeleton3D)

        # save 2D skeletons
        cv.imwrite("%s/%s_skeleton2D.jpg"%(demoDir,info), (self.draw_points_with_markers(points=self.skeleton2D,markerSize=8)*255).astype(np.uint8)[:,:,::-1])
        np.save("%s/%s_skeleton2D.npy"%(demoDir,info), self.skeleton2D)

        # save SMPL semantic segmentation mask (with KeepRatio)
        cv.imwrite("%s/%s_smplSem.jpg"%(demoDir,info), (self.smplSem*255).astype(np.uint8)[:,:,::-1])
        cv.imwrite("%s/%s_smplSemKeepRatio.jpg"%(demoDir,info), (self.smplSemKeepRatio*255).astype(np.uint8)[:,:,::-1])

        # save SMPL IUV
        cv.imwrite("%s/%s_smplIUV.jpg"%(demoDir,info), (self.smplIUV).astype(np.uint8)[:,:,::-1])

        # save smpl_normalized.obj, only for FRONT
        if info == "FRONT":
            assert(  os.path.exists(smplPath)  )
            call(["mv", smplPath, "%s/%s_smpl_normalized.obj"%(demoDir,info)])

        # save smplSemVoxels as .obj & .npy
        voxel_util.save_v_volume(self.smplSemVoxels, "%s/%s_smplSemVoxels.obj"%(demoDir,info), VOXEL_H, VOXEL_W, VOXEL_SIZE)
        np.save("%s/%s_smplSemVoxels.npy"%(demoDir,info), (self.smplSemVoxels*255).astype(np.uint8)) # 9.4 MB

        # log
        print("%s: rgbImage, normalRGB, maskImage, mesh, meshVoxels, maskFromVoxels(Resized), meshSem(KeepRatio), skeleton2D/3D, smplSem(KeepRatio), smplIUV, smplSemVoxels..." % (info))

    def draw_points_with_markers(self,points,markerSize=8):

        # init args.
        markersMap = np.ones((self.h,self.w,3), np.uint8)

        # draw each point onto (H,W)
        for ptIdx in range(points.shape[0]):

            # get point XY coord
            ptX = int(points[ptIdx,0])
            ptY = int(points[ptIdx,1])

            # FOV check, and draw the marker
            for rowOff in range(-markerSize,markerSize+1):
                for colOff in range(-markerSize,markerSize+1):
                    ptXDraw = ptX + colOff
                    ptYDraw = ptY + rowOff
                    if (0 <= ptXDraw < self.w) and (0 <= ptYDraw < self.h):
                        markersMap[ptYDraw,ptXDraw] = np.array([1,0,0]) # red

        return markersMap

    def create_dir(self,dirName):

        if not os.path.exists(dirName):
            os.makedirs(dirName)
            assert( os.path.exists(dirName) )

        return dirName

    def save_config(self,pathConfig):

        # init. vars.
        data = {}

        # args
        data['args'] = vars(args)

        # meshPath
        data["meshPath"] = self.meshPath

        # bgImgPath
        data["bgImgPath"] = self.bgImgPath

        # dataPrefix
        data["dataPrefix"] = self.dataPrefix

        # skyLightRotDegrees
        data["skyLightRotDegrees"] = self.skyLightRotDegrees

        # randomRot
        data["randomRot"] = self.randomRot.tolist()

        with open(pathConfig, 'w') as outfile:
            json.dump(data, outfile)

    def save_rendering(self, dataPrefix="demo", info="FRONT", numMeshVisualCheck=1):

        # get prefix
        prefixDict = {"FRONT": 0, "RIGHT": 1, "BACK":2, "LEFT": 3}
        assert(info in prefixDict.keys())
        prefix = int(dataPrefix)+int(prefixDict[info])

        # save config of {args, meshPath, bgImgPath, dataPrefix, lighting, randRot}
        dirName = self.create_dir(self.saveDir+"/config")
        self.save_config(pathConfig="%s/%06d.json"%(dirName,prefix))

        # save rgbImage
        dirName = self.create_dir(self.saveDir+"/rgbImage")
        cv.imwrite("%s/%06d.jpg"%(dirName,prefix), (self.rgbImage*255).astype(np.uint8)[:,:,::-1])

        # save normalRGB
        dirName = self.create_dir(self.saveDir+"/normalRGB")
        cv.imwrite("%s/%06d.jpg"%(dirName,prefix), (self.normalRGB*255).astype(np.uint8)[:,:,::-1])

        # save maskImage
        dirName = self.create_dir(self.saveDir+"/maskImage")
        cv.imwrite("%s/%06d.jpg"%(dirName,prefix), (self.maskImage*255).astype(np.uint8))

        # save meshVoxels as .obj & .npy
        dirName = self.create_dir(self.saveDir+"/meshVoxels")
        if self.visual_check_counter<numMeshVisualCheck*args.eachMeshWithBgNum*NUM_VIEW_RENDERING: voxel_util.save_volume(self.meshVoxels, "%s/%06d.obj"%(dirName,prefix), VOXEL_H, VOXEL_W, VOXEL_SIZE)
        np.save("%s/%06d.npy"%(dirName,prefix), self.meshVoxels) # 3.1 MB, dtype is np.bool of {True, False}

        # save maskFromVoxels
        dirName = self.create_dir(self.saveDir+"/maskFromVoxels")
        colNew = int((self.f/self.maskFromVoxels.shape[0])*self.maskFromVoxels.shape[1])
        rowNew = self.f
        maskFromVoxelsResized = cv.resize((self.maskFromVoxels*255).astype(np.uint8), (colNew,rowNew))
        cv.imwrite("%s/%06d_Resized.jpg"%(dirName,prefix), maskFromVoxelsResized)
        cv.imwrite("%s/%06d.jpg"%(dirName,prefix), (self.maskFromVoxels*255).astype(np.uint8))

        # save mesh semantic segmentation mask (with KeepRatio)
        dirName = self.create_dir(self.saveDir+"/meshSem") 
        cv.imwrite("%s/%06d.jpg"%(dirName,prefix), (self.meshSem*255).astype(np.uint8)[:,:,::-1])
        cv.imwrite("%s/%06d_KeepRatio.jpg"%(dirName,prefix), (self.meshSemKeepRatio*255).astype(np.uint8)[:,:,::-1])

        # save 3D skeletons
        dirName = self.create_dir(self.saveDir+"/skeleton3D")
        save_obj_data_binary({"v": self.skeleton3D}, "%s/%06d.obj"%(dirName,prefix))

        # save 2D skeletons
        dirName = self.create_dir(self.saveDir+"/skeleton2D")
        if self.visual_check_counter<numMeshVisualCheck*args.eachMeshWithBgNum*NUM_VIEW_RENDERING: cv.imwrite("%s/%06d.jpg"%(dirName,prefix), (self.draw_points_with_markers(points=self.skeleton2D,markerSize=8)*255).astype(np.uint8)[:,:,::-1])
        np.save("%s/%06d.npy"%(dirName,prefix), self.skeleton2D)

        # save SMPL semantic segmentation mask (with KeepRatio)
        dirName = self.create_dir(self.saveDir+"/smplSem") 
        cv.imwrite("%s/%06d.jpg"%(dirName,prefix), (self.smplSem*255).astype(np.uint8)[:,:,::-1])
        cv.imwrite("%s/%06d_KeepRatio.jpg"%(dirName,prefix), (self.smplSemKeepRatio*255).astype(np.uint8)[:,:,::-1])

        # save SMPL IUV
        dirName = self.create_dir(self.saveDir+"/smplIUV")
        cv.imwrite("%s/%06d.jpg"%(dirName,prefix), (self.smplIUV).astype(np.uint8)[:,:,::-1])

        # save smplSemVoxels as .obj & .npy
        dirName = self.create_dir(self.saveDir+"/smplSemVoxels")
        if self.visual_check_counter<numMeshVisualCheck*args.eachMeshWithBgNum*NUM_VIEW_RENDERING: voxel_util.save_v_volume(self.smplSemVoxels, "%s/%06d.obj"%(dirName,prefix), VOXEL_H, VOXEL_W, VOXEL_SIZE)
        np.save("%s/%06d.npy"%(dirName,prefix), (self.smplSemVoxels*255).astype(np.uint8)) # 9.4 MB

        # update self.visual_check_counter
        if self.visual_check_counter<numMeshVisualCheck*args.eachMeshWithBgNum*NUM_VIEW_RENDERING: self.visual_check_counter += 1

    def start_rendering(self, accuRotDegree, info, dataPrefix, meshPath):

        # rotate views for {RIGHT, BACK, LEFT}
        self.verts = self.inverseRotateY(points=self.verts,angle=accuRotDegree) # vertex of the mesh
        self.vertsNormal = self.inverseRotateY(points=self.vertsNormal,angle=accuRotDegree) # normal of the mesh
        self.vertsSmpl = self.inverseRotateY(points=self.vertsSmpl,angle=accuRotDegree) # vertex of the SMPL
        self.jointsFromSmplParams = self.inverseRotateY(points=self.jointsFromSmplParams,angle=accuRotDegree) # joints of the SMPL
        
        # init. the render
        self.rn.set(v=self.verts, f=self.faces, vc=self.vertsColor, bgcolor=np.ones(3))

        # set lighting
        self.set_lighting(initSkyLightLoc=np.array([0.,-2.5,-10]))

        # render rgbImage
        self.rgbImage = self.renderRGB() # (h,w,3), values in (0,1)

        # render {0,1} maskImage, 1-mask, 0-bg
        self.visMap, self.barycentricMap, self.maskImage = self.renderMask()
        
        # render normalMap
        self.normalMap, self.normalRGB = self.renderNormal()

        # voxelize the mesh
        self.meshVoxels, self.maskFromVoxels = self.voxelize(meshPath=meshPath,info=info)

        # render mesh semantic segmentation mask
        self.meshSem, self.meshSemKeepRatio = self.renderMeshSem()

        # render 2D skeleton
        self.skeleton2D, self.skeleton3D = self.renderSkeleton()

        # render SMPL semantic segmentation mask
        self.smplSem, self.smplSemKeepRatio = self.renderSmplSem()

        # render IUV map from SMPL
        self.smplIUV = self.renderSmplIUV()

        # render SMPL semantic voxels
        self.smplSemVoxels = self.semanticVoxelizationSMPL(meshPath=meshPath,info=info)

        # visualize or save rendered data
        if self.quickDemo:

            self.quick_demo(info=info,meshPath=meshPath.replace("mesh.obj","mesh_normalized.obj"),smplPath=meshPath.replace("mesh.obj","smpl_normalized.obj"))
        else:

            self.save_rendering(dataPrefix=dataPrefix,info=info)

    def renderSmplIUV(self):

        #----- init args -----
        smplIUV = np.zeros((self.h, self.w, 3), np.float32)

        #----- get IUV map -----

        # (numValidPixels,1), (numValidPixels,2), (self.h,self.w)
        validPixels_Idx, validPixels_U_V, fgPixels = self.barcentricSamplingIUV()

        # assign (validPixels_Idx, validPixels_U_V) of (numValidPixels,1+2) into smplIUV[self.maskImage.astype(np.bool)]
        smplIUV[fgPixels,0] = validPixels_U_V[:,1]*255. # V, [0., 255.]
        smplIUV[fgPixels,1] = validPixels_U_V[:,0]*255. # U, [0., 255.]
        smplIUV[fgPixels,2] = validPixels_Idx[:,0] # Idx, {0.,...,24}, 0. is background
        smplIUV = smplIUV.astype(np.uint8)

        return smplIUV

        # save_obj_data_binary({"v":self.vertsSmpl, "f":self.facesSmpl, "vc":self.smpl_vtx_std}, "./examplesRendering/smpl_iuv.obj")
        # iuvSmplArr = np.load("./im2smpl/smplify_public/code/models/uv_table_smpl.npy") # (6890,2), values in (0,1)
        # iuvSmplSize = 512
        # iuvSmplArr = np.round(iuvSmplArr*(iuvSmplSize-1))
        # assert(  np.all(0<=iuvSmplArr) and np.all(iuvSmplArr<iuvSmplSize)  )
        # iuvSmplMap = np.ones((iuvSmplSize,iuvSmplSize,3))
        # for ptIdx in range(iuvSmplArr.shape[0]):
        #     # get point UV coord
        #     ptU = int(iuvSmplArr[ptIdx,0])
        #     ptV = int(iuvSmplArr[ptIdx,1])
        #     # draw the marker
        #     markerSize = 8
        #     for rowOff in range(-markerSize,markerSize+1):
        #         for colOff in range(-markerSize,markerSize+1):
        #             rowIdx = (iuvSmplSize-1-ptU) + rowOff
        #             colIdx = ptV + colOff
        #     iuvSmplMap[rowIdx,colIdx] = self.smpl_vtx_std[ptIdx]
        # cv.imwrite("./examplesRendering/smpl_iuv.jpg", (iuvSmplMap*255).astype(np.uint8)[:,:,::-1])
        # print("save the colored SMPL, voxel normalized and IUV map, for consistency check...")
        # pdb.set_trace()

    def renderSkeleton(self):

        # apply orthographic projection on the 3D skeleton coords (24,3)
        skeleton2D = self.rn.camera.project_points(points=self.jointsFromSmplParams) # (24,2)

        # return the 2D/3D skeletons
        return skeleton2D, self.jointsFromSmplParams

    def barcentricSamplingIUV(self):
        """
        Input:
            facesAtlas_Idx_U_V: (numFaces, 1+6)
        Return:
            validPixels_Idx: (numValidPixels,1), {1.,...,24.}, np.float32
            validPixels_U_V: (numValidPixels,2), [0.,1.], np.float32
            fgPixels: (self.h,self.w), np.bool
        """

        #----- re-compute visMap, barycentricMap, maskImage of the SMPL model -----
        visMapSmpl, barycentricMapSmpl, maskImageSmpl = self.renderMask()
        fgPixels = maskImageSmpl.astype(np.bool)

        #----- obtain validPixels_Idx of (numValidPixels,1) -----
        assert(  self.facesAtlas_Idx_U_V.shape[0] == self.facesSmpl.shape[0]  )
        validPixels_Idx_U_V = self.facesAtlas_Idx_U_V[visMapSmpl[fgPixels]] # (numValidPixels,1+6)
        validPixels_Idx = validPixels_Idx_U_V[:,0:1] # (numValidPixels,1)

        #----- obtain validPixels_U_V of (numValidPixels,2) -----

        # target-UVs of 3-verts
        validPixels_U_V_ofVert0 = validPixels_Idx_U_V[:,1:3] # (numValidPixels,2)
        validPixels_U_V_ofVert1 = validPixels_Idx_U_V[:,3:5] # (numValidPixels,2)
        validPixels_U_V_ofVert2 = validPixels_Idx_U_V[:,5:7] # (numValidPixels,2)

        # barycentric-weights of 3-verts
        validFaceVertWeightArr = barycentricMapSmpl[fgPixels] # (numValidPixels,3)
        validFaceVertWeightArr_0 = validFaceVertWeightArr[:,0:1] # (numValidPixels,1)
        validFaceVertWeightArr_1 = validFaceVertWeightArr[:,1:2] # (numValidPixels,1)
        validFaceVertWeightArr_2 = validFaceVertWeightArr[:,2:3] # (numValidPixels,1)

        # multiply barycentric-weights with target-UVs, each (numValidPixels,2)
        validPixels_U_V_ofVert0 *= validFaceVertWeightArr_0
        validPixels_U_V_ofVert1 *= validFaceVertWeightArr_1
        validPixels_U_V_ofVert2 *= validFaceVertWeightArr_2

        # get the fused UV (numValidPixels,2)
        validPixels_U_V = validPixels_U_V_ofVert0+validPixels_U_V_ofVert1+validPixels_U_V_ofVert2 # (numValidPixels,2)

        # clip into [0,1]
        _ = np.clip(a=validPixels_U_V, a_min=0, a_max=1, out=validPixels_U_V)

        return validPixels_Idx, validPixels_U_V, fgPixels

    def barcentricSampling(self,samplingVertTargetArr):

        # get face indices (N,)
        validFaceIdxArr = self.visMap[self.maskImage.astype(np.bool)] # (N,)

        # use face indices to get vertex indices (N,3)
        validFaceVertArr = self.rn.f[validFaceIdxArr] # (N,3)
        validFaceVertArr_0 = validFaceVertArr[:,0] # (N,)
        validFaceVertArr_1 = validFaceVertArr[:,1] # (N,)
        validFaceVertArr_2 = validFaceVertArr[:,2] # (N,)

        # use vertex indices to obtain from target vector list, each (N,3)
        validFaceVertTargetArr_0 = samplingVertTargetArr[validFaceVertArr_0] # (N,3)
        validFaceVertTargetArr_1 = samplingVertTargetArr[validFaceVertArr_1] # (N,3)
        validFaceVertTargetArr_2 = samplingVertTargetArr[validFaceVertArr_2] # (N,3)

        # use face indices to obtain from barcentric map (N,3)
        validFaceVertWeightArr = self.barycentricMap[self.maskImage.astype(np.bool)] # (N,3)
        validFaceVertWeightArr_0 = validFaceVertWeightArr[:,0:1] # (N,1)
        validFaceVertWeightArr_1 = validFaceVertWeightArr[:,1:2] # (N,1)
        validFaceVertWeightArr_2 = validFaceVertWeightArr[:,2:3] # (N,1)

        # multiply batcentric weights with target vector list, each (N,3)
        validFaceVertTargetArr_0 *= validFaceVertWeightArr_0
        validFaceVertTargetArr_1 *= validFaceVertWeightArr_1
        validFaceVertTargetArr_2 *= validFaceVertWeightArr_2

        # get the fused face Target (N,3)
        validFaceTargetArr = validFaceVertTargetArr_0+validFaceVertTargetArr_1+validFaceVertTargetArr_2 # (N,3)

        return validFaceTargetArr

    def renderSmplSem(self):

        # init. args.
        smplSem = np.ones((self.h, self.w, 3)) # better change it to zeros
        smplSemKeepRatio = np.ones((self.h, self.w, 3)) # better change it to zeros

        # update render's {v, f, vc}, render smplSem
        self.rn.set(v=self.vertsSmpl, f=self.facesSmpl, vc=self.smpl_vtx_std)
        fgPixels = self.rn.visibility_image!=self.constBackground
        smplSem[fgPixels] = self.rn.r[fgPixels] # (H,W,3)

        # update render's {v, f, vc}, render smplSemKeepRatio
        self.rn.set(v=self.vertsSmpl, f=self.facesSmpl, vc=self.smpl_vtx_std_keepRatio)
        fgPixels = self.rn.visibility_image!=self.constBackground
        smplSemKeepRatio[fgPixels] = self.rn.r[fgPixels] # (H,W,3)

        return smplSem, smplSemKeepRatio

    def renderMeshSem(self):

        # rendering mesh semantic segmentation mask by barcentric sampling
        meshSemArr = self.barcentricSampling(samplingVertTargetArr=self.mesh_vertex_label) # (N,3)
        meshSemKeepRatioArr = self.barcentricSampling(samplingVertTargetArr=self.mesh_vertex_label_keepRatio) # (N,3)

        # clip into [0,1]
        _ = np.clip(         a=meshSemArr, a_min=0, a_max=1,          out=meshSemArr)
        _ = np.clip(a=meshSemKeepRatioArr, a_min=0, a_max=1, out=meshSemKeepRatioArr)

        # assign semantic labels to the semantic segmentation mask (H,W,3), invalid regions take [1,1,1] white color
        meshSem = np.ones((self.h, self.w, 3), np.float32)
        meshSemKeepRatio = np.ones((self.h, self.w, 3), np.float32)
        meshSem[self.maskImage.astype(np.bool)] = meshSemArr
        meshSemKeepRatio[self.maskImage.astype(np.bool)] = meshSemKeepRatioArr

        return meshSem, meshSemKeepRatio

    def semanticVoxelizationSMPL(self,meshPath,info):

        assert(info in ["FRONT", "RIGHT", "BACK", "LEFT"])
        if info == "FRONT":

            # save the normalized .obj
            smplNew = {'v':self.vertsSmpl, 'f':self.facesSmpl}
            smplPathNew = meshPath.replace("mesh.obj","smpl_normalized.obj")
            save_obj_data_binary(smplNew, smplPathNew) # notice that face's vertex idx should start from +1, not 0

            # voxelize the normalized .obj
            assert(os.path.exists(smplPathNew))
            voxels = voxel_util.voxelize_2(smplPathNew,VOXEL_H,VOXEL_W,VOXELIZER_PATH) # XYZ (128,192,128) voxels (not DHW, but WHD), 1 inside, 0 outside
            if not self.quickDemo: call(["rm", smplPathNew])
            voxels = voxel_util.binary_fill_from_corner_3D(voxels)

            # convert SPML voxels to semantic ones
            print("computing semantic SMPL voxels for %s..." % (smplPathNew))
            smplSemVoxels = voxel_util.calc_vmap_volume(voxels,self.vertsSmpl,VOXEL_H,VOXEL_W,VOXEL_SIZE)
            print("finished computing semantic SMPL voxels")

        else:
            # X <- Z
            # Y <- Y
            # Z <- (-X)
            # C <- C
            smplSemVoxels = np.transpose(self.smplSemVoxels, (2, 1, 0, 3))
            smplSemVoxels = np.flip(smplSemVoxels, axis=2)

        return smplSemVoxels

    def voxelize(self,meshPath,info):

        assert(info in ["FRONT", "RIGHT", "BACK", "LEFT"])
        if info == "FRONT":
            # save the normalized .obj
            meshNew = {'v':self.rn.v, 'f':self.rn.f}
            meshPathNew = meshPath.replace("mesh.obj","mesh_normalized.obj")
            print("saving %s..." % meshPathNew)
            save_obj_data_binary(meshNew, meshPathNew) # notice that face's vertex idx should start from +1, not 0
            print("finished saving normalized mesh .obj!")

            # voxelize the normalized .obj
            assert(os.path.exists(meshPathNew))
            voxels = voxel_util.voxelize_2(meshPathNew,VOXEL_H,VOXEL_W,VOXELIZER_PATH) # XYZ (128,192,128) voxels (not DHW, but WHD), 1 inside, 0 outside
            if not self.quickDemo: call(["rm", meshPathNew])
            voxels = voxel_util.binary_fill_from_corner_3D(voxels)

        else:
            # X <- Z
            # Y <- Y
            # Z <- (-X)
            voxels = np.transpose(self.meshVoxels, (2, 1, 0))
            voxels = np.flip(voxels, axis=2)

        # direct orthographic projection from voxels
        voxelsOrthoProjMask = np.max(voxels, axis=-1) # WH (128,192)
        voxelsOrthoProjMask = np.transpose(voxelsOrthoProjMask) # HW (192,128), 1 means occupied, 0 means background
        assert(  self.f%voxelsOrthoProjMask.shape[0]==0  )

        return voxels, voxelsOrthoProjMask

    def renderNormal(self):

        # rendering normal map by barcentric sampling
        validFaceNormalArr = self.barcentricSampling(samplingVertTargetArr=self.vertsNormal) # (N,3)
        normalizer = np.linalg.norm(validFaceNormalArr, ord=2, axis=1, keepdims=True) # (N, 1)
        validFaceNormalArr /= normalizer

        # assign normal values to the normal map (H,W,3), invalid regions take [0,0,0]
        normalMap = np.zeros((self.h, self.w, 3), np.float32)
        normalMap[self.maskImage.astype(np.bool)] = validFaceNormalArr

        # Z-direction canonization, make sure all(Z) >= 0, which means inwards
        zNegBool = normalMap[:,:,2] < 0
        zPosiBool = normalMap[:,:,2] > 0
        zNegNum = np.sum(zNegBool)
        zPosiNum = np.sum(zPosiBool)
        if zNegNum > zPosiNum: # usually True
            normalMap *= -1
            normalMap[zPosiBool] = np.array([0,0,0])
            normalInvalidMask = (1-self.maskImage).astype(np.bool) + zPosiBool
        else:
            normalMap[zNegBool] = np.array([0,0,0])
            normalInvalidMask = (1-self.maskImage).astype(np.bool) + zNegNum

        # map normal to RGB
        normalRGB = normalMap * (-1) # in the old normal coord: inwards -> outwards
        normalRGB[:,:,1] *= -1. # from the old normal coord to the new one: newY <- oldY*(-1)
        normalRGB[:,:,2] *= -1. # newZ <- oldZ*(-1)
        assert(  np.all(normalRGB[:,:,2]>=0.)  ) # all point outwards in the new normal coord.
        normalRGB = (normalRGB+1.) / 2. # normal -> RGB
        normalRGB[normalInvalidMask] = np.array([1.,1.,1.]) # set invalid region color

        return normalMap, normalRGB

    def renderMask(self):

        # (h,w), values are face-idx within self.f
        visMap = self.rn.visibility_image

        # (h,w,3), barycentric weights for each tri-face
        barycentricMap = self.rn.barycentric_image

        # (h,w): {0,1} maskImage, 1-mask, 0-bg
        maskImage = np.asarray(visMap != self.constBackground, np.uint32).reshape(visMap.shape)

        return visMap, barycentricMap, maskImage

    def renderRGB(self):

        # (h,w,3), values in (0,1)
        return self.rn.r 

    def inverseRotateY(self,points,angle):
        """
        Rotate the points by a specified angle., LEFT hand rotation
        """

        angle = np.radians(angle)
        ry = np.array([ [ np.cos(angle), 0., np.sin(angle)],
                        [            0., 1.,            0.],
                        [-np.sin(angle), 0., np.cos(angle)] ]) # (3,3)
        return np.dot(points, ry) # (N,3)

    def set_lighting(self,initSkyLightLoc=np.array([0.,-2.5,-5.])):
        """
        a mix-lighting of {front-left, front-right, back}
        """

        # Ground: front
        self.rn.vc = LambertianPointLight(f=self.rn.f,
                                          v=self.rn.v,
                                          num_verts=len(self.rn.v),
                                          light_pos=np.array([0,2.5,-5]),
                                          vc=self.vertsColor,
                                          light_color=np.array(self.colors['white']))

        # Sky: front-left
        if self.skyLightRotDegrees[0] != None:
            self.rn.vc += LambertianPointLight(f=self.rn.f,
                                               v=self.rn.v,
                                               num_verts=len(self.rn.v),
                                               light_pos=self.inverseRotateY(initSkyLightLoc,self.skyLightRotDegrees[0]),
                                               vc=self.vertsColor,
                                               light_color=np.array(self.colors['white']))

        # Sky: front-right
        if self.skyLightRotDegrees[1] != None:
            self.rn.vc += LambertianPointLight(f=self.rn.f,
                                               v=self.rn.v,
                                               num_verts=len(self.rn.v),
                                               light_pos=self.inverseRotateY(initSkyLightLoc,self.skyLightRotDegrees[1]),
                                               vc=self.vertsColor,
                                               light_color=np.array(self.colors['white']))

        # Sky: back
        self.rn.vc += LambertianPointLight(f=self.rn.f,
                                           v=self.rn.v,
                                           num_verts=len(self.rn.v),
                                           light_pos=self.inverseRotateY(initSkyLightLoc,self.skyLightRotDegrees[2]),
                                           vc=self.vertsColor,
                                           light_color=np.array(self.colors['white']))

    def voxelization_normalization(self,verts,useMean=True,useScaling=True):
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
            scaleArr = np.array([self.threshWD/abs(xyzMin[0]), self.threshH/abs(xyzMin[1]), self.threshWD/abs(xyzMin[2]), self.threshWD/xyzMax[0], self.threshH/xyzMax[1], self.threshWD/xyzMax[2]])
            scaleMin = np.min(scaleArr)
            vertsVoxelNorm *= scaleMin

        return vertsVoxelNorm, vertsMean, scaleMin

    def match_mesh_with_smpl(self,smpl_path,labelMin=0,labelMax=1):
        """
        obtain mesh vertex semantic labels, using SMPL rest shape under different normalizaiton methods
        """

        # load SMPL .obj & build KD-Tree from vertices
        assert(  os.path.exists(smpl_path)  )
        smpl = load_obj_data(smpl_path)
        kd_tree_smpl_v = scipy.spatial.KDTree(smpl['v']) # create KD-Tree from 6890 SMPL vertices

        # KNN searching from mesh-vertex to SMPL-vertex
        dist_list, id_list = kd_tree_smpl_v.query(self.vertsRaw, k=KNN_K) # (N,k), (N,k)

        # compute semantic label for each mesh vertex
        weight_list                 = np.exp(-np.square(dist_list)/SIGMA_SQUARE) # (N,k)
        weight_sum                  = np.zeros((weight_list.shape[0], 1)) # (N,1)
        mesh_vertex_label           = np.zeros((weight_list.shape[0], 3)) # (N,3)
        mesh_vertex_label_keepRatio = np.zeros((weight_list.shape[0], 3)) # (N,3)
        for ni in range(KNN_K):
            weight_sum[:, 0]            += weight_list[:, ni]
            mesh_vertex_label           += weight_list[:, ni:(ni+1)] *           self.smpl_vtx_std[id_list[:, ni], :]
            mesh_vertex_label_keepRatio += weight_list[:, ni:(ni+1)] * self.smpl_vtx_std_keepRatio[id_list[:, ni], :]
        mesh_vertex_label /= weight_sum
        mesh_vertex_label_keepRatio /= weight_sum

        # clip into [0,1]
        _ = np.clip(          a=mesh_vertex_label, a_min=labelMin, a_max=labelMax,           out=mesh_vertex_label)
        _ = np.clip(a=mesh_vertex_label_keepRatio, a_min=labelMin, a_max=labelMax, out=mesh_vertex_label_keepRatio)

        # sanity check
        assert(            np.all(mesh_vertex_label>=labelMin) and           np.all(mesh_vertex_label<=labelMax) )
        assert(  np.all(mesh_vertex_label_keepRatio>=labelMin) and np.all(mesh_vertex_label_keepRatio<=labelMax) )

        # get verts and faces of the registered SMPL
        vertsSmpl = smpl['v'] # (smpl['v'] - self.meshNormMean) * self.meshNormScale
        facesSmpl = smpl['f']

        return mesh_vertex_label, mesh_vertex_label_keepRatio, vertsSmpl, facesSmpl

    def parse_smpl_params(self,smpl_params_path):

        # loading code for smpl_params.txt
        assert(  os.path.exists(smpl_params_path)  )
        with open(smpl_params_path, 'r') as fp:
            lines = fp.readlines()

            # remove '\r\n'
            lines = [l[:-2] for l in lines]
            
            # shape coe.: m.betas, (10,)
            betas_data = filter(lambda s: len(s)!=0, lines[1].split(' '))
            betas = np.array([float(b) for b in betas_data])
            
            # [R|T] wrt world coord.
            root_mat_data = lines[3].split(' ') + lines[4].split(' ') +\
                            lines[5].split(' ') + lines[6].split(' ')
            root_mat_data = filter(lambda s: len(s)!=0, root_mat_data)
            root_mat = np.reshape(np.array([float(m) for m in root_mat_data]), (4, 4))
            root_rot = root_mat[:3, :3]
            root_trans = root_mat[:3, 3]

            # pose coe.: m.betas, (72,)
            theta_data = lines[8:80]
            theta = np.array([float(t) for t in theta_data])

        # Load SMPL model (here we load the female model)
        assert(self.quickDemo or ("_F/" in smpl_params_path) or ("_M/" in smpl_params_path))
        m = load_model( SMPL_FEMALE_PKL_PATH ) if (self.quickDemo or ("_F/" in smpl_params_path)) else load_model( SMPL_MALE_PKL_PATH )

        # Apply shape & pose parameters
        m.pose[:] = theta
        m.betas[:] = betas

        # apply [R|T] to get verts' and joints' XYZ locations
        vertsFromSmplParams = np.matmul(m.r, root_rot.transpose()) + np.reshape(root_trans, (1, -1)) # (6890,3)
        jointsFromSmplParams = np.matmul(m.J_transformed.r, root_rot.transpose()) + np.reshape(root_trans, (1, -1)) # (24,3)

        # normalize the verts and joints based on how we normalize the mesh: zeroMean & scaleing factor
        # vertsFromSmplParams = (vertsFromSmplParams - self.meshNormMean) * self.meshNormScale
        # jointsFromSmplParams = (jointsFromSmplParams - self.meshNormMean) * self.meshNormScale

        return vertsFromSmplParams, jointsFromSmplParams

    def generate_random_rot_matrix(self):

        if not self.loadedConfig:

            # init. vars.
            randomRot = np.zeros((3,3), np.float32) # (3,3)

            # rot around x (to right), > 0, lean forward wrt cam.
            RxDegree = 1.*np.random.randint(low=-15, high=15+1)
            Rx = cv.Rodrigues(np.array([[1],[0],[0]],dtype=float)*(RxDegree/180.*np.pi))[0] # (3,3)

            # rot around y (to bottom), > 0, turn left wrt cam.
            RyDegree = 1.*np.random.randint(low=-45, high=45+1)
            Ry = cv.Rodrigues(np.array([[0],[1],[0]],dtype=float)*(RyDegree/180.*np.pi))[0] # (3,3)

            # rot around z (to inside), > 0, tilt right wrt cam.
            RzDegree = 1.*np.random.randint(low=-15, high=15+1)
            Rz = cv.Rodrigues(np.array([[0],[0],[1]],dtype=float)*(RzDegree/180.*np.pi))[0] # (3,3)

            # the joint rot matrix
            randomRot = np.dot(Rz,np.dot(Ry,Rx)) # (3,3)

        else:

            # use the loaded random rotation matrix
            randomRot = np.array(self.loadedConfig["randomRot"], np.float64) # (3,3)

        return randomRot, np.transpose(randomRot)

    def set_v_f_vc_bgcolor(self,meshPath):

        # load the UnNormed-mesh, if this is the first time read this mesh
        assert(os.path.exists(meshPath))
        if self.newMeshToRead:

            # read mesh.obj
            mesh = load_obj_data(meshPath)

            # parse v, f, vc, vn
            self.vertsRaw = mesh['v'] # (N, 3)
            self.faces = mesh['f'] # (N, 3)
            self.vertsColor = mesh['vc'] # (N, 3)
            self.vertsNormalSourcePose = mesh['vn'] # (N, 3)

        # register the UnNormed-mesh with the UnNormed-SMPL, if this is the first time read this UnNormed-mesh
        if self.newMeshToRead:

            # obtain mesh vertex semantic labels, using SMPL rest shape under different normalizaiton methods
            self.mesh_vertex_label, self.mesh_vertex_label_keepRatio, self.vertsSmplSourcePose, self.facesSmpl = self.match_mesh_with_smpl(smpl_path=meshPath.replace("mesh.obj","smpl.obj"))

            # parse 24 3D-skeleton coords. of the SMPL model
            self.vertsFromSmplParamsSourcePose, self.jointsFromSmplParamsSourcePose = self.parse_smpl_params(smpl_params_path=meshPath.replace("mesh.obj", "smpl_params.txt"))

        # for each time of rendering
        # 1) apply random rotation to     {mesh-verts, mesh-normal, smpl-register-verts, smpl-param-3dJoints}
        # 2) voxel-based normalization to {mesh-verts,              smpl-register-verts, smpl-param-3dJoints}, normalize the mesh into H [-0.5,0.5]*(1-margin), W/D [-0.333,0.333]*(1-margin)
        # 3) reset                        {mesh-verts, mesh-normal, smpl-register-verts, smpl-param-3dJoints}
        vertsZeroMean, self.meshNormMean, _ = self.voxelization_normalization(self.vertsRaw,useScaling=False) # we want to determine scaling factor, after applying Rot jittering so that the mesh fits better into WHD
        self.randomRot, randomRotTrans = self.generate_random_rot_matrix()
        self.verts, _, self.meshNormScale = self.voxelization_normalization(np.dot(vertsZeroMean,randomRotTrans),useMean=False)
        self.vertsNormal          = copy.deepcopy(np.dot(self.vertsNormalSourcePose,randomRotTrans))
        self.vertsSmpl            = copy.deepcopy(np.dot(self.vertsSmplSourcePose-self.meshNormMean,randomRotTrans)*self.meshNormScale)
        self.jointsFromSmplParams = copy.deepcopy(np.dot(self.jointsFromSmplParamsSourcePose-self.meshNormMean,randomRotTrans)*self.meshNormScale)

        # save_obj_data_binary({"v":self.vertsRaw, "f":self.faces, "vc":self.vertsColor}, "./examplesRendering/debug_000.obj")
        # save_obj_data_binary({"v":vertsZeroMean, "f":self.faces, "vc":self.vertsColor}, "./examplesRendering/debug_111-0.obj")
        # save_obj_data_binary({"v":np.dot(vertsZeroMean,randomRotTrans), "f":self.faces, "vc":self.vertsColor}, "./examplesRendering/debug_111-1.obj")
        # save_obj_data_binary({"v":self.verts, "f":self.faces, "vc":self.vertsColor}, "./examplesRendering/debug_222.obj")
        # save_obj_data_binary({"v":self.vertsSmplSourcePose, "f":self.facesSmpl}, "./examplesRendering/debug_333.obj")
        # save_obj_data_binary({"v":self.vertsSmplSourcePose-self.meshNormMean, "f":self.facesSmpl}, "./examplesRendering/debug_444-0.obj")
        # save_obj_data_binary({"v":np.dot(self.vertsSmplSourcePose-self.meshNormMean,randomRotTrans), "f":self.facesSmpl}, "./examplesRendering/debug_444-1.obj")
        # save_obj_data_binary({"v":self.vertsSmpl, "f":self.facesSmpl}, "./examplesRendering/debug_555.obj")
        # save_obj_data_binary({"v":self.jointsFromSmplParamsSourcePose}, "./examplesRendering/debug_666.obj")
        # save_obj_data_binary({"v":self.jointsFromSmplParamsSourcePose-self.meshNormMean}, "./examplesRendering/debug_777-0.obj")
        # save_obj_data_binary({"v":np.dot(self.jointsFromSmplParamsSourcePose-self.meshNormMean,randomRotTrans)}, "./examplesRendering/debug_777-1.obj")
        # save_obj_data_binary({"v":self.jointsFromSmplParams}, "./examplesRendering/debug_888.obj")
        # save_obj_data_binary({"v":self.vertsRaw, "f":self.faces, "vc":self.vertsNormalSourcePose}, "./examplesRendering/debug_9990.obj")
        # save_obj_data_binary({"v":self.vertsRaw, "f":self.faces, "vc":self.vertsNormal}, "./examplesRendering/debug_9991.obj")
        # save_obj_data_binary({"v":self.verts, "f":self.faces, "vc":self.vertsNormalSourcePose}, "./examplesRendering/debug_9992.obj")
        # save_obj_data_binary({"v":self.verts, "f":self.faces, "vc":self.vertsNormal}, "./examplesRendering/debug_9993.obj")
        # print("need to check the (un)roted mesh and smpl by saving into obj files...")
        # pdb.set_trace()

        # colorField = copy.deepcopy(self.vertsSmpl)
        # for i in range(colorField.shape[0]):
        #     if colorField[i,0] < 0:
        #         if colorField[i,1] < 0:
        #             colorField[i] = np.array([0,0,1]) # blue
        #         else:
        #             colorField[i] = np.array([0,1,0]) # blue
        #     else:
        #         if colorField[i,1] < 0:
        #             colorField[i] = np.array([1,0,0]) # red
        #         else:
        #             colorField[i] = np.array([0,0,0]) # black
        # save_obj_data_binary({"v":self.vertsSmpl, "f":self.facesSmpl, "vc":colorField}, "./examplesRendering/smpl_iuv.obj")
        # iuvSmplArr = np.load("./im2smpl/smplify_public/code/models/uv_table_smpl.npy") # (6890,2), values in (0,1)
        # iuvSmplSize = 512
        # iuvSmplArr = np.round(iuvSmplArr*(iuvSmplSize-1))
        # assert(  np.all(0<=iuvSmplArr) and np.all(iuvSmplArr<iuvSmplSize)  )
        # iuvSmplMap = np.ones((iuvSmplSize,iuvSmplSize,3))
        # for ptIdx in range(iuvSmplArr.shape[0]):
        #     # get point UV coord
        #     ptU = int(iuvSmplArr[ptIdx,0])
        #     ptV = int(iuvSmplArr[ptIdx,1])
        #     # draw the marker
        #     markerSize = 1
        #     for rowOff in range(-markerSize,markerSize+1):
        #         for colOff in range(-markerSize,markerSize+1):
        #             rowIdx = (iuvSmplSize-1-ptV) + rowOff
        #             colIdx = ptU + colOff
        #             if (0 <= rowIdx < iuvSmplSize) and (0 <= colIdx < iuvSmplSize):
        #                 iuvSmplMap[rowIdx,colIdx] = colorField[ptIdx]
        # cv.imwrite("./examplesRendering/smpl_iuv.jpg", (iuvSmplMap*255).astype(np.uint8)[:,:,::-1])
        # print("!!! save the colored SMPL, voxel normalized and IUV map, for consistency check...")
        # pdb.set_trace()

    def visualize_image(self,img):
        
        # pre. processing
        if len(img.shape) == 2:
            img = img[:,:,None].astype(np.float32)
            img = np.concatenate((img, img, img), axis=2)
        assert(  len(img.shape)==3 )
        assert(  np.max(img)<=1 )

        plt.ion()
        plt.imshow(img)
        plt.draw()

    def load_image(self,imgPath):

        # load background image
        assert(os.path.exists(imgPath))
        img = cv.cvtColor(cv.imread(imgPath), cv.COLOR_BGR2RGB)
        img = np.float32(img)/255.0

        # resize and crop
        img = cv.resize(img, (self.h, self.h))
        edg = (self.h - self.w) // 2
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img[:, edg:-edg, :] if edg > 0 else img # crop along the width of this image
        assert(img.shape[0] == self.h)
        assert(img.shape[1] == self.w)

        return img

    def init_cam(self,w=128*2,h=192*2,rt=np.zeros(3),t=np.array([0.,0.,0.]),f=5000.,c=None):

        # init. args.
        c = np.array([w, h]) / 2. if c is None else c

        # init. cam.
        cam = ProjectPointsOrthogonal(rt=rt, t=t, f=np.array([f,f]), c=c, k=np.zeros(5))
        
        return cam

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--quickDemo', action='store_true', help='If enabled, do a quick demo of rendering a specified mesh')

    parser.add_argument('--w', type=int, default="256", help="128*2")
    parser.add_argument('--h', type=int, default="384", help="192*2")
    parser.add_argument('--f', type=int, default="384", help="better be 192*S, so that voxel projection can be aligned with ortho. renderings by scaling factor S")
    parser.add_argument('--near', type=float, default="0.5")
    parser.add_argument('--far', type=float, default="25")
    parser.add_argument('--resolutionScale', type=int, default="4")

    parser.add_argument('--meshDirSearch', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data")
    parser.add_argument('--bgDirSearch', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data")
    parser.add_argument('--saveDir', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data/humanRender")
    
    parser.add_argument('--useConfig', action='store_true', help='If enabled, load the pre-saved config.json for each mesh')
    parser.add_argument('--addionalType', type=str, default="smplSemVoxels", help="now supports: smplSemVoxels")

    parser.add_argument('--splitNum', type=int, default="30", help="for multi-process running")
    parser.add_argument('--splitIdx', type=int, default="0", help="{0, ..., splitNum-1}")

    parser.add_argument('--eachMeshWithBgNum', type=int, default="4", help="each mesh will be rendered with several background images")

    args = parser.parse_args()

    return args

def read_mesh_paths(meshDirSearch):
    """
    read a list of meshe paths, N (6795)
    """

    print("reading mesh paths...")
    personClothesDirs = glob.glob("%s/DeepHumanDataset/dataset/results_*" % (meshDirSearch)) # 202

    meshPaths = [] # 6795
    for dirEach in personClothesDirs:
        meshPathsTmp = glob.glob("%s/*" % (dirEach))
        meshPathsTmp = [p+"/mesh.obj" for p in meshPathsTmp]
        meshPaths.extend(meshPathsTmp)
    meshNum = len(meshPaths) # 6795

    return meshPaths, meshNum

def read_background_dict(bgDirSearch):
    """
    read a dict. of background image paths
    """

    backgroundDict = {}
    for nameEach in BACKGROUND_NAMES:
        print("reading background image paths of %s..." % (nameEach))
        jpgPaths = glob.glob("%s/LSUN/train_jpg/%s/*.jpg" % (bgDirSearch, nameEach))
        backgroundDict[nameEach] = jpgPaths

    return backgroundDict

def compute_split_range(dataNum, splitNum, splitIdx):
    """
    determine split range, for multi-process running
    """

    splitLen = int(np.ceil(1.*dataNum/splitNum))
    splitRange = [splitIdx*splitLen, min((splitIdx+1)*splitLen, dataNum)]

    return splitRange

def loaded_config_consistency_check(dataPrefix,meshPath):

    # load the config
    pathConfig = "%s/config/%06d.json" % (args.saveDir, dataPrefix)
    assert(os.path.exists(pathConfig))
    with open(pathConfig, 'r') as infile:
        loadedConfig = json.load(infile)

    # dataPrefix
    assert(loadedConfig["dataPrefix"] == dataPrefix)

    # meshPath
    assert(loadedConfig["meshPath"] == meshPath)

    # args
    args_loaded = loadedConfig['args']
    args_run = vars(args)
    for eachKey in args_loaded.keys():
        if eachKey in ["useConfig", "addionalType"]: continue
        assert(args_loaded[eachKey] == args_run[eachKey])

    return loadedConfig

def main(args):
    """
    total data number: N*M'*4 = 6795*4*4 = 108720
    dataset size (if use resolutionScale-4):  (4.1+9.4) (MB) * 108720. / 1024. / 1024.= 1.4 (T)
    estimated time (if with 30 processes in parallel): each process ~ 27+10 h(s)
    """

    # init. render
    render = render_mesh(w=args.w*args.resolutionScale,h=args.h*args.resolutionScale,f=args.f*args.resolutionScale,near=args.near,far=args.far,saveDir=args.saveDir)

    # read a list of meshe paths, N (6795)
    meshPaths, meshNum = read_mesh_paths(meshDirSearch=args.meshDirSearch)

    # read a list of background image paths, M
    backgroundDict = read_background_dict(bgDirSearch=args.bgDirSearch)

    # determine split range, for multi-process running
    splitRange = compute_split_range(dataNum=meshNum,splitNum=args.splitNum,splitIdx=args.splitIdx)

    # for each mesh
    t_used = 0.
    for meshIdx in range(meshNum):

        # split range check
        if not (splitRange[0] <= meshIdx < splitRange[1]): continue
        t_start = time.time()

        # select a subset of M' (maybe 4) background types
        selectedBgNamesIdx = np.random.choice(len(BACKGROUND_NAMES), args.eachMeshWithBgNum, replace=False)

        # for each selected background type
        for nameIdx in range(args.eachMeshWithBgNum):

            # randomly select an image from this background type
            sceneName = BACKGROUND_NAMES[selectedBgNamesIdx[nameIdx]]
            bgIdx = np.random.choice(len(backgroundDict[sceneName]), 1, replace=False)

            # call the render function, and save 4 views
            dataPrefix = int(meshIdx*args.eachMeshWithBgNum*NUM_VIEW_RENDERING + nameIdx*NUM_VIEW_RENDERING)
            if args.useConfig:
                loadedConfig = loaded_config_consistency_check(dataPrefix=dataPrefix,meshPath=meshPaths[meshIdx])
                render.render_colored_mesh_with_background_use_config(quickDemo=args.quickDemo,loadedConfig=loadedConfig,addionalType=args.addionalType)
            else:
                render.render_colored_mesh_with_background(meshPath=meshPaths[meshIdx],bgImgPath=backgroundDict[sceneName][bgIdx[0]],quickDemo=args.quickDemo,dataPrefix=dataPrefix,eachMeshWithBgNum=args.eachMeshWithBgNum)

        # log
        t_used += (time.time() - t_start) # sec.
        t_remains = (splitRange[1]-meshIdx-1) * (t_used/(meshIdx-splitRange[0]+1)) / 3600. # remaining hours
        print("\n\n###############################################################################")
        print("Split %d/%d | rendering mesh-%d within [%d, %d) | remains %.3f h(s) ..." % (args.splitIdx, args.splitNum, meshIdx, splitRange[0], splitRange[1], t_remains))
        print("###############################################################################\n\n")

def quick_rendering_demo(args):

    # init. render
    render = render_mesh(w=args.w*args.resolutionScale,h=args.h*args.resolutionScale,f=args.f*args.resolutionScale,near=args.near,far=args.far)

    # set mesh and background image paths
    meshPath = "./examplesRendering/mesh.obj" # results_gyx_20181011_zyq_1_F/10198/mesh.obj
    bgImgPath = "./examplesRendering/background.jpg" # from internet

    # start rendering
    assert( args.quickDemo==True )
    render.render_colored_mesh_with_background(meshPath=meshPath,bgImgPath=bgImgPath,quickDemo=True,dataPrefix="demo")

if __name__ == '__main__':

    # parse args.
    args = parse_args()

    if args.quickDemo:
        quick_rendering_demo(args=args)
    else:
        main(args=args)




