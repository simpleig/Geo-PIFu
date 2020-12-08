import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .UVConverter import UVConverter
import os
import pdb # pdb.set_trace()

class Normal2Atlas(UVConverter):
    def __init__(self, normal_size=512, atlas_size=200):
        # super().__init__()
        super(Normal2Atlas, self).__init__()
        self.normal_size = normal_size
        self.atlas_size = atlas_size
        self.atlas_tex = None
        self.atlas_ex = None
        self.file_path = Path(__file__).parent.resolve()
        self.normal2atlas_pickle_path = "%s/mapping_relations/normal2atlas_%d_%d_py2.pickle" % (self.file_path, normal_size, atlas_size)

        if os.path.exists(self.normal2atlas_pickle_path):
            with open(self.normal2atlas_pickle_path, mode='rb') as f:
                self.mapping_relation = pickle.load(f)
        else:
            self.mapping_relation = []
        # self.mapping_relation = []

    def obtain_facesAtlas_Idx_U_V(self, smpl_obj_path):
        """
        for each SMPL tri-face, we need to find its Atlas-{Idx,U,V}: (numFace,1) of Idx, and (numFace,3,2) of (U,V), which can directly be loaded later

        return:
            facesAtlas_Idx_U_V, (numFace, 1+6)
                k -> atlas_a[0] # atlas id indeed, within {1,...,24}
                k -> atlas_a,
                     atlas_b,
                     atlas_c
        """

        # load smpl
        assert(os.path.exists(smpl_obj_path))
        smpl = load_obj_data(smpl_obj_path)

        # init args.
        faceNum = smpl["f"].shape[0]
        facesAtlas_Idx_U_V = np.zeros((faceNum,1+6), np.float32)

        # get IUV for each smpl face
        for k in tqdm(range(faceNum)):

            #----- get 3-vert idx of a normal face -----
            face_vertex = smpl["f"][k]

            #----- get 3-vert atlas Idx & UV -----
            min_index = [0, 0, 0]
            flag = False
            for a, lst_a in enumerate(self.atlas_hash[face_vertex[0]]):
                for b, lst_b in enumerate(self.atlas_hash[face_vertex[1]]):
                    for c, lst_c in enumerate(self.atlas_hash[face_vertex[2]]):
                        if lst_a[0] == lst_b[0] and lst_b[0] == lst_c[0]: # three verts within the same atlas
                            min_index = [a, b, c]
                            flag = True
            if not flag:
                continue
            atlas_a = self.atlas_hash[face_vertex[0]][min_index[0]]  # vertex, normal UV loc get
            atlas_b = self.atlas_hash[face_vertex[1]][min_index[1]]
            atlas_c = self.atlas_hash[face_vertex[2]][min_index[2]]
            face_id = atlas_a[0]
            assert(1 <= face_id <= 24)

            facesAtlas_Idx_U_V[k][0] = face_id
            facesAtlas_Idx_U_V[k][1] = atlas_a[1]; facesAtlas_Idx_U_V[k][2] = atlas_a[2]
            facesAtlas_Idx_U_V[k][3] = atlas_b[1]; facesAtlas_Idx_U_V[k][4] = atlas_b[2]
            facesAtlas_Idx_U_V[k][5] = atlas_c[1]; facesAtlas_Idx_U_V[k][6] = atlas_c[2]

        return facesAtlas_Idx_U_V

        # # first need a sanity for SMPL face format
        # print("first need a sanity for SMPL face format...")
        # pdb.set_trace()
        # smpl = load_obj_data("/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv.obj")

        # smplFaceVertsListOfSet = []
        # for i in range(smpl["f"].shape[0]):
        #     vertsSetTmp = {smpl["f"][i][0], smpl["f"][i][1], smpl["f"][i][2]}
        #     assert(len(vertsSetTmp) == 3)
        #     assert(vertsSetTmp not in smplFaceVertsListOfSet)
        #     smplFaceVertsListOfSet.append(vertsSetTmp)
        # assert(len(smplFaceVertsListOfSet) == smpl["f"].shape[0])
        # convertFaceVertsListOfSet = []
        # for i in range(len(self.normal_faces)):
        #     vertsSetTmp = {self.normal_faces[i][0], self.normal_faces[i][1], self.normal_faces[i][2]}
        #     assert(len(vertsSetTmp) == 3)
        #     assert(vertsSetTmp not in convertFaceVertsListOfSet)
        #     convertFaceVertsListOfSet.append(vertsSetTmp)
        # assert(len(convertFaceVertsListOfSet) == len(self.normal_faces))
        # assert(len(convertFaceVertsListOfSet) == len(smplFaceVertsListOfSet))
        # jointVertsSet = []
        # import copy
        # smplFaceVertsListOfSetIterator = copy.deepcopy(smplFaceVertsListOfSet) 
        # for vertsSetEach in smplFaceVertsListOfSetIterator:
        #     if vertsSetEach in convertFaceVertsListOfSet:
        #         jointVertsSet.append(vertsSetEach)
        #         smplFaceVertsListOfSet.remove(vertsSetEach)
        #         convertFaceVertsListOfSet.remove(vertsSetEach)
        # jointVertsSetArr = np.zeros((len(jointVertsSet),3), np.int)
        # for iii in range(len(jointVertsSet)):
        #     print("{}/{}...".format(iii,len(jointVertsSet)))
        #     jointVertsSetArr[iii,0] = list(jointVertsSet[iii])[0]
        #     jointVertsSetArr[iii,1] = list(jointVertsSet[iii])[1]
        #     jointVertsSetArr[iii,2] = list(jointVertsSet[iii])[2]
        # save_obj_data_binary({"v":smpl["v"],"f":jointVertsSetArr,"vc":smpl["vc"]},"/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv_intersection.obj")
        # smplFaceVertsListOfSetArr = np.zeros((len(smplFaceVertsListOfSet),3), np.int)
        # for iii in range(len(smplFaceVertsListOfSet)):
        #     print("{}/{}...".format(iii,len(smplFaceVertsListOfSet)))
        #     smplFaceVertsListOfSetArr[iii,0] = list(smplFaceVertsListOfSet[iii])[0]
        #     smplFaceVertsListOfSetArr[iii,1] = list(smplFaceVertsListOfSet[iii])[1]
        #     smplFaceVertsListOfSetArr[iii,2] = list(smplFaceVertsListOfSet[iii])[2]
        # save_obj_data_binary({"v":smpl["v"],"f":smplFaceVertsListOfSetArr,"vc":smpl["vc"]},"/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv_justSmpl.obj")
        # convertFaceVertsListOfSetArr = np.zeros((len(convertFaceVertsListOfSet),3), np.int)
        # for iii in range(len(convertFaceVertsListOfSet)):
        #     print("{}/{}...".format(iii,len(convertFaceVertsListOfSet)))
        #     convertFaceVertsListOfSetArr[iii,0] = list(convertFaceVertsListOfSet[iii])[0]
        #     convertFaceVertsListOfSetArr[iii,1] = list(convertFaceVertsListOfSet[iii])[1]
        #     convertFaceVertsListOfSetArr[iii,2] = list(convertFaceVertsListOfSet[iii])[2]
        # save_obj_data_binary({"v":smpl["v"],"f":convertFaceVertsListOfSetArr,"vc":smpl["vc"]},"/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv_justConverter.obj")
        # print("joint: {}, smpl: {}, converter: {}...".format(1.*len(jointVertsSet)/smpl["f"].shape[0], 1.*len(smplFaceVertsListOfSet)/smpl["f"].shape[0], 1.*len(convertFaceVertsListOfSet)/smpl["f"].shape[0]))

        # save_obj_data_binary({"v":smpl["v"],"f":np.array(self.normal_faces),"vc":smpl["vc"]},"/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv_converter.obj")
        # colorVert = np.zeros(smpl["v"].shape, np.float64)
        # min_x = np.min(smpl["v"][:, 0])
        # max_x = np.max(smpl["v"][:, 0])
        # min_y = np.min(smpl["v"][:, 1])
        # max_y = np.max(smpl["v"][:, 1])
        # min_z = np.min(smpl["v"][:, 2])
        # max_z = np.max(smpl["v"][:, 2])
        # colorVert[:, 0] = (smpl["v"][:, 0]-min_x)/(max_x-min_x) # [0,1]
        # colorVert[:, 1] = (smpl["v"][:, 1]-min_y)/(max_y-min_y) # [0,1]
        # colorVert[:, 2] = (smpl["v"][:, 2]-min_z)/(max_z-min_z) # [0,1]
        # save_obj_data_binary({"v":smpl["v"],"f":smpl["f"],"vc":colorVert},"/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv_aaa.obj")
        # save_obj_data_binary({"v":smpl["v"],"f":np.array(self.normal_faces),"vc":colorVert},"/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv_bbb.obj")

        # numFaceSmpl = len(smpl['f'])
        # numFaceSmplConverter = len(self.normal_faces)
        # assert(numFaceSmpl==numFaceSmplConverter)
        # for faceIdxSeq in range(numFaceSmpl):
        #     faceVert_0 = smpl['f'][faceIdxSeq][0]
        #     faceVert_1 = smpl['f'][faceIdxSeq][1]
        #     faceVert_2 = smpl['f'][faceIdxSeq][2]
        #     faceVertConverter_0 = self.normal_faces[faceIdxSeq][0]
        #     faceVertConverter_1 = self.normal_faces[faceIdxSeq][1]
        #     faceVertConverter_2 = self.normal_faces[faceIdxSeq][2]
        #     always_true = ((faceVert_0==faceVertConverter_0) and (faceVert_1==faceVertConverter_1) and (faceVert_2==faceVertConverter_2))
        #     if not always_true:
        #         print("find problem!!!")
        #         print(faceVert_0,faceVert_1,faceVert_2,faceVertConverter_0,faceVertConverter_1,faceVertConverter_2)
        #         pdb.set_trace()
        # print("All good!!!")
        # pdb.set_trace()

    def convert(self, normal_tex, mask=None):
        self._mapping(normal_tex, mask)
        if len(self.mapping_relation) == 0:
            # for each face of SMPL, 13776 faces
            # smpl = load_obj_data("/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv.obj")
            for k in tqdm(range(len(self.normal_faces))):

                #----- get 3-vert idx of a normal face -----
                face_vertex = self.normal_faces[k]  # vertex id get
                # face_vertex = smpl["f"][k]

                #----- get 3-vert normal UV -----
                a_vertex, b_vertex, c_vertex = [], [], []
                min_index = [0, 0, 0]
                min_val = -1
                # For edge verts: if any vertex of this face has more than 1 normal UV coords., find the closest 3 UV coords.
                if len(self.normal_hash[face_vertex[0]]) > 1 or len(self.normal_hash[face_vertex[1]]) > 1 or len(self.normal_hash[face_vertex[2]]) > 1:

                    for ind_a, lst_a in enumerate(self.normal_hash[face_vertex[0]]):
                        for ind_b, lst_b in enumerate(self.normal_hash[face_vertex[1]]):
                            for ind_c, lst_c in enumerate(self.normal_hash[face_vertex[2]]):

                                total = np.sqrt(((np.array(lst_a) - np.array(lst_b))**2).sum()) + np.sqrt(((np.array(lst_b) - np.array(lst_c))**2).sum()) + np.sqrt(((np.array(lst_c) - np.array(lst_a))**2).sum())
                                if min_val == -1:
                                    min_val = total
                                elif total < min_val:
                                    min_val = total
                                    min_index = [ind_a, ind_b, ind_c]
                a_vertex = self.normal_hash[face_vertex[0]][min_index[0]] # normal UV of point-a
                b_vertex = self.normal_hash[face_vertex[1]][min_index[1]] # normal UV of point-b
                c_vertex = self.normal_hash[face_vertex[2]][min_index[2]] # normal UV of point-c

                # sanity check
                # always_true = (len(a_vertex) == 2) and (len(b_vertex) == 2) and (len(c_vertex) == 2)
                # if not always_true:
                #     print("how is this possible?")
                #     pdb.set_trace()
                if len(a_vertex) == 0 or len(b_vertex) == 0 or len(c_vertex) == 0:
                    continue

                #----- get 3-vert atlas UV -----
                min_index = [0, 0, 0]
                flag = False
                for a, lst_a in enumerate(self.atlas_hash[face_vertex[0]]):
                    for b, lst_b in enumerate(self.atlas_hash[face_vertex[1]]):
                        for c, lst_c in enumerate(self.atlas_hash[face_vertex[2]]):
                            if lst_a[0] == lst_b[0] and lst_b[0] == lst_c[0]: # three verts within the same atlas
                                min_index = [a, b, c]
                                flag = True
                if not flag:
                    continue
                atlas_a = self.atlas_hash[face_vertex[0]][min_index[0]]  # vertex, normal UV loc get
                atlas_b = self.atlas_hash[face_vertex[1]][min_index[1]]
                atlas_c = self.atlas_hash[face_vertex[2]][min_index[2]]

                #----- for each within face UV of atlas -----
                i_min = int(min([atlas_a[1], atlas_b[1], atlas_c[1]]) * self.atlas_size) # U_min
                i_max = int(max([atlas_a[1], atlas_b[1], atlas_c[1]]) * self.atlas_size) # U_max
                j_min = int(min([atlas_a[2], atlas_b[2], atlas_c[2]]) * self.atlas_size) # V_min
                j_max = int(max([atlas_a[2], atlas_b[2], atlas_c[2]]) * self.atlas_size) # V_max
                face_id = atlas_a[0] # atlas id indeed, within {1,...,24}
                for i in range(self.atlas_size): # row idx
                    if i < i_min or i > i_max:
                        continue
                    for j in range(self.atlas_size): # col idx
                        if j < j_min or j > j_max:
                            continue
                        ex = self.atlas_ex[face_id - 1, i, (self.atlas_size - 1) - j]
                        if ex == 0: # if this point not mapped yet

                            # if this point p is inside the triangle face
                            if self.barycentric_coordinates_exists(np.array(atlas_a[1:]), np.array(atlas_b[1:]), np.array(atlas_c[1:]), np.array([1.*i/(self.atlas_size-1), 1.*j/(self.atlas_size-1)])):
                                
                                a, b, c = self.barycentric_coordinates(np.array(atlas_a[1:]), np.array(atlas_b[1:]), np.array(atlas_c[1:]), np.array([1.*i/(self.atlas_size-1), 1.*j/(self.atlas_size-1)]))
                                
                                # a_vertex, b_vertex, c_vertex = [], [], []
                                # min_index = [0, 0, 0]
                                # min_val = -1
                                # # For edge verts: if any vertex of this face has more than 1 normal UV coords., find the closest 3 UV coords.
                                # if len(self.normal_hash[face_vertex[0]]) > 1 or len(self.normal_hash[face_vertex[1]]) > 1 or len(self.normal_hash[face_vertex[2]]) > 1:

                                #     for ind_a, lst_a in enumerate(self.normal_hash[face_vertex[0]]):
                                #         for ind_b, lst_b in enumerate(self.normal_hash[face_vertex[1]]):
                                #             for ind_c, lst_c in enumerate(self.normal_hash[face_vertex[2]]):

                                #                 total = np.sqrt(((np.array(lst_a) - np.array(lst_b))**2).sum()) + np.sqrt(((np.array(lst_b) - np.array(lst_c))**2).sum()) + np.sqrt(((np.array(lst_c) - np.array(lst_a))**2).sum())
                                #                 if min_val == -1:
                                #                     min_val = total
                                #                 elif total < min_val:
                                #                     min_val = total
                                #                     min_index = [ind_a, ind_b, ind_c]
                                # a_vertex = self.normal_hash[face_vertex[0]][min_index[0]] # normal UV of point-a
                                # b_vertex = self.normal_hash[face_vertex[1]][min_index[1]] # normal UV of point-b
                                # c_vertex = self.normal_hash[face_vertex[2]][min_index[2]] # normal UV of point-c

                                # # sanity check
                                # # always_true = (len(a_vertex) == 2) and (len(b_vertex) == 2) and (len(c_vertex) == 2)
                                # # if not always_true:
                                # #     print("how is this possible?")
                                # #     pdb.set_trace()
                                # if len(a_vertex) == 0 or len(b_vertex) == 0 or len(c_vertex) == 0:
                                #     continue

                                # UV coords. of the normal UV map
                                normal_tex_pos = a * np.array(a_vertex) + b * np.array(b_vertex) + c * np.array(c_vertex)

                                self.mapping_relation.append([i, # row idx of each atlas
                                                              (self.atlas_size - 1) - j, # col idx of each atlas
                                                              face_id - 1, # atlas idx - 1
                                                              (self.normal_size - 1) - int(normal_tex_pos[1] * self.normal_size), # row idx of normal UV map
                                                              int(normal_tex_pos[0] * self.normal_size)]) # col idx of normal UV map

            with open(self.normal2atlas_pickle_path, mode='wb') as f:
                pickle.dump(self.mapping_relation, f)

        painted_atlas_tex = np.copy(self.atlas_tex) # (24,200,200,3)
        painted_atlas_ex = np.copy(self.atlas_ex) # (24,200,200)

        # this is for the Dense Full Texture Mapping, not just the SMPL 6890 vertices
        for relation in self.mapping_relation: # len is 835485

            new_tex = normal_tex[relation[3], relation[4]]
            painted_atlas_tex[relation[2], relation[0], relation[1]] = new_tex / 255.

            if mask is not None:
                painted_atlas_ex[relation[2], relation[0], relation[1]] = mask[relation[3], relation[4]]

        if mask is not None:
            return painted_atlas_tex, painted_atlas_ex
        else:
            return painted_atlas_tex

    def _mapping(self, normal_tex, mask, return_exist_area=False):
        self.atlas_tex, self.atlas_ex = self._mapping_normal_to_atlas(normal_tex, mask)
        if return_exist_area:
            return self.atlas_tex, self.atlas_ex
        else:
            return self.atlas_tex

    def _mapping_to_each_atlas_parts(self, vertex_tex, vertex_mask, parts_num):
        """
        Function to convert normal texture for each part of atlas texture.

        params:
        vertex_tex: Stores the texture of each point of the SMPL model.
        parts_num: Part number. 1 ~ 24.
        """
        tex = np.zeros((self.atlas_size, self.atlas_size, 3)) # (200,200,3)
        tex_ex = np.zeros((self.atlas_size, self.atlas_size)) # (200,200)

        for k, v in self.atlas_hash.items(): # len is 6890, sequential key 0,...,6889

            for t in v:
                if t[0] == parts_num: # within {1,...,24}
                    smpl_pt_u_normalized_within_oneAtlas = t[1]
                    smpl_pt_v_normalized_within_oneAtlas = t[2]
                    each_atlas_texture_row = int(smpl_pt_u_normalized_within_oneAtlas * (self.atlas_size-1))
                    each_atlas_texture_col = (self.atlas_size-1) - int(smpl_pt_v_normalized_within_oneAtlas * (self.atlas_size-1))
                    tex[each_atlas_texture_row, each_atlas_texture_col, :] = vertex_tex[k] # assign RGB of this SMPL vert to an atlas location
                    if vertex_mask is not None:
                        tex_ex[each_atlas_texture_row, each_atlas_texture_col] = vertex_mask[k]
                    else:
                        tex_ex[each_atlas_texture_row, each_atlas_texture_col] = 1

        return tex / 255., tex_ex

    def _mapping_normal_to_atlas(self, normal_tex, mask):
        """
        Function to convert normal texture to atlas texture.

        params:
        normal_tex: before convert, normal texture
        """

        vertex_tex = {} # texture RGB of each SMPL vert, used as a bridge to assign RGB to each atlas
        vertex_mask = None
        if mask is not None:
            vertex_mask = {}
        h, w, _ = normal_tex.shape # (512,512,3)

        for k, v in self.normal_hash.items(): # len is 6890, sequential key 0,...,6889
            # There may be multiple candidates for one vertex, but the texture is assumed to be the same and the first candidate is used.
            smpl_pt_u_normalized = v[0][0]
            smpl_pt_v_normalized = v[0][1]
            smpl_texture_row = int(h - smpl_pt_v_normalized*(h-1))
            smpl_texture_col = int(smpl_pt_u_normalized*(w-1))
            vertex_tex[k] = normal_tex[smpl_texture_row, smpl_texture_col, :] # texture RGB of this SMPL vertex
            if vertex_mask is not None:
                vertex_mask[k] = mask[smpl_texture_row, smpl_texture_col]

        if 0:
            ##########################################################################
            ##########################################################################
            ##########################################################################

            smpl = load_obj_data("/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv.obj")


            iuvSmplSize = 512
            iuvSmplMap = np.ones((iuvSmplSize,iuvSmplSize,3))
            for k, v in self.normal_hash.items(): # len is 6890, sequential key 0,...,6889
                # repV = v[0]
                for repV in v:
                    # get point UV coord
                    ptU = int(round(repV[0]*(iuvSmplSize-1)))
                    ptV = int(round(repV[1]*(iuvSmplSize-1)))
                    # draw the marker
                    markerSize = 1
                    for rowOff in range(-markerSize,markerSize+1):
                        for colOff in range(-markerSize,markerSize+1):
                            rowIdx = (iuvSmplSize-1-ptV) + rowOff
                            colIdx = ptU + colOff
                            if (0 <= rowIdx < iuvSmplSize) and (0 <= colIdx < iuvSmplSize):
                                iuvSmplMap[rowIdx,colIdx] = smpl["vc"][k]
            import cv2 as cv
            cv.imwrite("/Users/tohe/Downloads/UVTextureConverter-master/input/smpl_iuv.jpg", (iuvSmplMap*255).astype(np.uint8)[:,:,::-1])

            print("can do a {SMPL.obj, UV map} visualization check here...")
            pdb.set_trace()

            ##########################################################################
            ##########################################################################
            ##########################################################################

        atlas_texture = np.zeros((24, self.atlas_size, self.atlas_size, 3)) # (24,200,200,3)
        atlas_ex = np.zeros((24, self.atlas_size, self.atlas_size)) # (24,200,200)
        for i in range(24): # for each semantic ID
            atlas_texture[i], atlas_ex[i] = self._mapping_to_each_atlas_parts(vertex_tex, vertex_mask, parts_num=i + 1)

        return atlas_texture, atlas_ex

def load_obj_data(filename):
    """load model data from .obj file"""
    v_list = []  # vertex coordinate
    vt_list = []  # vertex texture coordinate
    vc_list = []  # vertex color
    vn_list = []  # vertex normal
    f_list = []  # face vertex indices
    fn_list = []  # face normal indices
    ft_list = []  # face texture indices

    # read data
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        line_data = line.strip().split(' ')

        # parse vertex cocordinate
        if line_data[0] == 'v':
            v_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))
            if len(line_data) == 7:
                vc_list.append((float(line_data[4]), float(line_data[5]), float(line_data[6])))
            else:
                vc_list.append((0.5, 0.5, 0.5))

        # parse vertex texture coordinate
        if line_data[0] == 'vt':
            vt_list.append((float(line_data[1]), float(line_data[2])))

        # parse vertex normal
        if line_data[0] == 'vn':
            vn_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))

        # parse face
        if line_data[0] == 'f':
            # used for parsing face element data
            def segElementData(ele_str):
                fv = None
                ft = None
                fn = None
                eles = ele_str.strip().split('/')
                if len(eles) == 1:
                    fv = int(eles[0]) - 1
                elif len(eles) == 2:
                    fv = int(eles[0]) - 1
                    ft = int(eles[1]) - 1
                elif len(eles) == 3:
                    fv = int(eles[0]) - 1
                    fn = int(eles[2]) - 1
                    ft = None if eles[1] == '' else int(eles[1]) - 1
                return fv, ft, fn

            fv0, ft0, fn0 = segElementData(line_data[1])
            fv1, ft1, fn1 = segElementData(line_data[2])
            fv2, ft2, fn2 = segElementData(line_data[3])
            f_list.append((fv0, fv1, fv2))
            if ft0 is not None and ft1 is not None and ft2 is not None:
                ft_list.append((ft0, ft1, ft2))
            if fn0 is not None and fn1 is not None and fn2 is not None:
                fn_list.append((fn0, fn1, fn2))

    v_list = np.asarray(v_list)
    vn_list = np.asarray(vn_list)
    vt_list = np.asarray(vt_list)
    vc_list = np.asarray(vc_list)
    f_list = np.asarray(f_list)
    ft_list = np.asarray(ft_list)
    fn_list = np.asarray(fn_list)

    model = {'v': v_list, 'vt': vt_list, 'vc': vc_list, 'vn': vn_list,
             'f': f_list, 'ft': ft_list, 'fn': fn_list}
    return model

def save_obj_data_binary(model, filename):
    assert 'v' in model and model['v'].size != 0

    with open(filename, 'wb') as fp:
        if 'v' in model and model['v'].size != 0:
            if 'vc' in model and model['vc'].size != 0:
                for v, vc in zip(model['v'], model['vc']):
                    fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], vc[0], vc[1], vc[2]))
            else:
                for v in model['v']:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if 'vn' in model and model['vn'].size != 0:
            for vn in model['vn']:
                fp.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))

        if 'vt' in model and model['vt'].size != 0:
            for vt in model['vt']:
                fp.write('vt %f %f\n' % (vt[0], vt[1]))

        if 'f' in model and model['f'].size != 0:
            if 'fn' in model and model['fn'].size != 0 and 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['fn'].size
                assert model['f'].size == model['ft'].size
                for f_, ft_, fn_ in zip(model['f'], model['ft'], model['fn']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                             (f[0], ft[0], fn[0], f[1], ft[1], fn[1], f[2], ft[2], fn[2]))
            elif 'fn' in model and model['fn'].size != 0:
                assert model['f'].size == model['fn'].size
                for f_, fn_ in zip(model['f'], model['fn']):
                    f = np.copy(f_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d//%d %d//%d %d//%d\n' % (f[0], fn[0], f[1], fn[1], f[2], fn[2]))
            elif 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['ft'].size
                for f_, ft_ in zip(model['f'], model['ft']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fp.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
            else:
                for f_ in model['f']:
                    f = np.copy(f_) + 1
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

