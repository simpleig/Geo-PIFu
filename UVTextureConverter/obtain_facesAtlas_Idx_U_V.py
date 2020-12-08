# pip install pathlib
# pip install tqdm

from UVTextureConverter import Normal2Atlas
from UVTextureConverter import Atlas2Normal
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# try:
#     %matplotlib inline
# except Exception as e:
#     pass
import pdb # pdb.set_trace()

converter = Normal2Atlas(normal_size=512, atlas_size=200)
facesAtlas_Idx_U_V = converter.obtain_facesAtlas_Idx_U_V(smpl_obj_path="./input/smpl_iuv.obj") # (numFace, 1+6)
np.save('./input/facesAtlas_Idx_U_V.npy', facesAtlas_Idx_U_V)