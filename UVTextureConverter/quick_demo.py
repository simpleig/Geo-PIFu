# pip install pathlib
# pip install tqdm

# IUV parsing demo
if 0:
    from UVTextureConverter import UVConverter
    from UVTextureConverter import Normal2Atlas
    from UVTextureConverter import Atlas2Normal
    from PIL import Image
    import numpy as np
    from matplotlib import pyplot as plt
    try:
        %matplotlib inline
    except Exception as e:
        pass
    import pdb # pdb.set_trace()

    im = Image.open("./input/human.jpg")
    plt.imshow(im)
    pdb.set_trace()

    iuv = Image.open("./input/human_IUV.jpg")
    plt.imshow(iuv)
    pdb.set_trace()

    parts_size = 200
    tex_trans, mask_trans = UVConverter.create_texture("./input/human.jpg", "./input/human_IUV.jpg", parts_size=parts_size, concat=False)
    tex = UVConverter.concat_atlas_tex(tex_trans)  # 800 x 1200 x 3
    mask = UVConverter.concat_atlas_tex(mask_trans)  # 800 x 1200
    plt.imshow(tex)
    pdb.set_trace()
    plt.imshow(mask)
    pdb.set_trace()

    converter = Atlas2Normal(atlas_size=parts_size, normal_size=512)
    normal_tex, normal_ex = converter.convert((tex_trans*255).astype('int'), mask=mask_trans)
    plt.imshow(normal_tex)
    pdb.set_trace()
    plt.imshow(normal_ex)
    pdb.set_trace()

    converter = Normal2Atlas(atlas_size=parts_size, normal_size=512)
    atlas_tex, atlas_ex = converter.convert((normal_tex*255).astype('int'), normal_ex)
    plt.imshow(UVConverter.concat_atlas_tex(atlas_tex))
    pdb.set_trace()
    plt.imshow(UVConverter.concat_atlas_tex(atlas_ex))
    pdb.set_trace()

# normal & altas texture maps convertion demo
if 0:
    from UVTextureConverter import Normal2Atlas
    from UVTextureConverter import Atlas2Normal
    from PIL import Image
    import numpy as np
    from matplotlib import pyplot as plt
    try:
        %matplotlib inline
    except Exception as e:
        pass
    import pdb # pdb.set_trace()

    normal_tex = np.array(Image.open("./input/normal.jpg")) # (512, 512, 3)
    plt.imshow(normal_tex)
    pdb.set_trace()

    converter = Normal2Atlas(normal_size=512, atlas_size=200)
    atlas_texture = converter.convert(normal_tex)
    plt.imshow(Normal2Atlas.concat_atlas_tex(atlas_texture))
    pdb.set_trace()

    im = np.array(Image.open("./input/atlas.png").convert('RGB')).transpose(1, 0, 2)
    plt.imshow(im)
    pdb.set_trace()

    atlas_tex_stack = Atlas2Normal.split_atlas_tex(im)
    converter = Atlas2Normal(atlas_size=200, normal_size=512)
    normal_tex = converter.convert(atlas_tex_stack)
    plt.imshow(normal_tex)
    pdb.set_trace()

# py3 -> py2 pickle files convertion
if 0:
    import pickle

    path = "/Users/tohe/Downloads/UVTextureConverter-master/UVTextureConverter/mapping_relations/normal2atlas_512_200.pickle"
    with open(path, mode='rb') as f:
        aaa = pickle.load(f)
    with open(path.replace(".pickle","_py2.pickle"), mode="wb") as f:
        pickle.dump(aaa, f, protocol=2)

    path = "/Users/tohe/Downloads/UVTextureConverter-master/UVTextureConverter/mapping_relations/atlas2normal_200_512.pickle"
    with open(path, mode='rb') as f:
        aaa = pickle.load(f)
    with open(path.replace(".pickle","_py2.pickle"), mode="wb") as f:
        pickle.dump(aaa, f, protocol=2)

    path = "/Users/tohe/Downloads/UVTextureConverter-master/UVTextureConverter/config/normal_faces.pickle"
    with open(path, mode='rb') as f:
        aaa = pickle.load(f)
    with open(path.replace(".pickle","_py2.pickle"), mode="wb") as f:
        pickle.dump(aaa, f, protocol=2)

    path = "/Users/tohe/Downloads/UVTextureConverter-master/UVTextureConverter/config/normal.pickle"
    with open(path, mode='rb') as f:
        aaa = pickle.load(f)
    with open(path.replace(".pickle","_py2.pickle"), mode="wb") as f:
        pickle.dump(aaa, f, protocol=2)