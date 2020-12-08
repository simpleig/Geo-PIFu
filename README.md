# Geo-PIFu: Geometry and Pixel Aligned Implicit Functions for Single-view Human Reconstruction

This repository is the official PyTorch implementation of [Geo-PIFu: Geometry and Pixel Aligned Implicit Functions for Single-view Human Reconstruction](https://papers.nips.cc/paper/2020/file/690f44c8c2b7ded579d01abe8fdb6110-Paper.pdf), NeurIPS, 2020.
<p align="center">
<img src="https://github.com/simpleig/Geo-PIFu/blob/master/assests/pipeline.png" width="750">
</p>

If you find this code useful, please consider citing
```
@inproceedings{he2020geopifu,
  title     = {Geo-PIFu: Geometry and Pixel Aligned Implicit Functions for Single-view Human Reconstruction},
  author    = {Tong He and John Collomosse and Hailin Jin and Stefano Soatto},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}
```

Development log
- Environment configuration
  - [x] For `Geo-PIFu` and `PIFu` training, test and evaluation
  - [x] For `DeepHuman` data rendering using `opendr`
- Data preparation
  - [x] DeepHuman and LSUN datasets setup
  - [x] Clothed human mesh rendering and voxelization
- Training
  - [ ] Mesh query points offline sampling for fast and controled training
  - [ ] Train `PIFu` on the DeepHuman training dataset
  - [ ] Train `Geo-PIFu` on the DeepHuman training dataset
- Test
  - [ ] Test `PIFu` on the DeepHuman test dataset
  - [ ] Test `Geo-PIFu` on the DeepHuman test dataset
- Evaulation
  - [ ] Compute 4 metrics: CD, PSD, Normal Cosine, Normal L2

## 1. Requirements

We provide a conda `yml` environment file (you can modify `prefix` to change installation location). This conda env. is for `Geo-PIFu` and `PIFu` training, test and evaluation.

	conda env create -f geopifu_requirements.yml
	conda activate geopifu

We use `opendr` for mesh rendering. To accomodate its package requirements, we provide another conda `yml` environment file. If you have trouble with `yml` we also provide our installation commands log in `assests/create_opendrEnv.sh`.

	conda env create -f opendr_requirements.yml
	conda activate opendrEnv # plz make sure this works
	conda deactivate         # back to the geopifu env.

## 2. Dataset

Download the `DeepHuman` mesh dataset from [here](https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset) and move it into your preferred data folder. The downloaded data should be a zip file about 30.5 G.

	mv DeepHumanDataset.zip data/DeepHumanDataset/
	cd data/DeepHumanDataset
	unzip DeepHumanDataset.zip # plz visit https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset for password

Download the `LSUN` background image dataset and unzip the `zip` files.

	cd data/LSUN
	python3 Geo-PIFu/LSUN/download.py
	unzip "*.zip"

Parse the `lmdb` files into jpeg images. You might need to `pip install lmdb`.

	python Geo-PIFu/LSUN/data.py export ./*_val_lmdb --out_dir ./val_jpg --imageType jpg
	python Geo-PIFu/LSUN/data.py export ./*_train_lmdb --out_dir ./train_jpg --imageType jpg --limit 30000

## 3. Human Mesh Rendering and Voxelization

Please first activate the `opendrEnv` conda environment.
	
	conda activate opendrEnv

Generate SMPL-to-IUV mappings. This is for rendering `smplIUV` from parametric SMPL models. You might need `pip install pathlib` and `pip install tqdm`.

	cd Geo-PIFu/UVTextureConverter
	python obtain_facesAtlas_Idx_U_V.py

Compile the voxelizer cpp code. This is for voxelizing human meshes.

	cd Geo-PIFu/voxelizer
	mkdir build && cd build
	cmake ..
	make

You can specify `splitNum` and `splitIdx` (e.g. {0, ..., `splitNum`-1}) in order to split the full data and launch multiple rendering scripts in parallel. For example, with 30 splits the whole rendering process will take about 37 hrs and generate 1.4T rendered data. The rendering script generates various items besides the color human images: `config`, `rgbImage`, `normalRGB`, `maskImage`, `meshVoxels`, `maskFromVoxels`, `meshSem`, `skeleton3D`, `skeleton2D`, `smplSem`, `smplIUV`, `smplSemVoxels`. Not all the items are needed/used in this project. But they could be very useful for other relevant tasks. Please read `render_mesh.py` for detailed explanation of each item and modify the rendering script accordingly to fit your need.

	cd Geo-PIFu/
	python render_mesh.py --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --bgDirSearch ${PREFERRED_DATA_FOLDER}/data --saveDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resolutionScale 4 --splitNum 30 --splitIdx 0
	python render_mesh.py --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --bgDirSearch ${PREFERRED_DATA_FOLDER}/data --saveDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resolutionScale 4 --splitNum 30 --splitIdx 1
	...

Instead of rendering the data on your own, we suggest using [the provided `config` files](https://www.dropbox.com/s/kxx87kyfuewhx0i/config_split_001_000.zip?dl=0) in order to obtain the same set of images used in our work `Geo-PIFu`. For each rendered image, we recorded its rendering settings like camera angles, lighting directions and so on in `json`.

	unzip config_split_001_000.zip -d data/humanRender/config/ # should expect to see 108718 json files

We provide a demo script of parsing these recorded `json` rendering setting files. You might need to make a few simple modifications in `render_mesh.py` in order to properly launch the script and fully re-render our data for fair experiment comparisons. As an alternative, we want to directly provide a download link of our rendered data. But this requires some agreements with the `DeepHuman` dataset authors. We are working on it.

    python render_mesh.py --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --bgDirSearch ${PREFERRED_DATA_FOLDER}/data --saveDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resolutionScale 4 --useConfig --addionalType smplSemVoxels --splitNum 30 --splitIdx 0
    python render_mesh.py --meshDirSearch ${PREFERRED_DATA_FOLDER}/data --bgDirSearch ${PREFERRED_DATA_FOLDER}/data --saveDir ${PREFERRED_DATA_FOLDER}/data/humanRender --resolutionScale 4 --useConfig --addionalType smplSemVoxels --splitNum 30 --splitIdx 1
    ...

## 4. Query Points Offline Sampling

Prepare and save shape training query samples offline, because `--online_sampling` of `train_shape_iccv.py` is slow due to mesh reading, point sampling, ray tracing, etc. This query points offline sampling process only need to be done once. The sampled / saved query points can be used for both `PIFu` and `Geo-PIFu` training.

You can specify `splitNum` and `splitIdx` (e.g. {0, ..., `splitNum`-1}) in order to split the full data and launch multiple sampling scripts in parallel. For example, with 32 splits the whole sampling process will take about 4 hrs and generate 146.5 G query points in total: saved at {./occu_sigma3.5_pts5k_split32_00, ..., ./occu_sigma3.5_pts5k_split32_31}. We want to directly provide a download link of our sampled query points. But this requires some agreements with the `DeepHuman` dataset authors. We are working on it.

	python -m apps.prepare_shape_query

## 5. Acknowledgements

This repository is built on: [DeepHuman](https://github.com/ZhengZerong/DeepHuman) and [PIFu](https://github.com/shunsukesaito/PIFu). Thank the authors for sharing their code!