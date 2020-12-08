Code, data and results for Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image
==========
We supply an implementation of our paper in `fit_3d.py` and a set of example scripts to visualize the results of our experiments.

To try out the demo on LSP images, follow the directions in "Getting Started".

To learn more about the project, please visit our website: http://smplify.is.tue.mpg.de

You can find the paper at: http://files.is.tue.mpg.de/black/papers/BogoECCV2016.pdf

For comments or questions, please email us at: smplify@tuebingen.mpg.de

Please cite the paper if this code was helpful to your research:

```
@inproceedings{Bogo:ECCV:2016,
  title = {Keep it {SMPL}: Automatic Estimation of {3D} Human Pose and Shape
  from a Single Image},
  author = {Bogo, Federica and Kanazawa, Angjoo and Lassner, Christoph and
  Gehler, Peter and Romero, Javier and Black, Michael J.},
  booktitle = {Computer Vision -- ECCV 2016},
  series = {Lecture Notes in Computer Science},
  publisher = {Springer International Publishing},
  month = oct,
  year = {2016}
}
```

System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:

- [Numpy & Scipy](http://www.scipy.org/scipylib/download.html)
- [Chumpy](https://github.com/mattloper/chumpy) 
- [OpenCV](http://opencv.org/downloads.html)
- [OpenDR](https://github.com/mattloper/opendr)
- [SMPL](http://smpl.is.tue.mpg.de)   
Note that installing SMPL requires registration and manual download on the SMPL website.

Getting Started:
================

1. Extract the code:
--------------------
Extract the `mpips-smplify_public.zip` file to your home directory (or any other location you wish). 
This creates a directory `smplify_public/` which contains this README (plus a txt file used below to install dependencies) and the `code/` directory.


2. Get LSP data:
--------------------
a. Get the images  
Download images from the [LSP dataset](http://www.comp.leeds.ac.uk/mat4saj/lsp_dataset.zip).   
Make sure to download the cropped & scaled dataset, not the original scale images.  
Unzip this directory to any location you like.

b. Get the Deepcut detected joints  
Download `lsp_results.tar.gz` from our [website](http://smplify.is.tue.mpg.de). 
Extract this in the `smplify_public/` directory. This creates the `smplify_public/results` directory.

c. Create a symbolic link to LSP images  
Open a terminal window and type:
```
cd ${smplify_public/}
mkdir images
ln -s ${PATH_TO_LSP_DATASET_FOLDER}/images/ images/lsp
```

3. Install dependencies:
--------------------
We recommend using [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) to manage the python packages.  
To install virtualenv In Linux:  
`sudo apt-get install python-pip python-virtualenv`

To install virtualenv in OSX:  
`sudo pip install virtualenv`

Step 1. Setup your virtualenv:  
In `smplify_public/` or wherever you wish, do:  
`virtualenv --system-site-packages venv`   
where `venv` is the name of the virtualenv.

Step 2. To enter the virtualenv, type:  
`. ${PATH_TO_venv}/bin/activate`

Step 3. Install the packages
```
pip install -U pip
cd ${smplify_public/}
pip install -r requirements.txt
```

Step 4. OpenCV and SMPL require a different installation procedure. Please follow the procedure described on the corresponding websites.

4. Symlink female/male SMPL models from the SMPL package:
--------------------
From `smplify_public/`, do:  
`ln -s ${ABSOLUTE_PATH_WHERE_YOU_UNZIPPED_SMPL}/smpl/models/*.pkl code/models/`

5. Run fit_3d.py:
--------------------
In a new terminal window, navigate to the `smplify_public/code/` directory.
You can run the demo script now by typing the following (make sure you are inside the venv):

`python fit_3d.py ${smplify_public/} --viz`   
(or `python fit_3d.py ../ --viz`)

The script fits the model to LSP data, saving the results in `/tmp/smplify_lsp` (an alternative output folder can be set by passing the argument via command line).

Folder structure
==========
1. `smplify_public/code/` contains the main script running the fitter and visualization utilities (in python and MATLAB) to inspect the results (see also the following Section)
2. `smplify_public/code/lib` provides scripts implementing the objectives minimized during fitting
3. `smplify_public/code/models` provides additional data needed to run the fitter:   
   `gmm_08.pkl` stores the mixture of Gaussians info   
   `lbs_tj10smooth6_0fixed_normalized_locked_hybrid_model_novt.pkl` stores the gender-neutral SMPL model   
   `regressors_locked_normalized_*.npz` store the per-gender regressors to obtain capsule axis length and radius from shape parameters. 

Results
==========
On the [website](http://smplify.is.tue.mpg.de), we provide the detected joints and our fit results for LSP, HumanEva-I and Human3.6M. The following sections describe this data and how to use it.

DeepCut joints
--------------------
2D joints detected with DeepCut are saved as `est_joints.npz` in each dataset/sequence directory.
It holds 3 x 14 x N, where each column stores the (x, y, confidence)
value for the corresponding joint (see below for joint definition).  N is the number of images in the dataset (or frames in the
sequence). I.e. for LSP: N = 2000, where 0-999 is training data, 1000-1999 is test data.
For HEva/H36M: N = number of frames in the sequence.

The 14 DeepCut joints correspond to:

|index     |  joint name      |    corresponding SMPL joint ids   |
|----------|:----------------:|---- -----------------------------:|
| 0        |  Right ankle     |8                                  |
| 1        |  Right knee      |5                                  |
| 2        |  Right hip       |2                                  |
| 3        |  Left hip        |1                                  |
| 4        |  Left knee       |4                                  |
| 5        |  Left ankle      |7                                  |
| 6        |  Right wrist     |21                                 |
| 7        |  Right elbow     |19                                 |
| 8        |  Right shoulder  |17                                 |
| 9        |  Left shoulder   |16                                 |
| 10       |  Left elbow      |18                                 |
| 11       |  Left wrist      |20                                 |
| 12       |  Neck            |-                                  |
| 13       |  Head top        |vertex 411 (see line 233:fit_3d.py)|

SMPL does not have a joint corresponding to the head top so we picked an
appropriate vertex as the corresponding point for the head.

Fits
--------------------
We save the parameters of our fits in a pickle file `all_results.pkl` for each sequence/dataset.
This includes the shape coefficients, pose parameters, camera translation, as well as the internal camera parameters we used such as focal length and principal point.

Please see `show_humaneva.py` to learn how to use these results to set the shape and pose of SMPL.

For Human3.6M, we used a cropped version of the images (using a padded rectangle
on background subtraction) for efficiently running the joint detector. We save
this crop information as 'crop_box' in this pickle file. 

We also save the mesh information (vertices and faces) of the fitted model in
`meshes.hdf5` for each sequence/dataset. Please use `visualize_mesh_sequence.py`
or `visualize_mesh_sequence.m` for MATLAB version to see these meshes. 

Notes
======
- Code: Note that the public code is a demo code which doesn't include the C++ implementation used to compute SMPL derivatives with respect to pose and shape in the paper. For this reason it runs ~3x slower.

- LSP: We provide two results, one with gendered SMPL model `results/lsp/all_results.pkl` and
another with gender-neutral model `results/lsp/all_results_gender-neutral.pkl`. 

- Human3.6M: The camera we used is 'cam_3'. We experimented on all frames of the 15 actions.