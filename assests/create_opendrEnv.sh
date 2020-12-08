conda create --name opendrEnv python=2.7
conda activate opendrEnv
pip uninstall numpy
conda install numpy
pip install chumpy
conda install -c conda-forge opencv
conda install -c sunpy matplotlib
pip install opencv-python
conda install -c anaconda scikit-image
conda install -c anaconda scipy
pip install pip==9.0.3
pip install matplotlib==2.2.4
pip install opendr==0.77 # make sure "import opendr" works
pip install openmesh
pip uninstall numpy
pip install numpy # make sure "import cv2" and "import numpy as np" work
pip install tensorflow-gpu==1.14 # plz check assert(tf.__version__ == "1.14.0")
pip install --upgrade pip
pip install nvidia-ml-py==10.418.84
pip install -U Pillow
pip install ipdb # make sure "skimage.__version__" >= "0.13.X"
apt-get install zip unzip
pip install tensorboard
conda install pytorch==1.4.0 torchvision cudatoolkit=10.0 -c pytorch # use 10.1 or 10.0 or others based on your CUDA version, make sure "tf.test.is_gpu_available()" is True, e.g. Adobe Sensei is 10.1, UCLA Vision Lab Tong's Desktop is 10.0. Check torch.__version__ is correct or not.
