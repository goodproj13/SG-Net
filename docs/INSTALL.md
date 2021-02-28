## Installation

This repo was tested with Python 3.6, PyTorch 1.1.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=1.0.0. (0.4.x may be also ok)
```shell
python setup.py develop # OR python setup.py install
```

## Preparation

### Datasets

[Pittsburgh](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Torii_Visual_Place_Recognition_2013_CVPR_paper.pdf), [Tokyo 24/7](https://www.di.ens.fr/~josef/publications/Torii15.pdf) and [Tokyo Time Machine](https://arxiv.org/abs/1511.07247) datasets can be found [HERE](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/). The associated .mat files are available [HERE](https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz).

```shell
mkdir data
```
Download the raw datasets and then unzip them under the directory like
```shell
data
├── pitts
│   ├── raw
│   │   ├── pitts250k_test.mat
│   │   ├── pitts250k_train.mat
│   │   ├── pitts250k_val.mat
│   │   ├── pitts30k_test.mat
│   │   ├── pitts30k_train.mat
│   │   ├── pitts30k_val.mat
│   │   ├── Pittsburgh/
│   │   │   ├── images
│   └── └── └── queries
│
└── tokyo
    ├── raw
    │   ├── tokyo247/
    │   ├── tokyo247.mat
    │   ├── tokyoTM/
    │   ├── tokyoTM_train.mat
    └── └── tokyoTM_val.mat
```

### Use Custom Dataset (Optional)

1. Download your own dataset and save it under
```shell
data/my_dataset
  └── raw/ # save the images here
```

2. Define your own dataset following the [template](../DN/datasets/demo.py), and save it under [DN/datasets/](../DN/datasets/), e.g. `DN/datasets/my_dataset.py`.

3. Register it in [DN/datasets/\_\_init\_\_.py](../DN/datasets/__init__.py), e.g.
```shell
from .my_dataset import MyDataset # MyDataset is the class name
__factory = {
    'my_dataset': MyDataset,
}
```

4. (Optional) Read it by
```shell
from DN.datasets import create
dataset = create('my_dataset', 'data/my_dataset') # you can use this command for debugging
```

5. Use it for training/testing by adding args of `-d my_dataset` in the scripts.


### Pre-trained Weights

```shell
mkdir logs && cd logs
```
After preparing the pre-trained weights, the file tree should be
```shell
logs
├── DN_vgg16_conv_1_dim-[64/128/384].pth # refer to (1)
├── DN_vgg16_conv_m_dim-[64/128/256].pth # refer to (1)
├── DN_vgg16_conv_h_dim-[64/128/512].pth # refer to (1)
└── vgg16_pitts_64_desc_cen_[64/128/384]_[64/128/256]_[64/128/512].hdf5 # refer to (2)
```

**(1) Architecture-pretrained weights for VGG16 backbone**

Our VGG16-based feature extraction layers is pretrained on [Architecture dataset](https://www.kaggle.com/wwymak/architecture-dataset). Directly download from [Google Drive](https://drive.google.com/drive/folders/1is0pZ18PmKGxjgVD7mgCVSuRm2AAmvu8?usp=sharing), and save them under the path of `logs/`.

Or you can train the feature branch by running the `Feature_branch.py` (You should download the [Architecture dataset](https://www.kaggle.com/wwymak/architecture-dataset) and put them under the path of `Arch_test` and `Arch_train`):
```shell
# for small model (for one V100 GPU)
bash run_Feature_branch.sh
```

**Note:** the following models are not the original implementation from the paper, in order to accommodate small computation, we use channel dimensions of 64*3 and feature sizes of 128/2 X 128/2. We encourage you to change the channel dimensions (`L_DIM`, `M_DIM`, `H_DIM`) in `run_Feature_branch.sh` to increase the performance.

**(2) initial cluster centers for VLAD layer**

**Note:** it is important as the VLAD layer cannot work with random initialization.

The original cluster centers provided by NetVLAD are highly **recommended**. You could directly download from [Google Drive](https://drive.google.com/drive/folders/1E5ukbZiMFRM9pzUgqPHOnmREnY3s066I?usp=sharing) and save it under the path of `logs/`.

Or you could compute the centers by running the script (if you change the channel dimensions, you must recompute the centers):
```shell
bash cluster.sh
```
