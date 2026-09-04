# Patch-GSR: Fast Surface Reconstruction via 3D Gaussian Splatting


This project aim to achieve fast surface reconstruction from 3D Gaussian Splatting (3DGS). We will release all source code upon paper acceptance.

## Dataset

In our paper, DTU and Tanks and Temples dataset preprocess are based on [Neuralangelo scripts](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md). Evaluation scripts for DTU and Tanks and Temples dataset are based on [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation) respectively. We thank all the authors for their great work and repos. 


And the data structure should be organized as follows:

```shell
data/
├── 360_v2
│   ├── bicycle/
│   ├── bonsai/
│   ├── ...
├── DTU
│   ├── scan24
│   ├── ...
├── TNT
│   ├── Barn
│   ├── Caterpillar
│   ├── Courthouse
│   ├── Ignatius
│   ├── Meetingroom
│   ├── Truck
```


## Pipeline

![pipeline](assets/mesh-main.png)


## Run

### Environment

```shell
git clone https://github.com/liganggis/PatchGSR
cd PatchGSR

conda create -n PatchGSR-env python=3.10
conda activate PatchGSR-env

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

# install dependencies
pip install -r requirements.txt

# install submodules
Unzip the zip file in the submodules folder, then
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
### Train

We have provided the script in the folder (https://github.com/liganggis/PatchGSR/script/) that were used to generate the table in the paper.

## Acknowledgments

...


## BibTex


And thanks to the authors of [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) and [PGSR](https://github.com/zju3dv/PGSR) for their excellent code, please consider citing these repositories.
