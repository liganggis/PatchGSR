# Aniso-GS: Anisotropic Appearance Field for 3D Gaussian Splatting

![teaser](assets/synthetic0.png)

This project aim to enhance 3D Gaussian Splatting in modeling scenes with specular highlights. I hope this work can assist researchers who need to model specular highlights through splatting.

## Dataset

In our paper, we use:

- synthetic dataset from [NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), and [Anisotropic Synthetic Dataset] from [Spec-Gaussian](https://drive.google.com/drive/folders/1hH7qMSbTyR392PYgsqeMhAnaAxwxzemc?usp=drive_link)
- real-world dataset from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), and [Tandt\&Temple + Deepending] from [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).


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

### Dataset
Dataset setting refs to PGSR(https://github.com/zju3dv/PGSR).

## Acknowledgments

...


## BibTex


And thanks to the authors of [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) and [PGSR](https://github.com/zju3dv/PGSR) for their excellent code, please consider citing these repositories.
