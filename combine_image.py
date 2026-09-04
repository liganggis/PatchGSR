#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os, sys
from pathlib import Path
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

import torch
from scene import Scene
import json
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser 
import numpy as np

import cv2

def get_filename_in_folder(path):
    filenames = []
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):

            filenames.append(os.path.splitext(filename)[0])
    return filenames

if __name__ == "__main__":

    # scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']
    # scenes = ["scan24", "scan37", "scan63", "scan65", "scan69", "scan97", "scan105", "scan110"]
    scene_name = "scan24"
    images_folder_name = "Images37"
    # scene_name = "scan65"
    # images_folder_name = "DTU_images"

    gof_path = "../../Meshes/DTU/PatchGS/" + images_folder_name + "/tnt1/"
    pgsr_path = "../../Meshes/DTU/PatchGS/" + images_folder_name + "/tnt5"
    geosvr_path = "../../Meshes/DTU/PatchGS/" + images_folder_name + "/tnt6"
    patchgs_path = "../../Meshes/DTU/PatchGS/" + images_folder_name + "/tnt11"

    # print(os.path.abspath(os.path.join(gof_path,  "000001_simple.png")))
    output_path0 = "../../Meshes/DTU/PatchGS/" + images_folder_name + "/Temp"
    output_path = os.path.join(output_path0, scene_name)
    os.makedirs(output_path, exist_ok=True)

    filenames = get_filename_in_folder(os.path.abspath(gof_path))
    # print(filenames[0:10])
    for filename in filenames:
        if int(filename.split("_")[0]) >0:
            img_gof = cv2.imread(os.path.abspath(os.path.join(gof_path, filename + ".png")))
            img_pgsr = cv2.imread(os.path.abspath(os.path.join(pgsr_path, filename + ".png")))
            img_geosvr = cv2.imread(os.path.abspath(os.path.join(geosvr_path, filename + ".png")))
            img_patchgs = cv2.imread(os.path.abspath(os.path.join(patchgs_path, filename + ".png")))

            img_rows = np.concatenate([img_patchgs, img_geosvr, img_pgsr, img_gof], axis=1)
            cv2.imwrite(os.path.join(output_path, filename + "_row.png"), img_rows)