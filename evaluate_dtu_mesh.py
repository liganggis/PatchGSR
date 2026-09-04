import numpy as np
import torch
import torch.nn.functional as F
from scene import Scene
import cv2
import os
import random
from os import makedirs, path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import trimesh
from skimage.morphology import binary_dilation, disk

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=30_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--scan_id', type=str,  help='scan id of the input mesh')
    parser.add_argument('--DTU', type=str,  default='dtu_eval/Offical_DTU_Dataset', help='path to the GT DTU point clouds')
    parser.add_argument("-UDF", "--use_depth_filter", action="store_true")
    parser.add_argument('-f', "--model_file", type=str,  help='scan id of the input mesh')
    
    args = get_combined_args(parser)
    print("evaluating " + args.model_path)

    dataset = model.extract(args)

    mesh_dir = args.model_file
    if args.use_depth_filter:
        filename = "tsdf_depth_filter.ply"
    else:
        filename = "tsdf.ply"
    
    # load mesh
    mesh_file = os.path.join(dataset.model_path, mesh_dir, filename)
    output_dir = os.path.join(dataset.model_path, mesh_dir)
    scan = int(dataset.source_path.split("/")[-1][4:])
    
    cmd = f"python dtu_eval/evaluate_single_scene.py --input_mesh {mesh_file} --output_dir {output_dir} --scan_id {scan} --DTU {args.DTU} --mask_dir {dataset.source_path}"
    print(cmd)
    os.system(cmd)