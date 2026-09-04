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
# from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

def render_sets():
    
    with torch.no_grad():
        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # views = scene.getTrainCameras()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join('../../Depth-Anything-3', "DA3NESTED-GIANT-LARGE-1.1")
        model = DepthAnything3.from_pretrained(model_path, local_files_only=True).to("cuda")
        model.eval()
        image_paths = [
            "../../Depth-Anything-3/assets/examples/SOH/000.png",
            "../../Depth-Anything-3/assets/examples/SOH/010.png"
        ]
        # Run inference
        prediction = model.inference(
            image=image_paths,
            process_res=504,
            # process_res_method="upper_bound_resize",
            # export_dir=None,
            # export_format="glb"
        )
        # print(f"Depth shape: {prediction.depth.shape}")
        # print(f"Extrinsics: {prediction.extrinsics.shape if prediction.extrinsics is not None else 'None'}")
        # print(f"Intrinsics: {prediction.intrinsics.shape if prediction.intrinsics is not None else 'None'}")

        # Visualize input images and depth maps
        n_images = prediction.depth.shape[0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))

        if n_images == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_images):
            # Show original image
            if prediction.processed_images is not None:
                print('-------------')
                axes[0, i].imshow(prediction.processed_images[i])
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Show depth map
            depth_vis = visualize_depth(prediction.depth[i], cmap="Spectral")
            axes[1, i].imshow(depth_vis)
            axes[1, i].set_title(f"Depth {i+1}")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('./test.png')



if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--mesh_path', type=str,  default='')
    parser.add_argument('--output_dir', type=str,  default='')
    parser.add_argument("--iteration", default=-1, type=int)

    args = get_combined_args(parser)
    
    render_sets()