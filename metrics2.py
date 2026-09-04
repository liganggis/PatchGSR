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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from utils.image_utils import psnr
import time
import cv2
from utils.general_utils import update_config

def render_set(model_path, name, kernel_size, iteration, views, gaussians, pipeline, background):
    result_path= os.path.join(model_path, name)
    makedirs(result_path, exist_ok=True)

    model = "ours_{}".format(iteration)

    ssims = []
    psnrs = []
    lpipss = []

    # if os.path.exists(result_path + "/results.json"):
    #     with open(result_path + "/results.json", 'r') as fp:
    #         full_dict = json.load(fp)
    # else:
    #     full_dict = {}
    # if model not in full_dict.keys():
    #     full_dict[model] = {}
    full_dict = {}
    full_dict[model] = {}
    per_view_dict = {}
    per_view_dict[model] = {}
    nums_threshold = len(views) * 0.3
    

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = torch.clamp(render(view, gaussians, pipeline, background, kernel_size=kernel_size, app_model=None, return_depth=False, return_depth_normal=False)["render"], 0.0, 1.0)
        gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
        ssims.append(ssim(rendering, gt).mean().double())
        psnrs.append(psnr(rendering, gt).mean().double())
        lpipss.append(lpips(rendering, gt, net_type='vgg').mean().double())

    torch.cuda.empty_cache()
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    # start_time = time.time()
    iter_start.record()
    for view in views:
        render(view, gaussians, pipeline, background, kernel_size=kernel_size, app_model=None, return_depth=False, return_depth_normal=False)["render"]
    # end_time = time.time()
    iter_end.record()
    torch.cuda.synchronize()
    fps = len(views) / (iter_start.elapsed_time(iter_end) / 1000.0)

    for idx in range(len(psnrs)):
        per_view_dict[model][idx] = {"PSNR": psnrs[idx].item(),
                                            "SSIM": ssims[idx].item(),
                                            "LPIPS": lpipss[idx].item()
                                            }
    
    full_dict[model]['ema'] = {"PSNR": torch.tensor(psnrs).mean().item(),
                                            "SSIM": torch.tensor(ssims).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item(),
                                            "fps": fps}
    print(torch.tensor(psnrs).mean().item())
    print(torch.tensor(ssims).mean().item())
    print(torch.tensor(lpipss).mean().item())
    with open(result_path + "/results.json", 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(result_path + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    update_config(dataset, None, True, args.cfg_path)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", dataset.kernel_size, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--cfg_path', type=str,  default='')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)