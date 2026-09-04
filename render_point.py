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
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
# os.environ['OPEN#D_CPU_RENDERING'] = 'true'
import open3d as o3d
from scene.app_model import AppModel
import trimesh, copy
from collections import deque
from utils.loss_utils import l1_loss
from utils.image_utils import psnr
import cv2
from utils.general_utils import update_config



def render_sets(dataset : ModelParams, pipeline : PipelineParams, mesh_name: str, mesh_path: str, output_dir: str):
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gs_scene = Scene(dataset, gaussians, shuffle = False)
        views = gs_scene.getTrainCameras()
        w, h = views[0].image_width, views[0].image_height
        # w, h = 1958, 1090
        
        tsdf_path = os.path.join(mesh_path, mesh_name + ".ply")

        pcd = o3d.io.read_point_cloud(tsdf_path)
        
        points = np.asarray(pcd.points)
        points[:, 1] *= -1  
        points[:, 2] *= -1 
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        # pcd.paint_uniform_color([1.0,0.0, 0.0])
        # pcd.paint_uniform_color([0.0, 0.4, 0.8])
        # pcd.paint_uniform_color([176 / 255, 101 / 255, 0 / 255])
        # pcd.paint_uniform_color([0.5, 0.5, 0.5])
        # pcd.paint_uniform_color([0.0, 0, 1])
        # pcd.compute_vertex_normals()
    

        ############################ simple ##################################
        material = o3d.visualization.rendering.MaterialRecord()

        material.shader = "defaultLit"
        material.base_color = [24 / 255, 116 / 255, 205 / 255, 1.0] 
        # material.base_color = [176 / 255, 101 / 255, 0 / 255, 1.0] 
        material.base_roughness = 0.5
        material.base_metallic = 0.0
        material.base_reflectance = 0.2 
        material.point_size = 5.0

        renderer_simple = o3d.visualization.rendering.OffscreenRenderer(w, h)

        scene = renderer_simple.scene
        scene.add_geometry("mesh", pcd, material)

        renderer_simple.scene.show_skybox(False)
        renderer_simple.scene.view.set_shadowing(False)

        # renderer_simple.scene.set_lighting(renderer_simple.scene.LightingProfile.SOFT_SHADOWS, (0.5, 0.5, 0.5))
        renderer_simple.scene.scene.enable_sun_light(True)
        # scene.set_background([10.0, 10.0, 10.0, 1.0]) 

        # ############################ normal ##################################
        # normal_renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
        # mesh.compute_vertex_normals()
        
        # mat = o3d.visualization.rendering.MaterialRecord()
        # mat.shader = "normals"
        # mat.base_roughness = 1
        # mat.base_metallic = 0.0
        # mat.base_reflectance = 0
        
        # normal_renderer.scene.add_geometry("mesh", mesh, mat)
        # normal_renderer.scene.set_background(np.array([1, 1, 1, 1.0], dtype=np.float32))
        # normal_renderer.scene.scene.enable_sun_light(False)
        # normal_renderer.scene.view.set_post_processing(False)
        
        ############################ rendering ##################################
        colmap_to_opengl = np.eye(4)
        colmap_to_opengl[1, 1] = -1
        colmap_to_opengl[2, 2] = -1

        # render_mesh_path = os.path.join(output_dir, mesh_name)
        # os.makedirs(render_mesh_path, exist_ok=True)

        center = np.array([10.0, 10.0, 10.0])

        for idx, view in enumerate(tqdm(views)):
            if int(view.image_name) != int(mesh_name):
                continue
            # print(view.image_name)
            pose = np.eye(4)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T + 0.0
            
            pose = pose @ colmap_to_opengl

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=w,
                height=h,
                fx=view.Fx,
                fy=view.Fy,
                cx=view.Cx,
                cy=view.Cy
            )

            # eye = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])
            # light_dir = (center - eye) / np.linalg.norm(center - eye)

            rays_d = [(0-view.Cx) / view.Fx, (0-view.Cy) / view.Fy, 1]
            rays_d = rays_d @ pose[:3, :3]
            light_dir = rays_d / np.linalg.norm(rays_d)
            renderer_simple.scene.scene.set_sun_light(
                    direction = light_dir,
                    color = [1, 1, 1],
                    intensity = 110000
                )
            renderer_simple.setup_camera(intrinsic, pose)
            image = renderer_simple.render_to_image()
            o3d.io.write_image(os.path.join(output_dir, view.image_name + "_simple.png"), image)
            
            
            # normal_renderer.setup_camera(intrinsic, pose)
            # world_normal_image = normal_renderer.render_to_image()
            # o3d.io.write_image(os.path.join(render_mesh_path, view.image_name + "_normal.png"), world_normal_image)


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--cfg_path', type=str,  default='')
    parser.add_argument('--mesh_path', type=str,  default='')
    parser.add_argument('--output_dir', type=str,  default='')
    parser.add_argument('-n', '--mesh_name', type=str,  default='')

    args = parser.parse_args(sys.argv[1:])
    
    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args.mesh_name, args.mesh_path, args.output_dir)
    torch.cuda.empty_cache()