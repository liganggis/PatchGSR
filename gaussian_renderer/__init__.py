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
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image, depth_to_normal
import cupy as cp
import torch.nn.functional as F

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    # intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    # st = max(int(scale/2)-1,0)
    # if offset is not None:
    #     offset = offset[st::scale,st::scale]
    # normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
    #                                         intrinsic_matrix.to(depth.device), 
    #                                         extrinsic_matrix.to(depth.device), offset)
    
    normal_ref = depth_to_normal(viewpoint_cam, depth)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size = 0.3, override_color = None, 
           app_model: AppModel=None, return_depth = True, return_depth_normal = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = viewpoint_camera.tanfovx
    tanfovy = viewpoint_camera.tanfovy
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        kernel_size = kernel_size,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        return_depth=return_depth,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity
    
    scales = pc.get_scaling
    rotations = pc.get_rotation


    # shs = pc.get_features
    shs = None

    if not return_depth:
        _, pixes2D, conics, _, _, radii, tiles_touched = rasterizer.preprocess(
            means3D=means3D,
            shs = shs,
            scales=scales,
            rotations=rotations,
            opacities = opacity
        )
        mask = (radii > 0)
        dir_pp = (pc.get_xyz.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        colors_precomp = pc._nlgs(pc.get_features[mask], pc.get_features_rest[mask], dir_pp[mask], True)
        colors_precomps = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device='cuda')
        colors_precomps[mask] = colors_precomp
        
        rendered_image, out_observe, out_depth_normal = rasterizer(
            means2D = means2D,
            means2D_abs = means2D_abs,
            colors_precomp = colors_precomps,
            pixes2D = pixes2D,
            conics = conics,
            radii = radii,
            tiles_touched = tiles_touched)  

        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe} 
        
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image}) 

        return return_dict
    
    # global_normal = pc.get_normal(viewpoint_camera)

    _, pixes2D, conics, local_normal_distance, global_normal, radii, tiles_touched = rasterizer.preprocess(
        means3D=means3D,
        shs = shs,
        scales=scales,
        rotations=rotations,
        opacities = opacity,
    )
    mask = (radii > 0)
    dir_pp = (pc.get_xyz.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    colors_precomp = pc._nlgs(pc.get_features[mask], pc.get_features_rest[mask], dir_pp[mask], True)
    colors_precomps = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device='cuda')
    colors_precomps[mask] = colors_precomp

    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, out_observe, out_depth_normal = rasterizer(
        means2D = means2D,
        means2D_abs = means2D_abs,
        colors_precomp = colors_precomps,
        local_normal_distance = local_normal_distance,
        pixes2D = pixes2D,
        conics = conics,
        radii = radii,
        tiles_touched = tiles_touched) 

    rendered_normal = out_depth_normal[:3, :, :]
    alpha_depth =  out_depth_normal[4:5, :, :].abs()
    mid_depth = out_depth_normal[5:6, :, :]
    rendered_distance = out_depth_normal[3:4, :, :]

    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "alpha_depth": alpha_depth,
                    "rendered_distance": rendered_distance,
                    "mid_depth": mid_depth} 
    
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image}) 

    if return_depth_normal:
        # depth_normal = depth_to_normal(viewpoint_camera, alpha_depth.squeeze()).permute(2,0,1)
        # depth_normal = rasterizer.depth2normal_v2(alpha_depth, alpha_depth, alpha_depth, alpha_depth,
        #                                        viewpoint_camera.image_width, viewpoint_camera.image_height, tanfovx, tanfovy).permute(2,0,1)
        # depth_mid_normal = rasterizer.depth2normal_v2(mid_depth, mid_depth, mid_depth, mid_depth,
        #                                        viewpoint_camera.image_width, viewpoint_camera.image_height, tanfovx, tanfovy).permute(2,0,1)
        depth_normal = rasterizer.depth2normal(alpha_depth, alpha_depth, alpha_depth, alpha_depth, alpha_depth,
                                               viewpoint_camera.image_width, viewpoint_camera.image_height, tanfovx, tanfovy).permute(2,0,1)
        depth_mid_normal = rasterizer.depth2normal(mid_depth, mid_depth, mid_depth, mid_depth, mid_depth,
                                               viewpoint_camera.image_width, viewpoint_camera.image_height, tanfovx, tanfovy).permute(2,0,1)
        return_dict.update({"depth_normal": depth_normal})  
        return_dict.update({"depth_mid_normal": depth_mid_normal})  
        

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict

def compute_lncc(pc : GaussianModel, viewpoint_camera, nearest_camera, sample_num, pixels, 
                 normals, distances, depth, nearest_depth, patch_size, pixel_noise_th, ncc_threshold,
                 pipe, bg_color : torch.Tensor, kernel_size: float, return_depth = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        kernel_size = kernel_size,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        return_depth=return_depth,
        debug=pipe.debug
    )


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    _, nearest_image_gray = nearest_camera.get_image()
    _, gray_H, gray_W = nearest_image_gray.shape

    with torch.no_grad():
        d_mask, _, _ = rasterizer.depthWeight(depth, nearest_depth, nearest_camera.Fx, nearest_camera.Fy, 
                                                           nearest_camera.world_view_transform, pixel_noise_th)
        
        # weights, d_mask, pixel_noise = pc.compute_depth_loss(viewpoint_camera, nearest_camera, depth, nearest_depth, pixels, pixel_noise_th)

        temp_mask = torch.zeros_like(d_mask, dtype=torch.int32).cuda()
        valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
        if d_mask.sum() > sample_num:
            # index = np.random.choice(d_mask.sum().cpu().numpy(), 102400, replace = False)
            # valid_indices = valid_indices[index]
            index = cp.random.choice(d_mask.sum().item(), sample_num, replace = False)
            valid_indices = valid_indices[cp.asnumpy(index)]
        temp_mask[valid_indices] = torch.arange(valid_indices.shape[0], dtype=torch.int32, device=d_mask.device) + 1

        ref_to_neareast_r = nearest_camera.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_camera.world_view_transform[:3,:3]
        ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_camera.world_view_transform[3,:3] + nearest_camera.world_view_transform[3,:3]
    

    ref_patchs, nearest_patchs = rasterizer.homography(valid_indices.shape[0], patch_size, temp_mask, pixels, normals, distances, nearest_camera.ncc_scale, 
                                                                     gray_W, gray_H, nearest_camera.Fx, nearest_camera.Fy, 
                                                                     ref_to_neareast_r, ref_to_neareast_t)
    with torch.no_grad():
        _, ref_image_gray = viewpoint_camera.get_image()
        sampled_ref_gray = F.grid_sample(ref_image_gray[None], ref_patchs, align_corners=True).squeeze()
        
    # _, nearest_image_gray = nearest_camera.get_image()
    sampled_nearest_gray = F.grid_sample(nearest_image_gray[None], nearest_patchs, align_corners=True).squeeze()

    ncc, ncc_mask = rasterizer.lncc(valid_indices.shape[0], patch_size, ncc_threshold, sampled_ref_gray, sampled_nearest_gray)
    return ncc / (torch.sum(ncc_mask) + 0.000000001), valid_indices

def depth2normal(viewpoint_camera, depth):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    image_width=int(viewpoint_camera.image_width)
    image_height=int(viewpoint_camera.image_height)

    rasterizer = GaussianRasterizer(raster_settings=None)

    normal = rasterizer.depth2normal(depth, depth, depth, depth, depth,
                                               image_width, image_height, tanfovx, tanfovy)
    
    return normal

def render2(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, kernel_size = 0.3, 
           app_model: AppModel=None, return_depth = True, return_depth_normal = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return_dict = None
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_depth,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    if not return_depth:

        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image})
        return return_dict

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "alpha_depth": plane_depth,
                    "rendered_distance": rendered_distance
                    }
    
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})   

    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze())
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict