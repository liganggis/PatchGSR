# copy from 2DGS
import math
import torch
import numpy as np
import torch.nn.functional as F

def get_point_from_depth(view, depth):
    RT = view.world_view_transform[:3,:3].transpose(-1,-2)
    T = view.world_view_transform[3,:3]
    W, H = view.image_width, view.image_height
    intrins = view.intrins.cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.0, torch.arange(H, device='cuda').float() + 0.0, indexing='xy')
    pts = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = pts @ intrins.inverse().T
    # rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    pts = depth.reshape(-1, 1) * rays_d
    return (pts - T) @ RT

def get_depth_from_nearest_depth(view, nearest_view, depth, pts):
    W, H = view.image_width, view.image_height
    depth_view = depth[None, :, :, :]
    depth_view = depth_view[:H, :W]
    R = nearest_view.world_view_transform[:3,:3]
    RT = R.transpose(-1,-2)
    T = nearest_view.world_view_transform[3,:3]
    pts_cam = pts @ R + T
    pts_proj = torch.stack(
                        [pts_cam[:,0] * nearest_view.focal_x / pts_cam[:,2] + nearest_view.cx,
                         pts_cam[:,1] * nearest_view.focal_y / pts_cam[:,2] + nearest_view.cy], -1).float()
    mask = (pts_proj[:, 0] > 0) & (pts_proj[:, 0] < W) &\
            (pts_proj[:, 1] > 0) & (pts_proj[:, 1] < H) & (pts_cam[:,2] > 0.1)

    pts_proj[..., 0] /= ((W - 1) / 2)
    pts_proj[..., 1] /= ((H - 1) / 2)
    pts_proj -= 1
    pts_proj = pts_proj.view(1, -1, 1, 2)
    map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_proj,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
    pts_cam = pts_cam / pts_cam[:, 2:3]
    pts_cam = pts_cam * map_z.squeeze()[...,None]
    new_pts = (pts_cam - T) @ RT
    new_pts_cam = new_pts @ view.world_view_transform[:3,:3] + view.world_view_transform[3,:3]
    new_pts_proj = torch.stack(
                        [new_pts_cam[:,0] * view.focal_x / new_pts_cam[:,2] + view.cx,
                         new_pts_cam[:,1] * view.focal_y / new_pts_cam[:,2] + view.cy], -1).float()

    ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(depth.device)
    pixel_noise = torch.norm(new_pts_proj - pixels.reshape(*new_pts_proj.shape), dim=-1)
    
    return map_z, mask, pixel_noise

def depths_to_points_our(view, depthmap):
    # c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    intrins = view.intrins.cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.0, torch.arange(H, device='cuda').float() + 0.0, indexing='xy')
    # points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = torch.stack(
                    [(grid_x-view.cx) / view.focal_x,
                    (grid_y-view.cy) / view.focal_y,
                    torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    # rays_d = points @ intrins.inverse().T
    points = depthmap.reshape(-1, 1) * rays_d
   
    # R = torch.tensor(view.R).float().cuda()
    # T = torch.tensor(view.T).float().cuda()
    # points = (points-T)@R.transpose(-1,-2)
    return points

def depths_to_points_our2(view, depthmap):
    # c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    intrins = view.intrins.cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.0, torch.arange(H, device='cuda').float() + 0.0, indexing='xy')
    # points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = torch.stack(
                    [(grid_x-view.cx) / view.focal_x,
                    (grid_y-view.cy) / view.focal_y,
                    torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    # rays_d = points @ intrins.inverse().T
    points = depthmap.reshape(-1, 1) * rays_d
    return points, rays_d

# def depths_to_points_our(view, depthmap):
#     c2w = (view.world_view_transform.T).inverse()
#     W, H = view.image_width, view.image_height
#     fx = W / (2 * math.tan(view.FoVx / 2.))
#     fy = H / (2 * math.tan(view.FoVy / 2.))
#     intrins_inv = torch.tensor(
#         [[1/fx, 0.,-W/(2 * fx)],
#         [0., 1/fy, -H/(2 * fy),],
#         [0., 0., 1.0]]
#     ).float().cuda()
#     grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
#     points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
#     rays_d = points @ intrins_inv.T
#     # print('---------------------')
#     # print(depthmap)
#     # print(rays_d)
#     # rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
#     points = depthmap.reshape(-1,1) * rays_d
#     # print(points)
#     # points = points.T @ c2w[:3,:3].T
#     return points


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    # points = depths_to_points_our(view, depth).reshape(*depth.shape[1:], 3)
    points = depths_to_points_our(view, depth).reshape(*depth.shape[0:], 3)
    
    output = torch.zeros_like(points)   
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def depth_to_normal_d2(view, depth0, depth1):
    """
        view: view camera
        depth: depthmap 
    """
    points0 = depths_to_points_our(view, depth0).reshape(*depth0.shape[1:], 3)
    points1 = depths_to_points_our(view, depth1).reshape(*depth1.shape[1:], 3)

    points = torch.stack([points0, points1],dim=0)

    output = torch.zeros_like(points)
    # dx = torch.cat([points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]], dim=0)
    # dy = torch.cat([points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]], dim=1)
    center = points[..., 1:-1, 1:-1, :]
    dl = points[..., :-2, 1:-1, :]
    dr = points[..., 2:, 1:-1, :]
    dt = points[..., 1:-1, :-2, :]
    db = points[..., 1:-1, 2:, :]
    # normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)

    normal_map_lt = torch.nn.functional.normalize(torch.cross(center - dl, center - dt, dim=-1), dim=-1)
    normal_map_lb = torch.nn.functional.normalize(torch.cross(center - db, center - dl, dim=-1), dim=-1)
    normal_map_rt = torch.nn.functional.normalize(torch.cross(center - dt, center - dr, dim=-1), dim=-1)
    normal_map_rb = torch.nn.functional.normalize(torch.cross(center - dr, center - db, dim=-1), dim=-1)
    output[..., 1:-1, 1:-1, :] = torch.nn.functional.normalize((normal_map_lt + normal_map_lb + normal_map_rt + normal_map_rb) / 4, dim=-1)

    return output

def depth_to_normal_d(view, depth0, depth1):
    """
        view: view camera
        depth: depthmap 
    """

    points0 = depths_to_points_our(view, depth0).reshape(*depth0.shape[1:], 3)
    points1 = depths_to_points_our(view, depth1).reshape(*depth1.shape[1:], 3)

    points = torch.stack([points0, points1],dim=0)

    output = torch.zeros_like(points)
    dx = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]
    dy = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[..., 1:-1, 1:-1, :] = normal_map

    return output

def normal_to_depth(view, depth, normals, dist):
    points, rays_d = depths_to_points_our2(view, depth)
    points = points.reshape(*depth.shape[1:], 3)
    rays_d = rays_d.reshape(*depth.shape[1:], 3)
    normals = normals.permute(1, 2, 0)
    dist = dist.reshape(*depth.shape[1:], 1)
    nl = normals[:-2, 1:-1, :]
    nr = normals[2:, 1:-1, :]
    nt = normals[1:-1, :-2, :]
    nb = normals[1:-1, 2:, :]
    distl = dist[:-2, 1:-1, :]
    distr = dist[2:, 1:-1, :]
    distt = dist[1:-1, :-2, :]
    distb = dist[1:-1, 2:, :]
    center_rays = -rays_d[1:-1, 1:-1, :]
    dl = distl / (torch.sum(center_rays * nl, dim=-1).unsqueeze(-1) + 1e-8)
    dr = distr / (torch.sum(center_rays * nr, dim=-1).unsqueeze(-1) + 1e-8)
    dt = distt / (torch.sum(center_rays * nt, dim=-1).unsqueeze(-1) + 1e-8)
    db = distb / (torch.sum(center_rays * nb, dim=-1).unsqueeze(-1) + 1e-8)

    target = torch.ones_like(depth).reshape(*depth.shape[1:], 1)
    
    target[1:-1, 1:-1, :] = (dl + dr + dt + db) / 4
    return target
