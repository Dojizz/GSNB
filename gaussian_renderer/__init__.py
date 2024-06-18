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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh,eval_mlp
from basis_models.mlpmodel import MLPModel
from utils.direction_utils import az2xyz,xyz2az
from utils.general_utils import safe_state, get_linear_noise_func, get_linear_smooth_func

def render(iteration:int, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, basis: MLPModel = None , down_sampling=1 ,override_color = None, use_bakery = False, bakery_data = None, bakery_image_hw = 400):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height*down_sampling),
        image_width=int(viewpoint_camera.image_width*down_sampling),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        f_count=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
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
    if pipe.convert_SHs_python and pipe.compute_mlp_color:
        print("warning: cannot compute sh + neural basis color the same time, compute sh by default")
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        elif pipe.compute_mlp_color:
            if basis == None:
                print("error: require compute mlp color but basis model is none!")
                return
            # (points num, 3, 16)
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) #pc.get_features: [14462, 3, 16]
            shs_view = shs_view.to(torch.float32)
            
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)) # [14462, 3]
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) # [N, 3]
            # like deformable gs

            smooth_noise = get_linear_noise_func(lr_init=1.0, lr_final=0., lr_delay_mult=0.01, max_steps=pipe.noise_threshold_iter)
            smooth_noise_rate = smooth_noise(iteration)
            noise = torch.randn_like(dir_pp_normalized) * pipe.noise_value * smooth_noise_rate
            dir_pp_normalized_perturbed = dir_pp_normalized + noise
            dir_pp_normalized_perturbed = dir_pp_normalized_perturbed / dir_pp_normalized_perturbed.norm(dim=1, keepdim=True)

            value = basis(dir_pp_normalized_perturbed) # [N, 16/7]
            # DIMREC
            # result_bmm = torch.bmm(shs_view[:, :, 9:16], value.view(-1, 7, 1).to(torch.float32)).squeeze(-1) 
            result_bmm = torch.bmm(shs_view, value.view(-1, 16, 1).to(torch.float32)).squeeze(-1)

            smooth_term = get_linear_smooth_func(lr_init=0, lr_final=1.0, lr_delay_mult=0.01, max_steps=pipe.initial_iteration)
            smooth_rate = smooth_term(iteration)
            
            result = sh2rgb + result_bmm * smooth_rate + 0.5
            
            colors_precomp = torch.clamp_min(result, 0.0) 
            # directly render basis_value for debug
            # basis_value_debug = basis_value[:, 4]
            # basis_value_debug = basis_value_debug.reshape(basis_value_debug.shape[0], 1)
            # basis_value_debug = torch.stack([basis_value_debug, basis_value_debug, basis_value_debug], -1)
            # basis_value_debug = basis_value_debug.reshape(basis_value_debug.shape[0], basis_value_debug.shape[2])
            # colors_precomp = basis_value_debug.to(torch.float32)
        
        elif use_bakery:
            # (points num, 3, 16)
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) #pc.get_features: [14462, 3, 16]
            shs_view = shs_view.to(torch.float32)
            
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)) #[14462, 3]
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            azimuth = torch.atan(dir_pp_normalized[:, 1] / dir_pp_normalized[:, 0])
            zenith = torch.acos(dir_pp_normalized[:, 2])
            mask_0 = (dir_pp_normalized[:, 0] < 0) & (dir_pp_normalized[:, 1] > 0)
            mask_1 = (dir_pp_normalized[:, 0] < 0) & (dir_pp_normalized[:, 1] < 0)
            mask_2 = (dir_pp_normalized[:, 0] > 0) & (dir_pp_normalized[:, 1] < 0)
            azimuth[mask_0] = math.pi + azimuth[mask_0]
            azimuth[mask_1] = math.pi + azimuth[mask_1]
            azimuth[mask_2] = math.pi * 2 + azimuth[mask_2]
            hs = bakery_image_hw * azimuth / (2 * math.pi) # [points num]
            ws = bakery_image_hw * zenith / math.pi # [points num]
            hs = torch.clamp(hs, 0.0001, bakery_image_hw - 0.0001)
            ws = torch.clamp(ws, 0.0001, bakery_image_hw - 0.0001)
            hs = hs.int().to("cuda")
            ws = ws.int().to("cuda")
            basis_coff = bakery_data[hs, ws] # [points num, 7]
            # DIMREC
            #result_bmm = torch.bmm(shs_view[:, :, 9:16], basis_coff.view(-1, 7, 1).to(torch.float32)).squeeze(-1)
            result_bmm = torch.bmm(shs_view, basis_coff.view(-1, 16, 1).to(torch.float32)).squeeze(-1)
            result = sh2rgb + result_bmm + 0.5
            #result = sh2rgb + 0.5
            colors_precomp = torch.clamp_min(result, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def count_render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
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
        debug=pipe.debug,
        f_count=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
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
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    gaussians_count, opacity_important_score, T_alpha_important_score, rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "gaussians_count": gaussians_count,
        "opacity_important_score": opacity_important_score,
        "T_alpha_important_score": T_alpha_important_score
    }
