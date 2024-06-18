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

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import random_patch, l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from basis_models.mlpmodel import MLPModel
from basis_models.simplemlp import SimpleMLP
from basis_models.sirenmlp import SirenMLP
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import global_args
import torchvision.utils as vutils
import shutil
import lpips
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard is installed successfully.")
except ImportError:
    TENSORBOARD_FOUND = False
    print("Failed to import Tensorboard")

def training(args):
    lpips_model = lpips.LPIPS(net='vgg')
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(args.sh_degree)

    if args.mlp_type == "simple_mlp":
        basis = SimpleMLP().cuda()
        basis.initialize()
    elif args.mlp_type == "tcnn_mlp":
        basis=MLPModel().cuda()
        basis.initialize()
    elif args.mlp_type == "siren_mlp":
        basis = SirenMLP().cuda()
        basis.initialize()
    else:
        basis = None
    scene = Scene(args, gaussians, basis=basis)
    gaussians.training_setup(args)
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, args.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, args.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, args.convert_SHs_python, args.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, args, background, scaling_modifer, basis)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, args.source_path)
                if do_training and ((iteration < int(args.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        #basis.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == args.debug_from:
            args.debug = True
        else:
            args.debug = False

        bg = torch.rand((3), device="cuda") if args.random_background else background

        smooth_term = get_linear_noise_func(lr_init=args.initial_resolution, lr_final=1.0, lr_delay_mult=args.resolution_delay_mult, max_steps=args.max_steps)
        down_sampling = smooth_term(iteration)
        
        render_pkg = render(iteration, viewpoint_cam, gaussians, args, bg, basis=basis, down_sampling = down_sampling)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        cur_size = (int(gt_image.shape[1] * down_sampling), int(gt_image.shape[2] * down_sampling))
        gt_image_cur = F.interpolate(gt_image.unsqueeze(0), size=cur_size, mode='bilinear',
                                     align_corners=False).squeeze(0)    

        loss_local = 0
        loss_global= 0 
        Ll1_global = 0
        Ll1_local = 0

        Ll1_global = l1_loss(image, gt_image_cur)
        Lssim_global = ssim(image, gt_image_cur)
        loss_global = (1.0 - args.lambda_dssim) * Ll1_global + args.lambda_dssim * (1.0 - Lssim_global)

        if args.num_patches != 0:
            patch_size = (int(args.patch_ratio * image.size()[1]), int(args.patch_ratio * image.size()[2]))
            output_patches, positions = random_patch(image, patch_size, args.num_patches)
            real_patches = []
            for i, j in positions:
                real_patch, _= random_patch(gt_image_cur, patch_size, 1, i, j)
                real_patches.append(real_patch[0])
            
            for output_patch, real_patch in zip(output_patches, real_patches):
                Ll1_local = l1_loss(real_patch, output_patch)
                Lssim_local = ssim(output_patch, real_patch)
                loss_local += (1.0 - args.lambda_dssim) * Ll1_local + args.lambda_dssim * (1.0 - Lssim_local)

        loss = args.patch_importance_rate*loss_local/args.num_patches + loss_global
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Save images if needed
            if args.save_training_images:
                if iteration % args.save_training_images_interval == 0:
                    save_dir = os.path.join(args.model_path, "training_imgs")
                    os.makedirs(save_dir, exist_ok = True)
                    file_name = f"image_{iteration}.png"
                    save_path = os.path.join(save_dir, file_name)
                    vutils.save_image(image, save_path)
                    print(f"The image has been saved to {save_path}")

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == args.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            training_report(args, tb_writer, iteration, Ll1_global, loss, l1_loss, iter_start.elapsed_time(iter_end), args.test_iterations, scene, render, (args, background),lpips_model)
            

            # Densification
            if iteration < args.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None
                    gaussians.densify_and_prune(args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if args.compute_mlp_color:
                if iteration < args.iterations:
                    basis.optimizer.step()
                    basis.optimizer.zero_grad(set_to_none=True)

                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                        
            else:
                if iteration < args.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/gaussians_chkpnt" + str(iteration) + ".pth")
                if args.compute_mlp_color:
                    torch.save((basis.capture(),iteration), scene.model_path + "/basis_chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):  
    # if model_path not specified, choose a random path  
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    if os.path.exists(args.model_path):
        print("Output folder exists, try to remove the directory")
        shutil.rmtree(args.model_path, ignore_errors=True)

    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(args, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, lpips_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    headers = [
        "iteration",
        "set",
        "l1_loss",
        "psnr",
        "ssim",
        "lpips",
        "file_size",
        "elapsed",
    ]
    output_path= args.model_path
    csv_path = os.path.join(output_path, "metric.csv")
    # Check if the CSV file exists, if not, create it and write the header
    file_exists = os.path.isfile(csv_path)
    save_path = os.path.join(
        output_path,
        "point_cloud/iteration_" + str(iteration),
        "point_cloud.ply",
    )
    # Check if the file exists
    if os.path.exists(save_path):
        # Get the size of the file
        file_size = os.path.getsize(save_path)
        file_size_mb = file_size / 1024 / 1024  # Convert bytes to kilobytes
    else:
        file_size_mb = None

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(iteration, viewpoint, scene.gaussians, basis=scene.basis, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    
                    save_test_dir = os.path.join(output_path, f"test_{iteration}_imgs")
                    os.makedirs(save_test_dir, exist_ok=True)
                    img_name = f"image_{idx}.png"
                    save_path = os.path.join(save_test_dir, img_name)
                    vutils.save_image(image, save_path)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image = image.to("cuda")
                    gt_image = gt_image.to("cuda")
                    lpips_model = lpips_model.to("cuda")

                    lpips_test += lpips_model(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                
                if config["name"] == "test":
                    with open(csv_path, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=headers)
                        writer.writerow(
                            {
                                "iteration": iteration,
                                "set": config["name"],
                                "l1_loss": l1_test.item(),
                                "psnr": psnr_test.item(),
                                "ssim": ssim_test.item(),
                                "lpips": lpips_test.item(),
                                "file_size": file_size_mb,
                                "elapsed": elapsed,
                            }
                        )

                

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    training_parser = global_args.get_global_args("train")
    training_args = training_parser.parse_args(sys.argv[1:])

    # if model path not set, use dataset path to get model path
    if training_args.source_path == None:
        print("Error: no dataset source path, exit!")
        exit(0)
    if training_args.model_path == None:
        training_args.model_path = os.path.join("./output/", os.path.basename(
            os.path.normpath(training_args.source_path)
        ))
    print("Optimizing " + training_args.model_path)

    # Initialize system state (RNG)
    safe_state(training_args.quiet)
    # Start GUI server, configure and run training
    network_gui.init(training_args.ip, training_args.port)
    torch.autograd.set_detect_anomaly(training_args.detect_anomaly)
    training(training_args)

    # All done
    print("\nTraining complete.")
