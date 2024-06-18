# base on the source code of light gaussian
# this script should work for gaussian with neural expansion
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import torchvision.utils as vutils
from arguments import global_args
import math

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from icecream import ic
import random
import copy
import gc
from os import makedirs
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import csv
from basis_models.mlpmodel import MLPModel
from basis_models.simplemlp import SimpleMLP
from basis_models.sirenmlp import SirenMLP



# from the prune.py file
# return importance score with adaptive volume measure described in paper
def calculate_v_imp_score(gaussians, imp_list, v_pow):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component
    volume = torch.prod(gaussians.get_scaling, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list

# return gaussian counts + original importance score + T_alpha importance score, without volume
def prune_list(gaussians, scene, pipe, background):
    viewpoint_stack = scene.getTrainCameras().copy()
    gaussian_list, opacity_imp_list, T_alpha_imp_list = None, None, None
    viewpoint_cam = viewpoint_stack.pop()
    render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list, opacity_imp_list, T_alpha_imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["opacity_important_score"],
        render_pkg["T_alpha_important_score"]
    )

    for iteration in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gaussians_count, opacity_important_score, T_alpha_important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["opacity_important_score"].detach(),
            render_pkg["T_alpha_important_score"].detach()
        )
        gaussian_list += gaussians_count
        opacity_imp_list += opacity_important_score
        T_alpha_imp_list += T_alpha_important_score
        gc.collect()
    return gaussian_list, opacity_imp_list, T_alpha_imp_list

# from the logger_utils.py
def prepare_output_and_logger(args):

    # Set up output folder
    output_path = None
    if args.output_path == None:
        output_path = args.model_path
    else:
        output_path = args.output_path

    print("Output folder: {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    args,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    output_path = None
    if args.output_path == None:
        output_path = scene.model_path
    else:
        output_path = args.output_path

    # Report test and samples of training set
    if iteration in testing_iterations:
        ic("report")
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
                writer.writeheader()  # file doesn't exist yet, write a header

        torch.cuda.empty_cache()
        validation_configs = ({"name": "test", "cameras": scene.getTestCameras()},)
        #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(iteration, viewpoint, scene.gaussians, basis=scene.basis, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    
                    
                    save_test_dir = os.path.join(output_path, "test_prune_imgs")
                    os.makedirs(save_test_dir, exist_ok=True)
                    img_name = f"image_{idx}.png"
                    save_path = os.path.join(save_test_dir, img_name)
                    vutils.save_image(image, save_path)

                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type="vgg").mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                lpips_test /= len(config["cameras"])
                # sys.stderr.write(f"Iteration  {iteration} Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test} SSIM {ssim_test} LPIPS {lpips_test}\n")
                # sys.stderr.flush()
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                # print(
                #     "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                #         iteration,
                #         config["name"],
                #         l1_test,
                #         psnr_test,
                #         ssim_test,
                #         lpips_test,
                #     )
                # )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - lpips",
                        lpips_test,
                        iteration,
                    )
                if config["name"] == "test":
                    with open(csv_path, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=headers)
                        writer.writerow(
                            {
                                "iteration": iteration,
                                "set": "after prune",
                                "l1_loss": l1_test.item(),
                                "psnr": psnr_test.item(),
                                "ssim": ssim_test.item(),
                                "lpips": lpips_test.item(),
                                "file_size": file_size_mb,
                                "elapsed": elapsed,
                            }
                        )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )

        torch.cuda.empty_cache()

to_tensor = (
    lambda x: x.to("cuda")
    if isinstance(x, torch.Tensor)
    else torch.Tensor(x).to("cuda")
)
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(to_tensor([10.0]))
       
"""
example argument for reference: 
python prune_finetune.py -s ./datasets/cd/ -m ./output/cd
            --start_checkpoint ./output/cd/gaussians_chkpnt30000.pth
            --start_checkpoint_mlp ./output/cd/basis_chkpnt30000.pth
            --iteration 35000 
            --prune_percent 0.66 
            --prune_type v_important_score
            --prune_decay 1 
            --v_pow 0.1
            --prune_process single/hard/soft
"""

# 默认在第一个iter进行prune后，再优化
def pruning(args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    if not args.start_checkpoint:
        print("error: must specify prune target by start_checkpoint!")
        return
    if args.compute_mlp_color and not args.start_checkpoint_mlp:
        print("error: require mlp color but not start_checkpoint_mlp specified!")
        return
    output_path = None
    
    gaussians = GaussianModel(args.sh_degree)
    basis = None
    if args.compute_mlp_color:
        if args.mlp_type == "simple_mlp":
            basis = SimpleMLP().cuda()
        elif args.mlp_type == "tcnn_mlp":
            basis = MLPModel().cuda()
        elif args.mlp_type == "siren_mlp":
            basis = SirenMLP().cuda()
        mlp_checkpoint = torch.load(args.start_checkpoint_mlp)
        basis.load_state_dict(mlp_checkpoint[0]['model_state_dict'])
    scene = Scene(args, gaussians, basis, load_iteration=30000, shuffle=False)

    if args.output_path == None:
        output_path = scene.model_path
    else:
        output_path = args.output_path

    # load gaussian from pth, like light gaussian source code
    gaussians.training_setup(args)
    (model_params, first_iter) = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, args)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, args.iterations), desc="Training progress")
    first_iter += 1
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)

    # 30000~35000, by default, adapt for 5000 iteration, 
    # each iteration use one train image
    for iteration in range(first_iter, args.iterations + 1): 
        iter_start.record()
        gaussians.update_learning_rate(iteration)
         # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if iteration % 400 == 0:
            gaussians.scheduler.step()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == args.debug_from:
            args.debug = True
        bg = torch.rand((3), device="cuda") if args.random_background else background
        render_pkg = render(iteration, viewpoint_cam, gaussians, args, bg, basis=basis)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # save images if necessary
            if args.save_pruning_images and iteration % args.save_pruning_images_interval == 0:
                save_dir = os.path.join(output_path, "pruning_imgs", args.prune_type)
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
            if iteration in args.save_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                torch.save(
                    (gaussians.capture(), iteration),
                    output_path + "/gaussians_chkpnt" + str(iteration) + ".pth")
                if args.compute_mlp_color:
                    torch.save((basis.capture(),iteration), output_path + "/basis_chkpnt" + str(iteration) + ".pth")
                # no need to output importance list
                # if iteration == checkpoint_iterations[-1]:
                #     gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                #     v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                #     np.savez(os.path.join(scene.model_path,"imp_score"), v_list.cpu().detach().numpy()) 

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                args.test_iterations,
                scene,
                render,
                (args, background),
                args
            )

            # start pruning
            if iteration == args.prune_iterations[0]:
                ic("Before prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))
                i = args.prune_iterations.index(iteration)
                gaussian_list, opacity_imp_list, T_alpha_imp_list = prune_list(gaussians, scene, args, background)

                if args.prune_type == "important_score":
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, opacity_imp_list
                    )
                elif args.prune_type == "v_important_score":
                    # normalize scale
                    v_list = calculate_v_imp_score(gaussians, opacity_imp_list, args.v_pow)
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, v_list
                    )
                elif args.prune_type == "max_v_important_score":
                    v_list = opacity_imp_list * torch.max(gaussians.get_scaling, dim=1)[0]
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, v_list
                    )
                elif args.prune_type == "count":
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, gaussian_list
                    )
                elif args.prune_type == "opacity":
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent,
                        gaussians.get_opacity.detach(),
                    )
                # new importance score defined by doji
                elif args.prune_type == "T_alpha":
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent,
                        T_alpha_imp_list
                    )
                else:
                    raise Exception("Unsupportive prunning method")
                ic("After prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))

            # Optimizer step
            if iteration < args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if args.compute_mlp_color:
                    basis.optimizer.step()
                    basis.optimizer.zero_grad(set_to_none = True)    

if __name__ == "__main__":
    # Set up command line argument parser
    prune_parser = global_args.get_global_args("prune")
    prune_args = prune_parser.parse_args(sys.argv[1:])
    if not prune_args.model_path:
        print("error: no model path for pruning!")
        exit(0)
    print("Optimizing " + prune_args.model_path)

    # Initialize system state (RNG)
    safe_state(prune_args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(prune_args.ip, prune_args.port)
    torch.autograd.set_detect_anomaly(prune_args.detect_anomaly)

    if prune_args.prune_process == "single":
        pruning(prune_args)

    # All done
    print("\nPruning complete.")