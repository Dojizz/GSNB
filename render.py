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
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from arguments import global_args
from gaussian_renderer import GaussianModel
from basis_models.mlpmodel import MLPModel
from basis_models.simplemlp import SimpleMLP
from basis_models.sirenmlp import SirenMLP


def render_set(model_path, name, iteration, views, gaussians, basis_model, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(iteration, view, gaussians, pipeline, background, basis=basis_model)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def old_render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        basis_model = None
        if (pipeline.compute_mlp_color):
            basis_model = MLPModel().cuda()
            mlp_file_path = os.path.join(dataset.model_path, "basis_chkpnt30000.pth") # by default load 30000
            mlp_checkpoint = torch.load(mlp_file_path)
            basis_model.load_state_dict(mlp_checkpoint[0]['model_state_dict'])
            basis_model.eval()
        scene = Scene(dataset, gaussians, basis_model, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, basis_model, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, basis_model, pipeline, background)

def render_sets(args):
    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree)
        basis_model = None
        if args.compute_mlp_color:
            if args.mlp_type == "simple_mlp":
                basis_model = SimpleMLP().cuda()
            elif args.mlp_type == "tcnn_mlp":
                basis_model=MLPModel().cuda()
            elif args.mlp_type == "siren_mlp":
                basis_model = SirenMLP().cuda()
            mlp_file_path = os.path.join(args.model_path, "basis_chkpnt" + str(args.checkpoint_iterations[-1])\
                                          + ".pth") # by default load 30000
            mlp_checkpoint = torch.load(mlp_file_path)
            basis_model.load_state_dict(mlp_checkpoint[0]['model_state_dict'])
            basis_model.eval()
        # load gaussian from args.model_path
        scene = Scene(args, gaussians, basis_model, load_iteration=args.iteration, shuffle=False)

        bg_color = [1,1,1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not args.skip_train:
             render_set(args.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, basis_model, args, background)

        if not args.skip_test:
             render_set(args.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, basis_model, args, background)

if __name__ == "__main__":
    # Set up command line argument parser
    # parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
    # parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    # parser.add_argument("--quiet", action="store_true")
    # args = get_combined_args(parser)
    # print("Rendering " + args.model_path)
    render_parser = global_args.get_global_args("render")
    # combine original training argument stored in cfg_args
    render_args = get_combined_args(render_parser)

    # Initialize system state (RNG)
    safe_state(render_args.quiet)
    render_sets(render_args)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)