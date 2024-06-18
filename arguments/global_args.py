from argparse import ArgumentParser, Namespace
import sys
import os

def get_global_args(arg_type : str):
    parser = ArgumentParser(description=arg_type)
    #-------------- Model Params ----------------#
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("-s", "--source_path", type=str, default=None) # dataset path
    parser.add_argument("-m", "--model_path", type=str, default=None) # output model path
    parser.add_argument("-r", "--resolution", type=int, default=-1)
    parser.add_argument("-i", "--images", type=str, default="images")
    parser.add_argument("-w", "--white_background", action="store_true")
    parser.add_argument("--data_device", type=str, default="cuda")
    parser.add_argument("--eval", action="store_true")

    #-------------- Pipeline Params -------------#
    # by default, calculate the color in cuda by sh
    parser.add_argument("--compute_mlp_color", action="store_true")
    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument("--compute_cov3D_python", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mlp_type", default="tcnn_mlp") # tcnn_mlp, simple_mlp, siren_mlp
    parser.add_argument("--initial_iteration", type=int,default=5000) 
    parser.add_argument("--patch_importance_rate", type=float, default=0.2)

    #-------------- Optimization Params ---------#
    parser.add_argument("--iterations", type=int, default=30_000) # total iteration
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--position_lr_max_steps", type=int, default=30_000)
    parser.add_argument("--feature_lr", type=float, default=0.005)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)
    parser.add_argument("--percent_dense", type=float, default=0.01)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--densification_interval", type=int, default=100)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=15_000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--random_background", action="store_true")
    parser.add_argument("--noise_value", type=float,default=0.) # max value for gaussian noise 0.1
    parser.add_argument("--noise_threshold_iter", type=int, default=30_000)

    #-------------- Training Params ----------#
    if arg_type == "train":
        parser.add_argument('--ip', type=str, default="127.0.0.1")
        parser.add_argument('--port', type=int, default=6009)
        # specify the debug iteration, if -1, no debug
        parser.add_argument('--debug_from', type=int, default=-1) 
        parser.add_argument('--detect_anomaly', action='store_true', default=False)
        # tensorboard record iteration
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
        # save as ply file
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000]) 
        parser.add_argument("--quiet", action="store_true")
        # at checkpoint, save as pth file
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
        # specify the pth file location that load the gaussian pth file, if none, train a new gaussian
        parser.add_argument("--start_checkpoint", type=str, default = None)
        parser.add_argument("--save_training_images", action="store_true")
        parser.add_argument("--save_training_images_interval", type=int, default=1000)
        # coarse to fine
        parser.add_argument("--initial_resolution", type=float, default=0.125)
        parser.add_argument("--resolution_delay_mult", type=float, default=0.01)
        parser.add_argument("--max_steps", type=int, default=20_000)
        # patch
        parser.add_argument("--patch_ratio", type=float, default=0.5)
        parser.add_argument("--num_patches", type=int, default=1)

    if arg_type == "render":
        # by default, iteration = -1 means load the max iteration during rendering
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", default=True,action="store_true")
        # if no --eval during training, then use skip_test
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        #parser.add_argument("--mlp_type",default="siren_mlp")

    if arg_type == "prune":
        # specify the output path for pruned model & log, if none, use model path by default
        parser.add_argument("--output_path", type=str, default=None) 
        parser.add_argument("--ip", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=6009)
        parser.add_argument("--debug_from", type=int, default=-1)
        parser.add_argument("--detect_anomaly", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        # iterations for output test info
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_001, 40000])
        
        parser.add_argument("--prune_iterations", nargs="+", type=int, default=[30_001, 31_000,
                            32_000, 33_000, 34_000])
        # save as ply file
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[40_000])
        # save as pth file
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[40_000])
        # gaussian pth file location
        parser.add_argument("--start_checkpoint", type=str, default=None)
        # mlp pth file location
        parser.add_argument("--start_checkpoint_mlp", type=str, default=None)
        parser.add_argument("--start_pointcloud", type=str, default=None)
        parser.add_argument("--prune_percent", type=float, default=0.6)
        parser.add_argument("--prune_decay", type=float, default=1)
        parser.add_argument("--prune_type", type=str, default="T_alpha")
        parser.add_argument("--prune_process", type=str, default="single")
        parser.add_argument("--v_pow", type=float, default=0.1)
        # no densify
        parser.add_argument("--densify_iteration", nargs="+", type=int, default=[-1])
        # save image iteration
        parser.add_argument("--save_pruning_images", action="store_true")
        parser.add_argument("--save_pruning_images_interval", type=int, default=500)

    return parser
