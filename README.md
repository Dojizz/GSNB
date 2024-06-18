#  GSNB: Gaussian Splatting with Neural Basis Extension

This project tries to enhance original gaussian performance in highly view dependent scene with nerual basis function and maintain real-time rendering.

## Setup

#### Local Setup

Our default, provided install method is based on Conda package and environment management, this will set up the conda env and build the gaussian renderer in gaussian_viewers:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate gsnb
cd gaussian_viewers
cmake -Bbuild .
cmake --build build --target install --config RelWithDebInfo
```
Please note that this process assumes that you have CUDA SDK **12** installed. Currently we have only tested the code in Windowns. Also, GSNB uses tiny-cuda-nn to enable fast network training, the install tutorial can be found at https://github.com/NVlabs/tiny-cuda-nn. We have modified the original diff gaussian rasterization module, so a new conda environment instead of the original 3D-GS is needed.

Currently the anonymous repository is for double blind review, more detailed documentation will be released soon.

### Running

To run the optimizer, simply use
```shell
python train.py -s <path to dataset> -m <path to output model> --compute_mlp_color
```

To prune the trained model, use
```shell
python prune_finetune.py -s <path to dataset> -m <path to output model> --start_checkpoint <path to trained gaussian pth> --start_checkpoint_mlp <path to trained mlp pth> --iteration 40000 --compute_mlp_color
```
Checkpoint pth path is usually named as `gaussians_chkpnt30000.pth` and `basis_chkpnt30000.pth`. This will read the model trained by 30000 iterations, then prune the model and reoptimize for another 10000 iterations.

To bake the trained NBE, use
```shell
python bakery.py -m <path to output model> --bake_checkpoint_mlp <path to trained mlp pth> 
```
This will output a bakery image named as `basis.png` in the specified model path. More parameters setting can be found in `global_args.py`.

For the renderer, after installing the viewers, you can run the compiled `SIBR_gaussianViewer_app` app in `<SIBR install dir>/bin`, e.g.:
```shell
./<SIBR install dir>/bin/SIBR_gaussianViewer_app -m <path to trained model> --use_basis
```
By setting `--use_basis` flag, this will automatically use the bakery image `basis.png` to render the trained model.