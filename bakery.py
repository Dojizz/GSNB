from utils.direction_utils import az2xyz,xyz2az
from basis_models.simplemlp import SimpleMLP
from basis_models.mlpmodel import MLPModel
from basis_models.sirenmlp import SirenMLP
import torch as pt
import torch.nn as nn
import math
import numpy as np
from skimage import io
import os
from argparse import ArgumentParser
import sys

def makegrid(imgs, maxcol, first, last): #0 16 0 16
  big = []
  cols = []
  for i in range(first, last):
    cols.append(imgs[i, :, :, :4]) #imgs[i, :, :, :4]:(400, 400, 3)

    if len(cols) == maxcol:
      big.append(np.concatenate(cols, 1)) # (400, 6400, 3)
      cols = []

  return np.concatenate(big, 0)

def bake_model(model_path, bake_checkpoint_mlp, bakery_img_size, ss_ratio, mlp_type):
  checkpoint = pt.load(bake_checkpoint_mlp)
  if mlp_type == "simple_mlp":
      basis = SimpleMLP().cuda()
  elif mlp_type == "tcnn_mlp":
      basis=MLPModel().cuda()
  elif mlp_type == "siren_mlp":
      basis = SirenMLP().cuda()
  else:
      basis = None
      return

  basis.load_state_dict(checkpoint[0]['model_state_dict'])
  basis.eval()
  
  sh_v, sw_v = bakery_img_size, bakery_img_size
  azimuth, zenith= pt.meshgrid([
  pt.linspace(0, 2*math.pi, sw_v * ss_ratio),
  pt.linspace(0, math.pi, sh_v * ss_ratio)])
  viewing = pt.cat([azimuth[:,:,None], zenith[:,:,None]], -1) #[400 * ss, 400 * ss, 2]
  hinv_xy = viewing.view(sh_v*sw_v*ss_ratio*ss_ratio, -1) #[160000 * ss * ss, 2]
  xyz_coords = [az2xyz(az.item(), zn.item()) for az, zn in hinv_xy] #[160000 * ss * ss, 3]
  xyz_coords = pt.tensor(xyz_coords).cuda()#[160000* ss * ss, 3]

  out=basis(xyz_coords) #[160000, 16] float16
  out=out.transpose(0,1).cpu()#[16, 160000]
  # DIMREC
  #value = out.view(7, 1, sh_v * ss_ratio, sw_v * ss_ratio) #[16,1,400*ss,400*ss]
  value = out.view(16, 1, sh_v * ss_ratio, sw_v * ss_ratio)
  value = (value + 1) / 2 

  value = value.repeat([1, 3, 1, 1]) #[16,3,400*ss,400*ss]
  # down sample
  pooling = nn.AvgPool2d(ss_ratio, stride=ss_ratio)
  value = pooling(value)
  out = value.permute([0, 2, 3, 1]).detach().numpy() #(16, 400, 400, 3)
  
  mpi_reflection = makegrid(out, out.shape[0], 0, out.shape[0]) #(400, 6400, 3)
  io.imsave(os.path.join(model_path, "basis" + ".png"),np.floor(255 * mpi_reflection).astype(np.uint8))

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-m", "--model_path", type=str, default=None)
  parser.add_argument("--bake_checkpoint_mlp", type=str, default=None)
  parser.add_argument("--bakery_img_size", type=int, default=800)
  parser.add_argument("--ss_ratio", type=int, default=2)
  parser.add_argument("--mlp_type", type=str, default="tcnn_mlp")
  bake_args = parser.parse_args(sys.argv[1:])
  bake_model(bake_args.model_path, bake_args.bake_checkpoint_mlp, bake_args.bakery_img_size, bake_args.ss_ratio, bake_args.mlp_type)

