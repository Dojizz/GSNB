import math
import torch
def xyz2az(x,y,z):
    azimuth = torch.atan2(y, x)
    azimuth = (azimuth + 2 * math.pi) % (2 * math.pi)
    zenith = torch.acos(z.clamp(-1.0, 1.0))

    return azimuth, zenith

def az2xyz(azimuth,zenith):
    x = math.sin(zenith) * math.cos(azimuth)
    y = math.sin(zenith) * math.sin(azimuth)
    z = math.cos(zenith)
    return x,y,z
