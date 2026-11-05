"""Model-independent Sobol coordinates, preserving repeated discrete locations.
The leading arbitrary-B prefix is NOT claimed to retain every power-of-two
Sobol balance property. Inputs are spatial-domain facts supplied by adapters.
"""
import math
import numpy as np
from .core import parameters

def sobol_coordinates(width:int,height:int,crop_width:int,crop_height:int,count:int)->np.ndarray:
    if min(width,height,crop_width,crop_height,count)<1 or width<crop_width or height<crop_height:
        raise ValueError('INVALID_SPATIAL_DOMAIN')
    from scipy.stats import qmc
    pts=qmc.Sobol(2,scramble=True,seed=parameters()['candidate_seed']).random_base2(math.ceil(math.log2(max(count,2))))[:count]
    lengths=np.array([width-crop_width+1,height-crop_height+1])
    return np.minimum((pts*lengths).astype(np.int64),lengths-1)
