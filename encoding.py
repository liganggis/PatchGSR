import torch
import torch.nn as nn
import torch.nn.functional as F

def get_encoder(encoding, input_dim=3, 
                multires=6, # freq
                degree=4, # SH
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, # hash/tiled grid
                align_corners=False, interpolation='linear', # grid
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    elif encoding == 'sh':
        from submodules.shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sh, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim