from __future__ import print_function, division, absolute_import



import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from copy import copy
import yaml


with open("../configs/robot_config.yaml", "r") as file:
    config = yaml.safe_load(file)


asset_root = ".."
asset_path = config["robot"]["urdf_path"]


asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True

if sim_type == gymapi.SIM_FLEX:
    asset_options.max_angular_velocity = 40.

iiwa_asset = gym.load_asset(sim, asset_root, asset_root, asset_options)








