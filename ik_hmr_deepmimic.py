"""
Wrapper around the PyBullet DeepMimic's MoCap functions
to convert the motion reconstruction output file to the DeepMimic
rig JSON.

TODO: use argparse so that user can enter MoCap filename

"""
import numpy as np
import deepdish as dd
from scipy.spatial.transform import Rotation as R
import os
import glob
import pickle
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pred_path = '/home/huyueyue/Code/human_dynamics/demo_output/gBR_sBM_c01_d04_mBR1_ch04/hmmr_output/hmmr_output.pkl'
with open(pred_path, 'rb') as f:
    data = pickle.load(f)

## coco joints list
## we don't loop over these, but they're used as a lookup map
## for when we loop over the DeepMimic guy's bits
model_joints_map_hmr = [
    "right_ankle", "right_knee", "right_hip", "left_hip", "left_knee",
    "left_ankle", "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist", "neck", "head_top", "nose",
    "left_eye", "right_eye", "left_ear", "right_ear"
]

model_joints_map_hmmr = [
    'right_heel',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_heel',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head',
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_big_toe',
    'right_big_toe',
    'left_small_toe',
    'right_small_toe',
    'left_ankle',
    'right_ankle',
]

# theta_wanted = list(smpl_names_map.keys())

## One-dimensional angles

# "RightJoints": [3, 4, 5, 6, 7, 8],
# "LeftJoints": [9, 10, 11, 12, 13, 14],

# time (1), root pos(3), root orient(4), chest orient(4), neck orient(4),
# r.hip orient(4), r.knee orient(1), r.ankle(4), r.shoulder orient(4), r.elbow orient(1),
# l.hip orient(4), l.knee orient (1), l.ankle(4), l.shoulder orient(4), l.elbow orient(1)

json_mimic = {"Loop": "wrap", "Frames": []}

FRAME_DURATION = round(1 / 20, 6)


def match_coco_joint(pb_name, joints3d):
    """From PB joint name and Coco (HMR) joints3d, get
    the joint position that PB example expects."""
    try:
        # easy case
        j3d_id = model_joints_map_hmmr.index(pb_name)
        print("  - found index #%d (%s) in joints vec" %
              (j3d_id, model_joints_map_hmmr[j3d_id]))
        return joints3d[j3d_id]
    except ValueError:
        print("  - [W]", pb_name, "is not a Coco joint")
        if pb_name == "chest":
            ## HMR output REMOVES the chest from the SMPL joints3d which is
            ## fucking terrible
            return (.2 * joints3d[2] + .2 * joints3d[3] + .6 * joints3d[12])
        if pb_name == "eye":
            print("  - averaging eye positions")
            # only 1 eye in PB example: just get average
            return .5 * (joints3d[15] + joints3d[16])
        raise


from inverse_kinematics import get_angle, get_quaternion
from inverse_kinematics import PB_joint_info

pb_joint_names = PB_joint_info['joint_name']

pb_joint_frames = []

num_frames = data['cams'].shape[0]

for s in range(num_frames):
    s_int = int(s)
    joints = data['kps'][s_int]
    joints3d = data['joints'][s_int]

    cam = data['cams'][s_int]
    poses_rot = data['poses'][s_int]
    smpl = data['shapes'][s_int]

    joints3d[:, 1] *= -1
    joints3d[:, 2] *= -1

    cam_scales = cam[0]
    cam_transl = cam[1:]

    # proc_params = item['proc_param']
    # img_size = proc_params['target_size']  # processed image size
    # start_pt = proc_params['start_pt']
    # inv_proc_scale = 1. / np.asarray(proc_params['scale'])
    # bbox = proc_params[
    # 'bbox']  # bbox is obtained from OpenPose: bbox here is (cx, cy, scale, x, y, h, w)

    # principal_pt = np.array([img_size, img_size]) / 2.
    # flength = 500.
    # tz = flength / (0.5 * img_size * cam_scales)
    # trans = np.hstack([cam_transl, tz])  # camera translation vector ??
    # final_principal_pt = (principal_pt + start_pt) * inv_proc_scale
    # kp_original = ((joints + 1) * 0.5) * img_size  # in padded image.
    # kp_original = (kp_original + start_pt) * inv_proc_scale  # should be good

    # trans[1] *= -1
    # trans[2] *= -1

    # joints3d += trans
    # cx, cy = bbox[[0, 1]].astype(int)

    root_pos = .5 * (joints3d[2] + joints3d[3])
    root_pos_2d = .5 * (joints[2] + joints[3])  # left hip

    # DEBUG_FRAMES = []
    # pplot = s_int in DEBUG_FRAMES

    ## Loop over the names we want
    ## and fill in frame information for PB inverse_kinematics
    _new_frame_info = []

    # for k, name in enumerate(theta_wanted):
    # loop over joints
    for k, pb_name in enumerate(PB_joint_info['joint_name']):
        if pb_name == 'root':
            _new_frame_info.append(root_pos)
            continue  # already filled in root_pos
        # k_idx = theta_names.index(name)  # idx of the theta
        # dm_name = smpl_names_map[name]  # unneeded ?
        # dm_id = dm_name_to_id[dm_name]
        #if s_int==0:
        # print("Filling in DM_ID #%d: %s" % (dm_id, dm_name))
        print("Filling in PBID #%d: %s" % (k, pb_name))
        # print("  - got index #%d for SMPL %s" % (k_idx, theta_names[k_idx]))

        ## BUILD ROTATIONS FROM COORDINATES
        # grab joint coordinate for k_idx
        coord = match_coco_joint(pb_name, joints3d)
        _new_frame_info.append(coord)
    print()
    pb_joint_frames.append(_new_frame_info)

import json
from inverse_kinematics import coord_seq_to_rot_seq

rotation_sequence = coord_seq_to_rot_seq(pb_joint_frames, FRAME_DURATION)

json_mimic = {"Loop": "wrap", "Frames": rotation_sequence}

output_filename = '%s_pb_mimicfile.json' % base_name
with open(output_filename, 'w') as f:
    json.dump(json_mimic, f, indent=4)

os.system("cp %s ../DeepMimic/walken_est_pb.json" % output_filename)