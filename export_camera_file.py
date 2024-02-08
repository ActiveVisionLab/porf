import numpy as np
from path import Path
import os
import cv2
from scipy.spatial.transform import Rotation


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def umeyama_alignment(x, y, with_scale=True):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def pose_alignment(poses_pred, poses_gt):

    num_gt = poses_gt.shape[0]

    xyz_result = poses_pred[:num_gt, :3, 3].T
    xyz_gt = poses_gt[:, :3, 3].T

    r, t, scale = umeyama_alignment(xyz_result, xyz_gt, with_scale=True)

    align_transformation = np.eye(4)
    align_transformation[:3:, :3] = r
    align_transformation[:3, 3] = t

    for cnt in range(poses_pred.shape[0]):
        poses_pred[cnt][:3, 3] *= scale
        poses_pred[cnt] = align_transformation @ poses_pred[cnt]

    return poses_pred


def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    r_diff = Rotation.from_matrix(pose_error[:3, :3])
    pose_error = r_diff.as_matrix()
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error


def compute_RPE(gt, pred):
    trans_errors = []
    rot_errors = []
    for i in range(len(gt)-1):
        gt1 = gt[i]
        gt2 = gt[i+1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i+1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel

        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))

    return np.array(rot_errors), np.array(trans_errors)


def compute_ATE(gt, pred):
    """Compute RMSE of ATE
    Args:
        gt: ground-truth poses
        pred: predicted poses
    """
    r_errs = []
    t_errs = []

    for i in range(len(pred)):
        # cur_gt = np.linalg.inv(gt_0) @ gt[i]
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3]

        # cur_pred = np.linalg.inv(pred_0) @ pred[i]
        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz

        t_errs.append(np.sqrt(np.sum(align_err ** 2)))

        r_diff = np.linalg.inv(cur_gt[:3, :3]) @ cur_pred[:3, :3]
        r_errs.append(rotation_error(r_diff))

    # ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
    return np.array(r_errs), np.array(t_errs)


def generate_camera(scale_mats_np, intrinsics, poses, out_file):
    # write poses
    cameras = {}
    for idx in range(len(poses)):

        cameras["scale_mat_%d" % (idx)] = scale_mats_np[idx]

        K = intrinsics[idx]
        P = K @ np.linalg.inv(poses[idx])
        cameras["world_mat_%d" % (idx)] = P

    np.savez(out_file, **cameras)


def load_camera(cam_file, n_imgs):
    camera_dict = np.load(cam_file, allow_pickle=True)
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_imgs)]
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_imgs)]

    intrinsics = []
    poses = []
    for P in world_mats_np:
        intrinsic, pose = load_K_Rt_from_P(None, P[:3])
        poses.append(pose)
        intrinsics.append(intrinsic)
    poses = np.stack(poses)
    intrinsics = np.stack(intrinsics)

    return scale_mats_np, world_mats_np, intrinsics, poses


if __name__ == '__main__':

    root = 'exp_dtu'
    iters = 'poses_050000'

    method = 'dtu_sift_porf'
    out_name = 'cameras_refine_porf.npz'

    root_dir = Path('./porf_data/dtu/')
    scenes = [os.path.basename(s) for s in sorted(root_dir.dirs())]

    for s in scenes:
        scene_dir = root_dir/s

        pose_file = f'./{root}/{s}/{method}/{iters}/refined_pose.txt'
        if not os.path.exists(pose_file):
            continue

        poses_refine = np.loadtxt(pose_file).reshape(-1, 4, 4)

        # gt pose
        n_imgs = len((scene_dir/'image').files('*.png'))
        scale_mats_np, _, intrinsics, gt_poses = load_camera(scene_dir/'cameras.npz', n_imgs)

        # align pose to gt
        poses_refine = pose_alignment(poses_refine, gt_poses)

        r_err, t_err = compute_ATE(gt_poses, poses_refine)
        print('ate errs: ', np.mean(r_err) / 3.14 * 180, np.mean(t_err))

        r_err, t_err = compute_RPE(gt_poses, poses_refine)
        print('rpe errs: ', np.mean(r_err) / 3.14 * 180, np.mean(t_err))

        generate_camera(scale_mats_np, intrinsics, poses_refine, scene_dir/out_name)
