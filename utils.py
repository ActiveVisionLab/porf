import numpy as np
import torch
import kornia.geometry as KG
from scipy.spatial.transform import Rotation
import torch.nn.functional as F


def compute_P_from_KT(K, T):

    P = torch.matmul(K, torch.linalg.inv(T))
    return P


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

    xyz_result = poses_pred[:, :3, 3].T
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


def compute_rpe(gt, pred):
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

        rot_err = rotation_error(np.linalg.inv(cur_gt) @ cur_pred)

        r_errs.append(rot_err)
        t_errs.append(np.sqrt(np.sum(align_err ** 2)))

    # ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
    return np.array(r_errs), np.array(t_errs)



def compute_epipolar_err(ref_xy, src_xy, P1, P2):

    Fm = KG.epipolar.fundamental_from_projections(P1[None, :3], P2[None, :3])

    err = KG.symmetrical_epipolar_distance(ref_xy[None],
                                           src_xy[None],
                                           Fm,
                                           squared=False,
                                           eps=1e-08)

    return err.squeeze()


def evaluate_pose(intrinsic, pose, P_src_list, match_list, num_pairs, inlier_threshold):
    P_ref = compute_P_from_KT(intrinsic, pose)

    inlier_rates = []
    errs = []

    loss = 0
    for idx, m in enumerate(match_list):
        epi_err = compute_epipolar_err(m[:, 0:2],
                                       m[:, 2:4],
                                       P_ref,
                                       P_src_list[idx])

        inlier_mask = epi_err < inlier_threshold
        inlier_rate = inlier_mask.float().mean()

        inlier_rates.append(inlier_rate)

        if inlier_rate > 0:
            errs.append(epi_err)

            weight = inlier_rate * inlier_rate
            loss += weight * F.huber_loss(epi_err[inlier_mask], torch.zeros_like(epi_err[inlier_mask]))

        if len(errs) > num_pairs:
            break

    avg_inlier_rate = torch.stack(inlier_rates).mean()

    loss = loss / num_pairs

    return avg_inlier_rate, loss

