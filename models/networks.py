import torch
import torch.nn as nn
from kornia.geometry.conversions import axis_angle_to_rotation_matrix
from kornia.geometry.conversions import rotation_matrix_to_axis_angle


def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    c2w = torch.eye(4).type_as(r)
    R = axis_angle_to_rotation_matrix(r.unsqueeze(0))[0]  # (3, 3)
    c2w[:3, :3] = R
    c2w[:3, 3] = t

    return c2w


class LearnPose(nn.Module):
    def __init__(self, num_cams, init_c2w):
        """
        :param num_cams:
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()

        self.num_cams = num_cams
        self.init_c2w = init_c2w.clone().detach()
        self.init_r = []
        self.init_t = []
        for idx in range(num_cams):
            r_init = rotation_matrix_to_axis_angle(self.init_c2w[idx][:3, :3].reshape([1, 3, 3])).reshape(-1)
            t_init = self.init_c2w[idx][:3, 3].reshape(-1)
            self.init_r.append(r_init)
            self.init_t.append(t_init)
        self.init_r = torch.stack(self.init_r)  # nx3
        self.init_t = torch.stack(self.init_t)  # nx3

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=True)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=True)  # (N, 3)

    def get_init_pose(self, cam_id):
        return self.init_c2w[cam_id]

    def forward(self, cam_id):
        dr = self.r[cam_id]  # (3, ) axis-angle
        dt = self.t[cam_id]  # (3, )

        r = dr + self.init_r[cam_id]
        t = dt + self.init_t[cam_id]
        c2w = make_c2w(r, t)  # (4, 4)

        return c2w


class PoRF(nn.Module):
    def __init__(self, num_cams, init_c2w=None, layers=2, mode='porf', scale=1e-6):
        """
        :param num_cams:
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(PoRF, self).__init__()
        self.num_cams = num_cams
        self.scale = scale
        self.mode = mode

        if init_c2w is not None:
            self.init_c2w = init_c2w.clone().detach()
            self.init_r = []
            self.init_t = []
            for idx in range(num_cams):
                r_init = rotation_matrix_to_axis_angle(self.init_c2w[idx][:3, :3].reshape([1, 3, 3])).reshape(-1)
                t_init = self.init_c2w[idx][:3, 3].reshape(-1)
                self.init_r.append(r_init)
                self.init_t.append(t_init)
            self.init_r = torch.stack(self.init_r)  # nx3
            self.init_t = torch.stack(self.init_t)  # nx3
        else:
            self.init_r = torch.zeros(size=(num_cams, 3), dtype=torch.float32)
            self.init_t = torch.zeros(size=(num_cams, 3), dtype=torch.float32)

        d_in = 7  # 1 cam_id + 6 pose

        activation_func = nn.ELU(inplace=True)

        self.layers = nn.Sequential(nn.Linear(d_in, 256),
                                    activation_func)
        for i in range(layers):
            self.layers.append(nn.Sequential(nn.Linear(256, 256),
                                             activation_func))
        self.layers.append(nn.Linear(256, 6))

        print('init_r range: ', [self.init_r.min(), self.init_r.max()])
        print('init_t range: ', [self.init_t.min(), self.init_t.max()])

    def get_init_pose(self, cam_id):
        return self.init_c2w[cam_id]

    def forward(self, cam_id):
        cam_id_tensor = torch.tensor([cam_id]).type_as(self.init_c2w)
        cam_id_tensor = (cam_id_tensor / self.num_cams) * 2 - 1  # range [-1, +1]

        init_r = self.init_r[cam_id]
        init_t = self.init_t[cam_id]

        if self.mode == 'porf':
            inputs = torch.cat([cam_id_tensor, init_r, init_t], dim=-1)
        elif self.mode == 'index_only':
            inputs = torch.cat([cam_id_tensor,  torch.zeros_like(init_r), torch.zeros_like(init_t)], dim=-1)
        elif self.mode == 'pose_only':
            inputs = torch.cat([torch.zeros_like(cam_id_tensor),  init_r, init_t], dim=-1)

        out = self.layers(inputs) * self.scale

        # cat pose
        r = out[:3] + self.init_r[cam_id]
        t = out[3:] + self.init_t[cam_id]
        c2w = make_c2w(r, t)  # (4, 4)

        return c2w
