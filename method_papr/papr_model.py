"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
import os
import numpy as np
import yaml
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.scene_colliders import NearFarCollider

from method_papr.models.utils import normalize_vector, create_learning_rate_fn, add_points_knn, activation_func
from method_papr.models.mlp import get_mapping_mlp
from method_papr.models.tx import get_transformer
from method_papr.models.renderer import get_generator
from method_papr.models.lpips import LPNet
from method_papr.configs.config import args

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

import tinycudann as tcnn
# Model related configs
@dataclass
class PAPRModelConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: PAPR)
    """target class to instantiate"""
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    """parameters to instantiate scene collider with"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    """parameters to instantiate density field with"""
    eval_num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    prompt: Optional[str] = None
    """A prompt to be used in text to NeRF models"""
    # args: str = "demo.yml"

class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

class BasicLoss(nn.Module):
    def __init__(self, losses_and_weights):
        super(BasicLoss, self).__init__()
        self.losses_and_weights = losses_and_weights

    def forward(self, pred, target):
        loss = 0
        for name_and_weight, loss_func in self.losses_and_weights.items():
            name, weight = name_and_weight.split('/')
            cur_loss = loss_func(pred, target)
            # print(cur_loss.shape)
            # print(weight.shape)
            loss += float(weight) * torch.mean(cur_loss)
            # loss += cur_loss
            # print(name, weight, cur_loss, loss)
        return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_loss(loss_args):
    losses = nn.ModuleDict()
    for loss_name, weight in loss_args.items():
        if weight > 0:
            if loss_name == "mse":
                losses[loss_name + "/" +
                    str(format(weight, '.0e'))] = nn.MSELoss()
                # print("Using MSE loss, loss weight: ", weight)
            elif loss_name == "l1":
                losses[loss_name + "/" +
                    str(format(weight, '.0e'))] = nn.L1Loss()
                # print("Using L1 loss, loss weight: ", weight)
            elif loss_name == "lpips":
                lpips = LPNet()
                lpips.eval()
                losses[loss_name + "/" + str(format(weight, '.0e'))] = lpips
                # print("Using LPIPS loss, loss weight: ", weight)
            elif loss_name == "lpips_alex":
                lpips = lpips.LPIPS()
                lpips.eval()
                losses[loss_name + "/" + str(format(weight, '.0e'))] = lpips
                # print("Using LPIPS AlexNet loss, loss weight: ", weight)
            else:
                raise NotImplementedError(
                    'loss [{:s}] is not supported'.format(loss_name))
    return BasicLoss(losses)

def get_activation(act_type):
    activation = "LeakyReLU"
    if act_type == 'none':
        activation = "None"
    elif act_type == 'relu':
        activation = "ReLU"
    elif act_type == 'exponential':
        activation = "Exponential"
    elif act_type == 'sine':
        activation = "Sine"
    elif act_type == 'squareplus':
        activation = "Squareplus"
    elif act_type == 'tanh':
        activation = "Tanh"
    elif act_type == 'softplus':
        activation = "Softplus"
    elif act_type == 'sigmoid':
        activation = "Sigmoid"
    return activation
#TODO: refact PAPR model to fit it into nerf Base_Model
class PAPR(Model):

    config: PAPRModelConfig

    def _sphere_pc(self, center, num_pts, scale):
        xs, ys, zs = [], [], []
        phi = math.pi * (3. - math.sqrt(5.))
        for i in range(num_pts):
            y = 1 - (i / float(num_pts - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            xs.append(x * scale[0] + center[0])
            ys.append(y * scale[1] + center[1])
            zs.append(z * scale[2] + center[2])
        points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
        return torch.from_numpy(points).float()

    def _semi_sphere_pc(self, center, num_pts, scale, flatten="-z", flatten_coord=0.0):
        xs, ys, zs = [], [], []
        phi = math.pi * (3. - math.sqrt(5.))
        for i in range(num_pts):
            y = 1 - (i / float(num_pts - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            xs.append(x * scale[0] + center[0])
            ys.append(y * scale[1] + center[1])
            zs.append(z * scale[2] + center[2])
        points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
        points = torch.from_numpy(points).float()
        if flatten == "-z":
            points[:, 2] = torch.clamp(points[:, 2], min=flatten_coord)
        elif flatten == "+z":
            points[:, 2] = torch.clamp(points[:, 2], max=flatten_coord)
        elif flatten == "-y":
            points[:, 1] = torch.clamp(points[:, 1], min=flatten_coord)
        elif flatten == "+y":
            points[:, 1] = torch.clamp(points[:, 1], max=flatten_coord)
        elif flatten == "-x":
            points[:, 0] = torch.clamp(points[:, 0], min=flatten_coord)
        elif flatten == "+x":
            points[:, 0] = torch.clamp(points[:, 0], max=flatten_coord)
        else:
            raise ValueError("Invalid flatten type")
        return points

    def _cube_pc(self, center, num_pts, scale):
        xs = np.random.uniform(-scale[0], scale[0], num_pts) + center[0]
        ys = np.random.uniform(-scale[1], scale[1], num_pts) + center[1]
        zs = np.random.uniform(-scale[2], scale[2], num_pts) + center[2]
        points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
        return torch.from_numpy(points).float()

    def _cube_normal_pc(self, center, num_pts, scale):
        axis_num_pts = int(num_pts ** (1.0 / 3.0))
        xs = np.linspace(-scale[0], scale[0], axis_num_pts) + center[0]
        ys = np.linspace(-scale[1], scale[1], axis_num_pts) + center[1]
        zs = np.linspace(-scale[2], scale[2], axis_num_pts) + center[2]
        points = np.array([[i, j, k] for i in xs for j in ys for k in zs])
        rest_num_pts = num_pts - points.shape[0]
        if rest_num_pts > 0:
            rest_points = self._cube_pc(center, rest_num_pts, scale)
            points = np.concatenate([points, rest_points], axis=0)
        return torch.from_numpy(points).float()

    def _calculate_global_distances(self, rays_o, rays_d, points):
        """
        Select the top-k points with the smallest distance to the rays from all points

        Args:
            rays_o: (N, 3)
            rays_d: (N, H, W, 3)
            points: (num_pts, 3)
        Returns:
            select_k_ind: (N, H, W, select_k)
        """
        N, H, W, _ = rays_d.shape
        num_pts, _ = points.shape

        rays_d = rays_d.unsqueeze(-2)  # (N, H, W, 1, 3)
        rays_o = rays_o.reshape(N, 1, 1, 1, 3)
        points = points.reshape(1, 1, 1, num_pts, 3)

        v = points - rays_o    # (N, 1, 1, num_pts, 3)
        proj = rays_d * (torch.sum(v * rays_d, dim=-1) / (torch.sum(rays_d * rays_d, dim=-1) + self.eps)).unsqueeze(-1)
        D = v - proj    # (N, H, W, num_pts, 3)
        feature = torch.norm(D, dim=-1)

        _, select_k_ind = feature.topk(self.select_k, dim=-1, largest=False, sorted=self.args.geoms.points.select_k_sorted)  # (N, H, W, select_k)

        return select_k_ind

    def _calculate_distances(self, rays_o, rays_d, points):
        """
        Calculate the distances from top-k points to rays   TODO: redundant with _calculate_global_distances

        Args:
            rays_o: (N, 3)
            rays_d: (N, H, W, 3)
            points: (N, H, W, select_k, 3)
            c2w: (N, 4, 4)
        Returns:
            proj_dists: (N, H, W, select_k, 1)
            dists_to_rays: (N, H, W, select_k, 1)
            proj: (N, H, W, select_k, 3)    # the vector s in Figure 2
            D: (N, H, W, select_k, 3)    # the vector t in Figure 2
        """
        N, H, W, _ = rays_d.shape

        rays = normalize_vector(rays_d, eps=self.eps).unsqueeze(-2)  # (N, H, W, 1, 3)
        v = points - rays_o.reshape(N, 1, 1, 1, 3)    # (N, 1, 1, num_pts, 3)
        proj = rays * (torch.sum(v * rays, dim=-1) / (torch.sum(rays * rays, dim=-1) + self.eps)).unsqueeze(-1)
        D = v - proj    # (N, H, W, num_pts, 3)

        dists_to_rays = torch.norm(D, dim=-1, keepdim=True)
        proj_dists = torch.norm(proj, dim=-1, keepdim=True)
        
        return proj_dists, dists_to_rays, proj, D

    def _get_points(self, rays_o, rays_d, step=-1):
        """
        Select the top-k points with the smallest distance to the rays

        Args:
            rays_o: (N, 3)
            rays_d: (N, H, W, 3)
            c2w: (N, 4, 4)
        Returns:
            selected_points: (N, H, W, select_k, 3)
            select_k_ind: (N, H, W, select_k)
        """
        points = self.points
        N, H, W, _ = rays_d.shape
        if self.select_k >= points.shape[0] or self.select_k < 0:
            select_k_ind = torch.arange(points.shape[0], device=points.device).expand(N, H, W, -1)
        else:
            select_k_ind = self._calculate_global_distances(rays_o, rays_d, points)   # (N, H, W, num_pts)
        selected_points = points[select_k_ind, :]  # (N, H, W, select_k, 3)
        self.selected_points = selected_points
        
        return selected_points, select_k_ind

    def prune_points(self, thresh):
        if self.points_influ_scores is not None:
            if self.args.training.prune_type == '<':
                mask = (self.points_influ_scores[:, 0] > thresh)
            elif self.args.training.prune_type == '>':
                mask = (self.points_influ_scores[:, 0] < thresh)
            print(
                "@@@@@@@@@  pruned {}/{}".format(torch.sum(mask == 0), mask.shape[0]))

            cur_requires_grad = self.points.requires_grad
            self.points = nn.Parameter(self.points[mask, :], requires_grad=cur_requires_grad)
            print("@@@@@@@@@ New points: ", self.points.shape)

            cur_requires_grad = self.points_influ_scores.requires_grad
            self.points_influ_scores = nn.Parameter(self.points_influ_scores[mask, :], requires_grad=cur_requires_grad)
            print("@@@@@@@@@ New points_influ_scores: ", self.points_influ_scores.shape)

            if self.use_pc_feats:
                cur_requires_grad = self.pc_feats.requires_grad
                self.pc_feats = nn.Parameter(self.pc_feats[mask, :], requires_grad=cur_requires_grad)
                print("@@@@@@@@@ New pc_feats: ", self.pc_feats.shape)

            return torch.sum(mask == 0)
        return 0

    def add_points(self, add_num):
        points = self.points.detach().cpu()
        point_features = None
        cur_num_points = points.shape[0]

        if 'max_points' in self.args and self.args.max_points > 0 and (cur_num_points + add_num) >= self.args.max_points:
            add_num = self.args.max_points - cur_num_points
            if add_num <= 0:
                return 0

        if self.use_pc_feats:
            point_features = self.pc_feats.detach().cpu()
        
        new_points, num_new_points, new_influ_scores, new_point_features = add_points_knn(points, self.points_influ_scores.detach().cpu(), add_num=add_num,
                                                                                            k=self.args.geoms.points.add_k, comb_type=self.args.geoms.points.add_type,
                                                                                            sample_k=self.args.geoms.points.add_sample_k, sample_type=self.args.geoms.points.add_sample_type,
                                                                                            point_features=point_features)
        print("@@@@@@@@@  added {} points".format(num_new_points))

        if num_new_points > 0:
            cur_requires_grad = self.points.requires_grad
            self.points = nn.Parameter(torch.cat([points, new_points], dim=0).to(self.points.device), requires_grad=cur_requires_grad)
            print("@@@@@@@@@ New points: ", self.points.shape)

            if self.points_influ_scores is not None:
                cur_requires_grad = self.points_influ_scores.requires_grad
                self.points_influ_scores = nn.Parameter(torch.cat([self.points_influ_scores, new_influ_scores.to(self.points_influ_scores.device)], dim=0), requires_grad=cur_requires_grad)
                print("@@@@@@@@@ New points_influ_scores: ", self.points_influ_scores.shape)

            if self.use_pc_feats:
                cur_requires_grad = self.pc_feats.requires_grad
                self.pc_feats = nn.Parameter(torch.cat([self.pc_feats, new_point_features.to(self.pc_feats.device)], dim=0), requires_grad=cur_requires_grad)
                print("@@@@@@@@@ New pc_feats: ", self.pc_feats.shape)

        return num_new_points

    def _get_kqv(self, rays_o, rays_d, points, select_k_ind, step=-1):

        """
        Get the key, query, value for the proximity attention layer(s)
        """
        _, _, vec_p2o, vec_p2r = self._calculate_distances(rays_o, rays_d, points)

        k_type = self.args.models.transformer.k_type
        k_L = self.args.models.transformer.embed.k_L
        if k_type == 1:
            key = [points.detach(), vec_p2o, vec_p2r]
        else:
            raise ValueError('Invalid key type')
        assert len(key) == (len(k_L))

        q_type = self.args.models.transformer.q_type
        q_L = self.args.models.transformer.embed.q_L
        if q_type == 1:
            query = [rays_d.unsqueeze(-2)]
        else:
            raise ValueError('Invalid query type')
        assert len(query) == (len(q_L))

        v_type = self.args.models.transformer.v_type
        v_L = self.args.models.transformer.embed.v_L
        if v_type == 1:
            value = [vec_p2o, vec_p2r]
        else:
            raise ValueError('Invalid value type')
        assert len(value) == (len(v_L))

        # Add extra features that won't be passed through positional encoding
        k_extra = None
        q_extra = None
        v_extra = None
        if self.args.geoms.point_feats.use_ink:
            k_extra = [self.pc_feats[select_k_ind, :]]
        if self.args.geoms.point_feats.use_inq:
            q_extra = [self.pc_feats[select_k_ind, :]]
        if self.args.geoms.point_feats.use_inv:
            v_extra = [self.pc_feats[select_k_ind, :]]

        return key, query, value, k_extra, q_extra, v_extra

    def get_full_address(str):
        cwd = os.getcwd()
        args_path = os.path.join(cwd,str)
        return args_path


    def populate_modules(self, args=args):
        super().populate_modules()

        
        # with open(args, 'r') as f:
        #     self.args = DictAsMember(yaml.safe_load(f))
        self.args = DictAsMember(args)
        # print(type(self.args))
        self.eps = self.args.eps
        self.use_amp = self.args.use_amp
        self.amp_dtype = torch.float16 if self.args.amp_dtype == 'float16' else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        point_opt = self.args.geoms.points
        pc_feat_opt = self.args.geoms.point_feats
        bkg_feat_opt = self.args.geoms.background

        self.register_buffer('select_k', torch.tensor(
            point_opt.select_k, dtype=torch.int32))

        self.coord_scale = self.args.dataset.coord_scale

        if point_opt.load_path:
            if point_opt.load_path.endswith('.pth') or point_opt.load_path.endswith('.pt'):
                points = torch.load(point_opt.load_path, map_location='cpu')
                points = np.asarray(points).astype(np.float32)
                np.random.shuffle(points)
                points = points[:self.args.max_num_pts, :]
                points = torch.from_numpy(points).float()
            print("Loaded points from {}, shape: {}, dtype {}".format(point_opt.load_path, points.shape, points.dtype))
            print("Loaded points scale: ", points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max(), points[:, 2].min(), points[:, 2].max())
        else:
            # Initialize point positions
            pt_init_center = [i * self.coord_scale for i in point_opt.init_center]
            pt_init_scale = [i * self.coord_scale for i in point_opt.init_scale]
            if point_opt.init_type == 'sphere': # initial points on a sphere
                points = self._sphere_pc(pt_init_center, point_opt.num, pt_init_scale)
            elif point_opt.init_type == 'cube': # initial points in a cube
                points = self._cube_normal_pc(pt_init_center, point_opt.num, pt_init_scale)
            else:
                raise NotImplementedError("Point init type [{:s}] is not found".format(point_opt.init_type))
            print("Initialized points scale: ", points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max(), points[:, 2].min(), points[:, 2].max())
        self.points = torch.nn.Parameter(points, requires_grad=True)

        # Initialize point influence scores
        self.points_influ_scores = torch.nn.Parameter(torch.ones(
            points.shape[0], 1) * point_opt.influ_init_val, requires_grad=True)

        # Initialize mapping MLP, only if fine-tuning with IMLE for the exposure control
        # TODO: apply tiny-cuda-nn for optimize MLP
        self.mapping_mlp = None
        self.mlp_config = self.args.models.mapping_mlp
        if self.mlp_config.use:
            # self.mapping_mlp = get_mapping_mlp(
            #     self.args.models, use_amp=self.use_amp, amp_dtype=self.amp_dtype)
            act_type = self.mlp_config.act
            activation = get_activation(act_type)
            out_act_type = self.mlp_config.last_act
            out_activation = get_activation(out_act_type)

            print("---------------------------------------------having mlp with tcnn -----------------------------------------------------------------------")
            self.mapping_mlp = tcnn.Network(
                n_input_dims = self.args.models.shading_code_dim,
                n_output_dims = self.mlp_config.out_dim,
                network_config = {
                    "otype": "CutlassMLP",
                    "activation": activation,
                    "output_activation": out_activation,
                    "n_neurons": self.mlp_config.dim,
                    "n_hidden_layers": self.mlp_config.num_layers,
                }
            )

        # Initialize UNet
        # TODO: apply C++/CUDA for optimize UNet
        if self.args.models.use_renderer:
            tx_opt = self.args.models.transformer
            feat_dim = tx_opt.embed.d_ff_out if tx_opt.embed.share_embed else tx_opt.embed.value.d_ff_out
            self.renderer = get_generator(self.args.models.renderer.generator, in_c=feat_dim,
                                          out_c=3, use_amp=self.use_amp, amp_dtype=self.amp_dtype)
            print("Number of parameters of renderer: ", count_parameters(self.renderer))
        else:
            assert (self.args.models.transformer.embed.share_embed and self.args.models.transformer.embed.d_ff_out == 3) or \
                (not self.args.models.transformer.embed.share_embed and self.args.models.transformer.embed.value.d_ff_out == 3), \
                "Value embedding MLP should have output dim 3 if not using renderer"

        # Initialize background score and features
        if bkg_feat_opt.init_type == 'random':
            bkg_feat_init_func = torch.rand
        elif bkg_feat_opt.init_type == 'zeros':
            bkg_feat_init_func = torch.zeros
        elif bkg_feat_opt.init_type == 'ones':
            bkg_feat_init_func = torch.ones
        else:
            raise NotImplementedError(
                "Background init type [{:s}] is not found".format(bkg_feat_opt.init_type))
        feat_dim = 3
        self.bkg_feats = nn.Parameter(bkg_feat_init_func(bkg_feat_opt.seq_len, feat_dim) * bkg_feat_opt.init_scale, requires_grad=bkg_feat_opt.learnable)
        self.bkg_score = torch.tensor(bkg_feat_opt.constant, dtype=torch.float32).reshape(1)
        # Initialize point features
        self.use_pc_feats = pc_feat_opt.use_ink or pc_feat_opt.use_inq or pc_feat_opt.use_inv
        if self.use_pc_feats:
            self.pc_feats = nn.Parameter(torch.randn(points.shape[0], pc_feat_opt.dim), requires_grad=True)
            print("Point features: ", self.pc_feats.shape, self.pc_feats.min(), self.pc_feats.max(), self.pc_feats.mean(), self.pc_feats.std())

        v_extra_dim = 0
        k_extra_dim = 0
        q_extra_dim = 0
        if pc_feat_opt.use_inv:
            v_extra_dim = self.pc_feats.shape[-1]
            print("Using v_extra_dim: ", v_extra_dim)
        if pc_feat_opt.use_ink:
            k_extra_dim = self.pc_feats.shape[-1]
            print("Using k_extra_dim: ", k_extra_dim)
        if pc_feat_opt.use_inq:
            q_extra_dim = self.pc_feats.shape[-1]
            print("Using q_extra_dim: ", q_extra_dim)

        self.last_act = activation_func(self.args.models.last_act)

        # Initialize proximity attention layer(s)
        transformer = get_transformer(self.args.models.transformer,
                                      seq_len=point_opt.num,
                                      v_extra_dim=v_extra_dim,
                                      k_extra_dim=k_extra_dim,
                                      q_extra_dim=q_extra_dim,
                                      eps=self.eps,
                                      use_amp=self.use_amp,
                                      amp_dtype=self.amp_dtype)
        self.transformer = transformer

        # self.init_optimizers(total_steps=0)

        #TODO: initialize model here: 
        #TODO: set field and other thins to be used here
        # return super().populate_modules()
    def get_param_groups(self):
        param_groups={}
        param_groups["points"] = [self.points]
        param_groups["tx"] = list(self.transformer.parameters())
        param_groups["points_influ_scores"] = [self.points_influ_scores]
        if self.use_pc_feats:
            param_groups["pc_feats"] = [self.pc_feats]
        if self.mapping_mlp is not None:
            param_groups["mapping_mlp"] = list(self.mapping_mlp.parameters())

        if self.args.models.use_renderer:
            param_groups["renderer"] = list(self.renderer.parameters())

        if self.bkg_feats is not None and self.args.geoms.background.learnable:
            param_groups["bkg_feats"] = [self.bkg_feats]
        return param_groups
        # return super().get_param_groups()
    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        #TODO: not necessary
        return super().get_training_callbacks(training_callback_attributes)
    def get_outputs(self, ray_bundle: RayBundle, shading_code=None):
        #TODO: Convert RayBundle to rays_o, rays_d
        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions
        gamma, beta = None, None

        rays_d = torch.reshape(rays_d, (16,16,16,3))
        rays_o = rays_o[::256]


        if shading_code is not None and self.mapping_mlp is not None:  #????????
            affine = self.mapping_mlp(shading_code)
            affine_dim = affine.shape[-1]
            gamma, beta = affine[:affine_dim//2], affine[affine_dim//2:]
        
        points, select_k_ind = self._get_points(rays_o, rays_d)
        key, query, value, k_extra, q_extra, v_extra = self._get_kqv(rays_o, rays_d, points, select_k_ind)
        N, H, W, _ = rays_d.shape
        num_pts = points.shape[-2]

        cur_points_influ_scores = self.points_influ_scores[select_k_ind] if self.points_influ_scores is not None else None

        _, _, embedv, encode, scores = self.transformer(key, query, value, k_extra, q_extra, v_extra)


        embedv = embedv.reshape(N, H, W, -1, embedv.shape[-1])
        scores = scores.reshape(N, H, W, -1, 1)

        if cur_points_influ_scores is not None:
            # Multiply the influence scores to the attention scores
            scores = scores * cur_points_influ_scores

        if self.bkg_feats is not None:
            bkg_seq_len = self.bkg_feats.shape[0]
            self.bkg_score = self.bkg_score.to(self.device)
            scores = torch.cat([scores, self.bkg_score.expand(N, H, W, bkg_seq_len, -1)], dim=-2)
            attn = F.softmax(scores, dim=3) # (N, H, W, num_pts+bkg_seq_len, 1)
            topk_attn = attn[..., :num_pts, :]
            bkg_attn = attn[..., num_pts:, :]
            if self.args.models.normalize_topk_attn:
                topk_attn = topk_attn / torch.sum(topk_attn, dim=3, keepdim=True)
            fused_features = torch.sum(embedv * topk_attn, dim=3)   # (N, H, W, C)

            if self.args.models.use_renderer:
                foreground = self.renderer(fused_features.permute(0, 3, 1, 2), gamma=gamma, beta=beta).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
                if self.args.models.normalize_topk_attn:
                    rgb = foreground * (1 - bkg_attn) + self.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                else:
                    rgb = foreground + self.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                rgb = rgb.squeeze(-2)
            else:
                rgb = fused_features
        else:
            attn = F.softmax(scores, dim=3)
            fused_features = torch.sum(embedv * attn, dim=3)   # (N, H, W, C)
            if self.args.models.use_renderer:
                rgb = self.renderer(fused_features.permute(0, 3, 1, 2), gamma=gamma, beta=beta).permute(0, 2, 3, 1)   # (N, H, W, 3)
            else:
                rgb = fused_features
        output = {'rgb': rgb}

            
        return output


        
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        #TODO: loss function here
        # print(self.args.training.losses)
        loss_fn = get_loss(self.args.training.losses)
        loss_fn = loss_fn.to(self.device)
        loss_dict = {}

        gt_rgb = batch["image"]
        gt_rgb = torch.reshape( gt_rgb ,(16,16,16,3)).to(self.device)
        pred_rgb = outputs["rgb"].to(self.device)
        # print(gt_rgb.shape)
        # print(pred_rgb.shape)
        loss_dict['rgb_loss'] = loss_fn(pred_rgb,gt_rgb)

        return loss_dict


        # return super().get_loss_dict(outputs, batch, metrics_dict)
    def get_metrics_dict(self, outputs, batch) -> Dict[str, Any]:
        return super().get_metrics_dict(outputs, batch)
    

    


