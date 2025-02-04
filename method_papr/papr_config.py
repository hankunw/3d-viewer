"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from method_papr.papr_datamanager import (
    PAPRDataManagerConfig,
)
from method_papr.papr_model import PAPRModelConfig
from method_papr.papr_pipeline import (
    PAPRPipelineConfig,
)
from method_papr.configs.config import args
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
pc_feat_opt = args["geoms"]["point_feats"]
bkg_feat_opt = args["geoms"]["background"]
use_pc_feats = pc_feat_opt["use_ink"] or pc_feat_opt["use_inq"] or pc_feat_opt["use_inv"]
mapping_mlp = args["models"]["mapping_mlp"]["use"]
use_renderer = args["models"]["use_renderer"]
bkg_feats_type = bkg_feat_opt["init_type"]
use_bkg = (bkg_feats_type == 'random' or bkg_feats_type == 'zeros' or bkg_feats_type == 'ones' and bkg_feat_opt["learnable"] == True)

lr_opt = args["training"]["lr"]
eps = args["eps"]
max_steps = args["training"]["steps"]

points_lr = lr_opt["points"]["base_lr"] * lr_opt["lr_factor"]
tx_lr = lr_opt["transformer"]["base_lr"] * lr_opt["lr_factor"]
influ_lr = lr_opt["points_influ_scores"]["base_lr"] * lr_opt["lr_factor"]
trainerOptim = {
            # TODO: consider changing optimizers depending on your custom method
            "points": {
                "optimizer": AdamOptimizerConfig(lr=points_lr, eps=eps),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=max_steps),
            },
            "tx": {
                "optimizer": AdamOptimizerConfig(lr=tx_lr, eps=eps),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=max_steps),
            },
            "points_influ_scores": {
                "optimizer": AdamOptimizerConfig(lr=influ_lr, eps=eps),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=max_steps),
            },

        }
if use_pc_feats: 
    pc_feats_lr = lr_opt["feats"]["base_lr"] * lr_opt["lr_factor"]
    trainerOptim["pc_feats"] = {
        "optimizer": AdamOptimizerConfig(lr=pc_feats_lr, eps=eps),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=max_steps),
        }
if mapping_mlp:
    mlp_lr = lr_opt["mapping_mlp"]["base_lr"] * lr_opt["lr_factor"]
    trainerOptim["mapping_mlp"] = {
        "optimizer": AdamOptimizerConfig(lr=mlp_lr, eps=eps),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=max_steps),
        }
if use_renderer: 
    renderer_lr = lr_opt["generator"]["base_lr"] * lr_opt["lr_factor"]
    trainerOptim["renderer"] = {
        "optimizer": AdamOptimizerConfig(lr=renderer_lr, eps=eps),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=max_steps),
        }
if use_bkg:
    bkg_lr = lr_opt["bkg_feats"]["base_lr"] * lr_opt["lr_factor"]
    trainerOptim["bkg_feats"] = {
        "optimizer": AdamOptimizerConfig(lr=bkg_lr, eps=eps),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=max_steps),
        }
method_papr = MethodSpecification(
    config=TrainerConfig(
        method_name="papr",  
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=PAPRPipelineConfig(
            datamanager=PAPRDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=16
            ),
            model=PAPRModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers=trainerOptim,
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio method template.",
)
