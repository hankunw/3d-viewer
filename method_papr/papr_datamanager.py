"""
Template DataManager
"""

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

import torch
from torch.utils.data import Dataset
from typing import Dict, Literal, Tuple, Type, Union
import numpy as np
import imageio
from PIL import Image
from dataclasses import dataclass, field
from method_papr.dataset.utils import load_meta_data, get_rays, extract_patches

@dataclass
class PAPRDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: PAPRDataManager)
    patch_size: int = 16
    # potential hyperparameters
    # will this affect training? 


class PAPRDataManager(VanillaDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PAPRDataManagerConfig

    def __init__(
        self,
        config: PAPRDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.train_dataset

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        # print(image_batch["image"].shape)
        # print(image_batch["image_idx"].shape)
        batch = self.train_pixel_sampler.sample(image_batch)
        # print(batch["image"].shape)
        # print(batch["indices"].shape)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        # print(ray_bundle.directions.shape)
        return ray_bundle, batch

