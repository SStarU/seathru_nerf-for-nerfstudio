import torch
from torch import nn
from torch import Tensor

from typing import Dict, Literal, Optional, Tuple
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion

from seathru.seathru_fieldheadnames import SeathruHeadNames

try:
    import tinycudann as tcnn  # noqa
except ImportError:
    print("tinycudann is not installed! Please install it for faster training.")


class SeathruField(Field):
    """Field for Seathru-NeRF. Default configuration is the big model.

    Args:
        aabb: parameters of scene aabb bounds
        num_levels: number of levels of the hashmap for the object base MLP
        min_res: minimum resolution of the hashmap for the object base MLP
        max_res: maximum resolution of the hashmap for the object base MLP
        log2_hashmap_size: size of the hashmap for the object base MLP
        features_per_level: number of features per level of the hashmap for the object
                            base MLP
        num_layers: number of hidden layers for the object base MLP
        hidden_dim: dimension of hidden layers for the object base MLP
        bottleneck_dim: bottleneck dimension between object base MLP and object head MLP
        num_layers_colour: number of hidden layers for colour MLP
        hidden_dim_colour: dimension of hidden layers for colour MLP
        num_layers_medium: number of hidden layers for medium MLP
        hidden_dim_medium: dimension of hidden layers for medium MLP
        spatial_distortion: spatial distortion to apply to the scene
        implementation: implementation of the base mlp (tcnn or torch)
        use_viewing_dir_obj_rgb: whether to use viewing direction in object rgb MLP
        object_density_bias: bias for object density
        medium_density_bias: bias for medium density
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 8192,
        log2_hashmap_size: int = 21,
        features_per_level: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 256,
        bottleneck_dim: int = 63,
        num_layers_colour: int = 3,
        hidden_dim_colour: int = 256,
        num_layers_medium: int = 2,
        hidden_dim_medium: int = 128,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        use_viewing_dir_obj_rgb: bool = False,
        object_density_bias: float = 0.0,
        medium_density_bias: float = 0.0,
        num_cameras: int = 0,
        use_medium_c1: bool = False,
        medium_delta_dim: int = 8,
    ) -> None:
        super().__init__()

        # Register buffers
        self.register_buffer("aabb", aabb)
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.bottleneck_dim = bottleneck_dim
        self.spatial_distortion = spatial_distortion
        self.use_viewing_dir_obj_rgb = use_viewing_dir_obj_rgb
        self.object_density_bias = object_density_bias
        self.medium_density_bias = medium_density_bias
        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()

        # ------------------------Object network------------------------
        # Position encoding with trainable hash map
        self.hash_map = HashEncoding(
            num_levels=num_levels,
            min_res=min_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        # Slim mlp for object
        self.object_mlp_base_mlp = MLP(
            in_dim=self.hash_map.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.bottleneck_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # Object mlp_base
        self.object_mlp_base = torch.nn.Sequential(
            self.hash_map, self.object_mlp_base_mlp
        )

        # Object colour MLP
        direction_enc_out_dim = 0
        if self.use_viewing_dir_obj_rgb:
            direction_enc_out_dim = self.direction_encoding.get_out_dim()

        self.mlp_colour = MLP(
            in_dim=direction_enc_out_dim + self.bottleneck_dim,
            num_layers=num_layers_colour,
            layer_width=hidden_dim_colour,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        # ------------------------Medium network------------------------
        # Medium MLP
        self.medium_mlp = MLP(
            in_dim=self.direction_encoding.get_out_dim(),
            num_layers=num_layers_medium,
            layer_width=hidden_dim_medium,
            out_dim=9,
            activation=nn.Softplus(),
            out_activation=None,
            implementation=implementation,
        )

        # ------------------------Medium C1 (global + per-image residual)------------------------
        self.use_medium_c1 = use_medium_c1
        self.num_cameras = int(num_cameras)

        if self.use_medium_c1:
            if self.num_cameras <= 0:
                raise ValueError("num_cameras must be > 0 when use_medium_c1=True")

            # Pre-activation bias for [rgb(3), bs(3), attn(3)]
            self.medium_global_pre = nn.Parameter(torch.zeros(9))

            # Per-image residual embedding -> 9D pre-activation bias
            self.medium_delta_embed = nn.Embedding(self.num_cameras, medium_delta_dim)
            self.medium_delta_proj = nn.Linear(medium_delta_dim, 9, bias=False)

            # Small random init avoids a zero-gradient deadlock.
            nn.init.normal_(self.medium_delta_embed.weight, mean=0.0, std=1e-4)
            nn.init.normal_(self.medium_delta_proj.weight, mean=0.0, std=1e-4)
            self._c1_logged = False

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Compute output of object base MLP. (This function builds on the nerfacto
           implementation)

        Args:
            ray_samples: RaySamples object containing the ray samples.

        Returns:
            Tuple containing the object density and the bottleneck vector.
        """
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            # Normalize positions from [-2, 2] to [0, 1]
            positions = (positions + 2.0) / 4.0
        else:
            # If working with scene box instead of scene contraction
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )

        # Make sure the tcnn gets inputs between 0 and 1
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions

        # Make sure to turn gradients on for the sample locations
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        # Forward pass through the object base MLP
        positions_flat = positions.view(-1, 3)
        h_object = self.object_mlp_base(positions_flat).view(
            *ray_samples.frustums.shape, -1
        )
        density_before_activation, bottleneck_vector = torch.split(
            h_object, [1, self.bottleneck_dim], dim=-1
        )

        # From nerfacto: "Rectifying the density with an exponential is much more stable
        # than a ReLU or softplus, because it enables high post-activation (float32)
        # density outputs from smaller internal (float16) parameters."
        density_before_activation = density_before_activation + self.object_density_bias
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, bottleneck_vector

    def get_outputs(  # type: ignore
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[SeathruHeadNames, Tensor]:
        """Compute outputs of object and medium networks (except object density).

        Args:
            ray_samples: RaySamples object containing the ray samples.
            density_embedding: Bottleneck vector (output of object base MLP).

        Returns:
            Dictionary containing the outputs seathru network.
        """
        assert density_embedding is not None
        outputs = {}

        # Encode directions
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        if self.use_viewing_dir_obj_rgb:
            h_object = torch.cat(
                [directions_encoded, density_embedding.view(-1, self.bottleneck_dim)],
                dim=-1,
            )
        else:
            h_object = density_embedding.view(-1, self.bottleneck_dim)

        # Object colour MLP forward pass
        rgb_object = self.mlp_colour(h_object).view(*outputs_shape, -1).to(directions)
        outputs[FieldHeadNames.RGB] = rgb_object

        # Medium MLP forward pass
        medium_base_out = self.medium_mlp(directions_encoded)

        # C1: add global + per-image pre-activation bias
        if self.use_medium_c1:
            cam_idx = ray_samples.camera_indices
            if cam_idx is None:
                raise RuntimeError(
                    "ray_samples.camera_indices is None; C1 requires camera_indices to "
                    "index per-image embedding."
                )
            cam_idx = cam_idx.reshape(-1).to(device=medium_base_out.device, dtype=torch.long)

            if not self._c1_logged:
                print(f"[C1] cam_idx.max()={int(cam_idx.max())}, num_cameras={self.num_cameras}")
                self._c1_logged = True

            # Safety checks (fail fast if indexing is inconsistent)
            if cam_idx.numel() > 0:
                if int(cam_idx.min()) < 0 or int(cam_idx.max()) >= self.num_cameras:
                    raise RuntimeError(
                        f"camera_indices out of range: min={int(cam_idx.min())}, "
                        f"max={int(cam_idx.max())}, num_cameras={self.num_cameras}. "
                        "Check SeathruModelConfig.num_cameras."
                    )

            delta9 = self.medium_delta_proj(self.medium_delta_embed(cam_idx))
            bias9 = self.medium_global_pre.unsqueeze(0) + delta9
            medium_base_out = medium_base_out + bias9

        # different activations for different outputs
        medium_rgb = (
            self.colour_activation(medium_base_out[..., :3])
            .view(*outputs_shape, -1)
            .to(directions)
        )
        medium_bs = (
            self.sigma_activation(medium_base_out[..., 3:6] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )
        medium_attn = (
            self.sigma_activation(medium_base_out[..., 6:] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )

        outputs[SeathruHeadNames.MEDIUM_RGB] = medium_rgb
        outputs[SeathruHeadNames.MEDIUM_BS] = medium_bs
        outputs[SeathruHeadNames.MEDIUM_ATTN] = medium_attn

        return outputs
