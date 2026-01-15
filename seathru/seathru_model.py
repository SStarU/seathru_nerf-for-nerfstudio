from dataclasses import dataclass, field
from typing import Dict, List, Type, Literal, Tuple

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.renderers import AccumulationRenderer
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss, interlevel_loss
from nerfstudio.utils import colormaps
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)


from seathru.seathru_field import SeathruField
from seathru.seathru_fieldheadnames import SeathruHeadNames
from seathru.seathru_renderers import SeathruRGBRenderer
from seathru.seathru_losses import acc_loss, recon_loss
from seathru.seathru_utils import get_bayer_mask, save_debug_info, get_transmittance
from seathru.seathru_renderers import SeathruDepthRenderer


@dataclass
class SeathruModelConfig(ModelConfig):
    """SeaThru-NeRF Config."""

    _target: Type = field(default_factory=lambda: SeathruModel)
    near_plane: float = 0.05
    """Near plane of rays."""
    far_plane: float = 10.0
    """Far plane of rays."""
    num_levels: int = 16
    """Number of levels of the hashmap for the object base MLP."""
    min_res: int = 16
    """Minimum resolution of the hashmap for the object base MLP."""
    max_res: int = 8192
    """Maximum resolution of the hashmap for the object base MLP."""
    log2_hashmap_size: int = 21
    """Size of the hashmap for the object base MLP."""
    features_per_level: int = 2
    """Number of features per level of the hashmap for the object base MLP."""
    num_layers: int = 2
    """Number of hidden layers for the object base MLP."""
    hidden_dim: int = 256
    """Dimension of hidden layers for the object base MLP."""
    bottleneck_dim: int = 63
    """Bottleneck dimension between object base MLP and object head MLP."""
    num_layers_colour: int = 3
    """Number of hidden layers for colour MLP."""
    hidden_dim_colour: int = 256
    """Dimension of hidden layers for colour MLP."""
    num_layers_medium: int = 2
    """Number of hidden layers for medium MLP."""
    hidden_dim_medium: int = 128
    """Dimension of hidden layers for medium MLP."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Implementation of the MLPs (tcnn or torch)."""
    use_viewing_dir_obj_rgb: bool = False
    """Whether to use viewing direction in object rgb MLP."""
    object_density_bias: float = 0.0
    """Bias for object density."""
    medium_density_bias: float = 0.0
    """Bias for medium density (sigma_bs and sigma_attn)."""
    use_medium_c1: bool = False
    """Whether to use global + per-image medium pre-activation bias (C1)."""
    use_plan_b: bool = False
    """Whether to enable plan B variant for SeaThru."""
    lambda_clean: float = 0.003
    """Weight for clean RGB supervision loss."""
    lambda_depth: float = 0.005
    """Weight for depth supervision loss."""
    depth_loss_type: Literal["l1", "scale_shift_invariant"] = "scale_shift_invariant"
    """Depth loss type when depth supervision is available."""
    log_depth_stats: bool = True
    """Whether to log depth alignment error stats to metrics."""
    aux_ramp_start: int = 3000
    """Start step for ramping auxiliary supervision weights."""
    aux_ramp_end: int = 15000
    """End step for ramping auxiliary supervision weights."""
    acc_mask_thresh: float = 0.1
    """Threshold for gating depth supervision by NeRF accumulation."""
    keep_depth: float = 0.3
    """Target keep ratio for depth supervision by accumulation."""
    keep_clean: float = 0.5
    """Target keep ratio for clean supervision by accumulation."""
    peak_quantile: float = 0.7
    """Quantile for peak gating in depth/clean supervision."""
    peak_mask_fixed_thr: float = 0.02
    """Fixed threshold for peak gating when batch is tiny."""
    peak_min_batch: int = 256
    """Minimum batch size to use quantile-based peak gating."""
    medium_delta_dim: int = 8
    """Embedding dimension for per-image medium residuals."""
    lambda_delta: float = 1e-3
    """L2 regularization weight for per-image medium residuals."""
    lambda_zm: float = 1e-2
    """Zero-mean regularization weight for per-image medium residuals."""
    num_cameras: int = 0
    """Number of cameras/images for per-image medium residuals. 0 = auto."""
    max_num_cameras: int = 8192
    """Fallback number of cameras/images when auto-detection fails."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 128)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup."""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps."""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Whether to use the same proposal network."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 512,
                "use_linear": False,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 7,
                "max_res": 2048,
                "use_linear": False,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing (this gives an exploration at the \
        beginning of training)."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 15000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to \
        the camera."""
    initial_acc_loss_mult: float = 0.0001
    """Initial accuracy loss multiplier."""
    final_acc_loss_mult: float = 0.0001
    """Final accuracy loss multiplier."""
    acc_decay: int = 10000
    """Decay of the accuracy loss multiplier. (After this many steps, acc_loss_mult = \
        final_acc_loss_mult.)"""
    rgb_loss_use_bayer_mask: bool = False
    """Whether to use a Bayer mask for the RGB loss."""
    prior_on: Literal["weights", "transmittance"] = "transmittance"
    """Prior on the proposal weights or transmittance."""
    debug: bool = False
    """Whether to save debug information."""
    beta_prior: float = 100.0
    """Beta hyperparameter for the prior used in the acc_loss."""
    use_viewing_dir_obj_rgb: bool = False
    """Whether to use viewing direction in object rgb MLP."""
    use_new_rendering_eqs: bool = True
    """Whether to use the new rendering equations."""


class SeathruModel(Model):
    """Seathru model

    Args:
        config: SeaThru-NeRF configuration to instantiate the model with.
    """

    config: SeathruModelConfig  # type: ignore

    def populate_modules(self):
        """Setup the fields and modules."""
        super().populate_modules()
        self.use_plan_b = getattr(self.config, "use_plan_b", False)

        # Scene contraction
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Initialize SeaThru field
        num_cameras = 0
        if self.config.use_medium_c1:
            if self.config.num_cameras and self.config.num_cameras > 0:
                num_cameras = int(self.config.num_cameras)
            else:
                # Try to infer from available attributes across nerfstudio versions.
                for name in ["num_train_data", "num_eval_data", "num_cameras", "num_images"]:
                    v = getattr(self, name, None)
                    if isinstance(v, int) and v > 0:
                        num_cameras = int(v)
                        break
                if num_cameras <= 0:
                    num_cameras = int(self.config.max_num_cameras)
        self.field = SeathruField(
            aabb=self.scene_box.aabb,
            num_levels=self.config.num_levels,
            min_res=self.config.min_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            bottleneck_dim=self.config.bottleneck_dim,
            num_layers_colour=self.config.num_layers_colour,
            hidden_dim_colour=self.config.hidden_dim_colour,
            num_layers_medium=self.config.num_layers_medium,
            hidden_dim_medium=self.config.hidden_dim_medium,
            spatial_distortion=scene_contraction,
            implementation=self.config.implementation,
            use_viewing_dir_obj_rgb=self.config.use_viewing_dir_obj_rgb,
            object_density_bias=self.config.object_density_bias,
            medium_density_bias=self.config.medium_density_bias,
            num_cameras=num_cameras if self.config.use_medium_c1 else 0,
            use_medium_c1=self.config.use_medium_c1,
            medium_delta_dim=self.config.medium_delta_dim,
        )

        # Initialize proposal network(s) (this code snippet is taken from from nerfacto)
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert (
                len(self.config.proposal_net_args_list) == 1
            ), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )

        def update_schedule(step):
            return np.clip(
                np.interp(
                    step,
                    [0, self.config.proposal_warmup],
                    [0, self.config.proposal_update_every],
                ),
                1,
                self.config.proposal_update_every,
            )

        # Initial sampler
        initial_sampler = None  # None is for piecewise as default
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(
                single_jitter=self.config.use_single_jitter
            )

        # Proposal sampler
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )

        # Renderers
        self.renderer_rgb = SeathruRGBRenderer(
            use_new_rendering_eqs=self.config.use_new_rendering_eqs
        )
        self.renderer_depth = SeathruDepthRenderer(
            far_plane=self.config.far_plane, method="median"
        )
        self.renderer_accumulation = AccumulationRenderer()

        # Losses
        self.rgb_loss = MSELoss(reduction="none")

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Step member variable to keep track of the training step
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the parameter groups for the optimizer. (Code snippet from nerfacto)

        Returns:
            The parameter groups.
        """
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def step_cb(self, step) -> None:
        """Function for training callbacks to use to update training step.

        Args:
            step: The training step.
        """
        self.step = step

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Get the training callbacks.
           (Code of this function is from nerfacto but added step tracking for debugging.)

        Args:
            training_callback_attributes: The training callback attributes.

        Returns:
            List with training callbacks.
        """
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        # Additional callback to track the training step for decaying and
        # debugging purposes
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.step_cb,
            )
        )

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:  # type: ignore
        """Get outputs from the model.

        Args:
            ray_bundle: RayBundle containing the input rays to compute and render.

        Returns:
            Dict containing the outputs of the model.
        """

        ray_samples: RaySamples

        # Get output from proposal network(s)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )

        # Get output from Seathru field
        field_outputs = self.field.forward(ray_samples)
        field_outputs[FieldHeadNames.DENSITY] = torch.nan_to_num(
            field_outputs[FieldHeadNames.DENSITY], nan=1e-3
        )
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        if self.training and self.step == 0:
            w = weights.detach()
            print(
                "[DEBUG weights] shape",
                tuple(w.shape),
                "min",
                float(w.min()),
                "max",
                float(w.max()),
                "sum_ray_p50",
                float(w.sum(dim=1).median()),
                "sum_ray_mean",
                float(w.sum(dim=1).mean()),
            )
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        # Render rgb (only rgb in training and rgb, direct, bs, J in eval for
        # performance reasons as we do not optimize with respect to direct, bs, J)
        # ignore types to avoid unnecesarry pyright errors
        if self.training or not self.config.use_new_rendering_eqs:
            rgb = self.renderer_rgb(
                object_rgb=field_outputs[FieldHeadNames.RGB],
                medium_rgb=field_outputs[SeathruHeadNames.MEDIUM_RGB],  # type: ignore
                medium_bs=field_outputs[SeathruHeadNames.MEDIUM_BS],  # type: ignore
                medium_attn=field_outputs[SeathruHeadNames.MEDIUM_ATTN],  # type: ignore
                densities=field_outputs[FieldHeadNames.DENSITY],
                weights=weights,
                ray_samples=ray_samples,
            )
            direct = None
            bs = None
            J = None
        else:
            rgb, direct, bs, J = self.renderer_rgb(
                object_rgb=field_outputs[FieldHeadNames.RGB],
                medium_rgb=field_outputs[SeathruHeadNames.MEDIUM_RGB],  # type: ignore
                medium_bs=field_outputs[SeathruHeadNames.MEDIUM_BS],  # type: ignore
                medium_attn=field_outputs[SeathruHeadNames.MEDIUM_ATTN],  # type: ignore
                densities=field_outputs[FieldHeadNames.DENSITY],
                weights=weights,
                ray_samples=ray_samples,
            )

        # Render clean RGB (object only) and expected depth
        weights_detached = weights.detach()
        rgb_clean = torch.sum(weights_detached * field_outputs[FieldHeadNames.RGB], dim=1)
        t_mid = (ray_samples.frustums.starts + ray_samples.frustums.ends) * 0.5
        depth = torch.sum(weights * t_mid, dim=-2).to(torch.float32)
        acc = torch.sum(weights, dim=1).to(torch.float32)
        w_peak = weights.max(dim=1).values.to(torch.float32)

        # Render accumulation
        accumulation = self.renderer_accumulation(weights=weights)

        # Calculate transmittance and add to outputs for acc_loss calculation
        # Ignore type error that occurs because ray_samples can be initialized without deltas
        transmittance = get_transmittance(
            ray_samples.deltas, field_outputs[FieldHeadNames.DENSITY]  # type: ignore
        )
        outputs = {
            "rgb": rgb,
            "depth": depth,
            "rgb_clean": rgb_clean,
            "acc": acc,
            "w_peak": w_peak,
            "accumulation": accumulation,
            "transmittance": transmittance,
            "weights": weights,
            "direct": direct if not self.training else None,
            "bs": bs if not self.training else None,
            "J": J if not self.training else None,
        }

        # Add outputs from proposal network(s) to outputs if training for proposal loss
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        # Add proposed depth to outputs
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        return outputs

    def get_metrics_dict(self, outputs, batch):
        """Get evaluation metrics dictionary.
        (Compared to get_image_metrics_and_images(), this function does not render
        images and is executed at each training step.)

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Dict containing the metrics to log.
        """
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Calculate loss dictionary.

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Dict containing the loss values.
        """
        loss_dict = {}
        image = batch["image"].to(self.device)

        def _log_metric(name: str, value: torch.Tensor) -> None:
            if metrics_dict is not None:
                metrics_dict[name] = value.detach()
            else:
                loss_dict[name] = value

        start = self.config.aux_ramp_start
        end = self.config.aux_ramp_end
        if self.step <= start:
            w_aux = 0.0
        elif self.step >= end:
            w_aux = 1.0
        else:
            w_aux = float(self.step - start) / float(end - start)
        loss_dict["aux_weight"] = torch.tensor(w_aux, device=self.device)

        peak_mask = None
        if "w_peak" in outputs:
            peak = outputs["w_peak"].detach().view(-1)
            if peak.numel() > 0:
                peak_min = peak.min()
                peak_max = peak.max()
                if peak.numel() < self.config.peak_min_batch:
                    peak_thr = torch.as_tensor(
                        self.config.peak_mask_fixed_thr,
                        device=peak.device,
                        dtype=peak.dtype,
                    )
                    peak_mask = peak > peak_thr
                    peak_thr_mode = 0
                elif (peak_max - peak_min) < 1e-6:
                    peak_thr = peak_min
                    peak_mask = peak >= peak_thr
                    peak_thr_mode = 1
                else:
                    peak_thr = torch.quantile(peak, self.config.peak_quantile)
                    peak_mask = peak > peak_thr
                    if peak_mask.float().mean().item() == 0.0:
                        k = max(1, int(0.3 * peak.numel()))
                        topk_idx = torch.topk(peak, k, largest=True).indices
                        peak_mask = torch.zeros_like(peak, dtype=torch.bool)
                        peak_mask[topk_idx] = True
                        peak_thr_mode = 2
                    else:
                        peak_thr_mode = 3
                loss_dict["peak_thr"] = peak_thr
                loss_dict["peak_gate_ratio"] = peak_mask.float().mean()
                loss_dict["peak_p50"] = peak.median()
                loss_dict["peak_mean"] = peak.mean()
                loss_dict["peak_thr_mode"] = torch.as_tensor(
                    peak_thr_mode, device=peak.device, dtype=peak.dtype
                )

        # RGB loss
        if self.config.rgb_loss_use_bayer_mask:
            # Cut out camera/image indices and pass to get_bayer_mask
            bayer_mask = get_bayer_mask(batch["indices"][:, 1:].to(self.device))
            squared_error = self.rgb_loss(image, outputs["rgb"])  # clip or not clip?
            scaling_grad = 1 / (outputs["rgb"].detach() + 1e-3)
            loss = squared_error * torch.square(scaling_grad)
            denom = torch.sum(bayer_mask)
            loss_dict["rgb_loss"] = torch.sum(loss * bayer_mask) / denom
        else:
            loss_dict["rgb_loss"] = recon_loss(gt=image, pred=outputs["rgb"])

        # Clean RGB supervision (optional)
        if "clean_image" in batch and "rgb_clean" in outputs and "acc" in outputs:
            acc_flat = outputs["acc"].detach().view(-1)
            if acc_flat.numel() < 256:
                thr_clean = torch.as_tensor(
                    self.config.acc_mask_thresh, device=acc_flat.device, dtype=acc_flat.dtype
                )
            else:
                thr_clean = torch.quantile(acc_flat, 1.0 - self.config.keep_clean)
            acc_mask_clean = acc_flat > thr_clean
            if "depth_mask" in batch:
                final_mask = acc_mask_clean.view(-1, 1) & batch["depth_mask"].to(self.device).bool()
            else:
                final_mask = acc_mask_clean.view(-1, 1)
            if peak_mask is not None:
                final_mask = final_mask & peak_mask.view(-1, 1)
            loss_dict["acc_thr_clean"] = thr_clean
            loss_dict["acc_gate_ratio_clean"] = acc_mask_clean.float().mean()
            loss_dict["clean_valid_ratio"] = final_mask.float().mean()
            valid_mask = final_mask.view(-1)
            if valid_mask.sum().item() >= 64:
                clean_pred = outputs["rgb_clean"][valid_mask]
                clean_gt = batch["clean_image"].to(self.device)[valid_mask]
                clean_loss_raw = recon_loss(gt=clean_gt, pred=clean_pred)
                loss_dict["clean_loss"] = self.config.lambda_clean * clean_loss_raw
                loss_dict["clean_loss"] *= w_aux
                _log_metric("clean_loss_raw", clean_loss_raw)

        # Depth supervision (optional)
        if (
            "depth" in batch
            and "depth_mask" in batch
            and "depth" in outputs
            and "acc" in outputs
        ):
            pred = outputs["depth"].to(self.device)
            gt = batch["depth"].to(self.device)
            base_mask = batch["depth_mask"].to(self.device).bool()
            acc = outputs["acc"].detach().view(-1)
            if acc.numel() < 256:
                thr_depth = torch.as_tensor(
                    self.config.acc_mask_thresh, device=acc.device, dtype=acc.dtype
                )
            else:
                thr_depth = torch.quantile(acc, 1.0 - self.config.keep_depth)
            acc_mask = acc > thr_depth
            final_mask = base_mask & acc_mask.view(-1, 1)
            if peak_mask is not None:
                final_mask = final_mask & peak_mask.view(-1, 1)
            loss_dict["acc_min"] = acc.min()
            loss_dict["acc_max"] = acc.max()
            loss_dict["acc_p50"] = acc.median()
            loss_dict["acc_gate_ratio_depth"] = acc_mask.float().mean()
            loss_dict["acc_thr_depth"] = thr_depth
            loss_dict["depth_valid_ratio"] = final_mask.float().mean()
            loss_dict["acc_mean"] = outputs["acc"].mean()
            valid_mask = final_mask.view(-1)
            if valid_mask.sum().item() >= 64:
                try:
                    p = pred.view(-1)[valid_mask]
                    g = gt.view(-1)[valid_mask]
                    pd = p.detach()
                    gd = g.detach()
                    ones = torch.ones_like(pd)
                    A = torch.stack([pd, ones], dim=1)
                    try:
                        x = torch.linalg.lstsq(A, gd).solution
                    except Exception:
                        eps = 1e-6
                        AtA = A.T @ A
                        Atg = A.T @ gd
                        x = torch.linalg.solve(
                            AtA + eps * torch.eye(2, device=AtA.device, dtype=AtA.dtype),
                            Atg,
                        )
                    s = x[0]
                    t = x[1]
                    s_clamped = torch.clamp(s, 0.1, 10.0)
                    aligned = s_clamped * p + t
                    depth_err = torch.mean(torch.abs(aligned - g))
                    loss_dict["depth_err"] = depth_err
                    loss_dict["depth_scale"] = s_clamped
                    loss_dict["depth_shift"] = t
                    loss_dict["depth_loss"] = self.config.lambda_depth * depth_err
                    loss_dict["depth_loss"] *= w_aux
                except Exception:
                    pass

        if self.training:
            # Accumulation loss
            if self.step < self.config.acc_decay:
                acc_loss_mult = self.config.initial_acc_loss_mult
            else:
                acc_loss_mult = self.config.final_acc_loss_mult

            if self.config.prior_on == "weights":
                loss_dict["acc_loss"] = acc_loss_mult * acc_loss(
                    transmittance_object=outputs["weights"], beta=self.config.beta_prior
                )
            elif self.config.prior_on == "transmittance":
                loss_dict["acc_loss"] = acc_loss_mult * acc_loss(
                    transmittance_object=outputs["transmittance"],
                    beta=self.config.beta_prior,
                )
            else:
                raise ValueError(f"Unknown prior_on: {self.config.prior_on}")

            # Proposal loss
            loss_dict["interlevel_loss"] = (
                self.config.interlevel_loss_mult
                * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            )
            # ---- C1 regularization ----
            if self.config.use_medium_c1 and getattr(self.field, "use_medium_c1", False):
                cam_idx = batch["indices"][:, 0].to(self.device)
                if not hasattr(self, "_max_seen_camera_idx"):
                    self._max_seen_camera_idx = 0
                self._max_seen_camera_idx = max(
                    self._max_seen_camera_idx, int(cam_idx.max().item())
                )
                used_n = self._max_seen_camera_idx + 1
                delta9_used = self.field.medium_delta_proj(
                    self.field.medium_delta_embed.weight[:used_n]
                )
                loss_dict["medium_delta_l2"] = self.config.lambda_delta * (delta9_used ** 2).mean()
                loss_dict["medium_delta_zeromean"] = (
                    self.config.lambda_zm * (delta9_used.mean(dim=0) ** 2).sum()
                )
        # ---- C1 stats logging (delta_rms / global_rms) ----
        if self.config.use_medium_c1 and getattr(self.field, "use_medium_c1", False):
            # 1) global rms
            global_rms = torch.sqrt((self.field.medium_global_pre ** 2).mean())

            # 2) delta rms (only over used cameras)
            cam_idx = batch["indices"][:, 0].to(self.device)
            if not hasattr(self, "_max_seen_camera_idx"):
                self._max_seen_camera_idx = 0
            self._max_seen_camera_idx = max(
                self._max_seen_camera_idx, int(cam_idx.max().item())
            )
            used_n = self._max_seen_camera_idx + 1

            delta9_used = self.field.medium_delta_proj(
                self.field.medium_delta_embed.weight[:used_n]
            )
            delta_rms = torch.sqrt((delta9_used ** 2).mean())

            # Write into metrics_dict (viewer/tensorboard picks these up)
            if metrics_dict is not None:
                metrics_dict["global_rms"] = global_rms.detach()
                metrics_dict["delta_rms"] = delta_rms.detach()
        if self.training and self.step == 0:
            print("[DEBUG loss_dict keys]", sorted(loss_dict.keys()))
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Get evaluation metrics dictionary and images to log for eval batch.
        (extended from nerfacto)

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Tuple containing the metrics to log (as scalars) and the images to log.
        """
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]

        # Accumulation and depth maps
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Log the images
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        if self.config.use_new_rendering_eqs:
            # J (clean image), direct and bs images
            direct = outputs["direct"]
            bs = outputs["bs"]
            J = outputs["J"]

            combined_direct = torch.cat([direct], dim=1)
            combined_bs = torch.cat([bs], dim=1)
            combined_J = torch.cat([J], dim=1)

            # log the images
            images_dict["direct"] = combined_direct
            images_dict["bs"] = combined_bs
            images_dict["J"] = combined_J

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # Compute metrics
        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # Log the metrics (as scalars)
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        # Log the proposal depth maps
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(outputs[key])
            images_dict[key] = prop_depth_i

        # Debugging
        if self.config.debug:
            save_debug_info(
                weights=outputs["weights"],
                transmittance=outputs["transmittance"],
                depth=outputs["depth"],
                prop_depth=outputs["prop_depth_0"],
                step=self.step,
            )

        return metrics_dict, images_dict
