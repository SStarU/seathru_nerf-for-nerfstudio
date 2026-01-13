import copy

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from seathru.seathru_datamanager import SeathruDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from seathru.seathru_model import SeathruModelConfig


def _clone_method(
    base: MethodSpecification,
    *,
    method_name: str,
    description: str,
    use_medium_c1: bool,
    use_plan_b: bool = False,
) -> MethodSpecification:
    """Deep-copy a MethodSpecification and override C1/Plan-B toggles."""
    spec = copy.deepcopy(base)
    spec.description = description
    spec.config.method_name = method_name

    model_cfg = spec.config.pipeline.model
    model_cfg.use_medium_c1 = use_medium_c1
    model_cfg.use_plan_b = use_plan_b

    return spec


# Base method configuration
seathru_method = MethodSpecification(
    config=TrainerConfig(
        method_name="seathru-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=SeathruDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                use_clean_supervision=True,
                use_depth_supervision=True,
                #images_on_gpu=True,
            ),
            model=SeathruModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_medium_c1=False,
                num_cameras=0,
                max_num_cameras=8192,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "camera_opt": {
                "mode": "off",
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=6e-6, max_steps=500000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SeaThru-NeRF for underwater scenes.",
)

seathru_method_c1 = _clone_method(
    seathru_method,
    method_name="seathru-nerf-c1",
    description="SeaThru-NeRF for underwater scenes (C1 on).",
    use_medium_c1=True,
)

seathru_method_c1b = _clone_method(
    seathru_method,
    method_name="seathru-nerf-c1b",
    description="SeaThru-NeRF for underwater scenes (C1 + plan B).",
    use_medium_c1=True,
    use_plan_b=True,
)

# Lite method configuration
seathru_method_lite = MethodSpecification(
    config=TrainerConfig(
        method_name="seathru-nerf-lite",
        steps_per_eval_batch=1000,#500其实偏频繁，会拖慢
        steps_per_save=5000,#2000有点勤快
        max_num_iterations=30000,#50k
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=SeathruDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=8192,#8192
                eval_num_rays_per_batch=4096,
                use_clean_supervision=True,
                use_depth_supervision=True,
                #images_on_gpu=True,
            ),
            model=SeathruModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_nerf_samples_per_ray=64,
                num_proposal_samples_per_ray=(256, 128),
                max_res=2048,
                log2_hashmap_size=19,
                hidden_dim=64,
                bottleneck_dim=31,
                hidden_dim_colour=64,
                hidden_dim_medium=64,
                proposal_net_args_list=[
                    {
                        "hidden_dim": 16,
                        "log2_hashmap_size": 17,
                        "num_levels": 5,
                        "max_res": 128,
                        "use_linear": False,
                    },
                    {
                        "hidden_dim": 16,
                        "log2_hashmap_size": 17,
                        "num_levels": 5,
                        "max_res": 256,
                        "use_linear": False,
                    },
                ],
                use_medium_c1=False,#TrueFalse
                num_cameras=0,
                max_num_cameras=8192,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "camera_opt": {
                "mode": "off",
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=6e-6, max_steps=500000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Light SeaThru-NeRF for underwater scenes.",
)

seathru_method_lite_c1 = _clone_method(
    seathru_method_lite,
    method_name="seathru-nerf-lite-c1",
    description="Light SeaThru-NeRF for underwater scenes (C1 on).",
    use_medium_c1=True,
)

seathru_method_lite_c1b = _clone_method(
    seathru_method_lite,
    method_name="seathru-nerf-lite-c1b",
    description="Light SeaThru-NeRF for underwater scenes (C1 + plan B).",
    use_medium_c1=True,
    use_plan_b=True,
)
