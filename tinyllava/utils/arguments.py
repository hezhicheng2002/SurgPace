"""Arguments for the SurgPace main training objective."""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class DataArguments:
    data_path: str = field(default="")
    image_folder: str = field(default="")
    image_aspect_ratio: str = field(default="square")
    teacher_image_size: int = field(default=1024)
    curriculum_score_field: str = field(default="score_min_conf_lorentz_midrank_task")


@dataclass
class TrainingArguments(HFTrainingArguments):
    output_dir: str = field(default="outputs/surgpace")
    pretrained_model_path: str = field(default="")
    teacher_model_path: str = field(default="")
    teacher_code_root: Optional[str] = field(default=None)
    resume_from_checkpoint: Optional[str] = field(default=None)
    model_max_length: int = field(default=3072)
    max_steps: int = field(default=105266)
    num_train_epochs: float = field(default=1.0)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    tf32: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=0)
    logging_steps: float = field(default=100)
    save_steps: float = field(default=21053)
    save_total_limit: int = field(default=2)
    report_to: str = field(default="none")
    seed: int = field(default=42)
    training_recipe: str = field(default="lora")
    tune_type_llm: str = field(default="lora")
    tune_type_vision_tower: str = field(default="frozen")
    tune_type_connector: str = field(default="full")
    tune_vision_tower_from_layer: int = field(default=-1)
    bits: int = field(default=16)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    distill_use_image_for_teacher: bool = field(default=True)
    distill_temperature: float = field(default=1.0)
    distill_ce_weight: float = field(default=0.5)
    distill_kl_weight: float = field(default=1.0)
    distill_tl_weight: float = field(default=0.5)
    distill_entropy_min_weight: float = field(default=0.05)
    distill_feat_align_weight: float = field(default=0.2)
    task_loss_weights: str = field(default="grounding:1.2")
    curriculum_min_weight: float = field(default=0.7)
    curriculum_max_weight: float = field(default=1.3)
    curriculum_end_ratio: float = field(default=0.4)
    curriculum_phase_ratios: str = field(default="0,1,0")
    curriculum_ce_start: float = field(default=1.0)
    curriculum_ce_end: float = field(default=1.0)
    curriculum_distill_start: float = field(default=1.0)
    curriculum_distill_end: float = field(default=0.1)
    surgpace_ddp_static_graph: bool = field(default=True)
