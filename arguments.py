from dataclasses import dataclass, field
from typing import Literal, Optional
import transformers


@dataclass
class ModelArguments:
    checkpoint_path: Optional[str] = field(default="")
    only_attend_immediate_media: bool = field(default=True)
    model_path: Optional[str] = field(default="xyz9911/FLAME-init")

@dataclass
class DataArguments:
    legacy_navigation_mode: bool = field(default=True)
    train_if_data_path: str = field(default="")
    eval_if_data_path: str = field(default="")
    eval_split: str = field(default="dev")
    dataset: str = field(default="")
    img_db: str = field(default="/dataset/touchdown_feature")
    store_feature: bool = field(default=True)
    task: Literal['instruction_following', 'pretraining'] = 'instruction_following'
    lazy_preprocess: bool = True
    batch_size: Optional[int] = field(default=64)
    micro_batch_size: Optional[int] = field(default=1)
    env_batch_size: Optional[int] = field(default=4)
    eval_data_size: int = 128
    max_route_len: int = 60
    temperature: float = 0.0
    decoding_paths: int = 1


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default='checkpoints')
    learning_rate: float = field(default=1e-4)
    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=True)
    # fp16: bool = field(default=True)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    eval_loss_only: bool = field(default=True)
    warmup_ratio: Optional[int] = field(default=0.01)
    num_train_epochs: Optional[int] = field(default=50)
    save_strategy: Optional[str] = field(default='steps')
    evaluation_strategy: Optional[str] = field(default='steps')
    eval_steps: Optional[int] = field(default=100)
    save_steps: Optional[int] = field(default=100)
    lr_scheduler_type: Optional[str] = field(default='cosine')
    report_to: Optional[str] = field(default='wandb')
    wandb_project: Optional[str] = field(default='flame')
