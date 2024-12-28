import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"

import transformers

from llm_nav.trainer import FlameTrainer
from llm_nav.config import FlamingoConfig
from llm_nav.model.modeling_flamingo import FlamingoForConditionalGeneration
from arguments import ModelArguments, DataArguments, TrainingArguments
from llm_nav.dataset import make_supervised_data_module


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    gradient_accumulation_steps = data_args.batch_size // data_args.micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    print('ddp:', ddp)
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    model_config = FlamingoConfig.from_pretrained(model_args.model_path)
    model_config.only_attend_immediate_media = model_args.only_attend_immediate_media
    model_config.feature_as_input = data_args.store_feature
    model_name = "llama" if "llama" in model_args.model_path.lower() else "mpt"
    model = FlamingoForConditionalGeneration.from_pretrained(
        model_args.model_path,
        config=model_config
    )
    model.lang_encoder.config.vocab_size = len(model.text_tokenizer)
    tokenizer = model.text_tokenizer
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_args.batch_size = training_args.per_device_train_batch_size
    split = data_args.train_if_data_path.split('/')
    split_name = split[1] + '_' + split[2]
    task_name = data_args.task if data_args.task != 'instruction_following' else split_name

    training_args = transformers.TrainingArguments(
        remove_unused_columns=False,
        per_device_train_batch_size=data_args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=f'{training_args.output_dir}/ds_{task_name}_{model_name}_lr{training_args.learning_rate}',
        warmup_ratio=training_args.warmup_ratio,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        bf16=training_args.bf16,
        tf32=training_args.tf32,
        optim=training_args.optim,
        lr_scheduler_type=training_args.lr_scheduler_type,
        evaluation_strategy=training_args.evaluation_strategy,
        save_strategy=training_args.save_strategy,
        eval_steps=training_args.eval_steps,
        metric_for_best_model="TC",
        save_total_limit=25,
        save_steps=training_args.save_steps,
        report_to=training_args.report_to,
        deepspeed='ds_zero1_config.json'
    )

    trainer = FlameTrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)
    trainer.train()


if __name__ == "__main__":
    train()
