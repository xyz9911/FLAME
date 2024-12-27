from llm_nav.agent import run_navigation
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging
import json
import os
import torch.nn as nn
from transformers import Trainer
import transformers
from arguments import ModelArguments, DataArguments, TrainingArguments
from llm_nav.sim.env import TouchdownBatch
import shutil
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.get_logger(__name__)

parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

split = data_args.dataset.split('/')
split_name = split[1]
eval_env = TouchdownBatch(data_args, splits=[data_args.eval_split], name=split_name)


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class FlameTrainer(Trainer):
    def __init__(
            self,
            model=None,
            args=None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            preprocess_logits_for_metrics=None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        eval_loop_output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys,
                                                   metric_key_prefix)

        if data_args.task == 'instruction_following':
            epoch = 'finished'
            if metric_key_prefix == 'eval':
                epoch = self.state.epoch
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            tokenizer = self.data_collator.tokenizer
            metrics, trajs_record = run_navigation(eval_env, model, tokenizer, data_args)
            results_file = os.path.join(self.args.output_dir, f'{metric_key_prefix}_results_{epoch}.json')
            with open(results_file, 'w') as f:
                json.dump({'metrics': metrics, 'trajs': trajs_record}, f, indent=2)
            for key, value in metrics.items():
                eval_loop_output.metrics[f'{metric_key_prefix}_{key}'] = value
        return eval_loop_output

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if
                                "bias" not in name and "ff_gate" not in name and "attn_gate" not in name and "norm" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        for item in os.listdir(output_dir):
            if item.startswith('global_step') and os.path.isdir(os.path.join(output_dir, item)):
                shutil.rmtree(os.path.join(output_dir, item))
