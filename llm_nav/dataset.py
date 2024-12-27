import copy
import os
from dataclasses import dataclass
import json
import logging
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn

import transformers
from transformers import CLIPImageProcessor
from torch.utils.data import Dataset
import cv2
from PIL import Image
from functools import lru_cache

from llm_nav import conversation as conversation_lib

IGNORE_INDEX = -100

HEADING_FORWARD_KEY_WORD = "Front"
HEADING_LEFT_KEY_WORD = "Left"
HEADING_RIGHT_KEY_WORD = "Right"
STOP_KEY_WORD = "Stop"


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers, answer_token_id):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == conversation_lib.default_conversation.roles[0]:
            target[cur_idx:cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            label_idx = cur_idx
            while (label_idx < len(target) and target[label_idx] != answer_token_id):
                target[label_idx] = IGNORE_INDEX
                label_idx += 1
        cur_idx += tokenized_len
    target[target == answer_token_id] = IGNORE_INDEX


def _add_speaker_and_signal(header, source, concat_mode=False):
    """Add speaker and start/end signal on each round."""
    END_SIGNAL = "<|endofchunk|>"
    SEP = conversation_lib.default_conversation.sep
    conversation = f"{SEP}" if concat_mode else header
    for i, sentence in enumerate(source):
        from_str = sentence["from"]
        if from_str.lower() == "user":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        if from_str.lower() == "gpt":
            if "num_images" in sentence:
                from_str = '<image>' + from_str
            sentence["value"] = (from_str + ":<answer>" +
                                 sentence["value"] + END_SIGNAL + SEP)
        else:
            for i in range(sentence["num_images"]):
                from_str = '<image>' + from_str
            sentence["value"] = (from_str + ": " +
                                 sentence["value"] + SEP)
        conversation += sentence["value"]
    return conversation


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        inference_mode: bool = False,
        concat_mode: bool = False,
        task: str = "instruction_following",
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]

    conversations = []
    if task == "pretraining":
        header = f"{conversation_lib.default_conversation.system}"
    elif task == "instruction_following":
        header = f"{conversation_lib.default_conversation.system}"
    else:
        raise ValueError(f"Task {task} not supported.")
    for source in sources:
        conversation = _add_speaker_and_signal(header, source, concat_mode=concat_mode)
        if inference_mode:
            conversation += conversation_lib.default_conversation.roles[1] + ":<answer>"
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]

    if concat_mode:
        input_ids = [ids[1:] for ids in input_ids]

    targets = copy.deepcopy(input_ids)

    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        tokenized_lens = [length - 1 if i > 0 else length for i, length in enumerate(tokenized_lens)]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers, answer_token_id)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, img_db: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 store_feature: bool, inference_mode: bool, task: str = None, data_size: int = None):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))
        if data_size:
            list_data_dict = list_data_dict[:data_size]

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.image_processor = CLIPImageProcessor()
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.task = task
        self.store_feature = store_feature
        self.inference_mode = inference_mode
        if store_feature:
            self.ft_dir = img_db
        self.get_image_feature = lru_cache(maxsize=4096)(self._get_image_feature_uncached)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.list_data_dict[i]
        if 'images' in source:
            images = []
            for image_id in source['images']:
                if self.store_feature:
                    image = self.get_image_feature(image_id)
                else:
                    image = self.get_raw_image(image_id)
                images.append(image)

        conversations = copy.deepcopy(source['conversations'])
        data_dict = preprocess(
            [conversations],
            self.tokenizer,
            inference_mode=self.inference_mode,
            task=self.task)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'images' in source:
            if self.store_feature:
                data_dict['images_ft'] = images
            else:
                data_dict['images'] = images
        return data_dict

    def _get_image_feature_uncached(self, image_id):
        feature_path = os.path.join(self.ft_dir, f"{image_id}.pt")
        ft = torch.load(feature_path)
        return ft

    def get_raw_image(self, image_id):
        if image_id in self._image_cache:
            image = self._image_cache[image_id]
        else:
            with self.env.begin() as txn:
                img_bytes = txn.get(image_id.encode('ascii'))
            image_flt = np.frombuffer(img_bytes, dtype=np.uint8)
            image_flt = cv2.imdecode(image_flt, cv2.IMREAD_COLOR)
            image = image_flt.reshape(1500, 1500, 3)
            image = Image.fromarray(image)
            image = self.image_processor(image)["pixel_values"][0]
            self._image_cache[image_id] = image
        return image


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_processor = CLIPImageProcessor()

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            lang_x=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'images' in instances[0]:
            images_per_example = max(len(instance['images']) for instance in instances)
            batch_images = None
            for iexample, example in enumerate(instances):
                for iimage, image in enumerate(example['images']):
                    preprocessed = torch.from_numpy(image)
                    if batch_images is None:
                        batch_images = torch.zeros(
                            (len(instances), images_per_example, 1) + preprocessed.shape,
                            dtype=preprocessed.dtype,
                        )
                    batch_images[iexample, iimage, 0] = preprocessed
            batch['vision_x'] = batch_images
        elif 'images_ft' in instances[0]:
            batch['vision_x'] = [torch.stack(instance["images_ft"]) for instance in instances]
            batch['vision_x'] = nn.utils.rnn.pad_sequence(batch['vision_x'], batch_first=True)
            batch['vision_x'] = batch['vision_x'].unsqueeze(2)

        if 'gt_actions' in instances[0]:
            batch['gt_actions'] = [instance['gt_actions'] for instance in instances]

        if 'rl_input_ids' in instances[0]:
            rl_input_ids = [instance['rl_input_ids'] for instance in instances]
            rl_input_ids = torch.nn.utils.rnn.pad_sequence(rl_input_ids, batch_first=True,
                                                           padding_value=self.tokenizer.pad_token_id)
            batch['rl_lang_x'] = rl_input_ids
            batch['rl_attention_mask'] = rl_input_ids.ne(self.tokenizer.pad_token_id)

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.task == 'instruction_following':
        train_data_path = data_args.train_if_data_path
        eval_data_path = data_args.eval_if_data_path
    elif data_args.task == 'pretraining':
        train_data_path = data_args.train_pre_data_path
        eval_data_path = data_args.eval_pre_data_path
    else:
        raise Exception
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=train_data_path,
                                          img_db=data_args.img_db,
                                          task=data_args.task,
                                          store_feature=data_args.store_feature,
                                          inference_mode=False)
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                         data_path=eval_data_path,
                                         img_db=data_args.img_db,
                                         task=data_args.task,
                                         data_size=data_args.eval_data_size,
                                         store_feature=data_args.store_feature,
                                         inference_mode=False)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
