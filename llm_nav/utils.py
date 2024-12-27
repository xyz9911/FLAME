from typing import Dict
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import json


def load_datasets(data_path, eval_split):
    data = []
    if 'touchdown' in data_path:
        dataset = 'touchdown'
    else:
        dataset = 'map2seq'
    if 'unseen' in data_path:
        seen = 'unseen'
    else:
        seen = 'seen'
    if eval_split:
        split = eval_split
    else:
        if 'dev' in data_path:
            split = 'dev'
        else:
            split = 'test'
    with open('dataset/%s/%s/data/%s.json' % (dataset, seen, split)) as f:
        for line in f:
            data.append(json.loads(line))
    return data

class LazySupervisedDatasetForEvaluation(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, data_size: int = None, eval_split: str = None):
        super(LazySupervisedDatasetForEvaluation, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = load_datasets(data_path, eval_split)
        list_data_dict = json.load(open(data_path, 'r'))
        if data_size:
            list_data_dict = list_data_dict[:data_size]

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        data_dict = {'route_ids': [item['route_id'] if 'route_id' in item else item['id'] for item in sources]}
        return data_dict


@dataclass
class DataCollatorForSupervisedDatasetEvaluation(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        batch = dict(
            route_ids=[item for instance in instances for item in instance['route_ids']]
        )
        # batch['route_ids'] = [item for instance in instances for item in instance['route_ids']]
        return batch


def make_evaluation_data_module(opts):
    eval_dataset = LazySupervisedDatasetForEvaluation(
        data_path=opts.eval_if_data_path,
        data_size=opts.eval_data_size,
        eval_split=opts.eval_split)
    eval_data_collator = DataCollatorForSupervisedDatasetEvaluation()
    dataloader = DataLoader(eval_dataset, batch_size=opts.env_batch_size, collate_fn=eval_data_collator)
    return dataloader
