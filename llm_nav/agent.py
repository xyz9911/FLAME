import torch
from collections.abc import Mapping
from torch import nn
from tqdm import tqdm
import time

import copy
from llm_nav.dataset import preprocess
from llm_nav.utils import make_evaluation_data_module


class FLAME:
    def __init__(self, opts):
        self.max_route_len = opts.max_route_len
        self.use_feature = opts.store_feature
        self.legacy_navigation_mode = opts.legacy_navigation_mode
        self.temperature = opts.temperature
        self.decoding_paths = opts.decoding_paths
        self.autocast = torch.cuda.amp.autocast

    def _prepare_input(self, data, model):
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v, model) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v, model) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": model.device}
            return data.to(**kwargs, non_blocking=True)
        return data

    def _make_batch(self, conv_batch, tokenizer, input_ids=None):
        sources = copy.deepcopy([e["conversations"] for e in conv_batch])
        data_dict = preprocess(sources, tokenizer, inference_mode=True, concat_mode=input_ids is not None,
                               task="instruction_following")
        images_per_example = max(len(e['images']) for e in conv_batch)
        if self.use_feature:
            batch_images = [torch.stack(item['images']) for item in conv_batch]
            batch_images = nn.utils.rnn.pad_sequence(batch_images, batch_first=True)
            batch_images = batch_images.unsqueeze(2)
        else:
            batch_images = None
            for iexample, example in enumerate(conv_batch):
                for iimage, image in enumerate(example['images']):
                    preprocessed = torch.from_numpy(image)
                    if batch_images is None:
                        batch_images = torch.zeros(
                            (len(conv_batch), images_per_example, 1) + preprocessed.shape,
                            dtype=preprocessed.dtype,
                        )
                    batch_images[iexample, iimage, 0] = preprocessed
        reversed_input_ids = [torch.flip(ids, [0]) for ids in data_dict['input_ids']]
        padded_reversed_input_ids = torch.nn.utils.rnn.pad_sequence(
            reversed_input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        new_input_ids = torch.flip(padded_reversed_input_ids, [1])

        input_ids = torch.cat([input_ids, new_input_ids.to(input_ids.device)],
                              dim=1) if input_ids is not None else new_input_ids
        batch = dict(
            lang_x=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            vision_x=batch_images,
        )

        return batch

    def _get_action(self, response):
        if not response.endswith('.'):
            true_response = response.split('.')[-1]
        else:
            true_response = response.split('.')[-2]
        if 'stop' in true_response.lower():
            action = 'stop'
        elif 'turn around' in true_response.lower():
            action = 'turn_around'
        elif 'forward' in true_response.lower() or 'straight' in true_response.lower():
            action = 'forward'
        elif 'left' in true_response.lower():
            action = 'left'
        elif 'right' in true_response.lower():
            action = 'right'
        else:
            action = 'unk'

        return action

    def self_consistency(self, response_batch, outputs):
        actions = []
        for response in response_batch:
            action = self._get_action(response)
            actions.append(action)
        action_count = {action: 0 for action in actions}
        for i, action in enumerate(actions):
            action_count[action] += 1
        max_action = max(action_count, key=action_count.get)
        for i, action in enumerate(actions):
            if action == max_action:
                max_action_index = i
                break
        outputs['past_key_values'] = [(past_key_value[0][max_action_index].unsqueeze(0),
                                       past_key_value[1][max_action_index].unsqueeze(0)) for past_key_value in
                                      outputs['past_key_values']]
        outputs['sequences'] = outputs['sequences'][max_action_index].unsqueeze(0)
        return [response_batch[max_action_index]], outputs['sequences'], outputs['past_key_values']

    @torch.inference_mode()
    def rollout(self, env, model, tokenizer, route_ids=None):
        trajs = env.reset(route_ids=route_ids)  # a batch of the first panoid for each route_panoids
        batch_size = len(route_ids)
        ended = [0] * batch_size
        num_act_nav = [batch_size]
        total_steps = [0]
        conv_batch = [{"route_id": item['route_id'] if "route_id" in item else item['id']} for item in env.batch]

        if env.env.name.startswith("touchdown"):
            init_msg_legacy = "You are playing the role of an agent in a navigation task, and you will be providing navigation actions base on navigation text and visual observations. At the beginning of navigation, answer 'Turn around' to turn around and 'Head forward' to proceed. For each new observation, answer 'Head forward' to proceed and 'Stop' to end. At each intersection, answer 'Head forward' to proceed, 'Turn left' to turn left, 'Turn right' to turn right and 'Stop' to end."
            init_msg = "You are playing the role of an agent in a navigation task, and you will be providing navigation actions base on navigation text and visual observations. At the beginning of navigation, answer 'Turn around' to turn around and 'Forward' to proceed. For each new observation, answer 'Forward' to proceed and 'Stop' to end. At each intersection, you'll be asked to consider the next action (forward, left, right or stop)."
            begin_msg = "Turn around or head forward?"
            begin_msg_legacy = "Turn around or head forward?"

            user_msg = "Forward or stop?"
            user_msg_legacy = "Head forward or stop?"
        elif env.env.name.startswith("map2seq"):
            init_msg_legacy = "You are playing the role of an agent in a navigation task, and you will be providing navigation actions base on navigation text and visual observations. For each new observation, answer 'Head forward' to proceed and 'Stop' to end. At each intersection, answer 'Head forward' to proceed, 'Turn left' to turn left, 'Turn right' to turn right and 'Stop' to end."
            init_msg = "You are playing the role of an agent in a navigation task, and you will be providing navigation actions base on navigation text and visual observations. At the beginning of navigation, answer 'Turn around' to turn around and 'Forward' to proceed. For each new observation, answer 'Forward' to proceed and 'Stop' to end. At each intersection, you'll be asked to consider the next action (forward, left, right or stop)."
            begin_msg = "Forward or stop?"
            begin_msg_legacy = "Head forward or stop?"

            user_msg = "Forward or stop?"
            user_msg_legacy = "Head forward or stop?"
        else:
            raise ValueError(f"Unknown env name: {env.env.name}")

        input_ids = None
        past_key_values = None
        responses = [[] for _ in range(batch_size)]
        start_time = time.time()
        cnt = 0
        for step in range(self.max_route_len):
            observations = env.get_observations()
            for i in range(batch_size):
                obs = observations[i]
                viewpoint = obs['viewpoint']
                image = obs['image']

                if ended[i]:
                    conv_batch[i]['conversations'] = [{"from": "User", "value": "", "num_images": 0}]
                    continue
                if step == 0:
                    conv_batch[i]['obs'] = [viewpoint]
                else:
                    conv_batch[i]['obs'].append(viewpoint)
                if step == 0:
                    conv_batch[i]['images'] = [image]
                    if self.legacy_navigation_mode:
                        init = f"{init_msg_legacy} navigation_text: {env.batch[i]['navigation_text']}"
                        begin = begin_msg_legacy
                    else:
                        init = f"{init_msg} navigation_text: {env.batch[i]['navigation_text']}"
                        begin = begin_msg
                    conv_batch[i]['conversations'] = [{"from": "User", "value": init, "num_images": 0},
                                                      {"from": "User", "value": begin, "num_images": 1}]
                elif 'intersection' in obs:
                    vp = viewpoint[:viewpoint.rfind('_')]
                    neighbours_num = env.env.navs[i].graph.get_num_neighbors(vp)
                    conv_batch[i]['images'].append(image)
                    if self.legacy_navigation_mode:
                        user = f"You've arrived at a {neighbours_num}-way intersection. Head forward, turn left, turn right or stop?"
                    else:
                        user = f"You've arrived at a {neighbours_num}-way intersection. Consider your next move."
                    conv_batch[i]['conversations'] = [{"from": "User", "value": user, "num_images": 1}]
                else:
                    if self.legacy_navigation_mode:
                        user = user_msg_legacy
                    else:
                        user = user_msg
                    conv_batch[i]['images'].append(image)
                    conv_batch[i]['conversations'] = [{"from": "User", "value": user, "num_images": 1}]

            batch = self._make_batch(conv_batch, tokenizer, input_ids=input_ids)
            batch = self._prepare_input(batch, model)

            batch["past_key_values"] = past_key_values
            with self.autocast():
                if len(conv_batch) == 1 and 'intersection' in observations[0]:
                    outputs = model.generate_lightning(**batch, ended=ended, max_new_tokens=500,
                                                       num_return_sequences=self.decoding_paths,
                                                       temperature=self.temperature)
                else:
                    outputs = model.generate_lightning(**batch, ended=ended, max_new_tokens=500, num_return_sequences=1,
                                                       temperature=0.0)
            input_ids = outputs['sequences']
            response_batch = outputs['sequences'][:, len(batch["lang_x"][0]):]
            past_key_values = outputs['past_key_values']
            response_batch = tokenizer.batch_decode(response_batch, skip_special_tokens=True)

            if len(conv_batch) == 1 and (len(conv_batch) != len(response_batch)):
                response_batch, input_ids, past_key_values = self.self_consistency(response_batch, outputs)

            actions = []
            for i, response in enumerate(response_batch):
                if not ended[i]:
                    responses[i].append(response)
                action = self._get_action(response)
                if action == 'unk':
                    action = 'stop'
                actions.append(action)
            env.env.action_step(actions, ended, num_act_nav, trajs, total_steps)
            cnt += 1

            if not num_act_nav[0]:
                break

        end_time = time.time()
        execution_time = end_time - start_time
        latency = execution_time / cnt

        return trajs, [conv['obs'] for conv in conv_batch], responses


def inference(agent, env, dataloader, model, tokenizer):
    metrics = [0] * 3  # [TC, SPD, SED]
    trajs_record = []

    for inputs in tqdm(dataloader):
        route_ids = inputs.pop('route_ids')
        trajs, images, responses = agent.rollout(env=env, model=model, tokenizer=tokenizer, route_ids=route_ids)
        if trajs is None:
            continue
        env.eva_metrics(trajs, metrics)
        for i in range(len(trajs)):
            trajs_record.append(
                {'route_ids': route_ids[i], 'navigation_text': env.dict_data[route_ids[i]]['navigation_text'],
                 'trajs': trajs[i], 'images': images[i], 'responses': responses[i]})

    eval_dataset = getattr(dataloader, "dataset", None)
    num_samples = len(eval_dataset)
    metrics = [m / num_samples for m in metrics]
    metrics = [m * 100 if m < 1 else m for m in metrics]
    return metrics, trajs_record


def run_navigation(eval_env, model, tokenizer, opts):
    dataloader = make_evaluation_data_module(opts)
    split_name = eval_env.env.name
    eval_env.reset_epoch()
    agent = FLAME(opts)
    metrics, trajs_record = inference(agent, eval_env, dataloader, model, tokenizer)

    metrics = {"TC": metrics[0], f"TC_{split_name}": metrics[0], f"SPD_{split_name}": metrics[1],
               f"SED_{split_name}": metrics[2]}
    return metrics, trajs_record
