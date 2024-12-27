import random
import os
from PIL import Image
import lmdb
import networkx as nx
import numpy as np
from pyxdameraulevenshtein import damerau_levenshtein_distance as edit_dis
from transformers import CLIPImageProcessor

import torch
from llm_nav.sim.base_navigator import BaseNavigator, get_relative_angle, get_closest_heading
from llm_nav.sim.utils import load_datasets, load_nav_graph


_SUCCESS_THRESHOLD = 2


class EnvBatch:
    def __init__(self, opts, name=None):
        self.opts = opts
        self.name = name

        self.image_processor = CLIPImageProcessor()
        self.store_feature = opts.store_feature
        self.navs = []
        if opts.store_feature:
            self.img_ft_dir = opts.img_db
        else:
            self.img_ft_db = lmdb.open(opts.img_db, readonly=True)
        self._img_ft_cache = {}
        for i in range(opts.env_batch_size):
            nav = BaseNavigator(self.opts)
            self.navs.append(nav)
        print("=====Initializing %s navigators=====" % self.name)

    def newEpisodes(self, panoIds, headings):
        """ Iteratively initialize the simulators for # of batchsize"""
        for i, (panoId, heading) in enumerate(zip(panoIds, headings)):
            self.navs[i].init_state(panoId, heading)

    def _get_gt_action(self, batch):
        gt_action = []
        for i, item in enumerate(batch):
            nav = self.navs[i]
            gt_path = item['route_panoids']

            target_panoid = gt_path[-1]
            curr_panoid, curr_heading = nav.get_state()
            curr_node = nav.graph.nodes[curr_panoid]

            if curr_panoid in gt_path:
                num_occurrences = gt_path.count(curr_panoid)
                if num_occurrences == 1:
                    pano_index = gt_path.index(curr_panoid)
                else:  # if novel gold path visits panoid twice then select the correct one based on the current trajectory
                    num_occurrences_nav = nav.pano_path.count(curr_panoid)
                    nth_occurrence = min(num_occurrences, num_occurrences_nav) - 1
                    pano_index = [i for i, p in enumerate(gt_path) if p == curr_panoid][nth_occurrence]

                if pano_index == len(gt_path) - 1:
                    assert gt_path[pano_index] == target_panoid
                    gt_action.append('stop')
                    continue

                gt_next_panoid = gt_path[pano_index + 1]
                gt_next_heading = curr_node.get_neighbor_heading(gt_next_panoid)
            else:
                shortest_path = nav.graph.get_shortest_path(curr_panoid, target_panoid)
                if len(shortest_path) <= 1:
                    gt_action.append('stop')
                    continue
                gt_next_panoid = shortest_path[1]
                gt_next_heading = curr_node.get_neighbor_heading(gt_next_panoid)

            next_panoid, next_heading = nav.get_next_state('forward')
            if gt_next_panoid == next_panoid:
                # at 3-way intersection, "forward" AND "left"/"right" can be correct. Only chose forward as gold action
                # if it doesn't imply a rotation of over 45 degrees.
                if len(curr_node.neighbors) != 3 or abs(get_relative_angle(next_heading, gt_next_heading)) < 45:
                    gt_action.append('forward')
                    continue

            next_panoid, next_heading = nav.get_next_state('turn_around')
            if gt_next_heading == next_heading:
                gt_action.append('turn_around')
                continue

            next_panoid, next_heading_left = nav.get_next_state('left')
            if gt_next_heading == next_heading_left:
                gt_action.append('left')
                continue

            next_panoid, next_heading_right = nav.get_next_state('right')
            if gt_next_heading == next_heading_right:
                gt_action.append('right')
                continue

            # if multiple rotations are needed, choose direction which brings the agent closer to the correct next heading
            next_heading = get_closest_heading(gt_next_heading, [next_heading_left, next_heading_right])
            if next_heading == next_heading_left:
                gt_action.append('left')
                continue
            if next_heading == next_heading_right:
                gt_action.append('right')
                continue

            raise ValueError('gt_action not found')

        return gt_action

    def _get_observations(self, batch):
        """Get the observations for the current timestep."""
        obs = []
        for i, item in enumerate(batch):
            nav = self.navs[i]
            observations = dict()
            prev_state = None, None
            if len(nav.states) > 1:
                prev_state = nav.states[-2]
            panoid, heading = nav.states[-1]
            prev_panoid, prev_heading = prev_state

            # intersection
            num_neighbors = nav.graph.get_num_neighbors(panoid)
            if num_neighbors > 2 and panoid != prev_panoid:
                observations['intersection'] = num_neighbors
            # observations['available_actions'] = nav.get_available_actions()

            viewpoint = '{}_{}'.format(nav.states[-1][0], int(nav.states[-1][1]))
            observations['viewpoint'] = viewpoint
            observations['image'] = self._get_image_feature(viewpoint)
            obs.append(observations)
        return obs

    def _get_image_feature(self, image_id):
        # return None
        if image_id in self._img_ft_cache:
            image = self._img_ft_cache[image_id]
            return image
        if self.store_feature:
            feature_path = os.path.join(self.img_ft_dir, f"{image_id}.pt")
            image = torch.load(feature_path)
            self._img_ft_cache[image_id] = image
        else:
            with self.img_ft_db.begin() as txn:
                img_bytes = txn.get(image_id.encode('ascii'))
            image_flt = np.frombuffer(img_bytes, dtype=np.uint8)
            image_flt = cv2.imdecode(image_flt, cv2.IMREAD_COLOR)
            image = image_flt.reshape(1500, 1500, 3)
            image = Image.fromarray(image)
            image = self.image_processor(image)["pixel_values"][0]
            self._img_ft_cache[image_id] = image
        return image


    def _get_instructions(self, batch):
        instructions = []
        for i, item in enumerate(batch):
            instructions.append(item['navigation_text'])
        return instructions

    def cal_cls(self, graph, traj, gt_traj):
        PC = np.mean(np.exp([-np.min(
            [nx.dijkstra_path_length(graph, traj_point, gt_traj_point)
             for traj_point in traj])
                             for gt_traj_point in gt_traj]))
        EPL = PC * len(gt_traj)
        LS = EPL / (EPL + np.abs(EPL - len(traj)))
        return LS * PC

    def cal_dtw(self, graph, prediction, reference, success):
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction) + 1):
            for j in range(1, len(reference) + 1):
                best_previous_cost = min(
                    dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])
                cost = nx.dijkstra_path_length(graph, prediction[i - 1], reference[j - 1])
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]
        dtw_group = [dtw]
        ndtw = np.exp(-dtw / (_SUCCESS_THRESHOLD * np.sqrt(len(reference))))
        dtw_group += [ndtw, success * ndtw]
        return dtw_group

    def _eva_metrics(self, trajs, batch, graph, metrics):
        for i, item in enumerate(batch):
            success = 0
            traj = trajs[i]
            gt_traj = item["route_panoids"]
            ed = edit_dis(traj, gt_traj)
            ed = 1 - ed / max(len(traj), len(gt_traj))
            target_list = list(nx.all_neighbors(graph, gt_traj[-1])) + [gt_traj[-1]]
            if traj[-1] in target_list:
                success = 1
                metrics[0] += 1
                metrics[2] += ed
            metrics[1] += nx.dijkstra_path_length(graph, traj[-1], gt_traj[-1])
            # metrics[3] += self.cal_cls(graph, traj, gt_traj)
            # dtw_group = self.cal_dtw(graph, traj, gt_traj, success)
            # for j in range(-3, 0):
            #     metrics[j] += dtw_group[j]

    def action_step(self, target, ended, num_act_nav, trajs, total_steps):
        for i in range(len(ended)):
            nav = self.navs[i]
            if ended[i]:
                continue
            action = target[i]
            if action == "stop":
                ended[i] = 1
                num_act_nav[0] -= 1
            nav.step(action)
            if not nav.states[-1][0] == nav.states[-2][0]:
                new_pano, _ = nav.states[-1]
                trajs[i].append(new_pano)
            if nav.states[-1][0] is None or nav.states[-1][-1] is None:
                ended[i] = 1
                num_act_nav[0] -= 1
            total_steps[0] += 1


class TouchdownBatch:
    def __init__(self, opts, seed=10, splits=["train"],
                 name=None):
        self.env = EnvBatch(opts, name)
        self.data = []
        self.dict_data = {}
        self.opts = opts

        json_data = load_datasets(splits, opts)
        total_length = len(json_data)
        if total_length == 1:
            json_data = json_data[0]

        for i, item in enumerate(json_data):
            new_item = dict(item)
            self.data.append(new_item)
            if "route_id" in item:
                self.dict_data[item["route_id"]] = new_item
            elif "id" in item:
                self.dict_data[item["id"]] = new_item

        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = opts.env_batch_size
        self.splits = splits
        self._load_nav_graph()

    def _load_nav_graph(self):
        self.graph = load_nav_graph(self.opts)
        print("Loading navigation graph done.")

    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix + self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset(self, print_info=False, route_ids=None):
        if route_ids:
            self.batch = [self.dict_data[route_id] for route_id in route_ids]
        else:
            self._next_minibatch()
        panoIds = []
        headings = []
        trajs = []
        for i, item in enumerate(self.batch):
            curr_panoid = item["route_panoids"][0]
            next_panoid = item["route_panoids"][1]
            panoIds.append(curr_panoid)
            trajs.append([panoIds[-1]])
            neighbors = self.env.navs[0].graph.nodes[curr_panoid].neighbors
            if item["start_heading"] == 0:
                for key, value in neighbors.items():
                    if value.panoid == next_panoid:
                        gt_heading = key
                candidates = list(neighbors.keys())
                if not self.opts.legacy_navigation_mode:
                    reverse_heading = max(candidates, key=lambda h: 180 - abs(abs(gt_heading - h) - 180))
                    heading = reverse_heading
                    heading = gt_heading
                else:
                    heading = random.choice(candidates)
            else:
                heading = item["start_heading"]
            headings.append(heading)
        self.env.newEpisodes(panoIds, headings)

        return trajs  # returned a batch of the first panoid for each route_panoids

    def get_gt_action(self):
        return self.env._get_gt_action(self.batch)

    def get_instructions(self):
        return self.env._get_instructions(self.batch)

    def get_observations(self):
        return self.env._get_observations(self.batch)

    def reset_epoch(self):
        self.ix = 0

    def eva_metrics(self, trajs, metrics):
        self.env._eva_metrics(trajs, self.batch, self.graph, metrics)
