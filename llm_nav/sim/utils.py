import json, re, string
import numpy as np
import os, sys, torch
import warnings
from tensorboardX import SummaryWriter
import networkx as nx
import random
import shutil


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_datasets(splits, opts=None):
    data = []
    for split in splits:
        assert split in ['train', 'test', 'dev', 'union', 'meta']
        with open('%s/data/%s.json' % (opts.dataset, split)) as f:
            for line in f:
                data.append(json.loads(line))
    return data


def shortest_path(pano, Graph):
    dis = {}
    queue = []
    queue.append([Graph.graph.nodes[pano], 0])
    while queue:
        cur = queue.pop(0)
        cur_node = cur[0]
        cur_dis = cur[1]
        if cur_node.panoid not in dis.keys():
            dis[cur_node.panoid] = cur_dis
            cur_dis += 1
            for neighbors in cur_node.neighbors.values():
                queue.append([neighbors, cur_dis])

    with open("path/" + pano + ".json", "a") as f:
        json.dump(dis, f)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def set_tb_logger(log_dir, exp_name, resume):
    """ Set up tensorboard logger"""
    log_dir = log_dir + '/' + exp_name
    # remove previous log with the same name, if not resume
    if not resume and os.path.exists(log_dir):
        import shutil
        try:
            shutil.rmtree(log_dir)
        except:
            warnings.warn('Experiment existed in TensorBoard, but failed to remove')
    return SummaryWriter(log_dir=log_dir)


def load_nav_graph(opts):
    with open("%s/graph/links.txt" % opts.dataset) as f:
        G = nx.DiGraph()
        for line in f:
            pano_1, name, pano_2 = line.strip().split(",")
            G.add_edge(pano_1, pano_2)
    return G


def random_list(prob_torch, lists):
    x = random.uniform(0, 1)
    cum_prob = 0
    for i in range(len(lists) - 1):
        cum_prob += prob_torch[i]
        if x < cum_prob:
            return lists[i]
    return lists[len(lists) - 1]


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best_SPD, is_best_TC, epoch=-1):
    opts = state['opts']
    os.makedirs('{}/{}/{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name), exist_ok=True)
    filename = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '.pth.tar'))
    if opts.store_ckpt_every_epoch:
        filename = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '.%d.pth.tar' % epoch))
    torch.save(state, filename)
    if is_best_SPD:
        best_filename = (
            '{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '_model_SPD_best.pth.tar'))
        shutil.copyfile(filename, best_filename)
    if is_best_TC:
        best_filename = (
            '{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '_model_TC_best.pth.tar'))
        shutil.copyfile(filename, best_filename)


def input_img(pano, path):
    return np.load(path + "/" + pano + ".npy")
