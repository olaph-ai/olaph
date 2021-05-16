import os
import json
import subprocess
from functools import reduce
from copy import deepcopy
from glob import glob
from datetime import datetime
from collections import deque
import difflib as dl
import numpy as np
import matplotlib.pyplot as plt
import yaml
from generate_rego_policy import generate_rego_policy
from distance import compute_distances, compute_hd_distance
from run_opa import get_opa_denies
from generator import generate_policy, generate_policy_diff
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    pod_name = 'synheart-restapi-7674bbffdb-lt4k9'
    with open(os.getenv('CONFIG'), 'r') as f:
        config = yaml.safe_load(f)
    tasks_dir = config['paths']['tasks_dir']
    models_dir = config['paths']['models_dir']
    policies_dir = config['paths']['policies_dir']
    diffs_dir = config['paths']['diffs_dir']
    plots_dir = config['paths']['plots_dir']

    logger.info(config['settings'])
    max_attributes = int(config['settings']['max_attributes'])
    window_size = int(config['settings']['window_size'])
    relearn_threshold = float(config['settings']['relearn_threshold'])
    generalisation = int(config['settings']['generalisation'])
    decay = float(config['settings']['decay'])
    drop_threshold = float(config['settings']['drop_threshold'])
    warm_up = int(config['settings']['warm_up'])

    differ = dl.HtmlDiff(wrapcolumn=80)
    next_set = []
    avg_distances = []
    relearn_windows = []
    denies = []
    denieds = []
    cooldown = 0
    window = []
    w_i = 0
    f = subprocess.Popen(f'kubectl logs --tail 5000 -f pod/{pod_name} opa-istio', shell=True,
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    while warm_up > 0:
        line = f.stdout.readline().decode().strip()
        log = json.loads(line)
        if log['msg'] == 'Decision Log':
            window.append({'input': log['input']})
        if len(window) == window_size:
            distances = [relearn_threshold] * window_size
            next_set = list(filter(None,  # Remove empty lists
                           map(lambda w: [(r, decay * d) if d is not None else (r, None) for (r, d) in w
                                          if d is None or d > drop_threshold],  # Decay examples
                               next_set)
                           ))
            next_set.append(list(zip(window, distances)))
            w_i += 1
            warm_up -= 1
            window.clear()
    learned_requests, learned_distances = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
    logger.info('Learning first policy')
    curr_policy_path, curr_policy_time, curr_package = generate_policy(
        deepcopy(learned_requests), learned_distances, max_attributes,
        generalisation, f'{pod_name}_1', tasks_dir, models_dir, policies_dir, pod_name
    )
    p_i = 2
    w_i += 1
    while True:
        line = f.stdout.readline().decode().strip()
        log = json.loads(line)
        if log['msg'] == 'Decision Log':
            window.append({'input': log['input']})
        if len(window) == window_size:
            distances = compute_distances(deepcopy(window), deepcopy(learned_requests), max_attributes)
            cooldown = max(0, cooldown - 1)
            next_set = list(filter(None,  # Remove empty lists
                                   map(lambda w: [(r, decay * d) if d is not None else (r, None) for (r, d) in w
                                                  if d is None or d > drop_threshold],  # Decay examples
                                       next_set)
                                   ))
            next_set.append(list(zip(window, distances)))
            num_denies, denied_rs = get_opa_denies(window, curr_policy_path, curr_package)
            denieds.extend(denied_rs)
            denies.append((w_i, num_denies))
            hd_distances = list(map(lambda w: max(filter(lambda d: d is not None, list(zip(*w))[1])), next_set))
            avg_distance = sum(hd_distances) / len(hd_distances)
            avg_distances.append((w_i, avg_distance))
            logger.info(f'Window {w_i:3d} - Avg max distance: {avg_distance:.4f}, window_size: {len(window)}, '
                     f'learned_size: {len(learned_requests)}, next_size: {sum(map(len, next_set))}, '
                     f'denies: {num_denies}')
            if avg_distance > relearn_threshold and cooldown == 0:
                logger.info(f'Relearn policy as avg max distance {avg_distance} > {relearn_threshold} and not on CD')
                next_requests, next_distances = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
                new_policy_path, new_policy_time, new_package = generate_policy(
                    deepcopy(next_requests), next_distances, max_attributes,
                    generalisation, f'{pod_name}_{p_i}', tasks_dir, models_dir, policies_dir, pod_name
                )
                generate_policy_diff(new_policy_path, new_policy_time, curr_policy_path, curr_policy_time, p_i,
                                     differ, f'{pod_name}_{p_i-1}-{p_i}', policies_dir, diffs_dir)
                relearn_windows.append((w_i, avg_distance, num_denies))
                curr_policy_path, curr_policy_time, curr_package = new_policy_path, new_policy_time, new_package
                learned_requests = next_requests
                cooldown = len(hd_distances)
                p_i += 1
            w_i += 1
            window.clear()
