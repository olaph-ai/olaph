import os
import json
from functools import reduce
from copy import deepcopy
from glob import glob
from collections import deque
import difflib as dl
import numpy as np
import matplotlib.pyplot as plt
import yaml
from preprocess import get_requests_from_logs
from distance import compute_distances, compute_hd_distance
from run_opa import get_opa_denies
from generator import generate_policy, generate_policy_diff
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def clear_dir(path, name):
    for f in glob(f'{path}/{name}*'):
        os.remove(f)

def decay_examples(next_set, drop_threshold):
    return list(filter(None,  # Remove empty lists
                       map(lambda w: [(r, decay * d) if d is not None else (r, None) for (r, d) in w
                                      if d is None or d > drop_threshold],  # Decay examples
                           next_set)
                       ))

if __name__ == '__main__':
    with open(os.getenv('CONFIG'), 'r') as f:
        config = yaml.safe_load(f)
    data = config['paths']['data']
    data_base = data.rsplit('.', 1)[0]
    data_dir = config['paths']['data_dir']
    tasks_dir = config['paths']['tasks_dir']
    models_dir = config['paths']['models_dir']
    policies_dir = config['paths']['policies_dir']
    diffs_dir = config['paths']['diffs_dir']
    plots_dir = config['paths']['plots_dir']
    log.info(config['settings'])

    max_attributes = int(config['settings']['max_attributes'])
    window_size = int(config['settings']['window_size'])
    generalisation = int(config['settings']['generalisation'])
    decay = float(config['settings']['decay'])
    drop_threshold = float(config['settings']['drop_threshold'])
    warm_up = int(config['settings']['warm_up'])
    time_penalty = float(config['settings']['time_penalty'])
    differ = dl.HtmlDiff(wrapcolumn=80)

    all_requests = get_requests_from_logs(f'{data_dir}/{data}')[:20000]
    log.info(f'Total requests: {len(all_requests)}')

    next_set = []
    avg_distances_p, avg_distances = [], []
    relearn_windows = []
    denies = []
    denieds = []
    thresholds = []
    cooldown = 0
    for w_i in range(1, warm_up + 1):
        window = all_requests[(w_i-1) * window_size:w_i * window_size]
        distances = [1] * window_size
        next_set = decay_examples(next_set, drop_threshold)
        next_set.append(list(zip(window, distances)))
    learned_requests, learned_distances = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
    curr_policy_path, curr_policy_time, curr_package = generate_policy(
        deepcopy(learned_requests), learned_distances, max_attributes,
        generalisation, f'{data_base}_1', tasks_dir, models_dir, policies_dir, data_base
    )
    p_i, t_i = 2, 0
    w_i += 1
    window = all_requests[(w_i-1) * window_size:w_i * window_size]
    while window:
        distances = compute_distances(deepcopy(window), deepcopy(learned_requests), max_attributes)
        cooldown = max(0, cooldown - 1)
        next_set = decay_examples(next_set, drop_threshold)
        next_set.append(list(zip(window, distances)))
        num_denies, denied_rs = get_opa_denies(window, curr_policy_path, curr_package)
        denieds.extend(denied_rs)
        denies.append((w_i, num_denies))
        hd_distances = list(map(lambda w: max(filter(lambda d: d is not None, list(zip(*w))[1])), next_set))
        avg_distance = np.mean(hd_distances)
        avg_distances.append(avg_distance)
        avg_distance_p = avg_distance + t_i * time_penalty
        avg_distances_p.append((w_i, avg_distance_p))
        t_i += 1

        avg_distances = avg_distances[-len(hd_distances):]
        mean_avg_distance = np.mean(avg_distances)
        std_avg_distance = np.std(avg_distances)
        relearn_threshold = mean_avg_distance + 3 * std_avg_distance
        thresholds.append((w_i, relearn_threshold))
        log.info(f'Window {w_i:3d} - Avg max distance: {avg_distance_p:.4f}, '
                 f'learned_size: {len(learned_requests)}, next_size: {sum(map(len, next_set))}, '
                 f'threshold: {relearn_threshold}, denies: {num_denies}')
        if avg_distance_p > relearn_threshold and cooldown == 0:
            log.info(f'Relearn policy as avg max distance {avg_distance_p} > {relearn_threshold} and not on CD')
            next_requests, next_distances = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
            new_policy_path, new_policy_time, new_package = generate_policy(
                deepcopy(next_requests), next_distances, max_attributes,
                generalisation, f'{data_base}_{p_i}', tasks_dir, models_dir, policies_dir, data_base
            )
            generate_policy_diff(new_policy_path, new_policy_time, curr_policy_path, curr_policy_time, p_i, differ,
                                 f'{data_base}_{p_i-1}-{p_i}', policies_dir, diffs_dir)
            relearn_windows.append((w_i, avg_distance_p, num_denies))
            curr_policy_path, curr_policy_time, curr_package = new_policy_path, new_policy_time, new_package
            learned_requests = next_requests
            cooldown = len(hd_distances)
            p_i += 1
            t_i = 0
        w_i += 1
        window = all_requests[(w_i-1) * window_size:w_i * window_size]

    x, avg_distances_p = zip(*avg_distances_p)
    plt.plot(x, avg_distances_p)
    x, relearn_thresholds = zip(*thresholds)
    plt.plot(x, relearn_thresholds, 'k--', label='relearn threshold')
    if relearn_windows:
        x_relearn, y1_relearn, y2_relearn = zip(*relearn_windows)
        plt.plot(x_relearn, y1_relearn, 'ro', label='relearn')
    # plt.plot([2, 84], [0.23, 2], 'gs', label='new behaviour')
    # plt.hlines(relearn_threshold, x[0], x[-1], linestyles='dashed', label='relearn threshold', colors=['black'])
    plt.legend(loc='lower right')
    plt.title('Average max distance of incoming requests to the learning set')
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Average max distance (approx over the last {len(hd_distances)} windows)')
    plt.savefig(f'{plots_dir}/{data_base}-req_dist-{window_size}_{str(decay).replace(".", "_")}.png')
    plt.clf()
    x, w_denies = zip(*denies)
    plt.plot(x, w_denies)
    if relearn_windows:
        plt.plot(x_relearn, y2_relearn, 'ro', label='relearn')
    plt.title('Policy denies per window')
    plt.legend()
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Number of denies')
    plt.savefig(f'{plots_dir}/{data_base}-denies-{window_size}_{str(decay).replace(".", "_")}.png')
    with open(f'{plots_dir}/{data_base}-denieds-{window_size}_{str(decay).replace(".", "_")}.json', 'w') as f:
        f.write(json.dumps(denieds, indent=4))
