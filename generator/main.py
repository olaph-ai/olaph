import os
from functools import reduce
from copy import deepcopy
from glob import glob
from datetime import datetime
from collections import deque
import difflib as dl
import numpy as np
import matplotlib.pyplot as plt
import yaml
from generate_learning_task import generate_learning_task
from run_learning_task import run_task
from generate_rego_policy import generate_rego_policy
from preprocess import get_requests_from_logs
from distance import compute_distances, compute_hd_distance
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def clear_dir(path, name):
    for f in glob(f'{path}/{name}*'):
        os.remove(f)

def generate_policy(requests, distances, max_attributes, generalisation, name, tasks_dir, models_dir, policies_dir):
    task, body_cost = generate_learning_task(requests, distances, max_attributes, generalisation)
    task_path = f'{tasks_dir}/{name}.las'
    with open(task_path, 'w') as f:
        f.write(task)
    model, rule_confidences = run_task(task_path, body_cost)
    with open(f'{models_dir}/{name}.lp', 'w') as f:
        f.write(model)
    new_policy = generate_rego_policy(rule_confidences, data_base)
    with open(f'{policies_dir}/{name}.rego', 'w') as f:
        f.write(new_policy)
    now = datetime.now().strftime("%d/%m/%Y at %H:%M:%S")
    now_time = f'{name} on {now}'
    return new_policy, now_time

def generate_policy_diff(new_policy, new_policy_time, curr_policy, curr_policy_time, i,
                         name, policies_dir, diffs_dir):
    with open(f'{diffs_dir}/{name}.html', 'w') as f:
        f.write(differ.make_file(curr_policy.splitlines(True), new_policy.splitlines(True),
                                 fromdesc=curr_policy_time, todesc=new_policy_time))

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

    max_attributes = int(config['settings']['max_attributes'])
    window_size = int(config['settings']['window_size'])
    relearn_threshold = float(config['settings']['relearn_threshold'])
    generalisation = int(config['settings']['generalisation'])
    decay = float(config['settings']['decay'])
    drop_threshold = float(config['settings']['drop_threshold'])
    warm_up = int(config['settings']['warm_up'])
    # max_attributes = int(input(f'Max attributes [{max_attributes}]: ') or max_attributes)
    # window_size = int(input(f'Window size [{window_size}]: ') or window_size)
    # relearn_threshold = float(input(f'Relearn threshold [{relearn_threshold}]: ') or relearn_threshold)
    # generalisation = int(input(f'Generalisation [{generalisation}]: ') or generalisation)
    # decay = float(input(f'Example decay [{decay}]: ') or decay)
    # drop_threshold = float(input(f'Example drop threshold [{drop_threshold}]: ') or drop_threshold)
    differ = dl.HtmlDiff(wrapcolumn=80)

    all_requests = get_requests_from_logs(f'{data_dir}/{data}')
    log.info(f'Total requests: {len(all_requests)}')

    learned_requests, learned_distances = all_requests[0:window_size], [1] * window_size
    curr_policy, curr_policy_time = generate_policy(deepcopy(learned_requests), learned_distances, max_attributes,
                                                    generalisation, f'{data_base}_1',
                                                    tasks_dir, models_dir, policies_dir)
    next_set = []
    avg_distances = []
    relearn_windows = []
    cooldown = 0
    p_i, w_i = 2, 2
    window = all_requests[(w_i-1) * window_size:w_i * window_size]
    while window:
        cooldown = max(0, cooldown - 1)
        next_set = list(filter(None,  # Remove empty lists
                               map(lambda w: [(r, decay * d) if d is not None else (r, None) for (r, d) in w
                                              if d is None or d > drop_threshold],  # Decay examples
                                   next_set)
                               ))
        distances = compute_distances(deepcopy(window), deepcopy(learned_requests), max_attributes)
        next_set.append(list(zip(window, distances)))
        if w_i > warm_up:
            hd_distances = list(map(lambda w: max(filter(None, list(zip(*w))[1])), next_set))
            avg_distance = sum(hd_distances) / len(hd_distances)
            avg_distances.append((w_i, avg_distance))
            log.info(f'Window {w_i} - Avg distance: {avg_distance}, w_size: {len(window)}, '
                     f'l_size: {len(learned_requests)}, n_size: {sum(map(len, next_set))}')
            if avg_distance > relearn_threshold and cooldown == 0:
                log.info(f'Relearning policy as avg distance {avg_distance} > {relearn_threshold} and not on '
                         'cooldown/warmup')
                next_requests, next_distances = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
                new_policy, new_policy_time = generate_policy(deepcopy(next_requests), next_distances,
                                                              max_attributes, generalisation, f'{data_base}_{p_i}',
                                                              tasks_dir, models_dir, policies_dir)
                generate_policy_diff(new_policy, new_policy_time, curr_policy, curr_policy_time, p_i,
                                     f'{data_base}_{p_i-1}-{p_i}', policies_dir, diffs_dir)
                relearn_windows.append((w_i, avg_distance))
                curr_policy, curr_policy_time = new_policy, new_policy_time
                learned_requests = next_requests
                cooldown = len(hd_distances)
                p_i += 1
        w_i += 1
        window = all_requests[(w_i-1) * window_size:w_i * window_size]

    x, avg_distances = zip(*avg_distances)
    plt.plot(x, avg_distances)
    if relearn_windows:
        x_relearn, y_relearn = zip(*relearn_windows)
        plt.plot(x_relearn, y_relearn, 'ro', label='relearn')
    # plt.plot([2, 84], [0.23, 2], 'gs', label='new behaviour')
    plt.hlines(relearn_threshold, x[0], x[-1], linestyles='dashed', label='relearn threshold', colors=['black'])
    plt.legend(loc='lower right')
    plt.title('Average max distance of incoming requests to the learning set')
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Average max distance (approx over the last {len(hd_distances)} windows)')
    plt.savefig(f'{plots_dir}/{data_base}-req_dist-{str(relearn_threshold).replace(".", "_")}.png')
