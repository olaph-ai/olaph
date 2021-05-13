import os
import json
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
from run_opa import get_opa_denies
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
    new_policy, package = generate_rego_policy(rule_confidences, data_base)
    new_policy_path = f'{policies_dir}/{name}.rego'
    with open(new_policy_path, 'w') as f:
        f.write(new_policy)
    now = datetime.now().strftime("%d/%m/%Y at %H:%M:%S")
    now_time = f'{name} on {now}'
    return new_policy_path, now_time, package

def generate_policy_diff(new_policy_path, new_policy_time, curr_policy_path, curr_policy_time, i,
                         name, policies_dir, diffs_dir):
    with open(new_policy_path, 'r') as f:
        new_policy = f.readlines()
    with open(curr_policy_path, 'r') as f:
        curr_policy = f.readlines()
    with open(f'{diffs_dir}/{name}.html', 'w') as f:
        f.write(differ.make_file(curr_policy, new_policy, fromdesc=curr_policy_time, todesc=new_policy_time))

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
    relearn_threshold = float(config['settings']['relearn_threshold'])
    generalisation = int(config['settings']['generalisation'])
    decay = float(config['settings']['decay'])
    drop_threshold = float(config['settings']['drop_threshold'])
    warm_up = int(config['settings']['warm_up'])
    differ = dl.HtmlDiff(wrapcolumn=80)

    all_requests = get_requests_from_logs(f'{data_dir}/{data}')
    log.info(f'Total requests: {len(all_requests)}')

    next_set = []
    avg_distances = []
    relearn_windows = []
    denies = []
    denieds = []
    cooldown = 0
    for w_i in range(1, warm_up + 1):
        window = all_requests[(w_i-1) * window_size:w_i * window_size]
        distances = [relearn_threshold] * window_size
        next_set = list(filter(None,  # Remove empty lists
                       map(lambda w: [(r, decay * d) if d is not None else (r, None) for (r, d) in w
                                      if d is None or d > drop_threshold],  # Decay examples
                           next_set)
                       ))
        next_set.append(list(zip(window, distances)))
    learned_requests, learned_distances = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
    curr_policy_path, curr_policy_time, curr_package = generate_policy(
        deepcopy(learned_requests), learned_distances, max_attributes,
        generalisation, f'{data_base}_1', tasks_dir, models_dir, policies_dir
    )
    p_i = 2
    w_i += 1
    window = all_requests[(w_i-1) * window_size:w_i * window_size]
    while window:
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
        log.info(f'Window {w_i:3d} - Avg max distance: {avg_distance:.4f}, window_size: {len(window)}, '
                 f'learned_size: {len(learned_requests)}, next_size: {sum(map(len, next_set))}, '
                 f'denies: {num_denies}')
        if avg_distance > relearn_threshold and cooldown == 0:
            log.info(f'Relearn policy as avg max distance {avg_distance} > {relearn_threshold} and not on CD')
            next_requests, next_distances = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
            new_policy_path, new_policy_time, new_package = generate_policy(
                deepcopy(next_requests), next_distances, max_attributes,
                generalisation, f'{data_base}_{p_i}', tasks_dir, models_dir, policies_dir
            )
            generate_policy_diff(new_policy_path, new_policy_time, curr_policy_path, curr_policy_time, p_i,
                                 f'{data_base}_{p_i-1}-{p_i}', policies_dir, diffs_dir)
            relearn_windows.append((w_i, avg_distance, num_denies))
            curr_policy_path, curr_policy_time, curr_package = new_policy_path, new_policy_time, new_package
            learned_requests = next_requests
            cooldown = len(hd_distances)
            p_i += 1
        w_i += 1
        window = all_requests[(w_i-1) * window_size:w_i * window_size]

    x, avg_distances = zip(*avg_distances)
    plt.plot(x, avg_distances)
    if relearn_windows:
        x_relearn, y1_relearn, y2_relearn = zip(*relearn_windows)
        plt.plot(x_relearn, y1_relearn, 'ro', label='relearn')
    # plt.plot([2, 84], [0.23, 2], 'gs', label='new behaviour')
    plt.hlines(relearn_threshold, x[0], x[-1], linestyles='dashed', label='relearn threshold', colors=['black'])
    plt.legend(loc='lower right')
    plt.title('Average max distance of incoming requests to the learning set')
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Average max distance (approx over the last {len(hd_distances)} windows)')
    plt.savefig(f'{plots_dir}/{data_base}-req_dist-{str(relearn_threshold).replace(".", "_")}.png')
    plt.clf()
    x, w_denies = zip(*denies)
    plt.plot(x, w_denies)
    if relearn_windows:
        plt.plot(x_relearn, y2_relearn, 'ro', label='relearn')
    plt.title('Policy denies per window')
    plt.legend()
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Number of denies')
    plt.savefig(f'{plots_dir}/{data_base}-denies-{str(relearn_threshold).replace(".", "_")}.png')
    with open(f'{plots_dir}/{data_base}-denieds-{str(relearn_threshold).replace(".", "_")}.json', 'w') as f:
        f.write(json.dumps(denieds, indent=4))
