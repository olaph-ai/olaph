import os
from copy import deepcopy
from glob import glob
from datetime import datetime
from collections import deque
import difflib as dl
import numpy as np
import matplotlib.pyplot as plt
from generate_learning_task import generate_learning_task
from run_learning_task import run_task
from generate_rego_policy import generate_rego_policy
from preprocess import get_requests_from_logs
from distance import compute_distances, compute_hd_distance
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def clear_dir(path):
    for f in glob(f'{path}/*'):
        os.remove(f)

def generate_policy(requests, distances, max_attributes, name, tasks_dir, models_dir, policies_dir):
    task, body_cost = generate_learning_task(requests, distances, max_attributes)
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
    data = os.getenv('DATA', 'synheart-controller-opa-istio.log')
    data_base = data.rsplit('.', 1)[0]

    data_dir = os.getenv('DATA_DIR', '../data')
    tasks_dir = os.getenv('TASKS_DIR', '../tasks')
    clear_dir(tasks_dir)
    models_dir = os.getenv('MODELS_DIR', '../models')
    clear_dir(models_dir)
    policies_dir = os.getenv('POLICIES_DIR', '../policies')
    clear_dir(policies_dir)
    diffs_dir = os.getenv('DIFFS_DIR', '../diffs')
    clear_dir(diffs_dir)
    plots_dir = os.getenv('PLOTS_DIR', '../plots')
    # clear_dir(plots_dir)

    max_attributes = 20
    window_size = 50
    max_learning_window_size = 1000
    relearn_threshold = 3.5

    differ = dl.HtmlDiff(wrapcolumn=80)

    all_requests = get_requests_from_logs(f'{data_dir}/{data}')
    log.info(f'Total requests: {len(all_requests)}')
    learning_set = all_requests[:window_size]
    curr_policy, curr_policy_time = generate_policy(deepcopy(learning_set), None, max_attributes, f'{data_base}_1',
                                                    tasks_dir, models_dir, policies_dir)

    learning_window_start = window_size
    window_start = learning_window_start
    window_end = learning_window_start + window_size
    request_distances = deque(maxlen=max_learning_window_size)
    window = all_requests[window_start:window_end]
    hd_maxlen = max_learning_window_size // window_size
    hd_distances = deque(maxlen=hd_maxlen)
    avg_distances = []
    p_i, w_i = 1, 1
    relearn_windows = []
    cooldown = 0
    while window:
        w_i += 1
        cooldown = max(0, cooldown - 1)
        distances = compute_distances(deepcopy(window), deepcopy(learning_set), max_attributes)
        request_distances.extend(distances)
        hd_distances.append(distances.max())
        avg_distance = sum(hd_distances) / hd_maxlen
        avg_distances.append((w_i, avg_distance))
        log.info(f'Window {w_i} - Avg distance: {avg_distance}, w_size: {len(window)}, l_size: {len(learning_set)}')
        if avg_distance > relearn_threshold and cooldown == 0:
            p_i += 1
            log.info(f'Relearning policy as avg distance {avg_distance} > {relearn_threshold} and not on cooldown')
            learning_set = all_requests[learning_window_start:window_end]
            new_policy, new_policy_time = generate_policy(deepcopy(learning_set), request_distances, max_attributes,
                                                          f'{data_base}_{p_i}', tasks_dir, models_dir, policies_dir)
            request_distances.clear()
            relearn_windows.append((w_i, avg_distance))
            learning_window_start = window_end
            generate_policy_diff(new_policy, new_policy_time, curr_policy, curr_policy_time, p_i,
                                 f'{data_base}_{p_i-1}-{p_i}', policies_dir, diffs_dir)
            curr_policy, curr_policy_time = new_policy, new_policy_time
            cooldown = hd_maxlen
        window_start += window_size
        window_end += window_size
        if window_end - learning_window_start > max_learning_window_size:
            learning_window_start = window_end - max_learning_window_size
        window = all_requests[window_start:window_end]
    x, avg_distances = zip(*avg_distances)
    plt.plot(x, avg_distances)
    x_relearn, y_relearn = zip(*relearn_windows)
    plt.plot(x_relearn, y_relearn, 'ro', label='relearn')
    plt.plot([2, 84], [0.23, 2], 'gs', label='new behaviour')
    plt.hlines(relearn_threshold, x[0], x[-1], linestyles='dashed', label='relearn threshold', colors=['black'])
    plt.legend(loc='lower right')
    plt.title('Average distance of incoming requests to the learning set')
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Average distance (over the last {hd_maxlen} windows)')
    plt.savefig(f'{plots_dir}/req_dist.png')
