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
from sklearn.preprocessing import normalize
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def clear_dir(path, name):
    for f in glob(f'{path}/{name}*'):
        os.remove(f)

def decay_examples(next_set, drop_threshold, maxlen):
    return deque(filter(None,  # Remove empty lists
                       map(lambda w: [(r, decay * d) if d is not None else (r, None) for (r, d) in w
                                      if d is None or d > drop_threshold],  # Decay examples
                           next_set)
                        ), maxlen=maxlen)

if __name__ == '__main__':
    with open(os.getenv('CONFIG'), 'r') as f:
        config = yaml.safe_load(f)
    data = config['paths']['data']
    data_base = os.path.split(data)[1].split('.', 1)[0]
    data_dir = config['paths']['data_dir']
    tasks_dir = config['paths']['tasks_dir']
    models_dir = config['paths']['models_dir']
    policies_dir = config['paths']['policies_dir']
    diffs_dir = config['paths']['diffs_dir']
    plots_dir = config['paths']['plots_dir']
    log.info(config['settings'])
    restructure = True
    distance_measure = 'cityblock'
    # distance_measure = 'jaccard'

    max_attributes = int(config['settings']['max_attributes'])
    max_requests = int(config['settings']['max_requests'])
    base_window_size = int(config['settings']['base_window_size'])
    generalisation = int(config['settings']['generalisation'])
    decay = float(config['settings']['decay'])
    calibrate_interval = max(int(config['settings']['calibrate_interval']), 1)
    differ = dl.HtmlDiff(wrapcolumn=80)

    all_requests = get_requests_from_logs(f'{data_dir}/{data}', restructure)
    # all_requests = all_requests[:100000]
    # log.info('\n'.join([json.dumps(r, indent=4) for r in all_requests[:3]]))
    # all_requests = all_requests[(len(all_requests) // 2) + 5000:]
    # all_requests = all_requests[(len(all_requests) // 2) - 5000:]
    log.info(f'Total requests: {len(all_requests)}')

    window_size = base_window_size
    maxlen = max_requests // window_size
    next_set = deque(maxlen=maxlen)
    avg_distances = []
    relearn_windows = []
    relearn_schedule_ws = []
    denies = []
    denieds = []
    thresholds = []
    i = 0
    j = i + window_size
    window = all_requests[i:j]
    i = j
    j += window_size
    distances = [0] * len(window)
    next_set.append(list(zip(window, distances)))
    learned_requests, learned_distances = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
    curr_policy_path, curr_policy_time, curr_package = generate_policy(
        deepcopy(learned_requests), learned_distances, max_attributes,
        generalisation, f'{data_base}_1', tasks_dir, models_dir, policies_dir, data_base, restructure
    )
    p_i, r_i, u_i = 2, window_size, 0
    w_i = 2
    window = all_requests[i:j]
    i = j
    j += window_size
    cooldown = 0
    last_relearn = 0
    relearn_high, relearn_low = False, False
    low_thresh = 0
    while window:
        cooldown = max(0, cooldown - 1)
        r_i += len(window)
        maxlen = max_requests // window_size
        distances = compute_distances(deepcopy(window), deepcopy(learned_requests), distance_measure, max_attributes, restructure)
        next_set = decay_examples(next_set, low_thresh, maxlen)
        next_set.append(list(zip(window, distances)))
        if restructure:
            num_denies, denied_rs = get_opa_denies(window, curr_policy_path, curr_package)
            denieds.extend(denied_rs)
            denies.append((w_i, num_denies))
        else:
            num_denies = 0
        hd_distances = list(map(lambda w: max(filter(lambda d: d is not None, list(zip(*w))[1])), next_set))
        avg_distance = np.mean(hd_distances)
        avg_distances.append((w_i, avg_distance))
        curr_avg_distances = list(zip(*avg_distances[last_relearn:]))[1]
        mean_avg_d = np.mean(curr_avg_distances)
        std_avg_d = np.std(curr_avg_distances)
        high_thresh = mean_avg_d + 2 * std_avg_d
        low_thresh = max(mean_avg_d - 2 * std_avg_d, 0)
        thresholds.append((w_i, high_thresh, low_thresh))
        log.info(f'Window {w_i:3d} - w_size: {window_size}, Avg max distance: {avg_distance:.4f}, '
                 f'learned_size: {len(learned_requests)}, next_size: {sum(map(len, next_set)):4d} ({len(next_set)})'
                 f', high threshold: {high_thresh:.4f}, low threshold: {low_thresh:.4f}, denies: {num_denies}')
        if avg_distance > high_thresh and not relearn_high and not relearn_low:
            log.info(f'Schedule relearn of policy, as {avg_distance:.4f} > {high_thresh:.4f}')
            relearn_high = True
            relearn_schedule_ws.append((w_i, avg_distance))
            u_i = 0
        elif avg_distance < low_thresh and not relearn_low and not relearn_high and cooldown == 0:
            log.info(f'Schedule relearn of policy, as {avg_distance:.4f} < {low_thresh:.4f}')
            relearn_low = True
            relearn_schedule_ws.append((w_i, avg_distance))
            u_i = 0
        elif (relearn_high and avg_distance <= np.mean(curr_avg_distances)
              or relearn_low and avg_distance >= np.mean(curr_avg_distances)) or r_i % calibrate_interval == 0:
            log.info(f'Relearn {"high" if relearn_high else "low" if relearn_low else "calibrate"}')
            next_requests, next_distances = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
            new_policy_path, new_policy_time, new_package = generate_policy(
                deepcopy(next_requests), next_distances, max_attributes,
                generalisation, f'{data_base}_{p_i}', tasks_dir, models_dir, policies_dir, data_base, restructure
            )
            generate_policy_diff(new_policy_path, new_policy_time, curr_policy_path, curr_policy_time, p_i,
                                 differ, f'{data_base}_{p_i-1}-{p_i}', policies_dir, diffs_dir)
            relearn_windows.append((w_i, avg_distance, num_denies))
            curr_policy_path, curr_policy_time, curr_package = new_policy_path, new_policy_time, new_package
            learned_requests = next_requests
            cooldown = len(hd_distances)
            relearn_high, relearn_low = False, False
            last_relearn = len(avg_distances)
            p_i += 1
            window_size = min(window_size * u_i, max_requests // 2)
            r_i = 0
            u_i = 0
        w_i += 1
        wb = window_size - base_window_size
        cr = calibrate_interval  - r_i
        window_size = max(window_size - (cr + wb)//cr, base_window_size)
        u_i += 1
        window = all_requests[i:j]
        i = j
        j += window_size
    x, avg_distances = zip(*avg_distances)
    plt.plot(x, avg_distances)
    x, high_relearn_thresholds, low_relearn_thresholds = zip(*thresholds)
    plt.plot(x, high_relearn_thresholds, 'k--', label='relearn threshold', linewidth=0.5)
    plt.plot(x, low_relearn_thresholds, 'k--', linewidth=0.5)
    if relearn_schedule_ws:
        x_relearn_ws, y_relearn = zip(*relearn_schedule_ws)
        plt.plot(x_relearn_ws, y_relearn, 'go', label='schedule relearn')
    if relearn_windows:
        x_relearn, y1_relearn, y2_relearn = zip(*relearn_windows)
        plt.plot(x_relearn, y1_relearn, 'ro', label='relearn')
    decay_str = str(decay).replace(".", "_")
    name = f'{data_base}-{decay_str}-{distance_measure}'
    plt.legend()
    plt.title('Average max distance of incoming requests to the learning set')
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Average max distance (approx over the last {len(hd_distances)} windows)')
    plt.savefig(f'{plots_dir}/{name}-req_dist.png')
    if restructure:
        plt.clf()
        x, w_denies = zip(*denies)
        plt.plot(x, w_denies)
        if relearn_windows:
            plt.plot(x_relearn, y2_relearn, 'ro', label='relearn')
        plt.title('Policy denies per window')
        plt.legend()
        plt.xlabel(f'Window ({window_size} requests)')
        plt.ylabel(f'Number of denies')
        plt.savefig(f'{plots_dir}/{name}-denies.png')
        with open(f'{plots_dir}/{name}-denieds.json', 'w') as f:
            f.write(json.dumps(denieds, indent=4))
