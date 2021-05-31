import os
from math import ceil
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
from baseline import num_iforest_anomalies, num_oc_svm_anomalies, num_lof_anomalies
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def clear_dir(path, name):
    for f in glob(f'{path}/{name}*'):
        os.remove(f)

def decay_examples(next_set, drop_threshold, maxlen):
    return deque(filter(None,  # Remove empty lists
                       map(lambda w: [(r, decay * d, p) if not p else (r, d, p) for (r, d, p) in w
                                      if p or d > drop_threshold],  # Decay examples
                           next_set)
                        ), maxlen=maxlen)

if __name__ == '__main__':
    with open(os.getenv('CONFIG'), 'r') as f:
        config = yaml.safe_load(f)
    data = config['paths']['data']
    data_dir = config['paths']['data_dir']
    tasks_dir = config['paths']['tasks_dir']
    models_dir = config['paths']['models_dir']
    policies_dir = config['paths']['policies_dir']
    diffs_dir = config['paths']['diffs_dir']
    plots_dir = config['paths']['plots_dir']
    log.info(config['settings'])
    restructure = False
    distance_measure = 'cityblock'
    # distance_measure = 'jaccard'

    max_attributes = int(config['settings']['max_attributes'])
    max_requests = int(config['settings']['max_requests'])
    base_window_size = int(config['settings']['base_window_size'])
    generalisation = int(config['settings']['generalisation'])
    decay = float(config['settings']['decay'])
    calibrate_interval = max(int(config['settings']['calibrate_interval']), 1)
    differ = dl.HtmlDiff(wrapcolumn=80)
    clear_dir('/tasks', 'relearn')
    clear_dir('/tasks', 'trigger')

    data_path = f'{data_dir}/{data}'
    if os.path.isdir(data_path):
        all_requests = []
        i = 0
        data_type = config['paths']['data_type']
        data_base = os.path.split(data_type)[1].split('.', 1)[0]
        for p in sorted(glob(f'{data_path}/**', recursive=True)):
            if os.path.isfile(p) and os.path.basename(p) == data_type:
                if 12 <= i <= 15:
                    log.info(f'Loading requests from: {p}')
                    requests = get_requests_from_logs(p, restructure)
                    all_requests.extend(requests)
                i += 1
    else:
        data_base = os.path.split(data)[1].split('.', 1)[0]
        all_requests = get_requests_from_logs(data_path, restructure)
        # all_requests = all_requests[:100000]
        # log.info('\n'.join([json.dumps(r, indent=4) for r in all_requests[:3]]))
        # all_requests = all_requests[(len(all_requests) // 2) + 5000:]
        # all_requests = all_requests[(len(all_requests) // 2) - 5000:]
    all_requests = all_requests[(len(all_requests) // 4) * 2:(len(all_requests) // 4) * 3]
    # all_requests = all_requests[(len(all_requests) // 4) * 0:(len(all_requests) // 4) * 1 - 10000]
    with open('/tasks/reqs.tmp', 'w') as f:
        f.write(json.dumps(all_requests, indent=4))
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
    window_size = base_window_size
    i = j
    j += window_size
    distances = [0] * len(window)
    permanents = [False] * len(window)
    next_set.append(list(zip(window, distances, permanents)))
    learned_requests, learned_distances, _ = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
    curr_policy_path, curr_policy_time, curr_package = generate_policy(
        deepcopy(learned_requests), learned_distances, max_attributes,
        generalisation, f'{data_base}_1', tasks_dir, models_dir, policies_dir, data_base, restructure
    )
    p_i, total_r = 2, len(window)
    w_i, c_i = 2, 2
    window = all_requests[i:j]
    i = j
    j += window_size
    cooldown = 0
    last_relearn = 0
    relearn_high = False
    low_thresh = 0
    volatilities = []
    avg_denies = []
    baseline_anomalies = []
    while window:
        cooldown = max(0, cooldown - 1)
        total_r += len(window)
        maxlen = max_requests // window_size
        distances = compute_distances(deepcopy(window), deepcopy(learned_requests),
                                      distance_measure, max_attributes, restructure)
        next_set = decay_examples(next_set, low_thresh, maxlen)
        next_window = list(zip(window, distances, [False] * len(window)))
        next_set.append(next_window)
        ns = list(zip(*reduce(lambda a, b: a + b, next_set)))[0]
        num_denies, denied_rs = get_opa_denies(ns, curr_policy_path, curr_package, restructure)
        num_iforest, if_anomalies = num_iforest_anomalies(deepcopy(ns), max_attributes, restructure)
        baseline_anomalies.extend(if_anomalies)
        num_svm, svm_anomalies = num_oc_svm_anomalies(deepcopy(ns), max_attributes, restructure)
        baseline_anomalies.extend(svm_anomalies)
        num_lof, lof_anomalies = num_lof_anomalies(deepcopy(ns), max_attributes, restructure)
        baseline_anomalies.extend(lof_anomalies)
        denieds.extend(denied_rs)
        denies.append((w_i, num_denies, num_iforest, num_svm, num_lof))
        avg_denies.append((w_i, np.mean(list(zip(*denies))[1][last_relearn:])))
        hd_distances = list(map(lambda w: max(list(zip(*w))[1]), next_set))
        avg_distance = np.mean(hd_distances)
        avg_distances.append((w_i, avg_distance))
        curr_avg_distances = list(zip(*avg_distances[last_relearn if last_relearn != 0 else max(0,w_i-10):]))[1]
        mean_avg_d = np.mean(curr_avg_distances)
        std_avg_d = np.std(curr_avg_distances)
        high_thresh = mean_avg_d + 2 * std_avg_d
        low_thresh = max(mean_avg_d - 2 * std_avg_d, 0)
        thresholds.append((w_i, high_thresh, low_thresh))
        if len(curr_avg_distances) > 1:
            volatilities.append((w_i, std_avg_d))
        calibrate = c_i % calibrate_interval == 0
        log.info(f'Window {w_i:3d} ({int((total_r/len(all_requests)) * 100):2d}%) - w_size: {window_size}, '
                 f'Avg max distance: {avg_distance:.4f}, '
                 f'learned_size: {len(learned_requests)}, next_size: {sum(map(len, next_set)):4d} ({len(next_set)})'
                 f', high threshold: {high_thresh:.4f}, low threshold: {low_thresh:.4f}, denies: {num_denies}, '
                 f'iforest: {num_iforest}, oc_svm: {num_svm}, lof: {num_lof}')
        if avg_distance > high_thresh and not relearn_high:
            log.info(f'Schedule relearn of policy, as {avg_distance:.4f} > {high_thresh:.4f}')
            relearn_high = True
            relearn_schedule_ws.append((w_i, avg_distance))
            with open(f'/tasks/trigger{w_i}.json', 'w') as f:
                f.write(json.dumps(list(zip(*list(reduce(lambda a, b: a + b, next_set))))[0], indent=4))
        elif relearn_high and avg_distance <= np.mean(curr_avg_distances) or calibrate:
            log.info(f'Relearn {"high" if relearn_high else "calibrate"}')
            with open(f'/tasks/relearn{w_i}.json', 'w') as f:
                f.write(json.dumps(list(zip(*list(reduce(lambda a, b: a + b, next_set))))[0], indent=4))
            next_requests, next_distances, _ = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
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
            relearn_high = False
            last_relearn = len(avg_distances)
            if not calibrate:
                next_set.append(deepcopy([(r, d, True) for (r, d, _) in sorted(list(filter(lambda r: not r[2], reduce(lambda a, b: a + b, next_set))), key=lambda p: p[1], reverse=True)[:window_size]]))
            p_i += 1
            c_i = 0
        w_i += 1
        c_i += 1
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
    plt.title('Average distance of incoming window to learning set')
    plt.xlabel(f'Window (size {base_window_size})')
    plt.ylabel(f'Average distance')
    plt.savefig(f'{plots_dir}/{name}-req_dist.png')
    plt.clf()
    x, w_denies, if_denies, svm_denies, lof_denies = zip(*denies)
    plt.plot(x, w_denies, label='OLAPH')
    plt.plot(x, if_denies, label='iForest')
    plt.plot(x, svm_denies, label='OC-SVM')
    plt.plot(x, lof_denies, label='LOF')
    plt.legend()
    # if relearn_windows:
    #     plt.plot(x_relearn, y2_relearn, 'ro', label='relearn')
    plt.title('Anomalies per window')
    plt.legend()
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Anomalies')
    plt.savefig(f'{plots_dir}/{name}-denies.png')
    with open(f'{plots_dir}/{name}-denieds.json', 'w') as f:
        f.write(json.dumps(denieds, indent=4))
    with open(f'{plots_dir}/{name}-iforest_anomalies.json', 'w') as f:
        f.write(json.dumps(baseline_anomalies, indent=4))
    plt.clf()
    x, volatilities = zip(*volatilities)
    plt.plot(x, volatilities)
    plt.title('Average distance volatility per window')
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Standard deviation')
    plt.savefig(f'{plots_dir}/{name}-volatility.png')
    plt.clf()
    x, y_avg_denies = zip(*avg_denies)
    plt.plot(x, y_avg_denies)
    plt.title('Average denies since last relearn')
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Avg denies')
    plt.savefig(f'{plots_dir}/{name}-avg_denies.png')


            # if not calibrate:
            #     next_set.append([(r, d, True) for (r, d, p) in list(reduce(lambda a,b:a+b, next_set)) if not p])
            # if not calibrate:
            #     next_set.append(deepcopy([(r, d, True) for (r, d, _) in sorted(list(filter(lambda r: not r[2], reduce(lambda a, b: a + b, next_set))), key=lambda p: p[1], reverse=True)[:1]]))
