import os
import time
import sys
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
from main import clear_dir, decay_examples
import logging
from multiprocessing import Process

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def run():
    with open(os.getenv('CONFIG'), 'r') as f:
        config = yaml.safe_load(f)
    data = config['paths']['data']
    data_dir = config['paths']['data_dir']
    data_path = f'{data_dir}/{data}'
    data_base = os.path.split(data)[1].split('.', 1)[0]
    tasks_dir = config['paths']['tasks_dir']
    models_dir = config['paths']['models_dir']
    policies_dir = config['paths']['policies_dir']
    diffs_dir = config['paths']['diffs_dir']
    plots_dir = config['paths']['plots_dir']
    log.info(config['settings'])
    restructure = True
    distance_measure = 'cityblock'
    max_attributes = int(config['settings']['max_attributes'])
    max_buffer = int(config['settings']['max_buffer'])
    maxlen = max_buffer
    window_size = int(config['settings']['window_size'])
    g = max(int(config['settings']['g']), 0)
    k = max(int(config['settings']['k']), 1)
    c = max(int(config['settings']['c']), 0)
    preferred_attrs = list(config['settings']['preferred_attributes'])
    allowlist = str(config['settings']['always_allow'])
    decay = float(config['settings']['decay'])
    differ = dl.HtmlDiff(wrapcolumn=80)
    clear_dir('/tasks', 'relearn')
    clear_dir('/tasks', 'trigger')

    differ = dl.HtmlDiff(wrapcolumn=80)
    next_set = deque(maxlen=maxlen)
    avg_distances = []
    relearn_windows = []
    relearn_schedule_ws = []
    denies = []
    denieds = []
    thresholds = []
    window = []
    w_i = 0
    lf = open(data_path, 'rb')
    while len(window) < window_size:
        line = lf.readline().decode().strip()
        llog = json.loads(line)
        if llog['msg'] == 'Decision Log':
            window.append({'input': llog['input']})
    distances = [0] * len(window)
    permanents = [False] * len(window)
    next_set.append(list(zip(window, distances, permanents)))
    learned_requests, learned_distances, _ = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
    log.info('Learning first policy')
    allowed = [{'input': json.loads(al)['input']} for al in allowlist.split('\n')] if allowlist else []
    for al in allowed:
        if al not in learned_requests:
            learned_requests.append(al), learned_distances.append(0), permanents.append(True)
    curr_policy_path, curr_policy_time, curr_package = generate_policy(
        deepcopy(learned_requests), learned_distances, max_attributes,
        g, k, c, preferred_attrs, f'{data_base}_1', tasks_dir, models_dir, policies_dir, data_base, restructure
    )
    next_set.append(deepcopy([(r, d, True) for (r, d, _) in sorted(list(filter(lambda r: not r[2], reduce(lambda a, b: a + b, next_set))), key=lambda p: p[1], reverse=True)[:window_size]]))
    p_i, total_r = 2, len(window)
    w_i, c_i = 2, 2
    last_relearn = 0
    relearn_high, relearn_low = False, False
    low_thresh = 0
    volatilities = []
    avg_denies = []
    agreeds = []
    baseline_anomalies = []
    all_agrees = []
    tp, tn, fp, fn = 0, 0, 0, 0
    b_trues, b_scores = [], [[], [], []]
    window.clear()
    try:
        while True:
            line = lf.readline().decode().strip()
            try:
                llog = json.loads(line)
            except:
                log.error(line)
            if llog['msg'] == 'Decision Log':
                window.append({'input': llog['input']})
            if len(window) == window_size:
                distances = compute_distances(deepcopy(window), deepcopy(learned_requests),
                                              distance_measure, max_attributes, restructure)
                next_set = decay_examples(next_set, decay, low_thresh, maxlen)
                next_window = list(zip(window, distances, [False] * len(window)))
                next_set.append(next_window)
                total_r += len(window)
                hd_distances = list(map(lambda w: max(list(zip(*w))[1]), next_set))
                avg_distance = np.mean(hd_distances)
                avg_distances.append((w_i, avg_distance))
                curr_avg_distances = list(zip(*avg_distances[last_relearn if last_relearn != 0 else max(0,w_i-10):]))[1]
                mean_avg_d = np.mean(curr_avg_distances)
                std_avg_d = np.std(curr_avg_distances)
                high_thresh = mean_avg_d + 2 * std_avg_d
                low_thresh = max(mean_avg_d - 2 * std_avg_d, 0.001)
                thresholds.append((w_i, high_thresh, low_thresh))
                if len(curr_avg_distances) > 1:
                    volatilities.append((w_i, std_avg_d))
                log.info(
                    f'{data_base:<20}: Window {w_i:3d} - w_size: {window_size}, '
                    f'Avg max distance: {avg_distance:.4f}, '
                    f'learned_size: {len(learned_requests)}, next_size: {sum(map(len, next_set)):4d} ({len(next_set)})'
                    f', high thresh: {high_thresh:.4f}, low thresh: {low_thresh:.4f}')
                if avg_distance > high_thresh and not relearn_high and not relearn_low:
                    log.info(f'Schedule relearn of policy, as {avg_distance:.4f} > {high_thresh:.4f}')
                    relearn_high = True
                    relearn_schedule_ws.append((w_i, avg_distance))
                elif avg_distance < low_thresh and not relearn_low and not relearn_high:
                    log.info(f'Schedule relearn of policy, as {avg_distance:.4f} < {low_thresh:.4f}')
                    relearn_low = True
                    relearn_schedule_ws.append((w_i, avg_distance))
                elif (relearn_high and avg_distance <= high_thresh
                      or relearn_low and avg_distance >= low_thresh):
                    enforce = 'relearn'
                    p_i_old = p_i - 1
                    while enforce == 'relearn':
                        with open(os.getenv('CONFIG'), 'r') as f:
                            config = yaml.safe_load(f)
                        max_attributes = int(config['settings']['max_attributes'])
                        max_buffer = int(config['settings']['max_buffer'])
                        maxlen = max_buffer
                        window_size = int(config['settings']['window_size'])
                        g = max(int(config['settings']['g']), 0)
                        k = max(int(config['settings']['k']), 1)
                        c = max(int(config['settings']['c']), 0)
                        preferred_attrs = list(config['settings']['preferred_attributes'])
                        allowlist = str(config['settings']['always_allow'])
                        decay = float(config['settings']['decay'])
                        next_requests, next_distances, permanents = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
                        allowed = [{'input': json.loads(al)['input']} for al in allowlist.split('\n')] if allowlist else []
                        for al in allowed:
                            if al not in next_requests:
                                next_requests.append(al), next_distances.append(0), permanents.append(True)
                        outliers_fraction = max(min(permanents.count(False)/len(permanents), 0.5), 0.1)
                        strr = "high" if relearn_high else "low" if relearn_low else "calibrate"
                        log.info(f'Relearn {strr}, outliers frac: {outliers_fraction}')
                        new_policy_path, new_policy_time, new_package = generate_policy(
                            deepcopy(next_requests), next_distances, max_attributes, g, k, c, preferred_attrs,
                            f'{data_base}_{p_i}', tasks_dir, models_dir, policies_dir, data_base, restructure
                        )
                        generate_policy_diff(new_policy_path, new_policy_time, curr_policy_path, curr_policy_time, p_i,
                                             differ, f'{data_base}_{p_i_old}-{p_i}', policies_dir, diffs_dir)
                        with open(curr_policy_path, 'r') as f1:
                            with open(new_policy_path, 'r') as f2:
                                if f1.read() != f2.read():
                                    enforce = input(f'Approve policy {p_i}? y/n/relearn: ')
                                    while enforce not in ['y', 'n', 'relearn']:
                                        enforce = input(f'Approve policy {p_i}? y/n/relearn: ')
                                else:
                                    enforce = 'n'
                        p_i += 1
                    relearn_windows.append((w_i, avg_distance))
                    curr_policy_path, curr_policy_time, curr_package = new_policy_path, new_policy_time, new_package
                    learned_requests = next_requests
                    last_relearn = len(avg_distances)
                    if enforce == 'y':
                        next_set.append(deepcopy([(r, d, True) for (r, d, _) in sorted(list(filter(lambda r: not r[2], reduce(lambda a, b: a + b, next_set))), key=lambda p: p[1], reverse=relearn_high)[:window_size]]))
                    relearn_high, relearn_low = False, False
                w_i += 1
                window.clear()
    except KeyboardInterrupt:
        pass
    finally:
        while True:
            try:
                x, y_avg_distances = zip(*avg_distances)
                plt.plot(x, y_avg_distances)
                x, high_relearn_thresholds, low_relearn_thresholds = zip(*thresholds)
                plt.plot(x, high_relearn_thresholds, 'k--', label='relearn threshold', linewidth=0.5)
                plt.plot(x, low_relearn_thresholds, 'k--', linewidth=0.5)
                if relearn_schedule_ws:
                    x_relearn_ws, y_relearn = zip(*relearn_schedule_ws)
                    plt.plot(x_relearn_ws, y_relearn, 'go', label='schedule relearn')
                if relearn_windows:
                    x_relearn, y1_relearn = zip(*relearn_windows)
                    plt.plot(x_relearn, y1_relearn, 'ro', label='relearn')
                else:
                    x_relearn = []
                decay_str = str(decay).replace(".", "_")
                name = f'{data_base}-{decay_str}-{distance_measure}'
                plt.legend()
                plt.title('Average distance of learning set to learned set')
                plt.xlabel(f'Window (size {window_size})')
                plt.ylabel(f'Average distance')
                plt.savefig(f'{plots_dir}/{name}-req_dist.png')
                break
            except KeyboardInterrupt:
                pass
        lf.close()

if __name__ == '__main__':
    run()
