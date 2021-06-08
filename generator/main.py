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
from sklearn.metrics import roc_curve
from baseline import train_baselines, num_baseline_anomalies, offline_baseline_roc_aucs
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def clear_dir(path, name):
    for f in glob(f'{path}/{name}*'):
        os.remove(f)

def decay_examples(next_set, decay, drop_threshold, maxlen):
    return deque(filter(None,  # Remove empty lists
                       map(lambda w: [(r, decay * d, p) if not p else (r, d, p) for (r, d, p) in w
                                      if p or d > drop_threshold],  # Decay examples
                           next_set)
                        ), maxlen=maxlen)

def run(TPRs, FPRs, generalisation, run_i, third_party_anomalies):
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
    # generalisation = max(float(config['settings']['generalisation']), 0)
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
        data_base = os.path.split(data_type)[1].split('.', 1)[0] + f'_dist2_g{str(generalisation).replace(".", "_")}'
        for p in sorted(glob(f'{data_path}/**', recursive=True)):
            if os.path.isfile(p) and os.path.basename(p) == data_type:
                if 13 <= i <= 13:
                    log.info(f'Loading requests from: {p}')
                    requests = get_requests_from_logs(p, restructure)
                    all_requests.extend(requests)
                i += 1
    else:
        data_base = os.path.split(data)[1].split('.', 1)[0] + f'_dist2_g{str(generalisation).replace(".", "_")}'
        all_requests = get_requests_from_logs(data_path, restructure)
        # all_requests = all_requests[(len(all_requests) // 2) + 7300:][:1000]

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
    iforest, svm, lof, trained_attrs = train_baselines(deepcopy(learned_requests), 0.1, distance_measure,
                                                       max_attributes, restructure)
    p_i, total_r = 2, len(window)
    w_i, c_i = 2, 2
    window = all_requests[i:j]
    i = j
    j += window_size
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
    while window:
        maxlen = max_requests // window_size
        distances = compute_distances(deepcopy(window), deepcopy(learned_requests),
                                      distance_measure, max_attributes, restructure)
        next_set = decay_examples(next_set, decay, low_thresh, maxlen)
        next_window = list(zip(window, distances, [False] * len(window)))
        next_set.append(next_window)
        num_denies, denied_rs = get_opa_denies(window, curr_policy_path, curr_package, restructure)
        tpa = 0
        if total_r >= 1000:
            a_i = total_r
            for a_j, r in enumerate(window):
                curr_i = a_i + a_j
                true_anomaly = curr_i in third_party_anomalies
                if true_anomaly:
                    tpa += 1
                    b_trues.append(-1)
                    if r in denied_rs:
                        log.info(r)
                        tp += 1
                    else:
                        fn += 1
                else:
                    b_trues.append(1)
                    if r in denied_rs:
                        fp += 1
                    else:
                        tn += 1
        denieds.extend(denied_rs)
        b_anomalies = []
        num_iforest, if_anomalies, scores = num_baseline_anomalies(iforest, deepcopy(window), trained_attrs,
                                                                      max_attributes, restructure)
        if total_r >= 1000:
            b_scores[0].extend(list(scores))
        b_anomalies.append(if_anomalies)
        num_svm, svm_anomalies, scores = num_baseline_anomalies(svm, deepcopy(window), trained_attrs,
                                                        max_attributes, restructure)
        if total_r >= 1000:
            b_scores[1].extend(list(scores))
        b_anomalies.append(svm_anomalies)
        num_lof, lof_anomalies, scores = num_baseline_anomalies(lof, deepcopy(window), trained_attrs,
                                                        max_attributes, restructure)
        if total_r >= 1000:
            b_scores[2].extend(list(scores))
        b_anomalies.append(lof_anomalies)
        intersection_anomalies = list(reduce(lambda a, b: a.intersection(b),
                                             map(lambda ans: set(map(json.dumps, ans)), b_anomalies)))
        total_r += len(window)
        agrees = set(map(json.dumps, denied_rs)).intersection(intersection_anomalies)
        num_agrees = len(agrees)
        all_agrees.append(num_agrees)
        agreeds.extend(list(agrees))
        baseline_anomalies.extend(intersection_anomalies)
        num_int = len(intersection_anomalies)
        denies.append((w_i, num_denies, num_iforest, num_svm, num_lof, num_int, num_agrees, tpa))
        _, d_olaf, d_if, d_svm, d_lof, d_int, d_agrees, d_tpa = list(zip(*(denies[last_relearn:])))
        avg_denies.append((w_i, np.mean(d_olaf), np.mean(d_if), np.mean(d_svm),
                           np.mean(d_lof), np.mean(d_int), np.mean(d_agrees), np.mean(d_tpa)))
        hd_distances = list(map(lambda w: max(list(zip(*w))[1]), next_set))
        # if w_i == 24:
        #     mean_distances = list(map(lambda w: np.mean(list(zip(*w))[1]), next_set))
        #     import matplotlib.pyplot as plt
        #     plt.plot(range(len(hd_distances)), hd_distances)
        #     plt.title('Maximum distance of incoming windows to the learned set')
        #     plt.xlabel('Window')
        #     plt.ylabel('Directed Hausdorff distance')
        #     plt.savefig('/plots/hausdorff.png')
        #     import sys
        #     sys.exit(0)
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
        # calibrate = c_i % calibrate_interval == 0
        calibrate = False
        log.info(f'Window {w_i:3d} ({int((total_r/len(all_requests)) * 100):2d}%) - w_size: {window_size}, '
                 f'Avg max distance: {avg_distance:.4f}, '
                 f'learned_size: {len(learned_requests)}, next_size: {sum(map(len, next_set)):4d} ({len(next_set)})'
                 f', high thresh: {high_thresh:.4f}, low thresh: {low_thresh:.4f}, denies: {num_denies}, '
                 f'iforest: {num_iforest}, svm: {num_svm}, lof: {num_lof}, int: {num_int}, '
                 f'agree: {num_agrees}, tpa: {tpa}')
        if avg_distance > high_thresh and not relearn_high and not relearn_low:
            log.info(f'Schedule relearn of policy, as {avg_distance:.4f} > {high_thresh:.4f}')
            relearn_high = True
            relearn_schedule_ws.append((w_i, avg_distance))
            with open(f'/tasks/trigger{w_i}-high.json', 'w') as f:
                f.write(json.dumps(list(zip(*list(reduce(lambda a, b: a + b, next_set))))[0], indent=4))
        elif avg_distance < low_thresh and not relearn_low and not relearn_high:
            log.info(f'Schedule relearn of policy, as {avg_distance:.4f} < {low_thresh:.4f}')
            relearn_low = True
            relearn_schedule_ws.append((w_i, avg_distance))
            with open(f'/tasks/trigger{w_i}-low.json', 'w') as f:
                f.write(json.dumps(list(zip(*list(reduce(lambda a, b: a + b, next_set))))[0], indent=4))
        elif (relearn_high and avg_distance <= np.mean(curr_avg_distances)
              or relearn_low and avg_distance >= np.mean(curr_avg_distances)) or calibrate:
        # if relearn_high or relearn_low:
            next_requests, next_distances, permanents = list(zip(*list(reduce(lambda a, b: a + b, next_set))))
            outliers_fraction = max(min(permanents.count(False)/len(permanents), 0.5), 0.1)
            strr = "high" if relearn_high else "low" if relearn_low else "calibrate"
            log.info(f'Relearn {strr}, outliers frac: {outliers_fraction}')
            with open(f'/tasks/relearn{w_i}-{strr}.json', 'w') as f:
                f.write(json.dumps(next_requests, indent=4))
            new_policy_path, new_policy_time, new_package = generate_policy(
                deepcopy(next_requests), next_distances, max_attributes,
                generalisation, f'{data_base}_{p_i}', tasks_dir, models_dir, policies_dir, data_base, restructure
            )
            generate_policy_diff(new_policy_path, new_policy_time, curr_policy_path, curr_policy_time, p_i,
                                 differ, f'{data_base}_{p_i-1}-{p_i}', policies_dir, diffs_dir)
            iforest, svm, lof, trained_attrs = train_baselines(deepcopy(next_requests), outliers_fraction,
                                                               distance_measure, max_attributes, restructure)
            relearn_windows.append((w_i, avg_distance, num_denies))
            curr_policy_path, curr_policy_time, curr_package = new_policy_path, new_policy_time, new_package
            learned_requests = next_requests
            last_relearn = len(avg_distances)
            if not calibrate and not relearn_low:
                next_set.append(deepcopy([(r, d, True) for (r, d, _) in sorted(list(filter(lambda r: not r[2], reduce(lambda a, b: a + b, next_set))), key=lambda p: p[1], reverse=True)[:window_size]]))
            # if not calibrate:
            #     next_set.append(deepcopy([(r, d, True) for (r, d, _) in sorted(list(filter(lambda r: not r[2], reduce(lambda a, b: a + b, next_set))), key=lambda p: p[1], reverse=relearn_high)[:window_size]]))
            relearn_high, relearn_low = False, False
            p_i += 1
            c_i = 0
        w_i += 1
        c_i += 1
        window = all_requests[i:j]
        i = j
        j += window_size
    log.info(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        log.info(f'precision: {precision:.4f}')
    else:
        precision = 0
    if (tp + fn) > 0:
        TPR = recall = tp / (tp + fn)
        log.info(f'recall: {recall:.4f}')
        if (tn + fp) > 0:
            TNR = tn/(tn + fp)
            FPR = 1 - TNR
            TPRs.append(TPR)
            FPRs.append(FPR)
            b_acc = (TPR + TNR) / 2
            log.info(f'b_acc: {b_acc:.4f}')
    else:
        recall = 0
    if (tp + tn + fp + fn) > 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        log.info(f'accuracy: {accuracy:.4f}')
    if precision > 0 or recall > 0:
        f_score = 2 * (precision * recall)/(precision + recall)
        log.info(f'f1: {f_score:4f}')
    plt.clf()
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
    else:
        x_relearn = []
    decay_str = str(decay).replace(".", "_")
    name = f'{data_base}-{decay_str}-{distance_measure}'
    plt.legend()
    plt.title('Average distance of learning set to learned set')
    plt.xlabel(f'Window (size {base_window_size})')
    plt.ylabel(f'Average distance')
    plt.savefig(f'{plots_dir}/{name}-req_dist.png')
    # plt.clf()
    # x, w_denies, _, _, _, _, _, tpa_denies = zip(*denies)
    # plt.plot(x, w_denies, 'x', label='OLAPH')
    # plt.plot(x, tpa_denies, 'X', label='Third party')
    # plt.plot(x, if_denies, label='iForest')
    # plt.plot(x, svm_denies, label='OC-SVM')
    # plt.plot(x, lof_denies, label='LOF')
    # plt.legend()
    # # if relearn_windows:
    # #     plt.plot(x_relearn, y2_relearn, 'ro', label='relearn')
    # plt.title('Anomalies per window')
    # plt.legend()
    # plt.xlabel(f'Window ({window_size} requests)')
    # plt.ylabel(f'Anomalies')
    # plt.savefig(f'{plots_dir}/{name}-denies.png')
    with open(f'{plots_dir}/{name}-denieds.json', 'w') as f:
        f.write(json.dumps(denieds, indent=4))
    with open(f'{plots_dir}/{name}-intersection_anomalies.json', 'w') as f:
        f.write(json.dumps(baseline_anomalies, indent=4))
    plt.clf()
    x, volatilities = zip(*volatilities)
    plt.plot(x, volatilities)
    for (i, w) in enumerate(x_relearn):
        if i == 0:
            plt.axvline(w, 0, 1, label='Relearn', color='k', linestyle='--', linewidth=0.5)
        else:
            plt.axvline(w, 0, 1, color='k', linestyle='--', linewidth=0.5)
        plt.legend()
    plt.title('Average distance volatility since last relearn')
    plt.xlabel(f'Window (size {window_size})')
    plt.ylabel(f'Standard deviation')
    plt.savefig(f'{plots_dir}/{name}-volatility.png')
    plt.clf()
    half = 0
    x, olaf_d, if_d, svm_d, lof_d, int_d, agrees_d, tpa_d = map(lambda av: av[half:], list(zip(*avg_denies)))
    plt.plot(x, olaf_d, label='OLAPH')
    # plt.plot(x, if_d, label='iForest')
    # plt.plot(x, svm_d, label='OC-SVM')
    # plt.plot(x, lof_d, label='LOF')
    plt.plot(x, int_d, label='intersection')
    for (i, w) in enumerate(x_relearn):
        if w in x:
            if i == 0:
                plt.axvline(w, 0, 1, label='Relearn', color='k', linestyle='--', linewidth=0.5)
            else:
                plt.axvline(w, 0, 1, color='k', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title('Average denies since last relearn')
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Avg denies')
    plt.savefig(f'{plots_dir}/{name}-avg_denies.png')
    plt.clf()
    plt.plot(x, olaf_d, label='OLAPH')
    plt.plot(x, if_d, label='iForest')
    plt.plot(x, svm_d, label='OC-SVM')
    plt.plot(x, lof_d, label='LOF')
    plt.plot(x, tpa_d, label='Third party')
    i = 0
    for w in x_relearn:
        if w in x:
            if i == 0:
                plt.axvline(w, 0, 1, label='Relearn', color='k', linestyle='--', linewidth=0.5)
            else:
                plt.axvline(w, 0, 1, color='k', linestyle='--', linewidth=0.5)
            i += 1
    plt.title('Average denies since last relearn')
    plt.xlabel(f'Window ({window_size} requests)')
    plt.ylabel(f'Avg denies')
    plt.legend()
    plt.savefig(f'{plots_dir}/{name}-avg_denies_all.png')
    # plt.clf()
    # plt.plot(x, agrees_d)
    # plt.title('Average agreed anomalies since last relearn')
    # plt.xlabel(f'Window ({window_size} requests)')
    # plt.ylabel(f'Num agreed anomalies')
    # plt.savefig(f'{plots_dir}/{name}-agrees.png')
    with open(f'{plots_dir}/{name}-agreeds.json', 'w') as f:
        f.write(json.dumps(agreeds, indent=4))
    return roc_curve(b_trues, b_scores[0]), roc_curve(b_trues, b_scores[1]), roc_curve(b_trues, b_scores[2])


if __name__ == '__main__':
    third_party_anomalies = [133] + [914] + [1202] + [1670] + [1747]
    third_party_anomalies = [tpa + 1000 for tpa in third_party_anomalies]
    with open(os.getenv('CONFIG'), 'r') as f:
        config = yaml.safe_load(f)
    data = config['paths']['data']
    data_dir = config['paths']['data_dir']
    data_path = f'{data_dir}/{data}'
    restructure = False
    all_requests = get_requests_from_logs(data_path, restructure)
    labels = [-1 if i in third_party_anomalies else 1 for i, _ in enumerate(all_requests)][1000:]
    if_rc, svm_rc, lof_rc = offline_baseline_roc_aucs(all_requests, labels, 0.5, 'cityblock', 30, restructure)
    if_fpr, if_tpr, _ = if_rc
    svm_fpr, svm_tpr, _ = svm_rc
    lof_fpr, lof_tpr, _ = lof_rc
    if_auc = np.trapz(if_tpr, if_fpr)
    svm_auc = np.trapz(svm_tpr, svm_fpr)
    lof_auc = np.trapz(lof_tpr, lof_fpr)
    log.info(f'if auc: {if_auc}, svm auc: {svm_auc}, lof auc: {lof_auc}')

    TPRs, FPRs = [], []
    # gs = list(np.arange(0.25, 1.75, 0.25))
    gs = list(np.arange(0.25, 1.15, 0.15))
    # gs = list(np.arange(0.25, 2.25, 0.25))
    # gs = list(np.arange(0.25, 10.3, 2))
    # gs = [0.25, 2.0, 2.5, 2.75, 3]
    # gs = [2.75, 3, 3.5]
    # gs = list(np.arange(0, 17, 2))
    log.info(f'gs: {gs}')
    for run_i, g in enumerate(gs):
        log.info(f'Running for generalisation: {g}')
        if_roc, svm_roc, lof_roc = run(TPRs, FPRs, g, run_i, third_party_anomalies)
        log.info(TPRs)
        log.info(FPRs)
    plt.clf()
    s_FPR, s_TPR = zip(*list(sorted(list(zip(FPRs, TPRs)) + [(1, 1)] + [(0, 0)], key=lambda x: x[0])))
    AUC = np.trapz(s_TPR, s_FPR)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.plot(s_FPR, s_TPR, label=f'OLAPH (auc = {AUC:.2f})')
    fpr, tpr, thresholds = if_roc
    AUC = np.trapz(tpr, fpr)
    plt.plot(fpr, tpr, label=f'iForest (auc = {AUC:.2f})')
    fpr, tpr, thresholds = svm_roc
    AUC = np.trapz(tpr, fpr)
    plt.plot(fpr, tpr, label=f'OC-SVM (auc = {AUC:.2f})')
    fpr, tpr, thresholds = lof_roc
    AUC = np.trapz(tpr, fpr)
    plt.plot(fpr, tpr, label=f'LOF (auc = {AUC:.2f})')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.legend()
    plt.title('AUC-ROC curve (online baselines)')
    plt.xlabel(f'FPR')
    plt.ylabel(f'TPR')
    plt.savefig(f'/plots/test_roc-auc_online-dist2.png')
    plt.clf()

    s_FPR, s_TPR = zip(*list(sorted(list(zip(FPRs, TPRs)) + [(1, 1)] + [(0, 0)], key=lambda x: x[0])))
    AUC = np.trapz(s_TPR, s_FPR)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.plot(s_FPR, s_TPR, label=f'OLAPH (auc = {AUC:.2f})')
    plt.plot(if_fpr, if_tpr, label=f'iForest (auc = {if_auc:.2f})')
    plt.plot(svm_fpr, svm_tpr, label=f'OC-SVM (auc = {svm_auc:.2f})')
    plt.plot(lof_fpr, lof_tpr, label=f'LOF (auc = {lof_auc:.2f})')
    plt.legend()
    plt.title('AUC-ROC curve (offline baselines)')
    plt.xlabel(f'FPR')
    plt.ylabel(f'TPR')
    plt.savefig(f'/plots/test_roc-auc_offline-dist2.png')
