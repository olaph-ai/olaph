import os
from copy import deepcopy
from glob import glob
from datetime import datetime
import difflib as dl
from generate_learning_task import generate_learning_task
from run_learning_task import run_task
from generate_rego_policy import generate_rego_policy
from preprocess import get_requests_from_logs
from distance import compute_distances
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def clear_dir(path):
    for f in glob(f'{path}/*'):
        os.remove(f)

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

    max_attributes = 20
    window_size = 300

    differ = dl.HtmlDiff(tabsize=16)
    all_requests = get_requests_from_logs(f'{data_dir}/{data}')
    prev_time = None
    prev_requests = None
    for i, w in enumerate(range(0, len(all_requests), window_size)):
        log.info(f'Learning a policy for window {i}/{len(all_requests)//window_size}...')
        requests = all_requests[w:w+window_size]
        distances = compute_distances(prev_requests, requests) if prev_requests else None
        prev_requests = deepcopy(requests)
        task, body_cost = generate_learning_task(requests, distances, max_attributes)
        task_path = f'{tasks_dir}/{data_base}{i}.las'
        with open(task_path, 'w') as f:
            f.write(task)
        model, rule_confidences = run_task(task_path, body_cost)
        with open(f'{models_dir}/{data_base}{i}.lp', 'w') as f:
            f.write(model)
        new_policy = generate_rego_policy(rule_confidences, data_base)
        with open(f'{policies_dir}/{data_base}{i}.rego', 'w') as f:
            f.write(new_policy)
        if i > 0:
            with open(f'{policies_dir}/{data_base}{i-1}.rego', 'r') as f:
                prev_policy = f.readlines()
            with open(f'{diffs_dir}/{data_base}{i-1}_{i}.html', 'w') as f:
                now = datetime.now().strftime("%d/%m/%Y at %H:%M:%S")
                now_time = f'{data_base} on {now}'
                f.write(differ.make_file(prev_policy, new_policy.splitlines(True),
                                         fromdesc=prev_time, todesc=now_time))
                prev_time = now_time
