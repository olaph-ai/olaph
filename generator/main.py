import os
import logging
import difflib as dl
from generate_learning_task import generate_learning_task
from run_learning_task import run_task
from generate_rego_policy import generate_rego_policy
from preprocess import get_requests_from_logs

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    data = os.getenv('DATA', 'synheart-controller-opa-istio.log')
    data_base = data.rsplit('.', 1)[0]

    data_dir = os.getenv('DATA_DIR', '../data')
    tasks_dir = os.getenv('TASKS_DIR', '../tasks')
    models_dir = os.getenv('MODELS_DIR', '../models')
    policies_dir = os.getenv('POLICIES_DIR', '../policies')
    diffs_dir = os.getenv('DIFFS_DIR', '../diffs')

    max_attributes = 10
    window_size = 300

    all_requests = get_requests_from_logs(f'{data_dir}/{data}')
    for i, w in enumerate(range(0, len(all_requests), window_size)):
        log.info(f'Processing window {i}/{len(all_requests)//window_size}...')
        requests = all_requests[w:w+window_size]
        task, body_cost = generate_learning_task(requests, max_attributes)
        task_path = f'{tasks_dir}/{data_base}{i}.las'
        with open(task_path, 'w') as f:
            f.write(task)
        model, rule_confidences = run_task(task_path, body_cost)
        model_path = f'{models_dir}/{data_base}{i}.lp'
        with open(model_path, 'w') as f:
            f.write(model)
        new_policy = generate_rego_policy(rule_confidences, data_base)
        with open(f'{policies_dir}/{data_base}{i}.rego', 'w') as f:
            f.write(new_policy)
        if i > 0:
            with open(f'{policies_dir}/{data_base}{i-1}.rego', 'r') as f:
                prev_policy = f.readlines()
            with open(f'{diffs_dir}/{data_base}{i-1}_{i}.diff', 'w') as f:
                f.write('\n'.join([diff for diff in dl.unified_diff(prev_policy, new_policy.splitlines(True))]))
