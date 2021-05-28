from datetime import datetime
from generate_learning_task import generate_learning_task
from run_learning_task import run_task
from generate_rego_policy import generate_rego_policy

def generate_policy(requests, distances, max_attributes, generalisation, name,
                    tasks_dir, models_dir, policies_dir, data_base, restructure):
    task, body_cost = generate_learning_task(requests, distances, max_attributes, generalisation, restructure)
    task_path = f'{tasks_dir}/{name}.las'
    with open(task_path, 'w') as f:
        f.write(task)
    model_path = f'{models_dir}/{name}.lp'
    model, rule_confidences = run_task(task_path, model_path, body_cost)
    with open(model_path, 'w') as f:
        f.write(model)
    new_policy, package = generate_rego_policy(rule_confidences, data_base)
    new_policy_path = f'{policies_dir}/{name}.rego'
    with open(new_policy_path, 'w') as f:
        f.write(new_policy)
    now = datetime.now().strftime("%d/%m/%Y at %H:%M:%S")
    now_time = f'{name} on {now}'
    return new_policy_path, now_time, package

def generate_policy_diff(new_policy_path, new_policy_time, curr_policy_path, curr_policy_time, i, differ,
                         name, policies_dir, diffs_dir):
    with open(new_policy_path, 'r') as f:
        new_policy = f.readlines()
    with open(curr_policy_path, 'r') as f:
        curr_policy = f.readlines()
    with open(f'{diffs_dir}/{name}.html', 'w') as f:
        f.write(differ.make_file(curr_policy, new_policy, fromdesc=curr_policy_time, todesc=new_policy_time))
