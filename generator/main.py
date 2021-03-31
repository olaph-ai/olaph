import os
from generate_learning_task import generate_learning_task
from run_learning_task import run_task
from generate_rego_policy import generate_rego_policy

if __name__ == '__main__':
    data = os.getenv('DATA', 'synheart-controller-opa-istio.log')
    data_base = data.rsplit('.', 1)[0]

    data_dir = os.getenv('DATA_DIR', '../data')
    tasks_dir = os.getenv('TASKS_DIR', '../tasks')
    models_dir = os.getenv('MODELS_DIR', '../models')
    policies_dir = os.getenv('POLICIES_DIR', '../policies')

    num_body_attributes = os.getenv('N_BODY_ATTRIBUTES') or 4

    task_path = generate_learning_task(data, data_base, data_dir, tasks_dir, num_body_attributes)
    model = run_task(task_path, data_base, models_dir)
    generate_rego_policy(model, data_base, policies_dir)
