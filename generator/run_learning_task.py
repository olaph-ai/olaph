from subprocess import run

def run_task(task, data_base, models_dir):
    out = run(['FastLAS', task], capture_output=True)
    model = out.stdout.decode().strip()
    model_path = f'{models_dir}/{data_base}.lp'
    with open(model_path, 'w') as f:
        f.write(model)
    return model
