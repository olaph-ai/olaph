from subprocess import run
from re import split

def run_task(task, body_cost, data_base, models_dir):
    not_in_quotes = '(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)'
    out = run(['FastLAS', '--d', task], capture_output=True)
    debug = out.stdout.decode().strip()
    prev, rest = split(fr'Solving...{not_in_quotes}', debug)
    model = split(fr'{{{not_in_quotes}', rest)[0].strip()
    model_path = f'{models_dir}/{data_base}.lp'
    with open(model_path, 'w') as f:
        f.write(model)
    # Calculate confidence
    rules = model.split('\n')
    S_Ms = split(fr'S_M:{not_in_quotes}', prev)[1].strip().split('\n')
    S_Ms = [split(fr' ~ {not_in_quotes}', S_M) for S_M in S_Ms]
    rule_penalties = [(r, int(p)) for p, r in S_Ms if r in rules]
    rule_confidences = []
    for rule, p in rule_penalties:
        rule_s = split(rf' :- {not_in_quotes}', rule)
        if len(rule_s) > 1:
            head, body = rule_s
        else:
            rule_confidences.append((rule, 1))
            continue
        num_body_atoms = len(split(rf', {not_in_quotes}', body[:-1]))
        bc, num_examples, max_body_literals = body_cost(num_body_atoms)
        num_covered_examples = p - bc
        bc_rescaled = (bc * num_examples) / (max_body_literals ** 2)
        bc_conf = num_examples - bc_rescaled
        rule_confidences.append((rule, (0.5 * num_covered_examples + 0.5 * bc_conf)/num_examples))
    return rule_confidences
