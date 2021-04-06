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
            if rule_s[0]:
                rule_confidences.append(1)
            break
        num_body_atoms = len(split(rf', {not_in_quotes}', body[:-1]))
        rule_confidences.append((rule, p - 2 * body_cost(num_body_atoms)))
    total_confidence = sum([c for _, c in rule_confidences])
    model = [(r, (c / (total_confidence + 0.000000001))) for r, c in rule_confidences]
    return model
