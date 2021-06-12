from functools import reduce
from preprocess import preprocess_data
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def _escape_str_bias(s):
    return s.replace("'", "\\'")

def _example_to_las(atoms):
    return '\n'.join([f'  {k}({", ".join(terms)}).' for k, terms in atoms])

def _normalise_distance(distances, lower, upper, dmin, dmax, i):
    dnorm = lower + (((distances[i] - dmin) * (upper - lower)) / (dmax - dmin))
    return int(dnorm)

def get_example_penalty(distances, i):
    if distances[i] is not None:
        distances_filtered = list(filter(lambda d: d is not None, distances))
        dmin = min(distances_filtered)
        dmax = max(distances_filtered)
        if dmin == dmax:
            return ''
        else:
            distance = _normalise_distance(distances, 10, 100, dmin, dmax, i)
            return f'@{distance}'
    else:
        return ''

def examples_to_las(examples, distances):
    return '\n'.join([f'''
#pos(eg(id{i}){get_example_penalty(distances, i)}, {{allow}}, {{}}, {{
{_example_to_las(atoms)}
}}).''' for i, atoms in enumerate(examples)])

def _example_to_bias(i, atoms):
    return '\n'.join([f"#bias('user(eg(id{i}), {k}({', '.join(map(_escape_str_bias, terms))})).')."
                      for k, terms in atoms])

def examples_to_bias(examples):
    return '\n'.join([f"#bias('user(eg(id{i})).').\n{_example_to_bias(i, atoms)}"
                      for i, atoms in enumerate(examples)])

def generate_mode_bias(atoms, g, k_param, c, required_attrs, variables_in_bias, examples_in_bias):
    mode_bias = []
    for k, terms in reduce(lambda a, b: a + b, atoms):
        combs = [['const'] * len(terms)]
        if variables_in_bias:
            combs.append((['const'] * (len(terms)-1)) + ['var'])
        for comb in combs:
            placeholders = ', '.join([f'{t}({k})' for t in comb])
            mb = f'#modeb(1, {k}({placeholders})).'
            if mb not in mode_bias:
                mode_bias.append(mb)
        for term in terms:
            mb = f'#constant({k}, {term}).'
            if mb not in mode_bias:
                mode_bias.append(mb)
    max_body = max(map(len, atoms))
    min_body = min(map(len, atoms))
    normalised = int(0.5 * (max_body + min_body))
    avg_min_max_body_literals = min(int(g), normalised)
    if examples_in_bias:
        mode_bias.append(examples_to_bias(atoms))
    body_lits_cost = lambda n: (abs((n - avg_min_max_body_literals)**k_param) + c, len(atoms))
    if variables_in_bias:
        mode_bias.append('\n#maxv(1).')
    if required_attrs:
        required_atoms = '#bias("penalty(1, body(X)) :- in_body(X), not required(X).").\n'
        required_atoms += '\n'.join([f"#bias('required(X) :- in_body(X), X = {ra}.')." for ra in required_attrs])
    else:
        required_atoms = ''
    mode_bias.append(f'''
#modeh(allow).

#bias("penalty(|(N - {avg_min_max_body_literals})**{k_param}| + {c}, rule) :- N = #count{{X: in_body(X)}}.").

% Prefer rules that cover fewer examples
#bias("n(U) :- user(U), not user(U, BodyLit), in_body(BodyLit).").
#bias("penalty(1, U) :- user(U), not n(U).").
{required_atoms}
''')
    return '\n'.join(mode_bias), body_lits_cost


def generate_learning_task(requests, distances, max_attributes, g, k, c, required_attrs, restructure):
    example_atoms = preprocess_data(requests, max_attributes, restructure)
    las_examples = examples_to_las(example_atoms, distances)
    las_mode_bias, body_cost = generate_mode_bias(example_atoms, g, k, c, required_attrs,
                                                  variables_in_bias=False, examples_in_bias=True)
    task = las_examples + '\n\n' + las_mode_bias
    return task, body_cost
