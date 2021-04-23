import json
from itertools import groupby, product
from functools import reduce
from re import escape

from preprocess import preprocess_data

def _escape_str(s):
    return s.replace("\\", "").replace('"', '\\"')

def _val_to_las(v):
    return str(v).lower() if isinstance(v, int) else f'"{_escape_str(str(v))}"'

def _context_to_atoms(ck, context):
    atoms = []
    def ctl_aux(x, keys):
        if type(x) is dict:
            for k in x:
                ctl_aux(x[k], keys + [_val_to_las(k)])
        elif type(x) is list:
            for i, e in enumerate(x):
                ctl_aux(e, keys + [i])
        else:
            atoms.append((ck, keys + [_val_to_las(x)]))
    ctl_aux(context, [])
    return atoms

def example_to_atoms(example):
    access, context = example
    atoms = []
    for k, v in access.items():
        atoms.append((k, [_val_to_las(v)]))
    for k, v in context.items():
        atoms.extend(_context_to_atoms(k, v))
    return atoms

def _example_to_las(atoms):
    return '\n'.join([f'  {k}({", ".join(terms)}).' for k, terms in atoms])

def examples_to_las(examples):
    return '\n'.join([f'''
#pos(eg(id{i}), {{allow}}, {{}}, {{
{_example_to_las(atoms)}
}}).''' for i, atoms in enumerate(examples)])

def _example_to_bias(i, atoms):
    return '\n'.join([f"#bias('user(eg(id{i}), {k}({', '.join(terms)})).')." for k, terms in atoms])

def examples_to_bias(examples):
    return '\n'.join([f"#bias('user(eg(id{i})).').\n{_example_to_bias(i, atoms)}"
                      for i, atoms in enumerate(examples)])

def generate_mode_bias(atoms, variables_in_bias, examples_in_bias):
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
    max_body_literals = max(map(len, atoms))
    if examples_in_bias:
        mode_bias.append(examples_to_bias(atoms))
    body_lits_cost = lambda n: ((n - max_body_literals)**2, len(atoms))
    if variables_in_bias:
        mode_bias.append('\n#maxv(1).')
    mode_bias.append(f'''
#modeh(allow).

% Prefer rules with the maximum number of body literals in the examples
#bias("penalty((N - {max_body_literals})**2, rule) :- N = #count{{X: in_body(X)}}.").

% Prefer rules that cover fewer examples
#bias("n(U) :- user(U), not user(U, BodyLit), in_body(BodyLit).").
#bias("penalty(1, U) :- user(U), not n(U).").
''')
    return '\n'.join(mode_bias), body_lits_cost


def generate_learning_task(data, data_base, data_dir, tasks_dir):
    access_examples = preprocess_data(f'{data_dir}/{data}')
    example_atoms = list(map(example_to_atoms, access_examples))
    las_examples = examples_to_las(example_atoms)
    las_mode_bias, body_cost = generate_mode_bias(example_atoms, variables_in_bias=False, examples_in_bias=True)
    task_path = f'{tasks_dir}/{data_base}.las'
    with open(task_path, 'w') as f:
        f.write(las_examples + '\n\n' + las_mode_bias)
    return task_path, body_cost
