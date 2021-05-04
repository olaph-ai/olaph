from functools import reduce

from preprocess import preprocess_data

def _escape_str_bias(s):
    return s.replace("'", "\\'")

def _example_to_las(atoms):
    return '\n'.join([f'  {k}({", ".join(terms)}).' for k, terms in atoms])

def examples_to_las(examples):
    return '\n'.join([f'''
#pos(eg(id{i}), {{allow}}, {{}}, {{
{_example_to_las(atoms)}
}}).''' for i, atoms in enumerate(examples)])

def _example_to_bias(i, atoms):
    return '\n'.join([f"#bias('user(eg(id{i}), {k}({', '.join(map(_escape_str_bias, terms))})).')."
                      for k, terms in atoms])

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
    body_lits_cost = lambda n: ((n - max_body_literals)**2 + 1, len(atoms))
    if variables_in_bias:
        mode_bias.append('\n#maxv(1).')
    mode_bias.append(f'''
#modeh(allow).

% Prefer rules with the maximum number of body literals in the examples
% Add 1 to encourage learning fewer rules
#bias("penalty((N - {max_body_literals})**2 + 1, rule) :- N = #count{{X: in_body(X)}}.").

% Prefer rules that cover fewer examples
#bias("n(U) :- user(U), not user(U, BodyLit), in_body(BodyLit).").
#bias("penalty(1, U) :- user(U), not n(U).").
''')
    return '\n'.join(mode_bias), body_lits_cost


def generate_learning_task(data, data_base, data_dir, tasks_dir, max_attributes, max_examples):
    example_atoms = preprocess_data(f'{data_dir}/{data}', max_attributes, max_examples)
    las_examples = examples_to_las(example_atoms)
    las_mode_bias, body_cost = generate_mode_bias(example_atoms, variables_in_bias=False, examples_in_bias=True)
    task_path = f'{tasks_dir}/{data_base}.las'
    with open(task_path, 'w') as f:
        f.write(las_examples + '\n\n' + las_mode_bias)
    return task_path, body_cost
