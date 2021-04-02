from re import split, sub

def generate_rego_policy(model, data_base, policies_dir):
    not_in_quotes = '(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)'
    rego_policy = []
    for rule in split(rf'\n{not_in_quotes}', model):
        rule = split(rf' :- {not_in_quotes}', rule)
        if len(rule) > 1:
            head, body = rule
        else:
            rego_policy.append(f"\n{rule[0].strip()[:-1]} = true\n")
            break
        body_atoms = split(rf', {not_in_quotes}', body[:-1])
        rego_body = []
        for atom in body_atoms:
            s = split(rf'\({not_in_quotes}', atom)
            name, terms = sub(rf'\_\_{not_in_quotes}', '.', s[0]), split(rf',{not_in_quotes}', s[1][:-1])
            rego_atom = []
            rego_atom.append(name)
            terms, result = terms[:-1], terms[-1]
            for term in terms:
                if '"' not in term:
                    rego_atom.append(f'[{term}]')
                else:
                    term = term[1:][:-1]
                    rego_atom.append(f'.{term}')
            rego_atom.append(f' == {result}')
            rego_body.append(''.join(rego_atom))
        rego_body = '\n    '.join(rego_body)
        rego_policy.append(f"""
{head.strip()} {{
    {rego_body}
}}
""")
    rego_policy = f"""package {data_base.replace('-', '_')}

default allow = false
""" + '\n'.join(rego_policy)
    with open(f'{policies_dir}/{data_base}.rego', 'w') as f:
        f.write(rego_policy)
