from re import split, sub

def generate_rego_policy(model, data_base, policies_dir):
    not_in_quotes = '(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)'
    preamble = f"""package {data_base.replace('-', '')}

import input.attributes.source.address.socketAddress as source
import input.attributes.destination.address.socketAddress as destination
import input.attributes.request.http as request
import input.attributes.request.http.headers
import input.parsed_body
import input.parsed_path
import input.parsed_query

default allow = {{
    "allowed": false,
    "confidence": 1
}}
"""
    rego_policy = []
    for rule, confidence in model:
        rule = split(rf' :- {not_in_quotes}', rule)
        if len(rule) > 1:
            head, body = rule
        else:
            if rule[0]:
                rego_policy.append(f"""
{rule[0].strip()[:-1]} = {{
    \"allowed\": true,
    \"confidence\": {round(confidence, 2)}
}}
""")
            else:
                preamble = ''
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
                    if name == 'headers':
                        rego_atom.append(f'["{term}"]')
                    else:
                        rego_atom.append(f'.{term}')
            rego_atom.append(f' == {result}')
            rego_body.append(''.join(rego_atom))
        rego_body.sort()
        rego_body = '\n    '.join(rego_body)
        rego_policy.append(f"""
{head.strip()} = response {{
    {rego_body}
    response := {{
        \"allowed\": true,
        \"confidence\": {round(confidence, 4)}
    }}
}}
""")
    rego_policy.sort(reverse=True)
    rego_policy = preamble + '\n'.join(rego_policy)
    with open(f'{policies_dir}/{data_base}.rego', 'w') as f:
        f.write(rego_policy)
