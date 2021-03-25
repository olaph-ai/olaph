def generate_rego_policy(model, data_base, policies_dir):
    rego_policy = []
    for rule in model.split('\n'):
        head, body = rule.split(' :- ', 1)
        rego_body = body[:-1].replace(', ', '\n    ')
        rego_policy.append(f"""
{head} {{
    {rego_body}
}}
""")
    rego_policy = f"""package {data_base.replace('-', '_')}

default allow = false
""" + '\n'.join(rego_policy)
    with open(f'{policies_dir}/{data_base}.rego', 'w') as f:
        f.write(rego_policy)
