import json
import gzip
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def flatten(request):
    example = {}
    def flatten_aux(x, name):
        if type(x) is dict:
            for k in x:
                flatten_aux(x[k], name + (k,))
        elif type(x) is list:
            for i, e in enumerate(x):
                flatten_aux(e, name + (i,))
        else:
            example[name] = x
    flatten_aux(request, tuple())
    return example

def _restructure_request(request):
    # Remove redundant attributes
    request['input']['attributes']['request'].pop('time')
    request['input']['attributes']['request']['http'].pop('id')
    request['input']['attributes']['request']['http'].pop('path')
    request['input'].pop('truncated_body')
    request['input'].pop('version')
    # Shortcuts for Rego imports
    request['source'] = request['input']['attributes']['source']['address'].pop('socketAddress')
    request['destination'] = request['input']['attributes']['destination']['address'].pop('socketAddress')
    request['headers'] = request['input']['attributes']['request']['http'].pop('headers')
    request['request'] = request['input']['attributes']['request'].pop('http')

    user_input = {}
    user_input['headers'] = request.pop('headers')
    if d := request['input'].pop('parsed_body', None):
        user_input['parsed_body'] = d
    if d := request['input'].pop('parsed_query', None):
        user_input['parsed_query'] = d
    if l := request['input'].pop('parsed_path', None):
        user_input['parsed_path'] = l
    return flatten(request), flatten(user_input)

def get_requests_from_logs(path, restructure):
    logs = _get_logs(path, restructure)
    if restructure:
        return list(map(lambda d: {'input': d['input']},
                        list(filter(lambda l: l['msg'] == 'Decision Log', logs))
                        ))
    else:
        return logs

def _get_logs(path, restructure):
    logs = []
    with open(path, 'r') if restructure else gzip.open(path, 'rb') as f:
        for i, l in enumerate(f.readlines()):
            try:
                log = json.loads(l)
                logs.append(log)
            except:
                if l.replace('\n', ''):
                    print(f'error at {i}:\n{l}')
    return logs

def _escape_str(s):
    return s.replace("\\", "").replace('"', '\\"')

def _val_to_las(v):
    return str(v).lower() if isinstance(v, int) else f'"{_escape_str(str(v))}"'

def _user_val_to_las(v):
    return f'"{str(v).lower()}"' if isinstance(v, int) else f'"{_escape_str(str(v))}"'

def example_to_atoms(example):
    request, user_input = example
    atoms = []
    for k, v in request.items():
        atoms.append(('__'.join(k), (_val_to_las(v),)))
    for k, v in user_input.items():
        atoms.append((k[0], tuple(map(_val_to_las, k[1:])) + (_user_val_to_las(v),)))
    return atoms

def _select_features(ds, max_attributes):
    data = pd.DataFrame(ds)
    encoded = pd.DataFrame(OrdinalEncoder().fit_transform(data.astype(str)), columns=data.columns)
    heavy_tailedness = encoded.kurtosis().sort_values(ascending=False)
    chosen_attributes = sorted(heavy_tailedness.index.sort_values().to_list())[:max_attributes]
    return list(map(lambda d: {k: v for k, v in d.items() if k in chosen_attributes}, ds)), len(chosen_attributes)

def select_features(data, max_attributes):
    requests, user_input = zip(*data)
    requests, n_features = _select_features(requests, max_attributes)
    user_input, _ = _select_features(user_input, max_attributes - n_features)
    return zip(requests, user_input)

def process_examples(requests, max_attributes, restructure):
    if restructure:
        processed = list(map(_restructure_request, requests))
    else:
        processed = list(map(lambda r: (dict(), {(k[0].lower(),) + k[1:]: v for k, v in flatten(r).items()}), requests))
    return select_features(processed, max_attributes)

def preprocess_data(requests, max_attributes, restructure):
    examples = process_examples(requests, max_attributes, restructure)
    return list(map(example_to_atoms, examples))
