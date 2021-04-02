import json

def flatten(request):
    example = {}
    def flatten_aux(x, name):
        if type(x) is dict:
            for k in x:
                flatten_aux(x[k], f'{name}{k}__')
        elif type(x) is list:
            for i, e in enumerate(x):
                flatten_aux(e, f'{name[:-1]}{i}__')
        else:
            example[name[:-2]] = x
    flatten_aux(request, '')
    return example

def _dictify_lists(x):
    if type(x) is dict:
        for k in x:
            if type(x[k]) is list:
                x[k] = dict(enumerate(x[k]))
            else:
                _dictify_lists(x[k])
    return x

def _process_request(request):
    # Remove attributes that change in every request
    request['input']['attributes']['request'].pop('time', None)
    request['input']['attributes']['request']['http'].pop('id', None)
    request['input']['attributes']['request']['http']['headers'].pop('x-request-id', None)
    request['input']['attributes']['source']['address']['socketAddress'].pop('portValue', None)

    context = {'input': {}}

    # Shortcuts for Rego imports
    request['source'] = request['input']['attributes']['source']['address'].pop('socketAddress')
    request['destination'] = request['input']['attributes']['destination']['address'].pop('socketAddress')
    context['headers'] = request['input']['attributes']['request']['http'].pop('headers')
    request['request'] = request['input']['attributes']['request'].pop('http')

    if d := request['input'].pop('parsed_body', None):
        context['parsed_body'] = _dictify_lists(d)
    if d := request['input'].pop('parsed_query', None):
        context['parsed_query'] = _dictify_lists(d)
    if l := request['input'].pop('parsed_path', None):
        context['parsed_path'] = dict(enumerate(l))
    return flatten(request), context

def get_requests_from_logs(logs):
    return list(
        map(lambda d: _process_request({'input': d['input']}),
            filter(lambda l: l['msg'] == 'Decision Log', logs)
        )
    )

def preprocess_data(path):
    logs = []
    with open(path, 'r') as f:
        for i, l in enumerate(f.readlines()):
            try:
                log = json.loads(l)
                if log not in logs:
                    logs.append(log)
            except:
                if l.replace('\n', ''):
                    print(f'error at {i}:\n{l}')
    with open(f'{path}.json', 'w') as f:
        f.write(json.dumps(logs, indent=4))
    return get_requests_from_logs(logs)
