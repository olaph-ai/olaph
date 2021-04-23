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

def restructure_request(request):
    # Remove attributes that change in every request
    request['input']['attributes']['request'].pop('time', None)
    request['input']['attributes']['request']['http'].pop('id', None)
    request['input']['attributes']['request']['http']['headers'].pop('x-request-id', None)
    request['input']['attributes']['source']['address']['socketAddress'].pop('portValue', None)

    # Shortcuts for Rego imports
    request['source'] = request['input']['attributes']['source']['address'].pop('socketAddress')
    request['destination'] = request['input']['attributes']['destination']['address'].pop('socketAddress')
    request['headers'] = request['input']['attributes']['request']['http'].pop('headers')
    request['request'] = request['input']['attributes']['request'].pop('http')

    if d := request['input'].pop('parsed_body', None):
        request['parsed_body'] = d
    if d := request['input'].pop('parsed_query', None):
        request['parsed_query'] = d
    if l := request['input'].pop('parsed_path', None):
        request['parsed_path'] = l
    return request

def _process_request(request):
    request = restructure_request(request)
    context = {}
    context['headers'] = request.pop('headers')

    if d := request.pop('parsed_body', None):
        context['parsed_body'] = _dictify_lists(d)
    if d := request.pop('parsed_query', None):
        context['parsed_query'] = _dictify_lists(d)
    if l := request.pop('parsed_path', None):
        context['parsed_path'] = dict(enumerate(l))
    return flatten(request), context

def get_requests_from_logs(path):
    logs = _get_logs(path)
    return list(
        map(lambda d: {'input': d['input']},
            filter(lambda l: l['msg'] == 'Decision Log', logs)
        )
    )

def _get_logs(path):
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
    return logs

def preprocess_data(path):
    return list(map(_process_request, get_requests_from_logs(path)))
