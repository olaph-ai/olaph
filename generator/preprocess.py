import json

def _cleanup_name(name):
    clean = name.split('_')
    if len(clean) <= 2:
        clean = '_'.join(clean)
    else:
        clean = '_'.join([clean[0], clean[-1]])
    return clean.replace(':', '').replace('-', '_').lower()

def flatten(request):
    example = {}
    def flatten_aux(x, name):
        if type(x) is dict:
            for k in x:
                flatten_aux(x[k], f'{name}{k}_')
        elif type(x) is list:
            for i, e in enumerate(x):
                flatten_aux(e, f'{name[:-1]}{i}_')
        else:
            example[_cleanup_name(name[:-1])] = x
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
    # request.pop('version', None)
    # request.pop('truncated_body', None)
    # request['attributes']['request'].pop('time', None)
    # request['attributes']['request']['http'].pop('id', None)
    # request['attributes']['request']['http']['headers'].pop('x-request-id', None)
    # request['attributes']['request']['http']['headers'].pop(':path', None)
    # request['attributes']['request']['http']['headers'].pop(':authority', None)
    # request['attributes']['request']['http']['headers'].pop('accept-encoding', None)
    # request['attributes']['request']['http']['headers'].pop('accept-language', None)

    # request['attributes']['request']['http'].pop('path', None)
    # request['attributes']['request']['http'].pop('protocol', None)
    # request['attributes']['source']['address']['socketAddress'].pop('portValue', None)
    # for key in list(request['attributes']['request']['http']['headers'].keys()):
    #     if 'x-' in key and key != 'x-forwarded-host' or 'sec-' in key:
    #         request['attributes']['request']['http']['headers'].pop(key, None)
    # req = request

    req = {}
    req['destination__address'] = request['attributes']['destination']['address']['socketAddress']['address']
    req['destination__portValue'] = request['attributes']['destination']['address']['socketAddress']['portValue']
    req['source__address'] = request['attributes']['source']['address']['socketAddress']['address']
    req['request__host'] = request['attributes']['request']['http']['host']
    req['request__method'] = request['attributes']['request']['http']['method']
    # req['request_origin'] = request['attributes']['request']['http']['headers']['origin']
    # req['request_referer'] = request['attributes']['request']['http']['headers']['referer']
    # req['request_user_agent'] = request['attributes']['request']['http']['headers']['user-agent']

    context = {}
    if d := request.pop('parsed_body', None):
        context['parsed_body'] = _dictify_lists(d)
    if d := request.pop('parsed_query', None):
        context['parsed_query'] = _dictify_lists(d)
    if l := request.pop('parsed_path', None):
        context['parsed_path'] = dict(enumerate(l))
    for k in [k for k in request if k != 'attributes']:
        request['attributes'][k] = request.pop(k)
    return req, context
    # return flatten(request['attributes']), context

def get_requests_from_logs(logs):
    return list(
        map(lambda d: _process_request(d['input']),
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
