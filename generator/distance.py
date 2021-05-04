import pandas as pd
from preprocess import process_examples
from scipy.spatial import distance

def _tupleify_lists(x):
    if type(x) is dict:
        for k in x:
            if type(x[k]) is list:
                x[k] = tuple(x[k])
            else:
                _tupleify_lists(x[k])
    return x

def compute_distance(data, data_dir):
    # data = list(map(_tupleify_lists, map(restructure_request, get_requests_from_logs(f'{data_dir}/{data}'))))
    # data = process_requests_from_logs(f'{data_dir}/{data}', max_attributes=30, max_examples=5000)
    data = process_examples(f'{data_dir}/{data}', max_attributes=30, max_examples=5000)
    data = list(map(lambda r: r[0] | r[1], data))
    # Perturb input for testing
    # data[0]['destination']['portValue'] = 9090
    # data[0]['request']['method'] = 'POST'

    data = pd.DataFrame(data)
    print(data)
    print(data.columns)
    data = data.astype(str)
    data = pd.get_dummies(data)
    print(data)
    test, data = data[:1], data[1:]
    d = distance.cdist(test, data, 'euclidean')[0]
    d = d.min()
    print(f'min distance: {d}')

if __name__ == '__main__':
    data = 'synheart-controller-opa-istio.log'
    data_dir = '../data'
    compute_distance(data, data_dir)
window_size
