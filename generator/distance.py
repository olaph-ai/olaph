import pandas as pd
from preprocess import get_requests_from_logs, restructure_request
from scipy.spatial import distance

# Same as policy
def discrete_distance(r, T):
    return int(r not in T)

def num_attributes_distance(r, T):
    return min([abs(len(r) - len(t)) for t in T])


def _tupleify_lists(x):
    if type(x) is dict:
        for k in x:
            if type(x[k]) is list:
                x[k] = tuple(x[k])
            else:
                _tupleify_lists(x[k])
    return x

def compute_distance(data, data_dir):
    data = list(map(_tupleify_lists, map(restructure_request, get_requests_from_logs(f'{data_dir}/{data}'))))
    # data[0]['destination']['portValue'] = 9090
    # data[0]['request']['method'] = 'POST'
    data = pd.json_normalize(data)
    data = data.astype(str)
    data = pd.get_dummies(data)
    test, data = data[:1], data[1:]
    d = distance.cdist(test, data, 'euclidean')
    print(f'min distance: {d[0].min()}')

if __name__ == '__main__':
    data = 'synheart-controller-opa-istio.log'
    data_dir = '../data'
    compute_distance(data, data_dir)
